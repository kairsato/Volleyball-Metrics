import json
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

GAME_STATUS_LOG_NAME = "game_status.json"

# Fine-tuned checkpoint from the open-source volleyball_analytics project
# (github.com/masouduut94/volleyball_analytics), base model
# MCG-NJU/videomae-base-finetuned-kinetics (Hugging Face, CC-BY-NC-4.0 -
# non-commercial use only). Classifies a 16-frame clip into no-play / play /
# service - see checkpoint/config.json's id2label. This is now the sole
# source of truth for rally segmentation; the older ball-speed heuristic
# (inferring "live" from how fast the tracked ball was moving) has been
# retired in its favour.
MODEL_DIR = Path(__file__).resolve().parent / "Models" / "VolleyballAnalytics" / "3-states" / "checkpoint"

LIVE_LABELS = {"play", "service"}

# The checkpoint's own config.json fixes num_frames=16 (tubelet_size=2) -
# not tunable without retraining. WINDOW_FRAMES is how much raw video each
# inference call looks at before subsampling down to those 16 frames (~2.1s
# at 30fps - the stride-4-over-64-frames convention VideoMAE's Kinetics
# finetunes are typically trained with). STRIDE_FRAMES is how far the
# window advances between inferences: the label granularity of the
# resulting timeline, and how much window-to-window overlap there is for
# smoother rally boundaries (75% overlap at these settings).
CLIP_NUM_FRAMES = 16
WINDOW_FRAMES = 64
STRIDE_FRAMES = 16

# Same merge/drop tolerances the old ball-speed heuristic used - a brief
# no-play blip (a misclassified window, a lull mid-rally) shouldn't
# fragment one continuous rally, and a "rally" shorter than this is almost
# certainly a misclassification rather than real play.
MAX_GAP_SECONDS_WITHIN_RALLY = 1.0
MIN_RALLY_DURATION_SECONDS = 1.0

# Frames are buffered at this size (shortest edge) rather than full
# resolution - the model only ever looks at 224x224 crops anyway, and
# holding WINDOW_FRAMES of full-resolution video in memory would burn
# hundreds of MB for no benefit.
BUFFER_SHORTEST_EDGE = 256


def _patch_legacy_attention_bias(model, checkpoint_dir: Path):
    """The checkpoint was saved by an older transformers version whose
    VideoMAE attention module used separate `q_bias`/`v_bias` parameters
    (BEiT-style: query and value each get a learnable bias, key doesn't)
    instead of a `.bias` on the query/key/value Linear layers themselves.
    The installed transformers version expects the latter, so
    from_pretrained silently leaves query.bias/key.bias/value.bias
    randomly initialised and drops q_bias/v_bias on the floor without
    raising anything - verified via its own load report, which lists
    q_bias/v_bias as "unexpected" and query.bias/key.bias/value.bias as
    "missing". Left unpatched, every attention layer runs on random
    biases instead of the fine-tuned ones, which is not "using the
    downloaded model" in any meaningful sense even though it loads
    without error and produces (wrong) predictions. This copies the real
    trained values across; key's bias is correctly left at zero, matching
    the original architecture rather than being randomly initialised.
    """
    state_dict = load_file(str(checkpoint_dir / "model.safetensors"))

    for i, layer in enumerate(model.videomae.encoder.layer):
        prefix = f"videomae.encoder.layer.{i}.attention.attention."
        q_bias = state_dict.get(prefix + "q_bias")
        v_bias = state_dict.get(prefix + "v_bias")
        if q_bias is None or v_bias is None:
            raise RuntimeError(
                f"Expected legacy q_bias/v_bias for encoder layer {i} in {checkpoint_dir} - "
                "checkpoint format may have changed, this patch needs re-checking."
            )

        layer.attention.attention.query.bias.data.copy_(q_bias)
        layer.attention.attention.value.bias.data.copy_(v_bias)
        layer.attention.attention.key.bias.data.zero_()


def _resize_shortest_edge(frame_bgr, target):
    h, w = frame_bgr.shape[:2]
    if min(h, w) <= target:
        return frame_bgr
    scale = target / min(h, w)
    return cv2.resize(frame_bgr, (round(w * scale), round(h * scale)))


def _classify_window(model, processor, device, frames_bgr):
    """frames_bgr: a buffered window of BGR frames, oldest first. Returns
    the predicted label string for the whole window."""
    indices = np.linspace(0, len(frames_bgr) - 1, CLIP_NUM_FRAMES).round().astype(int)
    clip_rgb = [cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2RGB) for i in indices]

    inputs = processor(list(clip_rgb), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits

    predicted = int(logits.argmax(dim=-1).item())
    return model.config.id2label[predicted]


def detectGameStatus(video_path, output_path):
    """
    Segments the video into rallies (ball actively in play) vs. dead-ball
    stretches using the fine-tuned VideoMAE game-state classifier in
    MODEL_DIR: a sliding window of raw video is classified as no-play, play
    or service every STRIDE_FRAMES frames, and "play"/"service" chunks
    within MAX_GAP_SECONDS_WITHIN_RALLY of each other are merged into one
    rally.
    """
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Game-state model checkpoint not found at {MODEL_DIR} - game status "
            "detection requires it and has no other detection path to fall back to."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = VideoMAEImageProcessor.from_pretrained(str(MODEL_DIR))
    model = VideoMAEForVideoClassification.from_pretrained(str(MODEL_DIR))
    _patch_legacy_attention_bias(model, MODEL_DIR)
    model = model.to(device)
    model.eval()

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # chunk_labels[i] is the label for frames
    # [i * STRIDE_FRAMES, min((i + 1) * STRIDE_FRAMES, total_frames) - 1].
    chunk_labels = []
    buffer = []
    frame_idx = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        buffer.append(_resize_shortest_edge(frame, BUFFER_SHORTEST_EDGE))
        if len(buffer) > WINDOW_FRAMES:
            buffer.pop(0)

        frame_idx += 1
        if frame_idx % STRIDE_FRAMES == 0:
            chunk_labels.append(_classify_window(model, processor, device, buffer))

        if frame_idx % (STRIDE_FRAMES * 50) == 0:
            print(f"game status - frame {frame_idx}/{total_frames}: {len(chunk_labels)} chunks classified")

    cap.release()

    # Anything left over (video length not a multiple of STRIDE_FRAMES, or
    # shorter than one full window) still needs a label - classify whatever
    # was buffered so no trailing frames are silently dropped.
    if frame_idx % STRIDE_FRAMES != 0 and buffer:
        chunk_labels.append(_classify_window(model, processor, device, buffer))

    rally_log = []
    if chunk_labels:
        is_live = [label in LIVE_LABELS for label in chunk_labels]

        max_gap_chunks = max(1, round(MAX_GAP_SECONDS_WITHIN_RALLY * fps / STRIDE_FRAMES))
        min_rally_chunks = max(1, round(MIN_RALLY_DURATION_SECONDS * fps / STRIDE_FRAMES))

        live_chunk_indices = [i for i, live in enumerate(is_live) if live]

        rallies = []
        if live_chunk_indices:
            start = live_chunk_indices[0]
            prev = live_chunk_indices[0]
            for idx in live_chunk_indices[1:]:
                if idx - prev > max_gap_chunks:
                    rallies.append((start, prev))
                    start = idx
                prev = idx
            rallies.append((start, prev))

        rallies = [(s, e) for s, e in rallies if (e - s + 1) >= min_rally_chunks]

        for i, (start_chunk, end_chunk) in enumerate(rallies):
            start_frame = start_chunk * STRIDE_FRAMES
            end_frame = min((end_chunk + 1) * STRIDE_FRAMES, total_frames) - 1
            rally_log.append({
                "rally_index": i,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_s": start_frame / fps,
                "end_time_s": end_frame / fps,
                "duration_s": (end_frame - start_frame) / fps,
            })

    status_log_file = Path(output_path) / GAME_STATUS_LOG_NAME
    with open(status_log_file, "w") as f:
        json.dump({
            "fps": fps,
            "total_frames": total_frames,
            "rallies": rally_log,
        }, f, indent=2)

    print(f"Detected {len(rally_log)} rally segment(s) via {MODEL_DIR.parent.name} game-state model.")
    print(f"Game status saved: {status_log_file}")
