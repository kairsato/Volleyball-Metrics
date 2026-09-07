import argparse
import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import cv2
from PIL import Image, ImageTk

GAME_STATUS_LABELS_NAME = "game_status_labels.json"

STATE_KEYS = {"1": "play", "2": "no-play", "3": "service"}
STATE_COLOR = {
    "play": "#00aa00",
    "no-play": "#8c8c8c",
    "service": "#ff8c00",
}
UNLABELED_COLOR = "#323232"
BREAKPOINT_COLOR = "#ffffff"
BREAKPOINT_DRAG_COLOR = "#00ffff"
PLAYHEAD_COLOR = "#ff3b3b"

BG = "#1e1e1e"
BG_RAISED = "#2b2b2b"
FG = "#e0e0e0"
FG_MUTED = "#9a9a9a"
BORDER = "#3f3f3f"
SELECT_BG = "#3a6ea5"

TIMELINE_HEIGHT = 50
HANDLE_RADIUS = 6
DRAG_TOLERANCE_PX = 8
DISPLAY_MAX_WIDTH = 1632

SPEED_OPTIONS = [0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 8.0]
ZOOM_MIN = 1.0
ZOOM_MAX = 30.0


def _load_breakpoints(labels_file):
    if not labels_file.exists():
        return []
    with open(labels_file) as f:
        data = json.load(f)
    return sorted((seg["start_frame"], seg["state"]) for seg in data.get("segments", []))


def _state_at(breakpoints, frame_idx):
    state = None
    for bp_frame, bp_state in breakpoints:
        if bp_frame <= frame_idx:
            state = bp_state
        else:
            break
    return state


def _upsert_breakpoint(breakpoints, frame_idx, state):
    breakpoints = [bp for bp in breakpoints if bp[0] != frame_idx]
    breakpoints.append((frame_idx, state))
    breakpoints.sort(key=lambda bp: bp[0])
    return breakpoints


def _remove_breakpoint(breakpoints, frame_idx):
    return [bp for bp in breakpoints if bp[0] != frame_idx]


def _nearest_breakpoint(breakpoints, frame_idx, max_distance_frames):
    if not breakpoints:
        return None
    nearest = min(breakpoints, key=lambda bp: abs(bp[0] - frame_idx))
    return nearest[0] if abs(nearest[0] - frame_idx) <= max_distance_frames else None


def _frame_to_x(frame_idx, view_start, visible_frames, width):
    frac = (frame_idx - view_start) / max(1, visible_frames - 1)
    return int(frac * width)


def _x_to_frame(x, view_start, visible_frames, width):
    frac = x / max(1, width)
    frame = view_start + frac * (visible_frames - 1)
    return max(view_start, min(view_start + visible_frames - 1, round(frame)))


def _export(breakpoints, total_frames, fps, output_path):
    segments = []
    bounds = [bp[0] for bp in breakpoints[1:]] + [total_frames]
    for (start_frame, state), next_start in zip(breakpoints, bounds):
        end_frame = next_start - 1
        segments.append({
            "segment_index": len(segments),
            "state": state,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time_s": start_frame / fps,
            "end_time_s": (end_frame + 1) / fps,
            "duration_s": (end_frame + 1 - start_frame) / fps,
        })

    labels_file = Path(output_path) / GAME_STATUS_LABELS_NAME
    with open(labels_file, "w") as f:
        json.dump({"fps": fps, "total_frames": total_frames, "segments": segments}, f, indent=2)
    print(f"Labels exported: {labels_file}")
    return labels_file


class LabelApp:
    """Real, separate tkinter widgets (video label, a slider for the video
    scrubber, native buttons, a dedicated canvas for the label timeline)
    instead of one drawn-on image with manually hit-tested regions."""

    def __init__(self, video_path, output_path):
        self.output_path = output_path
        self.labels_file = Path(output_path) / GAME_STATUS_LABELS_NAME
        self.breakpoints = _load_breakpoints(self.labels_file)

        self.cap = cv2.VideoCapture(str(video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        self.frame_idx = 0
        self.paused = True
        self.dragging = None
        self.photo_image = None
        self.state_buttons = {}
        self.speed = 1.0
        self.zoom = ZOOM_MIN
        self.visible_frames = self.total_frames
        self.view_start = 0
        self._decoder_pos = -1

        self.root = tk.Tk()
        self.root.title("Label play / no-play / service")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._apply_dark_theme()
        self._build_ui()
        self.root.focus_set()
        self._show_frame(0)
        self._apply_view()
        self._refresh_controls()
        self._tick()

    def _apply_dark_theme(self):
        self.root.configure(bg=BG)
        self.root.option_add("*Menu.background", BG_RAISED)
        self.root.option_add("*Menu.foreground", FG)
        self.root.option_add("*Menu.activeBackground", SELECT_BG)
        self.root.option_add("*Menu.activeForeground", FG)

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background=BG, foreground=FG, fieldbackground=BG_RAISED,
                         bordercolor=BORDER, darkcolor=BG_RAISED, lightcolor=BG_RAISED,
                         troughcolor=BG_RAISED)
        style.map(".", background=[("active", BG_RAISED)], foreground=[("disabled", FG_MUTED)])

        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG)
        style.configure("TLabelframe", background=BG, foreground=FG, bordercolor=BORDER)
        style.configure("TLabelframe.Label", background=BG, foreground=FG)
        style.configure("TButton", background=BG_RAISED, foreground=FG, bordercolor=BORDER)
        style.map("TButton", background=[("active", SELECT_BG), ("pressed", SELECT_BG)])
        style.configure("TSeparator", background=BORDER)
        style.configure("TScale", background=BG, troughcolor=BG_RAISED)
        style.configure("Horizontal.TScrollbar", background=BG_RAISED, troughcolor=BG,
                         bordercolor=BORDER, arrowcolor=FG)
        style.configure("TCombobox", fieldbackground=BG_RAISED, background=BG_RAISED, foreground=FG,
                         arrowcolor=FG, selectbackground=BG_RAISED, selectforeground=FG)
        style.map("TCombobox", fieldbackground=[("readonly", BG_RAISED)],
                   selectbackground=[("readonly", BG_RAISED)], selectforeground=[("readonly", FG)])
        self.root.option_add("*TCombobox*Listbox.background", BG_RAISED)
        self.root.option_add("*TCombobox*Listbox.foreground", FG)
        self.root.option_add("*TCombobox*Listbox.selectBackground", SELECT_BG)

    def _build_ui(self):
        self.video_label = tk.Label(self.root, bd=2, relief="groove", bg=BG_RAISED)
        self.video_label.pack(padx=8, pady=(8, 4))

        self.status_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.status_var, anchor="w", bg=BG, fg=FG).pack(fill="x", padx=8)

        scrubber_frame = ttk.LabelFrame(self.root, text="Video position")
        scrubber_frame.pack(fill="x", padx=8, pady=4)
        self.scrubber_var = tk.DoubleVar(value=0)
        self.scrubber = ttk.Scale(scrubber_frame, from_=0, to=max(0, self.total_frames - 1),
                                   orient="horizontal", variable=self.scrubber_var, command=self._on_scrub)
        self.scrubber.pack(fill="x", padx=6, pady=6)

        transport = ttk.Frame(self.root)
        transport.pack(pady=4)
        ttk.Button(transport, text="<< Prev", command=self._prev_frame).pack(side="left", padx=4)
        self.play_button = ttk.Button(transport, text="Play", command=self._toggle_play)
        self.play_button.pack(side="left", padx=4)
        ttk.Button(transport, text="Next >>", command=self._next_frame).pack(side="left", padx=4)

        ttk.Label(transport, text="  Speed:").pack(side="left")
        self.speed_var = tk.StringVar(value="1.0x")
        speed_box = ttk.Combobox(transport, textvariable=self.speed_var, width=5, state="readonly",
                                  values=[f"{s}x" for s in SPEED_OPTIONS])
        speed_box.pack(side="left", padx=4)
        speed_box.bind("<<ComboboxSelected>>", self._on_speed_change)

        zoom_frame = ttk.Frame(self.root)
        zoom_frame.pack(fill="x", padx=8, pady=(4, 0))
        ttk.Label(zoom_frame, text="Zoom:").pack(side="left")
        self.zoom_var = tk.DoubleVar(value=ZOOM_MIN)
        self.zoom_scale = ttk.Scale(zoom_frame, from_=ZOOM_MIN, to=ZOOM_MAX, orient="horizontal",
                                     variable=self.zoom_var, command=self._on_zoom_change)
        self.zoom_scale.pack(side="left", fill="x", expand=True, padx=6)
        self.zoom_label = ttk.Label(zoom_frame, text="1.0x", width=6)
        self.zoom_label.pack(side="left")

        timeline_frame = ttk.LabelFrame(self.root, text="Labels (play / no-play / service) - drag a point to move it, right-click anywhere for a label menu")
        timeline_frame.pack(fill="x", padx=8, pady=4)
        self.timeline = tk.Canvas(timeline_frame, height=TIMELINE_HEIGHT, bg=UNLABELED_COLOR, highlightthickness=0,
                                   highlightbackground=BORDER)
        self.timeline.pack(fill="x", padx=6, pady=(6, 2))
        self.timeline.bind("<Button-1>", self._timeline_click)
        self.timeline.bind("<B1-Motion>", self._timeline_drag)
        self.timeline.bind("<ButtonRelease-1>", self._timeline_release)
        self.timeline.bind("<Button-3>", self._timeline_right_click)
        self.timeline.bind("<Configure>", lambda _e: self._redraw_timeline())

        self.hscroll = ttk.Scrollbar(timeline_frame, orient="horizontal", command=self._on_scroll)
        self.hscroll.pack(fill="x", padx=6, pady=(0, 6))

        state_frame = ttk.Frame(self.root)
        state_frame.pack(pady=4)
        for label, state_name in (("Play", "play"), ("No-play", "no-play"), ("Service", "service")):
            button = tk.Button(state_frame, text=label, bg=STATE_COLOR[state_name], fg="white",
                                activebackground=STATE_COLOR[state_name], activeforeground="white",
                                highlightbackground=BG, highlightthickness=0, bd=0,
                                command=lambda s=state_name: self._set_state(s))
            button.pack(side="left", padx=4)
            self.state_buttons[state_name] = button
        ttk.Separator(state_frame, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(state_frame, text="Export", command=self._on_export_click).pack(side="left", padx=4)

        legend = ("1 / 2 / 3 or the buttons above set a point at the current frame.   "
                  "X / Delete clears the point at the current frame.   Points can't be dragged past each other.   "
                  "Right-click the timeline for a menu (relabel or delete a point, or add one wherever you click).")
        tk.Label(self.root, text=legend, anchor="w", bg=BG, fg=FG_MUTED).pack(fill="x", padx=8, pady=(0, 8))

        self.root.bind("<Key>", self._on_key)

    def _on_close(self):
        self.cap.release()
        self.root.destroy()

    def _tick(self):
        if not self.paused:
            new_idx = self.frame_idx + 1
            if new_idx >= self.total_frames - 1:
                new_idx = self.total_frames - 1
                self.paused = True
            self._seek(new_idx)
        interval = max(1, int(1000 / (self.fps * self.speed))) if not self.paused else 30
        self.root.after(interval, self._tick)

    def _show_frame(self, idx):
        # Only seek when actually jumping - cap.set() forces most codecs
        # (long-GOP H.264/H.265) to rewind to the nearest keyframe and
        # decode forward to reach idx, which is far slower than a plain
        # sequential read. During ordinary forward playback the decoder is
        # already sitting exactly at idx from the previous read, so a seek
        # here is pure wasted work - repeated on every single frame, that's
        # what made playback feel much slower than real time.
        if idx != self._decoder_pos:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return
        self._decoder_pos = idx + 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if rgb.shape[1] > DISPLAY_MAX_WIDTH:
            scale = DISPLAY_MAX_WIDTH / rgb.shape[1]
            rgb = cv2.resize(rgb, (DISPLAY_MAX_WIDTH, int(rgb.shape[0] * scale)))
        self.photo_image = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.video_label.configure(image=self.photo_image)

    def _clamp_view_start(self, view_start):
        return max(0, min(self.total_frames - self.visible_frames, view_start))

    def _apply_view(self):
        view_end = self.view_start + self.visible_frames - 1
        self.scrubber.configure(from_=self.view_start, to=view_end)
        first = self.view_start / self.total_frames
        last = (self.view_start + self.visible_frames) / self.total_frames
        self.hscroll.set(first, last)
        self._redraw_timeline()

    def _seek(self, new_idx):
        new_idx = max(0, min(self.total_frames - 1, int(new_idx)))
        if new_idx != self.frame_idx:
            self.frame_idx = new_idx
            self._show_frame(new_idx)
        if not (self.view_start <= new_idx < self.view_start + self.visible_frames):
            self.view_start = self._clamp_view_start(new_idx - self.visible_frames // 2)
            self._apply_view()
        self._refresh_controls()

    def _refresh_controls(self):
        self.scrubber_var.set(self.frame_idx)
        current_state = _state_at(self.breakpoints, self.frame_idx)
        self.status_var.set(f"frame {self.frame_idx}/{self.total_frames - 1}   "
                             f"({self.frame_idx / self.fps:.2f}s)   state here: {current_state or 'unlabeled'}")
        for state_name, button in self.state_buttons.items():
            button.configure(relief="sunken" if state_name == current_state else "raised")
        self._redraw_timeline()

    def _redraw_timeline(self):
        self.timeline.delete("all")
        width = self.timeline.winfo_width()
        if width <= 1:
            return
        view_start, visible = self.view_start, self.visible_frames
        view_end = view_start + visible

        bounds = [bp[0] for bp in self.breakpoints[1:]] + [self.total_frames]
        for (bp_frame, bp_state), next_start in zip(self.breakpoints, bounds):
            seg_start, seg_end = max(bp_frame, view_start), min(next_start, view_end)
            if seg_end <= seg_start:
                continue
            x1 = _frame_to_x(seg_start, view_start, visible, width)
            x2 = _frame_to_x(seg_end - 1, view_start, visible, width) + 1
            self.timeline.create_rectangle(x1, 0, x2, TIMELINE_HEIGHT,
                                            fill=STATE_COLOR.get(bp_state, UNLABELED_COLOR), width=0)

        for bp_frame, _ in self.breakpoints:
            if not (view_start <= bp_frame < view_end):
                continue
            x = _frame_to_x(bp_frame, view_start, visible, width)
            colour = BREAKPOINT_DRAG_COLOR if self.dragging == bp_frame else BREAKPOINT_COLOR
            self.timeline.create_line(x, 0, x, TIMELINE_HEIGHT, fill=colour, width=2)
            self.timeline.create_oval(x - HANDLE_RADIUS, -HANDLE_RADIUS, x + HANDLE_RADIUS, HANDLE_RADIUS,
                                       fill=colour, outline=colour)

        if view_start <= self.frame_idx < view_end:
            px = _frame_to_x(self.frame_idx, view_start, visible, width)
            self.timeline.create_line(px, -8, px, TIMELINE_HEIGHT + 8, fill=PLAYHEAD_COLOR, width=2)

    def _drag_tolerance_frames(self):
        width = self.timeline.winfo_width()
        return max(1, round(DRAG_TOLERANCE_PX / max(1, width) * self.visible_frames))

    def _timeline_click(self, event):
        clicked_frame = _x_to_frame(event.x, self.view_start, self.visible_frames, self.timeline.winfo_width())
        self.dragging = _nearest_breakpoint(self.breakpoints, clicked_frame, self._drag_tolerance_frames())
        self._redraw_timeline()

    def _timeline_drag(self, event):
        if self.dragging is None:
            return
        new_frame = _x_to_frame(event.x, self.view_start, self.visible_frames, self.timeline.winfo_width())

        ordered = [bp[0] for bp in self.breakpoints]
        idx = ordered.index(self.dragging)
        lower_bound = ordered[idx - 1] + 1 if idx > 0 else 0
        upper_bound = ordered[idx + 1] - 1 if idx + 1 < len(ordered) else self.total_frames - 1
        new_frame = max(lower_bound, min(upper_bound, new_frame))

        bp_state = dict(self.breakpoints)[self.dragging]
        bps = _remove_breakpoint(self.breakpoints, self.dragging)
        self.breakpoints = _upsert_breakpoint(bps, new_frame, bp_state)
        self.dragging = new_frame
        self.paused = True
        self._seek(new_frame)

    def _timeline_release(self, _event):
        self.dragging = None
        self._redraw_timeline()

    def _timeline_right_click(self, event):
        clicked_frame = _x_to_frame(event.x, self.view_start, self.visible_frames, self.timeline.winfo_width())
        nearest = _nearest_breakpoint(self.breakpoints, clicked_frame, self._drag_tolerance_frames())
        target_frame = nearest if nearest is not None else clicked_frame

        menu = tk.Menu(self.root, tearoff=0, bg=BG_RAISED, fg=FG, activebackground=SELECT_BG, activeforeground=FG)
        for label, state_name in (("Play", "play"), ("No-play", "no-play"), ("Service", "service")):
            menu.add_command(label=label, command=lambda s=state_name, f=target_frame: self._apply_label_at(f, s))
        if nearest is not None:
            menu.add_separator()
            menu.add_command(label="Delete point", command=lambda f=nearest: self._delete_point_at(f))
        menu.tk_popup(event.x_root, event.y_root)

    def _apply_label_at(self, frame_idx, state_name):
        self.breakpoints = _upsert_breakpoint(self.breakpoints, frame_idx, state_name)
        self._refresh_controls()

    def _delete_point_at(self, frame_idx):
        self.breakpoints = _remove_breakpoint(self.breakpoints, frame_idx)
        self._refresh_controls()

    def _on_scroll(self, action, value, *rest):
        if action == "moveto":
            new_start = round(float(value) * self.total_frames)
        elif action == "scroll":
            step = self.visible_frames if (rest and rest[0] == "pages") else max(1, self.visible_frames // 20)
            new_start = self.view_start + int(value) * step
        else:
            return
        self.view_start = self._clamp_view_start(new_start)
        self._apply_view()

    def _on_zoom_change(self, value):
        self.zoom = max(ZOOM_MIN, float(value))
        self.zoom_label.configure(text=f"{self.zoom:.1f}x")
        self.visible_frames = max(1, min(self.total_frames, round(self.total_frames / self.zoom)))
        self.view_start = self._clamp_view_start(self.frame_idx - self.visible_frames // 2)
        self._apply_view()

    def _on_speed_change(self, _event):
        try:
            self.speed = float(self.speed_var.get().rstrip("x"))
        except ValueError:
            self.speed = 1.0

    def _on_scrub(self, value):
        self.paused = True
        self._seek(round(float(value)))

    def _prev_frame(self):
        self.paused = True
        self._seek(self.frame_idx - 1)

    def _next_frame(self):
        self.paused = True
        self._seek(self.frame_idx + 1)

    def _toggle_play(self):
        self.paused = not self.paused
        self.play_button.configure(text="Play" if self.paused else "Pause")

    def _set_state(self, state_name):
        self.breakpoints = _upsert_breakpoint(self.breakpoints, self.frame_idx, state_name)
        self._refresh_controls()

    def _on_export_click(self):
        _export(self.breakpoints, self.total_frames, self.fps, self.output_path)

    def _on_key(self, event):
        key = event.keysym
        if key in STATE_KEYS:
            self._set_state(STATE_KEYS[key])
        elif key == "space":
            self._toggle_play()
        elif key in ("Left", "a", "A"):
            self._prev_frame()
        elif key in ("Right", "d", "D"):
            self._next_frame()
        elif key in ("x", "X", "Delete", "BackSpace"):
            self.breakpoints = _remove_breakpoint(self.breakpoints, self.frame_idx)
            self._refresh_controls()
        elif key in ("e", "E"):
            self._on_export_click()
        elif key == "Escape":
            self._on_close()


def labelVideo(video_path, output_path):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    app = LabelApp(video_path, output_path)
    app.root.mainloop()
    return app.labels_file


def _parse_args():
    parser = argparse.ArgumentParser(description="Manually label a video's play / no-play / service timeline.")
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Defaults to the video's own folder.")
    return parser.parse_args()


def _choose_video():
    initial_dir = "C:\\Users\\Kai\\Documents\\Volleyball Footage"
    if not Path(initial_dir).exists():
        initial_dir = str(Path.home())

    root = tk.Tk()
    root.withdraw()
    chosen = filedialog.askopenfilename(
        title="Choose a video to label",
        initialdir=initial_dir,
        filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
    )
    root.destroy()

    if not chosen:
        raise SystemExit("No video selected.")
    return Path(chosen)


if __name__ == "__main__":
    args = _parse_args()
    video = args.video or _choose_video()
    output = args.output or video.parent
    labelVideo(str(video), str(output))
