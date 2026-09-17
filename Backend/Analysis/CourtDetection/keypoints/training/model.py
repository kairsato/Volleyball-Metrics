"""Base checkpoint selection for the court-keypoint model.

Unlike ActionDetection's hand-rolled classifier, a YOLO-pose model's
architecture/training loop is entirely handled by ultralytics' own
`YOLO(...).train(data=..., task="pose")` - there's no custom nn.Module or
DataLoader to write here, just which pretrained checkpoint to fine-tune
from. "n" (nano) is the default: this project's court-keypoint task is a
single, geometrically simple object (one court, 4 points) with a training
set in the low thousands of images, where a small model trains faster and
resists overfitting better than the detector-sized backbones this project
uses for players/ball - accuracy can be revisited with a larger variant
once real footage shows the nano model actually falling short.
"""

BASE_CHECKPOINT = "yolo11n-pose.pt"
