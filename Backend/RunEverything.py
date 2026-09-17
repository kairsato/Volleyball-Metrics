import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent / "Analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from CourtDetection.court import detectCourt
from BallDetection.ballDetection import detectBall
from PlayerDetection.tracker import track_and_chain, consolidate_tracklets
from GameStatusDetection.gameStatusDetection import detectGameStatus
from ActionDetection.actionDetection import detectActions
from PostProcessing.consolidate import consolidateStats
from PostProcessing.renderVideo import renderAnnotatedVideo
from PostProcessing.generate_dashboard import generateDashboard
from label import labelVideo

video_path = "C:\\Users\\Kai\\Documents\\VolleyballArea\\verycut.mp4"
output_path = "C:\\Users\\Kai\\Documents\\Volleyball Footage\\generatedStuff"

# Player tracking is two stages (see PlayerDetection/tracker.py's module
# docstring): track_and_chain does the expensive detect+track+embed pass and
# saves its raw tracklets to disk; consolidate_tracklets does the graph
# matching and can be re-run on its own straight from that saved file - flip
# PLAYER_IDENTIFICATION_PREVIEW and re-run just this line to replay the
# result without repeating track_and_chain.
PLAYER_IDENTIFICATION_PREVIEW = True

#detectCourt(video_path,output_path)
#track_and_chain(video_path, output_path)
#consolidate_tracklets(video_path, output_path, show_preview=PLAYER_IDENTIFICATION_PREVIEW, save_video=False)
#detectBall(video_path,output_path, show_preview=True)
#detectGameStatus(video_path,output_path)
detectActions(video_path,output_path)
#consolidateStats(output_path)
#generateDashboard(output_path)
renderAnnotatedVideo(video_path,output_path)
#labelVideo(video_path,output_path)

