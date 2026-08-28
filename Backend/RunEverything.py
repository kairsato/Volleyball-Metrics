import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent / "Analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from CourtDefinition.court import courtDefine
from CourtDefinition.BallDetection.ballDetection import detectBall
from PlayerDetection.tracker_offline import trackplayers_offline
from GameStatusDetection.gameStatusDetection import detectGameStatus
from ActionDetection.actionDetection import detectActions
from PostProcessing.consolidate import consolidateStats
from PostProcessing.renderVideo import renderAnnotatedVideo
from PostProcessing.generate_dashboard import generateDashboard

video_path = "C:\\Users\\Kai\\Documents\\VolleyballArea\\verycut.mp4"
output_path = "C:\\Users\\Kai\\Documents\\Volleyball Footage\\generatedStuff"

courtDefine(video_path,output_path)
trackplayers_offline(video_path,output_path, True,False)
#detectBall(video_path,output_path)
#detectGameStatus(video_path,output_path)
#detectActions(video_path,output_path)
#consolidateStats(output_path)
#generateDashboard(output_path)
#renderAnnotatedVideo(video_path,output_path)

