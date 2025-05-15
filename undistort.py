import cv2
import numpy as np
import argparse
import sys
from pathlib import Path

# ------------- command-line arguments ---------------------------------------
parser = argparse.ArgumentParser(description="Undistort fisheye video.")
parser.add_argument("input_video",  type=Path, help="path to input video")
parser.add_argument("output_video", type=Path, help="path to save result")
parser.add_argument("--balance", type=float, default=0.3,
                    help="0=crop more, 1=keep everything (default 0.3)")
args = parser.parse_args()

# ------------- utilities ----------------------------------------------------
click_pts = []
frame_for_clicks = None


def _click(event, x, y, flags, param):
    global click_pts, frame_for_clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        click_pts.append((x, y))
        cv2.circle(frame_for_clicks, (x, y), 4, (0, 255, 0), -1)
        cv2.imshow("Pick pts on *one* straight line, then press q", frame_for_clicks)


def get_user_points(frame):
    """Let user click a few points that should be collinear in reality."""
    global click_pts, frame_for_clicks
    click_pts.clear()
    frame_for_clicks = frame.copy()
    cv2.imshow("Pick pts on *one* straight line, then press q", frame_for_clicks)
    cv2.setMouseCallback("Pick pts on *one* straight line, then press q", _click)
    while True:
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):  # q or ESC
            break
    cv2.destroyAllWindows()
    return np.asarray(click_pts, np.float32)


def score_points(points):
    """How bent are the selected points? 0 = perfectly straight."""
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    # perpendicular distance of each point to the fitted line
    dist = np.abs(vx * (points[:, 1] - y0) - vy * (points[:, 0] - x0))
    return dist.sum()


def find_best_distortion(points, K, search=(-.35, .35, 15)):
    """
    Grid-search
    """
    lo, hi, N = search
    k_range = np.linspace(lo, hi, N)
    best_D, best_err = None, np.inf
    for k1 in k_range:
        for k2 in k_range:
            D = np.array([[k1],[k2],[0.],[0.]], np.float32)
            # undistort only the clicked points (cheap)
            undist = cv2.fisheye.undistortPoints(points.reshape(-1,1,2), K, D,
                                                 R=np.eye(3), P=K).reshape(-1,2)
            e = score_points(undist)
            if e < best_err:
                best_err, best_D = e, D.copy()
    print(f"[INFO] best k1,k2 = {best_D.ravel()[:2]}, geometric error {best_err:.1f}")
    return best_D


cap = cv2.VideoCapture(str(args.input_video))
if not cap.isOpened():
    sys.exit(f"Cannot open {args.input_video}")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

ret, first = cap.read()
if not ret:
    sys.exit("Could not read first frame")

pts = get_user_points(first)
if len(pts) < 4:
    sys.exit("Need at least 4 points — run again and click more")

K = np.array([[w, 0, w/2],
              [0, w, h/2],
              [0, 0, 1]], np.float32)

D = find_best_distortion(pts, K)
K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3), balance=args.balance)
print(f"[INFO] K_new =\n{K_new}")

map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K_new, (w, h), cv2.CV_16SC2)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(args.output_video), fourcc, fps, (w, h))

print("[INFO] writing undistorted video … (press Ctrl-C to abort)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    undist = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
    out.write(undist)

cap.release()
out.release()
print(f"[DONE] saved → {args.output_video}")


