"""
Local garment point selector — run this on your PC before uploading to Kaggle.

Usage:
    python select_garment_points.py path/to/image.jpg

Controls:
    U  — switch to UPPER garment mode (red dots)
    L  — switch to LOWER garment mode (blue dots)
    Z  — undo last point
    S  — save points and quit
    Q  — quit without saving

Output:
    garment_points.json  (same folder as the image)

Upload BOTH the image and garment_points.json to Kaggle.
"""

import cv2
import json
import sys
import os
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python select_garment_points.py <image_path>")
    sys.exit(1)

img_path = sys.argv[1]
img = cv2.imread(img_path)
if img is None:
    print(f"Could not load image: {img_path}")
    sys.exit(1)

# Scale down for display if too large
MAX_DIM = 900
h, w = img.shape[:2]
scale = min(MAX_DIM / w, MAX_DIM / h, 1.0)
disp_w, disp_h = int(w * scale), int(h * scale)
display = cv2.resize(img, (disp_w, disp_h))

print(f"Image size: {w}×{h} (displaying at {disp_w}×{disp_h})")

points = {"upper": [], "lower": []}
mode = "upper"          # current mode: "upper" or "lower"
COLORS = {"upper": (0, 0, 255), "lower": (255, 0, 0)}   # BGR: red / blue

def redraw():
    canvas = display.copy()
    # Draw all points
    for pt in points["upper"]:
        px, py = int(pt[0] * scale), int(pt[1] * scale)
        cv2.circle(canvas, (px, py), 7, COLORS["upper"], -1)
        cv2.circle(canvas, (px, py), 8, (255, 255, 255), 1)
    for pt in points["lower"]:
        px, py = int(pt[0] * scale), int(pt[1] * scale)
        cv2.circle(canvas, (px, py), 7, COLORS["lower"], -1)
        cv2.circle(canvas, (px, py), 8, (255, 255, 255), 1)
    # Status bar
    mode_color = COLORS[mode]
    label = f"Mode: {mode.upper()}  |  U=upper  L=lower  Z=undo  S=save  Q=quit"
    label += f"  |  upper:{len(points['upper'])}  lower:{len(points['lower'])}"
    label += f"  |  {w}×{h} -> {disp_w}×{disp_h}"
    cv2.rectangle(canvas, (0, 0), (disp_w, 28), (30, 30, 30), -1)
    cv2.putText(canvas, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 1, cv2.LINE_AA)
    cv2.imshow("Garment Point Selector", canvas)

def on_mouse(event, x, y, flags, param):
    global mode
    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert display coords back to original image coords
        orig_x = int(x / scale)
        orig_y = int(y / scale)
        points[mode].append([orig_x, orig_y])
        redraw()

cv2.namedWindow("Garment Point Selector", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Garment Point Selector", disp_w, disp_h + 30)
cv2.setMouseCallback("Garment Point Selector", on_mouse)
redraw()

print("\n=== Garment Point Selector ===")
print("Click on the UPPER garment first (mode starts as UPPER - red dots)")
print("Press L to switch to LOWER garment mode (blue dots)")
print("Press S to save and quit.\n")

while True:
    key = cv2.waitKey(20) & 0xFF
    if key == ord('u') or key == ord('U'):
        mode = "upper"
        print("Mode → UPPER")
        redraw()
    elif key == ord('l') or key == ord('L'):
        mode = "lower"
        print("Mode → LOWER")
        redraw()
    elif key == ord('z') or key == ord('Z'):
        if points[mode]:
            removed = points[mode].pop()
            print(f"Undo: removed {mode} point {removed}")
            redraw()
    elif key == ord('s') or key == ord('S'):
        out_path = os.path.join(os.path.dirname(os.path.abspath(img_path)), "garment_points.json")
        data = {
            "image_size": [w, h],   # original image size — used to rescale points in Kaggle
            "upper": points["upper"],
            "lower": points["lower"],
        }
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n✓ Saved {len(points['upper'])} upper + {len(points['lower'])} lower points ({w}x{h})")
        print(f"  → {out_path}")
        print("\nUpload BOTH the image and garment_points.json to Kaggle.")
        break
    elif key == ord('q') or key == ord('Q'):
        print("Quit without saving.")
        break

cv2.destroyAllWindows()
