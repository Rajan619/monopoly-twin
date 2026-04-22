import cv2
import numpy as np
import json
import asyncio
import websockets
import threading

from flask import Flask, Response, request, render_template
from board_mapper import get_cell_from_position

# -----------------------------
# GLOBAL STATE
# -----------------------------
latest_pawns = []

transform = {
    "x": 0,
    "y": 0,
    "sx": 1.0,
    "sy": 1.0
}

# EVENT SYSTEM
pawn_last_cell = {}
pawn_candidate = {}
pawn_candidate_count = {}

event_queue = []

STABILITY_FRAMES = 3

# -----------------------------
# FLASK
# -----------------------------
app = Flask(__name__)

# -----------------------------
# CAMERA
# -----------------------------
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
parameters = aruco.DetectorParameters()

detector = aruco.ArucoDetector(dictionary, parameters)

BOARD = 1000

# -----------------------------
# GRID
# -----------------------------
def draw_grid(board):

    corner = int(BOARD * (2.25 / 18))
    edge   = int(BOARD * (1.5  / 18))
    t = 2

    cv2.rectangle(board,(0,BOARD-corner),(corner,BOARD),(255,0,0),t)
    cv2.rectangle(board,(BOARD-corner,BOARD-corner),(BOARD,BOARD),(255,0,0),t)
    cv2.rectangle(board,(BOARD-corner,0),(BOARD,corner),(255,0,0),t)
    cv2.rectangle(board,(0,0),(corner,corner),(255,0,0),t)

    x = corner
    for _ in range(9):
        cv2.rectangle(board,(x,BOARD-edge),(x+edge,BOARD),(255,0,0),t)
        x += edge

    y = corner
    for _ in range(9):
        cv2.rectangle(board,(BOARD-edge,y),(BOARD,y+edge),(255,0,0),t)
        y += edge

    x = corner
    for _ in range(9):
        cv2.rectangle(board,(x,0),(x+edge,edge),(255,0,0),t)
        x += edge

    y = corner
    for _ in range(9):
        cv2.rectangle(board,(0,y),(edge,y+edge),(255,0,0),t)
        y += edge

# -----------------------------
# DETECT PAWNS
# -----------------------------
def detect_pawns_aruco(corners, ids):

    pawns = []

    if ids is None:
        return pawns

    for i in range(len(ids)):
        marker_id = ids[i][0]

        if marker_id > 5:
            continue

        pts = corners[i][0]
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())

        pawns.append({
            "id": int(marker_id),
            "pos": (cx, cy)
        })

    return pawns

# -----------------------------
# TRANSFORM
# -----------------------------
def transform_point(pt, M):

    px, py = pt
    vec = np.array([[[px, py]]], dtype='float32')
    transformed = cv2.perspectiveTransform(vec, M)

    x = transformed[0][0][0]
    y = transformed[0][0][1]

    x = (x * transform["sx"]) + transform["x"]
    y = (y * transform["sy"]) + transform["y"]

    return int(x), int(y)

# -----------------------------
# CV LOOP
# -----------------------------
def cv_loop():

    while True:

        success, frame = camera.read()
        if not success:
            continue

        corners, ids, _ = detector.detectMarkers(frame)

        if ids is None:
            ids = []
            corners = []

        marker_pts = {}

        for i in range(len(ids)):
            marker_id = int(ids[i][0])
            pt = corners[i][0][0]
            marker_pts[marker_id] = pt

        if not all(k in marker_pts for k in [6,7,8,9]):
            continue

        src = np.float32([
            marker_pts[6],
            marker_pts[7],
            marker_pts[8],
            marker_pts[9]
        ])

        margin = int(BOARD * (2.25 / 18))

        dst = np.float32([
            [margin, margin],
            [BOARD-margin, margin],
            [BOARD-margin, BOARD-margin],
            [margin, BOARD-margin]
        ])

        M = cv2.getPerspectiveTransform(src, dst)

        pawns = detect_pawns_aruco(corners, ids)

        for p in pawns:

            pawn_id = p["id"]

            px, py = transform_point(p["pos"], M)
            cell = get_cell_from_position(px, py)

            if cell is None:
                continue

            cell = int(cell)

            if pawn_id not in pawn_candidate:
                pawn_candidate[pawn_id] = cell
                pawn_candidate_count[pawn_id] = 1
            else:
                if pawn_candidate[pawn_id] == cell:
                    pawn_candidate_count[pawn_id] += 1
                else:
                    pawn_candidate[pawn_id] = cell
                    pawn_candidate_count[pawn_id] = 1

            if pawn_candidate_count[pawn_id] >= STABILITY_FRAMES:

                last_cell = pawn_last_cell.get(pawn_id)

                if last_cell != cell:

                    event = {
                        "type": "PAWN_MOVED",
                        "id": pawn_id,
                        "from": last_cell,
                        "to": cell
                    }

                    print("EVENT:", event)

                    event_queue.append(event)
                    pawn_last_cell[pawn_id] = cell

# -----------------------------
# VIDEO DEBUG
# -----------------------------
def generate_frames():

    while True:

        success, frame = camera.read()
        if not success:
            continue

        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None and len(ids) > 0:

            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            marker_pts = {}

            for i in range(len(ids)):
                marker_id = int(ids[i][0])
                pt = corners[i][0][0]
                marker_pts[marker_id] = pt

            if all(k in marker_pts for k in [6,7,8,9]):

                src = np.float32([
                    marker_pts[6],
                    marker_pts[7],
                    marker_pts[8],
                    marker_pts[9]
                ])

                margin = int(BOARD * (2.25 / 18))

                dst = np.float32([
                    [margin, margin],
                    [BOARD-margin, margin],
                    [BOARD-margin, BOARD-margin],
                    [margin, BOARD-margin]
                ])

                M = cv2.getPerspectiveTransform(src, dst)

                board = cv2.warpPerspective(frame, M, (BOARD, BOARD))

                draw_grid(board)

                pawns = detect_pawns_aruco(corners, ids)

                for p in pawns:
                    px, py = transform_point(p["pos"], M)
                    cell = get_cell_from_position(px, py)

                    if cell is None:
                        continue

                    cv2.putText(board,
                                f"ID:{p['id']} C:{cell}",
                                (px, py-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0,255,255), 2)

                frame = board

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# -----------------------------
# ROUTES
# -----------------------------
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video')
def video():
    return Response(generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_transform', methods=['POST'])
def set_transform():
    data = request.json
    transform["x"] = float(data["x"])
    transform["y"] = float(data["y"])
    transform["sx"] = float(data["sx"])
    transform["sy"] = float(data["sy"])
    return {"status": "ok"}

# -----------------------------
# WEBSOCKET
# -----------------------------
async def ws_handler(websocket):
    print("Unreal connected")

    try:
        while True:

            state_msg = {
                "type": "STATE_SYNC",
                "pawns": [
                    {"id": int(pid), "cell": int(cell)}
                    for pid, cell in pawn_last_cell.items()
                ]
            }

            msg = json.dumps(state_msg)
            await websocket.send(msg)

            print("SYNC:", msg)

            await asyncio.sleep(0.5)

    except Exception as e:
        print("Client disconnected:", e)

async def ws_main():
    async with websockets.serve(ws_handler, "0.0.0.0", 8765):
        print("WS running on ws://localhost:8765")
        await asyncio.Future()

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    threading.Thread(target=cv_loop, daemon=True).start()
    threading.Thread(target=lambda: asyncio.run(ws_main()), daemon=True).start()

    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
