import cv2
import time
import sqlite3
import threading
from datetime import datetime
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import easyocr

app = Flask(__name__)

# --- CAMERA CONFIGURATION ---
# Replace with your actual phone stream addresses
ENTRY_CAM_SRC = 1
EXIT_CAM_SRC = 0

# 20 Taka per hour -> approx 0.33 Taka per minute (minimum charge 5 Taka for toy car test)
RATE_PER_HOUR = 20.0  

# --- EVENT STATE HOLDERS FOR USER SCREENS ---
latest_entry_event = None
latest_exit_event = None

# --- LOAD MODELS ---
print("[INFO] Loading YOLO and EasyOCR...")
model = YOLO("best.pt")
reader = easyocr.Reader(['en'], gpu=False)
print("[INFO] Models loaded successfully!")

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parking_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            entry_time TIMESTAMP NOT NULL,
            exit_time TIMESTAMP,
            duration_minutes REAL,
            amount_due REAL,
            status TEXT DEFAULT 'ACTIVE'
        )
    """)
    conn.commit()
    conn.close()

init_db()

latest_frames = {"entry": None, "exit": None}

def read_plate_text(crop):
    h, w = crop.shape[:2]
    crop_up = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    ocr_res = reader.readtext(
        crop_up,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        detail=1
    )
    if not ocr_res:
        return None
    ocr_res.sort(key=lambda item: item[0][0][0])
    text = "".join([t[1] for t in ocr_res if t[2] > 0.30])
    return text if len(text) >= 4 else None

def process_entry(plate_number):
    global latest_entry_event
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM parking_logs WHERE plate_number = ? AND status = 'ACTIVE'", (plate_number,))
    if not cursor.fetchone():
        now = datetime.now()
        cursor.execute("INSERT INTO parking_logs (plate_number, entry_time) VALUES (?, ?)", (plate_number, now))
        conn.commit()
        log_id = cursor.lastrowid
        print(f"[ENTRY LOGGED] Plate: {plate_number}")
        latest_entry_event = {
            "id": log_id,
            "plate": plate_number,
            "time": now.strftime("%H:%M:%S")
        }
    conn.close()

def process_exit(plate_number):
    global latest_exit_event
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, entry_time FROM parking_logs WHERE plate_number = ? AND status = 'ACTIVE'", (plate_number,))
    record = cursor.fetchone()
    if record:
        record_id, entry_time_str = record
        entry_time = datetime.fromisoformat(entry_time_str)
        exit_time = datetime.now()
        
        duration_sec = (exit_time - entry_time).total_seconds()
        duration_min = max(round(duration_sec / 60.0, 2), 0.1)
        
        # Calculate bill (20 Taka/hr, minimum test bill 5 Taka)
        hours = duration_sec / 3600.0
        total_bill = max(round(hours * RATE_PER_HOUR, 2), 5.0)

        cursor.execute("""
            UPDATE parking_logs 
            SET exit_time = ?, duration_minutes = ?, amount_due = ?, status = 'COMPLETED'
            WHERE id = ?
        """, (exit_time, duration_min, total_bill, record_id))
        conn.commit()
        print(f"[EXIT LOGGED] Plate: {plate_number} | Bill: {total_bill} Taka")
        
        latest_exit_event = {
            "id": record_id,
            "plate": plate_number,
            "duration": duration_min,
            "bill": total_bill
        }
    conn.close()

def camera_worker(src, mode):
    global latest_frames
    cap = cv2.VideoCapture(src)
    last_detect_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(1)
            cap = cv2.VideoCapture(src)
            continue

        current_time = time.time()
        if current_time - last_detect_time > 1.5:
            last_detect_time = current_time
            results = model(frame, conf=0.45, verbose=False)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    plate = read_plate_text(crop)
                    if plate:
                        if mode == "entry":
                            process_entry(plate)
                        elif mode == "exit":
                            process_exit(plate)

        _, buffer = cv2.imencode('.jpg', frame)
        latest_frames[mode] = buffer.tobytes()

threading.Thread(target=camera_worker, args=(ENTRY_CAM_SRC, "entry"), daemon=True).start()
threading.Thread(target=camera_worker, args=(EXIT_CAM_SRC, "exit"), daemon=True).start()

# --- WEB ROUTES ---

# 1. Admin Master Dashboard
@app.route('/')
def admin_dashboard():
    return render_template('dashboard.html')

# 2. Gate 1 User Screen
@app.route('/entry')
def entry_kiosk():
    return render_template('entry_kiosk.html')

# 3. Gate 2 User Screen
@app.route('/exit')
def exit_kiosk():
    return render_template('exit_kiosk.html')

# --- API ENDPOINTS ---
@app.route('/api/latest_entry')
def get_latest_entry():
    return jsonify(latest_entry_event or {})

@app.route('/api/latest_exit')
def get_latest_exit():
    return jsonify(latest_exit_event or {})

@app.route('/api/records')
def get_records():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, plate_number, entry_time, exit_time, duration_minutes, amount_due, status FROM parking_logs ORDER BY id DESC LIMIT 10")
    rows = cursor.fetchall()
    conn.close()
    return jsonify(rows)

def stream_generator(mode):
    while True:
        frame_bytes = latest_frames.get(mode)
        if frame_bytes is None:
            time.sleep(0.1)
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)

@app.route('/video_entry')
def video_entry():
    return Response(stream_generator("entry"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_exit')
def video_exit():
    return Response(stream_generator("exit"), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)