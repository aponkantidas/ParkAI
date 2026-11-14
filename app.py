import cv2
import time
import sqlite3
import threading
from datetime import datetime, timedelta
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
import easyocr

app = Flask(__name__)

# --- CAMERA CONFIGURATION ---
ENTRY_CAM_SRC = "http://172.20.10.8:8080/video"
EXIT_CAM_SRC = "http://172.20.10.2:8080/video"

# 20 Taka per hour (minimum charge 5 Taka for short test)
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
            paid_amount REAL DEFAULT 0.0,
            paid_until TIMESTAMP,
            amount_due REAL DEFAULT 0.0,
            status TEXT DEFAULT 'ACTIVE'
        )
    """)
    conn.commit()
    conn.close()

init_db()

latest_frames = {"entry": None, "exit": None}

def calculate_current_fee(entry_time, now):
    duration_sec = (now - entry_time).total_seconds()
    hours = duration_sec / 3600.0
    return max(round(hours * RATE_PER_HOUR, 2), 5.0)

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
    cursor.execute("""
        SELECT id, entry_time, paid_amount, paid_until 
        FROM parking_logs 
        WHERE plate_number = ? AND status = 'ACTIVE'
    """, (plate_number,))
    record = cursor.fetchone()
    
    if record:
        record_id, entry_time_str, paid_amount, paid_until_str = record
        entry_time = datetime.fromisoformat(entry_time_str)
        now = datetime.now()
        duration_min = max(round((now - entry_time).total_seconds() / 60.0, 2), 0.1)
        
        total_required_fee = calculate_current_fee(entry_time, now)
        paid_until = datetime.fromisoformat(paid_until_str) if paid_until_str else None

        # CASE 1: Never made any payment
        if paid_amount <= 0:
            latest_exit_event = {
                "event_key": f"{record_id}_UNPAID_{int(time.time())}",
                "status": "BLOCKED",
                "plate": plate_number,
                "title": "PAYMENT REQUIRED",
                "reason": f"Parked for {duration_min} mins. Please pay via phone.",
                "due": total_required_fee
            }
            print(f"[EXIT BLOCKED - UNPAID] {plate_number} owes {total_required_fee} Taka")

        # CASE 2: Paid earlier, but stayed past paid_until time
        elif paid_until and now > paid_until:
            extra_due = max(round(total_required_fee - paid_amount, 2), 5.0)
            latest_exit_event = {
                "event_key": f"{record_id}_OVERSTAY_{int(time.time())}",
                "status": "BLOCKED",
                "plate": plate_number,
                "title": "OVERSTAY EXTRA FEE",
                "reason": f"Pre-paid time expired at {paid_until.strftime('%H:%M')}. Pay extra balance.",
                "due": extra_due
            }
            print(f"[EXIT BLOCKED - OVERSTAY] {plate_number} owes extra {extra_due} Taka")

        # CASE 3: Paid & within allowed time -> OPEN GATE
        else:
            cursor.execute("""
                UPDATE parking_logs 
                SET exit_time = ?, duration_minutes = ?, amount_due = 0, status = 'COMPLETED'
                WHERE id = ?
            """, (now, duration_min, record_id))
            conn.commit()
            latest_exit_event = {
                "event_key": f"{record_id}_OPEN_{int(time.time())}",
                "status": "OPEN",
                "plate": plate_number
            }
            print(f"[EXIT APPROVED] {plate_number} - Gate Opening!")

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
@app.route('/')
def admin_dashboard():
    return render_template('dashboard.html')

@app.route('/entry')
def entry_kiosk():
    return render_template('entry_kiosk.html')

@app.route('/exit')
def exit_kiosk():
    return render_template('exit_kiosk.html')

@app.route('/pay')
def mobile_pay_page():
    return render_template('pay.html')

# --- MOBILE PAYMENT APIs ---
@app.route('/api/check_plate/<plate>')
def check_plate_status(plate):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT entry_time, paid_amount, paid_until 
        FROM parking_logs 
        WHERE plate_number = ? AND status = 'ACTIVE'
    """, (plate.upper(),))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "No active parked car found with this plate."})

    entry_time = datetime.fromisoformat(row[0])
    paid_amount = row[1]
    paid_until = datetime.fromisoformat(row[2]) if row[2] else None
    now = datetime.now()

    duration_mins = round((now - entry_time).total_seconds() / 60.0, 1)
    total_fee = calculate_current_fee(entry_time, now)

    if paid_until and now <= paid_until:
        due_now = 0.0
    else:
        due_now = max(round(total_fee - paid_amount, 2), 5.0)

    return jsonify({
        "entry_time": entry_time.strftime("%H:%M:%S"),
        "duration_mins": duration_mins,
        "paid_amount": paid_amount,
        "due_now": due_now
    })

@app.route('/api/pay', methods=['POST'])
def process_mobile_payment():
    data = request.json
    plate = data.get("plate", "").upper()
    hours_selected = float(data.get("hours", 0))

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, entry_time, paid_amount, paid_until 
        FROM parking_logs 
        WHERE plate_number = ? AND status = 'ACTIVE'
    """, (plate,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return jsonify({"success": False, "error": "Car not found in active parking."})

    record_id, entry_time_str, paid_amount, paid_until_str = row
    entry_time = datetime.fromisoformat(entry_time_str)
    now = datetime.now()

    if hours_selected > 0:
        base_time = datetime.fromisoformat(paid_until_str) if paid_until_str else entry_time
        new_paid_until = base_time + timedelta(hours=hours_selected)
        amount_to_pay = round(hours_selected * RATE_PER_HOUR, 2)
    else:
        total_fee = calculate_current_fee(entry_time, now)
        amount_to_pay = max(round(total_fee - paid_amount, 2), 5.0)
        new_paid_until = now + timedelta(minutes=10)

    new_total_paid = round(paid_amount + amount_to_pay, 2)

    cursor.execute("""
        UPDATE parking_logs 
        SET paid_amount = ?, paid_until = ? 
        WHERE id = ?
    """, (new_total_paid, new_paid_until, record_id))
    conn.commit()
    conn.close()

    return jsonify({
        "success": True,
        "paid": amount_to_pay,
        "valid_until": new_paid_until.strftime("%H:%M:%S")
    })

# --- DASHBOARD & KIOSK APIs ---
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
    cursor.execute("SELECT id, plate_number, entry_time, exit_time, duration_minutes, paid_amount, status FROM parking_logs ORDER BY id DESC LIMIT 10")
    rows = cursor.fetchall()
    conn.close()
    return jsonify(rows)

@app.route('/api/delete/<int:record_id>', methods=['DELETE'])
def delete_record(record_id):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM parking_logs WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()
    print(f"[INFO] Deleted parking record ID: {record_id}")
    return jsonify({"success": True})

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
    app.run(host='0.0.0.0', port=8000, debug=False)