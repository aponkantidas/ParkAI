import cv2
import time
import sqlite3
import easyocr
from datetime import datetime
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# --- CONFIGURATION ---
RATE_PER_MINUTE = 2.0  # Adjust hourly/minutely rate as needed
# For phones: install "IP Webcam" app (Android) or use RTSP / HTTP URL
# Example: "http://192.168.0.101:8080/video"
# For testing with laptop webcams, use 0 and 1
ENTRY_CAM_SRC = 0  
EXIT_CAM_SRC = 1   

# --- LOAD MODELS ---
model = YOLO("best.pt")
reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True if your local machine has an NVIDIA GPU

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

# --- OCR HELPER FUNCTION ---
def read_plate_text(crop):
    """Upscales crop and extracts clean plate characters."""
    h, w = crop.shape[:2]
    crop_up = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    
    ocr_res = reader.readtext(
        crop_up,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        detail=1
    )
    if not ocr_res:
        return None
    
    # Sort left-to-right
    ocr_res.sort(key=lambda item: item[0][0][0])
    text = "".join([t[1] for t in ocr_res if t[2] > 0.35])
    return text if len(text) >= 4 else None

# --- PARKING BUSINESS LOGIC ---
def process_entry(plate_number):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    
    # Check if plate is already parked without exit
    cursor.execute("SELECT id FROM parking_logs WHERE plate_number = ? AND status = 'ACTIVE'", (plate_number,))
    existing = cursor.fetchone()
    
    if not existing:
        now = datetime.now()
        cursor.execute("INSERT INTO parking_logs (plate_number, entry_time) VALUES (?, ?)", (plate_number, now))
        conn.commit()
        print(f"[ENTRY] Plate {plate_number} entered at {now.strftime('%H:%M:%S')}")
    conn.close()

def process_exit(plate_number):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, entry_time FROM parking_logs WHERE plate_number = ? AND status = 'ACTIVE'", (plate_number,))
    record = cursor.fetchone()
    
    bill_info = None
    if record:
        record_id, entry_time_str = record
        entry_time = datetime.fromisoformat(entry_time_str)
        exit_time = datetime.now()
        
        # Calculate time and charge
        duration_seconds = (exit_time - entry_time).total_seconds()
        duration_minutes = max(round(duration_seconds / 60.0, 2), 0.1) # at least 0.1 min for test
        total_bill = round(duration_minutes * RATE_PER_MINUTE, 2)
        
        cursor.execute("""
            UPDATE parking_logs 
            SET exit_time = ?, duration_minutes = ?, amount_due = ?, status = 'COMPLETED'
            WHERE id = ?
        """, (exit_time, duration_minutes, total_bill, record_id))
        conn.commit()
        
        bill_info = {
            "plate": plate_number,
            "duration": duration_minutes,
            "bill": total_bill
        }
        print(f"[EXIT] Plate {plate_number} exited! Total: {total_bill} BDT")
        
    conn.close()
    return bill_info

# --- VIDEO GENERATOR PIPELINE ---
def generate_frames(camera_source, mode="entry"):
    cap = cv2.VideoCapture(camera_source)
    last_detected_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        current_time = time.time()
        # Run detection every 1.5 seconds to conserve CPU/GPU
        if current_time - last_detected_time > 1.5:
            results = model(frame, conf=0.5, verbose=False)[0]
            
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    plate_text = read_plate_text(crop)
                    if plate_text:
                        last_detected_time = current_time
                        if mode == "entry":
                            process_entry(plate_text)
                        elif mode == "exit":
                            process_exit(plate_text)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- FLASK ROUTES ---
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/video_entry')
def video_entry():
    return Response(generate_frames(ENTRY_CAM_SRC, mode="entry"), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_exit')
def video_exit():
    return Response(generate_frames(EXIT_CAM_SRC, mode="exit"), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/records')
def get_records():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, plate_number, entry_time, exit_time, duration_minutes, amount_due, status FROM parking_logs ORDER BY id DESC LIMIT 10")
    rows = cursor.fetchall()
    conn.close()
    return jsonify(rows)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)