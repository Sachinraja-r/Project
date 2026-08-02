import csv
import os
import datetime
import collections
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas

import database

LOG_DIR    = os.path.join(os.path.dirname(__file__), 'logs')
REPORT_DIR = os.path.join(os.path.dirname(__file__), 'reports')
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

W, H = A4   # 595 x 842 pts

def generate_pdf_report(session_id: str = None):
    # Try fetching from SQLite DB first
    sess_data = None
    if session_id:
        sess_data = database.get_session_db(session_id)
        
    if sess_data:
        student = sess_data.get('student_name', 'Unknown')
        start_t = sess_data.get('start_time', 'N/A')
        end_t   = sess_data.get('end_time', 'N/A')
        total_frames = sess_data.get('total_frames', 0) or 1
        score = sess_data.get('integrity_score', 100.0)
        events = sess_data.get('events', [])
        
        wrong_face_n = sum(1 for e in events if e.get('reason') == 'wrong_face')
        no_face_n    = sum(1 for e in events if e.get('reason') == 'no_face')
        gaze_n       = sum(1 for e in events if e.get('reason') == 'gaze')
        emotion_n    = sum(1 for e in events if e.get('reason') == 'emotion')
        phone_n      = sum(1 for e in events if e.get('reason') in ('phone', 'multi_person'))
        
        high_n  = sum(1 for e in events if e.get('alert_level') == 'HIGH')
        med_n   = sum(1 for e in events if e.get('alert_level') == 'MEDIUM')
        low_n   = sum(1 for e in events if e.get('alert_level') == 'LOW')
        
        report_filename = f"{session_id}_report.pdf"
    else:
        # Fallback to latest CSV file
        log_files = sorted([f for f in os.listdir(LOG_DIR) if f.endswith('.csv')], reverse=True)
        if not log_files:
            print("No CSV logs or DB sessions found.")
            return None
        
        log_path = os.path.join(LOG_DIR, log_files[0])
        with open(log_path, newline='') as f:
            rows = list(csv.DictReader(f))
            
        if not rows:
            print("Log file empty.")
            return None
            
        total_frames = len(rows)
        reasons      = [r['reason'] for r in rows]
        alerts       = [r['alert_level'] for r in rows]
        faces        = [r['face_detected'] for r in rows]
        
        wrong_face_n = reasons.count('wrong_face')
        no_face_n    = reasons.count('no_face')
        gaze_n       = reasons.count('gaze')
        emotion_n    = reasons.count('emotion')
        phone_n      = reasons.count('phone') + reasons.count('multi_person')
        
        high_n  = alerts.count('HIGH')
        med_n   = alerts.count('MEDIUM')
        low_n   = alerts.count('LOW')
        
        start_t = rows[0]['time']
        end_t   = rows[-1]['time']
        name_counter = collections.Counter(n for n in faces if n not in ('Unknown', 'No face', 'Detecting...'))
        student = name_counter.most_common(1)[0][0] if name_counter else 'Unknown'
        
        penalty = high_n * 3 + med_n * 1.5 + low_n * 0.5
        max_penalty = total_frames * 3
        score = max(0.0, round(100 - (penalty / max_penalty * 100), 1)) if max_penalty else 100.0
        
        session_tag = log_files[0].replace('.csv', '')
        report_filename = f"{session_tag}_report.pdf"

    verdict = "PASS" if score >= 75 else "REVIEW REQUIRED"
    score_color = colors.HexColor("#1a6e30") if score >= 75 else colors.HexColor("#8b1a1a")
    pdf_path = os.path.join(REPORT_DIR, report_filename)

    c = canvas.Canvas(pdf_path, pagesize=A4)

    def text(txt, x, y, size=10, color='#2c2c2c', bold=False):
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.setFillColor(colors.HexColor(color))
        c.drawString(x, y, str(txt))

    def rtext(txt, x, y, size=10, color='#2c2c2c'):
        c.setFont("Helvetica", size)
        c.setFillColor(colors.HexColor(color))
        c.drawRightString(x, y, str(txt))

    def hline(y, lx=50, rx=None):
        rx = rx or W - 50
        c.setStrokeColor(colors.HexColor("#dddddd"))
        c.setLineWidth(0.5)
        c.line(lx, y, rx, y)

    def section(title, y):
        text(title, 50, y, size=12, bold=True, color='#1a1a2e')
        hline(y - 6)
        return y - 22

    def bar(value, max_val, x, y, bar_w=300, bar_h=14, fill='#4a90d9'):
        ratio = value / max_val if max_val else 0
        c.setFillColor(colors.HexColor("#eeeeee"))
        c.rect(x, y, bar_w, bar_h, fill=1, stroke=0)
        c.setFillColor(colors.HexColor(fill))
        c.rect(x, y, int(bar_w * ratio), bar_h, fill=1, stroke=0)

    # ── HEADER ──
    c.setFillColor(colors.HexColor("#0f172a"))
    c.rect(0, H - 75, W, 75, fill=1, stroke=0)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, H - 44, "ProctorEye Pro")
    c.setFont("Helvetica", 11)
    c.drawString(50, H - 62, "AI-Powered Exam Integrity & Proctored Session Report")
    c.setFont("Helvetica", 9)
    c.drawRightString(W - 50, H - 52, f"Generated: {datetime.datetime.now().strftime('%d %b %Y %H:%M')}")

    y = H - 105

    # ── SESSION OVERVIEW ──
    y = section("Session Overview", y)
    pairs = [
        ("Student Name",   student),
        ("Session Start",  start_t),
        ("Session End",    end_t),
        ("Total Frames",   f"{total_frames:,}"),
        ("Session ID",     session_id or "Local Session"),
    ]
    for label, val in pairs:
        text(f"{label:<20}: {val}", 55, y, size=10)
        y -= 17
    y -= 8

    # ── INTEGRITY SCORE CARD ──
    card_h = 60
    bg = "#e8f5e9" if score >= 75 else "#fce8e8"
    c.setFillColor(colors.HexColor(bg))
    c.roundRect(50, y - card_h, W - 100, card_h, 8, fill=1, stroke=0)
    c.setFillColor(score_color)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(68, y - 28, f"Integrity Score: {score}/100")
    c.setFont("Helvetica", 11)
    c.drawString(68, y - 48, f"Status: {verdict}")
    rtext(f"HIGH: {high_n}  MED: {med_n}  LOW: {low_n}", W - 65, y - 36, size=9, color='#555555')
    y -= card_h + 20

    # ── ALERT SUMMARY ──
    y = section("Alert & Infraction Summary", y)
    alert_data = [
        ("Wrong Face Identified",  wrong_face_n, "#c0392b"),
        ("Face Absent (>3 sec)",    no_face_n,    "#e67e22"),
        ("Gaze / Pupil Deviation", gaze_n,       "#2980b9"),
        ("Object / Phone Detected",phone_n,      "#d35400"),
        ("Suspicious Emotion",      emotion_n,    "#8e44ad"),
    ]
    top_count = max((v for _, v, _ in alert_data), default=1) or 1
    for label, count, clr in alert_data:
        text(f"{label:<28}", 55, y, size=10)
        bar(count, top_count, 240, y - 2, bar_w=220, bar_h=12, fill=clr)
        rtext(str(count), W - 55, y, size=10, color=clr)
        y -= 22
    y -= 8

    # ── FOOTER ──
    c.setFillColor(colors.HexColor("#f5f5f5"))
    c.rect(0, 0, W, 32, fill=1, stroke=0)
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#999999"))
    c.drawString(50, 12,
        "ProctorEye Pro System  |  For academic integrity use only  |  "
        "AI results should be reviewed by an invigilator.")

    c.save()
    print(f"Generated PDF Report at: {pdf_path}")
    return pdf_path

if __name__ == "__main__":
    generate_pdf_report()
