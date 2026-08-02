import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "proctor.db")

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            student_name TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            duration TEXT,
            total_frames INTEGER DEFAULT 0,
            integrity_score REAL DEFAULT 100.0,
            status TEXT DEFAULT 'ACTIVE'
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            frame_num INTEGER,
            alert_level TEXT NOT NULL,
            reason TEXT NOT NULL,
            details TEXT,
            snapshot_path TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    ''')

    conn.commit()
    conn.close()

def register_student_db(name: str):
    init_db()
    conn = get_db()
    cursor = conn.cursor()
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        cursor.execute("INSERT OR REPLACE INTO students (name, created_at) VALUES (?, ?)", (name, created_at))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error registering student: {e}")
        return False
    finally:
        conn.close()

def get_all_students_db():
    init_db()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT name, created_at FROM students ORDER BY name ASC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def create_session_db(session_id: str, student_name: str):
    init_db()
    conn = get_db()
    cursor = conn.cursor()
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO sessions (session_id, student_name, start_time, status) VALUES (?, ?, ?, 'ACTIVE')",
        (session_id, student_name, start_time)
    )
    conn.commit()
    conn.close()

def log_event_db(session_id: str, timestamp: str, frame_num: int, alert_level: str, reason: str, details: str = "", snapshot_path: str = None):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO events 
           (session_id, timestamp, frame_num, alert_level, reason, details, snapshot_path)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (session_id, timestamp, frame_num, alert_level, reason, details, snapshot_path)
    )
    conn.commit()
    conn.close()

def update_session_db(session_id: str, end_time: str, duration: str, total_frames: int, integrity_score: float):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """UPDATE sessions 
           SET end_time = ?, duration = ?, total_frames = ?, integrity_score = ?, status = 'COMPLETED'
           WHERE session_id = ?""",
        (end_time, duration, total_frames, integrity_score, session_id)
    )
    conn.commit()
    conn.close()

def get_session_db(session_id: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
    session = cursor.fetchone()
    if not session:
        conn.close()
        return None
    
    cursor.execute("SELECT * FROM events WHERE session_id = ? ORDER BY id ASC", (session_id,))
    events = cursor.fetchall()
    conn.close()
    
    res = dict(session)
    res['events'] = [dict(ev) for ev in events]
    return res

def get_all_sessions_db():
    init_db()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions ORDER BY start_time DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
