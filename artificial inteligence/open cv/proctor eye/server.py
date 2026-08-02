import cv2
import os
import base64
import json
import asyncio
import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import database
from proctor_engine import ProctorEngine
from audio_monitor import AudioMonitor
import report

app = FastAPI(title="ProctorEye Pro API", version="2.0")

BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

database.init_db()

class StartSessionRequest(BaseModel):
    student_name: str

class StopSessionRequest(BaseModel):
    session_id: str

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/students")
async def get_students():
    return database.get_all_students_db()

@app.get("/api/sessions")
async def get_sessions():
    return database.get_all_sessions_db()

@app.post("/api/session/start")
async def start_session(req: StartSessionRequest):
    student_name = req.student_name.strip()
    if not student_name:
        raise HTTPException(status_code=400, detail="Student name required")
    
    database.register_student_db(student_name)
    session_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    session_id = f"session_{session_time}"
    
    database.create_session_db(session_id, student_name)
    return {"session_id": session_id, "student_name": student_name}

@app.post("/api/session/stop")
async def stop_session(req: StopSessionRequest):
    sess = database.get_session_db(req.session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
        
    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start_dt = datetime.datetime.strptime(sess["start_time"], "%Y-%m-%d %H:%M:%S")
    duration = str(datetime.datetime.now() - start_dt).split('.')[0]
    
    database.update_session_db(
        session_id=req.session_id,
        end_time=end_time,
        duration=duration,
        total_frames=sess.get("total_frames", 0),
        integrity_score=sess.get("integrity_score", 100.0)
    )
    
    # Generate PDF report
    pdf_path = report.generate_pdf_report(req.session_id)
    return {"status": "success", "pdf_path": pdf_path}

@app.get("/api/reports/{session_id}")
async def download_report(session_id: str):
    pdf_path = os.path.join(REPORTS_DIR, f"{session_id}_report.pdf")
    if not os.path.exists(pdf_path):
        pdf_path = report.generate_pdf_report(session_id)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Report generation failed")
    return FileResponse(pdf_path, media_type="application/pdf", filename=f"{session_id}_report.pdf")

@app.websocket("/ws/proctor")
async def websocket_proctor(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    sess = database.get_session_db(session_id)
    if not sess:
        await websocket.close(code=4000, reason="Session not found")
        return
        
    student_name = sess["student_name"]
    engine = ProctorEngine(student_name=student_name, session_id=session_id)
    audio = AudioMonitor()
    audio.start()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        await websocket.send_json({"error": "Webcam unavailable"})
        await websocket.close()
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame, telemetry = engine.process_frame(frame)
            
            audio_status = audio.get_status()
            if audio_status["speech_detected"]:
                telemetry["alert_reason"] = "audio_speech"
                if telemetry["alert_level"] == "CLEAR":
                    telemetry["alert_level"] = "LOW"
            
            if telemetry["alert_level"] != "CLEAR" and telemetry["alert_reason"]:
                database.log_event_db(
                    session_id=session_id,
                    timestamp=telemetry["time"],
                    frame_num=telemetry["frame"],
                    alert_level=telemetry["alert_level"],
                    reason=telemetry["alert_reason"],
                    details=f"Emotion: {telemetry['emotion']}, Gaze: {telemetry['gaze']}"
                )

            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            await websocket.send_json({
                "type": "frame",
                "frame": jpg_as_text,
                "telemetry": telemetry
            })

            await asyncio.sleep(0.01)
            
    except WebSocketDisconnect:
        print(f"WebSocket client disconnected for session {session_id}")
    except Exception as e:
        print(f"WebSocket processing error: {e}")
    finally:
        engine.stop()
        cap.release()
        audio.stop()

if __name__ == "__main__":
    import uvicorn
    print("Launching ProctorEye Pro Server on http://localhost:8000 ...")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
