import asyncio
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="FaceFusion Microservice API",
    description="High-performance, GPU-accelerated Face & Deep Swapper API for microservice dashboards with real-time live streaming.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = DATA_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JOBS: Dict[str, Dict[str, Any]] = {}


class DfmSwapRequest(BaseModel):
    model: str = Field(default="iperov/elon_musk_224", description="DFM model identifier")
    target_path: str = Field(description="Path to target image or video file")
    output_path: Optional[str] = Field(default=None, description="Path where output should be written")
    morph: int = Field(default=100, ge=0, le=100, description="Morph percentage (0-100)")
    face_detector_model: str = Field(default="scrfd")
    face_detector_size: str = Field(default="320x320")
    face_mask_types: List[str] = Field(default=["box", "occlusion"])
    execution_providers: List[str] = Field(default=["cuda"])
    execution_thread_count: int = Field(default=4)
    video_memory_strategy: str = Field(default="strict")
    async_job: bool = Field(default=False, description="Run in background and return job_id immediately")


class FaceSwapRequest(BaseModel):
    source_paths: List[str] = Field(description="Paths to source face image(s)")
    target_path: str = Field(description="Path to target image or video file")
    output_path: Optional[str] = Field(default=None, description="Path where output should be written")
    model: str = Field(default="hyperswap_1a_256", description="Face swapper model name")
    face_detector_model: str = Field(default="scrfd")
    face_detector_size: str = Field(default="320x320")
    face_mask_types: List[str] = Field(default=["box", "occlusion"])
    execution_providers: List[str] = Field(default=["cuda"])
    execution_thread_count: int = Field(default=4)
    video_memory_strategy: str = Field(default="strict")
    async_job: bool = Field(default=False, description="Run in background and return job_id immediately")


class LiveStreamEngine:
    """Manages persistent in-memory GPU models for low-latency live video streaming."""
    def __init__(self):
        self._initialized = False
        self._current_processor = None
        self._current_dfm_model = None

    def initialize(self):
        if self._initialized:
            return
        from facefusion.program import create_program
        from facefusion.args import apply_args
        from facefusion import state_manager

        program = create_program()
        args = vars(program.parse_args([
            'run',
            '--processors', 'deep_swapper',
            '--deep-swapper-model', 'iperov/elon_musk_224',
            '--face-detector-model', 'scrfd',
            '--face-detector-size', '320x320',
            '--face-selector-mode', 'one',
            '--face-mask-types', 'box', 'occlusion',
            '--execution-providers', 'cuda',
            '--execution-thread-count', '4',
            '--video-memory-strategy', 'strict'
        ]))
        apply_args(args, state_manager.init_item)
        self._initialized = True
        self.set_dfm_model('iperov/elon_musk_224')

    def set_dfm_model(self, model_name: str, morph: int = 100):
        from facefusion import state_manager
        from facefusion.processors.core import get_processors_modules

        if not self._initialized:
            self.initialize()

        if self._current_dfm_model != model_name or self._current_processor is None:
            state_manager.set_item('processors', ['deep_swapper'])
            state_manager.set_item('deep_swapper_model', model_name)
            state_manager.set_item('deep_swapper_morph', morph)

            modules = get_processors_modules(['deep_swapper'])
            if modules:
                self._current_processor = modules[0]
                self._current_processor.clear_inference_pool()
                self._current_processor.pre_check()
                self._current_processor.pre_process('stream')
            self._current_dfm_model = model_name

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        from facefusion.streamer import process_stream_frame
        if not self._initialized:
            self.initialize()
        return process_stream_frame([], frame_bgr)


stream_engine = LiveStreamEngine()


@app.on_event("startup")
def startup_event():
    # Pre-warm GPU stream engine in a worker thread on boot
    import threading
    t = threading.Thread(target=stream_engine.initialize, daemon=True)
    t.start()


def run_headless_process(args_list: List[str]) -> Dict[str, Any]:
    cmd = [sys.executable, "facefusion.py", "headless-run"] + args_list
    start_time = time.time()

    import subprocess
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    duration = time.time() - start_time

    if proc.returncode != 0:
        error_msg = proc.stderr.strip() or proc.stdout.strip() or f"Process failed with exit code {proc.returncode}"
        raise RuntimeError(error_msg)

    return {
        "duration_seconds": round(duration, 3),
        "stdout": proc.stdout.strip(),
    }


def execute_background_job(job_id: str, args_list: List[str], output_path: str):
    JOBS[job_id]["status"] = "processing"
    JOBS[job_id]["started_at"] = time.time()
    try:
        result = run_headless_process(args_list)
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["completed_at"] = time.time()
        JOBS[job_id]["duration_seconds"] = result["duration_seconds"]
        JOBS[job_id]["output_path"] = output_path
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)


@app.get("/health")
def health_check():
    import onnxruntime
    available_providers = onnxruntime.get_available_providers()
    cuda_available = "CUDAExecutionProvider" in available_providers

    return {
        "status": "healthy",
        "service": "facefusion",
        "cuda_available": cuda_available,
        "available_providers": available_providers,
        "active_jobs": sum(1 for j in JOBS.values() if j.get("status") == "processing")
    }


@app.get("/api/models/dfm")
def list_dfm_models():
    from facefusion.processors.modules.deep_swapper import choices

    models_dir = Path(".assets/models")
    downloaded = []
    if models_dir.exists():
        for dfm_file in models_dir.rglob("*.dfm"):
            rel = dfm_file.relative_to(models_dir)
            downloaded.append(str(rel).replace("\\", "/").removesuffix(".dfm"))

    return {
        "total_available": len(choices.deep_swapper_models),
        "downloaded_models": downloaded,
        "all_models": choices.deep_swapper_models,
    }


@app.get("/api/models/face-swapper")
def list_face_swapper_models():
    from facefusion.processors.modules.face_swapper import choices

    models_dir = Path(".assets/models")
    downloaded = []
    if models_dir.exists():
        for onnx_file in models_dir.glob("*.onnx"):
            name = onnx_file.stem
            if name in choices.face_swapper_models:
                downloaded.append(name)

    return {
        "downloaded_models": downloaded,
        "all_models": choices.face_swapper_models,
    }


@app.post("/api/swap/dfm")
def swap_dfm(request: DfmSwapRequest, background_tasks: BackgroundTasks):
    target = Path(request.target_path).resolve()
    if not target.exists():
        raise HTTPException(status_code=400, detail=f"Target file not found: {request.target_path}")

    if request.output_path:
        output = Path(request.output_path).resolve()
    else:
        output = OUTPUT_DIR / f"dfm_output_{uuid.uuid4().hex[:8]}{target.suffix}"
    output.parent.mkdir(parents=True, exist_ok=True)

    args = [
        "--processors", "deep_swapper",
        "--deep-swapper-model", request.model,
        "--deep-swapper-morph", str(request.morph),
        "-t", str(target),
        "-o", str(output),
        "--face-detector-model", request.face_detector_model,
        "--face-detector-size", request.face_detector_size,
        "--face-mask-types", *request.face_mask_types,
        "--execution-providers", *request.execution_providers,
        "--execution-thread-count", str(request.execution_thread_count),
        "--video-memory-strategy", request.video_memory_strategy,
    ]

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "job_id": job_id,
        "type": "dfm",
        "model": request.model,
        "target_path": str(target),
        "output_path": str(output),
        "status": "queued",
        "created_at": time.time(),
    }

    if request.async_job:
        background_tasks.add_task(execute_background_job, job_id, args, str(output))
        return {
            "job_id": job_id,
            "status": "queued",
            "output_path": str(output),
            "message": "Job queued for background execution."
        }

    try:
        result = run_headless_process(args)
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["duration_seconds"] = result["duration_seconds"]
        return {
            "job_id": job_id,
            "status": "completed",
            "output_path": str(output),
            "duration_seconds": result["duration_seconds"],
        }
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/swap/face")
def swap_face(request: FaceSwapRequest, background_tasks: BackgroundTasks):
    target = Path(request.target_path).resolve()
    if not target.exists():
        raise HTTPException(status_code=400, detail=f"Target file not found: {request.target_path}")

    for s_path in request.source_paths:
        if not Path(s_path).resolve().exists():
            raise HTTPException(status_code=400, detail=f"Source file not found: {s_path}")

    if request.output_path:
        output = Path(request.output_path).resolve()
    else:
        output = OUTPUT_DIR / f"face_output_{uuid.uuid4().hex[:8]}{target.suffix}"
    output.parent.mkdir(parents=True, exist_ok=True)

    args = [
        "--processors", "face_swapper",
        "--face-swapper-model", request.model,
        "-s", *[str(Path(p).resolve()) for p in request.source_paths],
        "-t", str(target),
        "-o", str(output),
        "--face-detector-model", request.face_detector_model,
        "--face-detector-size", request.face_detector_size,
        "--face-mask-types", *request.face_mask_types,
        "--execution-providers", *request.execution_providers,
        "--execution-thread-count", str(request.execution_thread_count),
        "--video-memory-strategy", request.video_memory_strategy,
    ]

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "job_id": job_id,
        "type": "face_swapper",
        "model": request.model,
        "target_path": str(target),
        "output_path": str(output),
        "status": "queued",
        "created_at": time.time(),
    }

    if request.async_job:
        background_tasks.add_task(execute_background_job, job_id, args, str(output))
        return {
            "job_id": job_id,
            "status": "queued",
            "output_path": str(output),
            "message": "Job queued for background execution."
        }

    try:
        result = run_headless_process(args)
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["duration_seconds"] = result["duration_seconds"]
        return {
            "job_id": job_id,
            "status": "completed",
            "output_path": str(output),
            "duration_seconds": result["duration_seconds"],
        }
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    dest_path = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{file.filename}"
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {
        "filename": file.filename,
        "file_path": str(dest_path.resolve()),
        "relative_path": f"/data/uploads/{dest_path.name}"
    }


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]


@app.get("/api/jobs")
def list_jobs():
    return list(JOBS.values())


@app.get("/api/files/{filename}")
def download_output_file(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        file_path = DATA_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


# =========================================================================
# REAL-TIME LIVE WEBSOCKET STREAMING
# =========================================================================

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            message = await websocket.receive()
            msg_type = message.get("type")

            if msg_type == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                np_arr = np.frombuffer(raw_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is not None:
                    # Offload GPU processing to threadpool to avoid blocking event loop
                    out_frame = await asyncio.to_thread(stream_engine.process_frame, frame)
                    _, buffer = cv2.imencode('.jpg', out_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    await websocket.send_bytes(buffer.tobytes())

            elif "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                    if data.get("action") == "set_model" and data.get("model"):
                        model_name = data["model"]
                        morph = data.get("morph", 100)
                        await asyncio.to_thread(stream_engine.set_dfm_model, model_name, morph)
                        await websocket.send_text(json.dumps({"type": "status", "status": "model_updated", "model": model_name}))
                except Exception as e:
                    await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket session closed: {e}")


@app.get("/live", response_class=HTMLResponse)
def live_preview_page():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FaceFusion Live DFM Stream</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: #0f172a; color: #f8fafc; display: flex; flex-direction: column; align-items: center; min-height: 100vh; padding: 20px; }
        header { text-align: center; margin-bottom: 20px; }
        h1 { font-size: 1.8rem; font-weight: 700; color: #38bdf8; }
        p { color: #94a3b8; font-size: 0.95rem; margin-top: 4px; }
        .controls { display: flex; gap: 12px; margin-bottom: 20px; align-items: center; flex-wrap: wrap; justify-content: center; }
        button, select { background: #1e293b; color: #f8fafc; border: 1px solid #334155; padding: 10px 16px; border-radius: 8px; font-size: 0.9rem; cursor: pointer; transition: all 0.2s; }
        button:hover { background: #334155; }
        button.primary { background: #0284c7; border-color: #0284c7; font-weight: 600; }
        button.primary:hover { background: #0369a1; }
        button.danger { background: #dc2626; border-color: #dc2626; }
        .stats { display: flex; gap: 20px; margin-bottom: 15px; font-size: 0.9rem; color: #38bdf8; font-weight: 600; }
        .stream-container { position: relative; width: 100%; max-width: 960px; background: #000; border-radius: 12px; overflow: hidden; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5); border: 1px solid #1e293b; aspect-ratio: 16/9; display: flex; align-items: center; justify-content: center; }
        #canvasOutput { width: 100%; height: 100%; object-fit: contain; }
        .pip-video { position: absolute; bottom: 16px; right: 16px; width: 200px; aspect-ratio: 16/9; border-radius: 8px; border: 2px solid #38bdf8; overflow: hidden; background: #000; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.5); }
        .pip-video video { width: 100%; height: 100%; object-fit: cover; }
        .pip-label { position: absolute; top: 4px; left: 8px; font-size: 0.7rem; background: rgba(0,0,0,0.7); padding: 2px 6px; border-radius: 4px; color: #38bdf8; }
        .placeholder { color: #64748b; font-size: 1.1rem; }
    </style>
</head>
<body>
    <header>
        <h1>FaceFusion Real-Time Live Stream</h1>
        <p>Ultra-low-latency DFM Character Face Swapping directly via WebSockets</p>
    </header>

    <div class="controls">
        <label for="modelSelect" style="font-size:0.9rem; color:#94a3b8;">DFM Model:</label>
        <select id="modelSelect">
            <option value="iperov/elon_musk_224">Elon Musk (iperov/elon_musk_224)</option>
        </select>
        <button id="btnStart" class="primary" onclick="startStream()">Start Camera</button>
        <button id="btnStop" class="danger" onclick="stopStream()" disabled>Stop</button>
    </div>

    <div class="stats">
        <div>FPS: <span id="fpsVal">0</span></div>
        <div>Latency: <span id="latencyVal">0</span> ms</div>
        <div>Status: <span id="statusVal" style="color:#22c55e;">Ready</span></div>
    </div>

    <div class="stream-container">
        <div id="placeholder" class="placeholder">Click "Start Camera" to begin real-time streaming</div>
        <canvas id="canvasOutput" style="display:none;"></canvas>
        <div class="pip-video" id="pipBox" style="display:none;">
            <div class="pip-label">Local Camera</div>
            <video id="videoInput" autoplay playsinline muted></video>
        </div>
    </div>

    <canvas id="canvasCapture" style="display:none;"></canvas>

    <script>
        const video = document.getElementById('videoInput');
        const canvasOutput = document.getElementById('canvasOutput');
        const canvasCapture = document.getElementById('canvasCapture');
        const placeholder = document.getElementById('placeholder');
        const pipBox = document.getElementById('pipBox');
        const btnStart = document.getElementById('btnStart');
        const btnStop = document.getElementById('btnStop');
        const fpsVal = document.getElementById('fpsVal');
        const latencyVal = document.getElementById('latencyVal');
        const statusVal = document.getElementById('statusVal');
        const modelSelect = document.getElementById('modelSelect');

        let ws = null;
        let stream = null;
        let isStreaming = false;
        let lastFrameTime = performance.now();
        let frameCount = 0;
        let pendingFrame = false;
        let sendTime = 0;

        fetch('/api/models/dfm')
            .then(res => res.json())
            .then(data => {
                modelSelect.innerHTML = '';
                const models = data.downloaded_models.length > 0 ? data.downloaded_models : data.all_models.slice(0, 20);
                models.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m;
                    opt.textContent = m;
                    if (m === 'iperov/elon_musk_224') opt.selected = true;
                    modelSelect.appendChild(opt);
                });
            })
            .catch(err => console.error('Failed to load models:', err));

        modelSelect.addEventListener('change', () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ action: 'set_model', model: modelSelect.value }));
                statusVal.textContent = 'Switching model...';
            }
        });

        async function startStream() {
            try {
                statusVal.textContent = 'Connecting camera...';
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
                    audio: false
                });
                video.srcObject = stream;
                await video.play();

                const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${proto}//${window.location.host}/ws/stream`);
                ws.binaryType = 'blob';

                ws.onopen = () => {
                    statusVal.textContent = 'Streaming Live';
                    placeholder.style.display = 'none';
                    canvasOutput.style.display = 'block';
                    pipBox.style.display = 'block';
                    btnStart.disabled = true;
                    btnStop.disabled = false;
                    isStreaming = true;

                    // Send initial model configuration
                    ws.send(JSON.stringify({ action: 'set_model', model: modelSelect.value }));

                    // Start sending frames
                    sendNextFrame();
                };

                ws.onmessage = async (event) => {
                    if (typeof event.data === 'string') {
                        try {
                            const info = JSON.parse(event.data);
                            if (info.status === 'model_updated') {
                                statusVal.textContent = 'Active: ' + info.model;
                            }
                        } catch (e) {}
                        return;
                    }

                    if (event.data instanceof Blob) {
                        const roundtrip = Math.round(performance.now() - sendTime);
                        latencyVal.textContent = roundtrip;

                        const img = new Image();
                        const url = URL.createObjectURL(event.data);
                        img.onload = () => {
                            canvasOutput.width = img.width;
                            canvasOutput.height = img.height;
                            const ctx = canvasOutput.getContext('2d');
                            ctx.drawImage(img, 0, 0);
                            URL.revokeObjectURL(url);

                            frameCount++;
                            const now = performance.now();
                            if (now - lastFrameTime >= 1000) {
                                fpsVal.textContent = frameCount;
                                frameCount = 0;
                                lastFrameTime = now;
                            }

                            pendingFrame = false;
                            if (isStreaming) {
                                sendNextFrame();
                            }
                        };
                        img.src = url;
                    }
                };

                ws.onerror = (err) => {
                    console.error('WebSocket Error:', err);
                    statusVal.textContent = 'Connection Error';
                };

                ws.onclose = () => {
                    stopStream();
                };

            } catch (err) {
                alert('Could not access webcam: ' + err.message);
                statusVal.textContent = 'Camera Error';
            }
        }

        function sendNextFrame() {
            if (!isStreaming || pendingFrame || !ws || ws.readyState !== WebSocket.OPEN) return;

            if (video.videoWidth > 0 && video.videoHeight > 0) {
                canvasCapture.width = video.videoWidth;
                canvasCapture.height = video.videoHeight;
                const ctx = canvasCapture.getContext('2d');
                ctx.drawImage(video, 0, 0);

                canvasCapture.toBlob((blob) => {
                    if (blob && ws && ws.readyState === WebSocket.OPEN && !pendingFrame) {
                        pendingFrame = true;
                        sendTime = performance.now();
                        ws.send(blob);
                    }
                }, 'image/jpeg', 0.8);
            } else {
                requestAnimationFrame(sendNextFrame);
            }
        }

        function stopStream() {
            isStreaming = false;
            pendingFrame = false;
            if (ws) { ws.close(); ws = null; }
            if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
            placeholder.style.display = 'block';
            canvasOutput.style.display = 'none';
            pipBox.style.display = 'none';
            btnStart.disabled = false;
            btnStop.disabled = true;
            statusVal.textContent = 'Stopped';
            fpsVal.textContent = '0';
            latencyVal.textContent = '0';
        }
    </script>
</body>
</html>
    """


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting FaceFusion Live Stream API Service on {host}:{port}...")
    uvicorn.run("service:app", host=host, port=port, reload=False)
