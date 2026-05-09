const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const modelSelect = document.getElementById('model-select');
const confSlider = document.getElementById('conf-slider');
const confVal = document.getElementById('conf-val');
const iouSlider = document.getElementById('iou-slider');
const iouVal = document.getElementById('iou-val');
const fpsVal = document.getElementById('fps-val');
const countVal = document.getElementById('count-val');

let sessions = {}; // Store loaded models
let currentMode = "models/yolov8_best_fold_3.onnx";
let isDetecting = false;
let animationId;
let lastTime = 0;

const classNames = ["gizi_buruk", "normal"];
const colors = ["#FF4B4B", "#21C354"];

// Configure ONNX Runtime to use WASM backend (Pinned version)
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";

async function initModel(mode) {
    currentMode = mode;
    startBtn.disabled = true;
    startBtn.innerText = "Memuat Model... (Mohon Tunggu)";
    try {
        const pathsToLoad = mode === "ensemble" ? [
            "models/yolov8_best_fold_3.onnx",
            "models/yolov11_best_fold_3.onnx",
            "models/yolov26_best_fold_3.onnx"
        ] : [mode];

        for (const path of pathsToLoad) {
            if (!sessions[path]) {
                console.log(`Loading model: ${path}`);
                sessions[path] = await ort.InferenceSession.create(path, { executionProviders: ['wasm'] });
                console.log(`Loaded: ${path}`);
            }
        }
        
        if (!isDetecting) {
            startBtn.disabled = false;
            startBtn.innerText = "Mulai Kamera";
        }
    } catch (e) {
        console.error("Failed to load model", e);
        startBtn.innerText = "Gagal Memuat Model";
        alert("Gagal memuat model. Pastikan file model ONNX tersedia dan jaringan internet lancar.");
    }
}

confSlider.addEventListener('input', (e) => confVal.innerText = e.target.value);
iouSlider.addEventListener('input', (e) => iouVal.innerText = e.target.value);

modelSelect.addEventListener('change', async (e) => {
    await initModel(e.target.value);
});

startBtn.addEventListener('click', async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        video.srcObject = stream;
        
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            isDetecting = true;
            startBtn.disabled = true;
            stopBtn.disabled = false;
            detectFrame();
        };
    } catch (err) {
        console.error("Error accessing webcam:", err);
        alert("Tidak dapat mengakses kamera. Mohon izinkan akses kamera di browser Anda.");
    }
});

stopBtn.addEventListener('click', () => {
    isDetecting = false;
    cancelAnimationFrame(animationId);
    const stream = video.srcObject;
    if (stream) {
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
    }
    video.srcObject = null;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
});

async function detectFrame(currentTime) {
    if (!isDetecting) return;

    // Check if the required sessions are loaded
    let isReady = true;
    if (currentMode === "ensemble") {
        if (!sessions["models/yolov8_best_fold_3.onnx"] || !sessions["models/yolov11_best_fold_3.onnx"] || !sessions["models/yolov26_best_fold_3.onnx"]) isReady = false;
    } else {
        if (!sessions[currentMode]) isReady = false;
    }

    if (!isReady) {
        animationId = requestAnimationFrame(detectFrame);
        return;
    }

    // Calculate FPS
    const dt = currentTime - lastTime;
    if (dt > 0) {
        fpsVal.innerText = Math.round(1000 / dt);
    }
    lastTime = currentTime;

    const inputSize = 640;
    const offscreenCanvas = document.createElement('canvas');
    offscreenCanvas.width = inputSize;
    offscreenCanvas.height = inputSize;
    const offscreenCtx = offscreenCanvas.getContext('2d');
    
    offscreenCtx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, 0, 0, inputSize, inputSize);
    const imgData = offscreenCtx.getImageData(0, 0, inputSize, inputSize).data;
    
    const float32Data = new Float32Array(3 * inputSize * inputSize);
    for (let i = 0; i < inputSize * inputSize; i++) {
        float32Data[i] = imgData[i * 4] / 255.0; // R
        float32Data[inputSize * inputSize + i] = imgData[i * 4 + 1] / 255.0; // G
        float32Data[2 * inputSize * inputSize + i] = imgData[i * 4 + 2] / 255.0; // B
    }

    const inputTensor = new ort.Tensor('float32', float32Data, [1, 3, inputSize, inputSize]);
    
    try {
        const confThresh = parseFloat(confSlider.value);
        const iouThresh = parseFloat(iouSlider.value);
        let allBoxes = [];

        if (currentMode === "ensemble") {
            const paths = [
                "models/yolov8_best_fold_3.onnx",
                "models/yolov11_best_fold_3.onnx",
                "models/yolov26_best_fold_3.onnx"
            ];
            const promises = paths.map(path => {
                const sess = sessions[path];
                const feeds = {};
                feeds[sess.inputNames[0]] = inputTensor;
                return sess.run(feeds).then(results => {
                    const output = results[sess.outputNames[0]].data;
                    const dims = results[sess.outputNames[0]].dims;
                    return extractBoxes(output, dims, confThresh);
                });
            });
            const boxArrays = await Promise.all(promises);
            for (const arr of boxArrays) allBoxes.push(...arr);
        } else {
            const sess = sessions[currentMode];
            const feeds = {};
            feeds[sess.inputNames[0]] = inputTensor;
            const results = await sess.run(feeds);
            const output = results[sess.outputNames[0]].data;
            const dims = results[sess.outputNames[0]].dims;
            allBoxes = extractBoxes(output, dims, confThresh);
        }

        // 3. Apply NMS and Draw
        const finalBoxes = nms(allBoxes, iouThresh);
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const scaleX = video.videoWidth / inputSize;
        const scaleY = video.videoHeight / inputSize;
        
        countVal.innerText = finalBoxes.length;

        finalBoxes.forEach(box => {
            const x1 = box.x1 * scaleX;
            const y1 = box.y1 * scaleY;
            const w = (box.x2 - box.x1) * scaleX;
            const h = (box.y2 - box.y1) * scaleY;
            
            const color = colors[box.classId];
            const name = classNames[box.classId];
            
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.strokeRect(x1, y1, w, h);
            
            ctx.fillStyle = color;
            ctx.font = "16px Arial";
            const text = `${name} ${box.score.toFixed(2)}`;
            const textWidth = ctx.measureText(text).width;
            
            ctx.fillRect(x1, y1 - 20, textWidth + 10, 20);
            ctx.fillStyle = "#FFF";
            ctx.fillText(text, x1 + 5, y1 - 5);
        });

    } catch (e) {
        console.error("Inference Error:", e);
    }

    animationId = requestAnimationFrame(detectFrame);
}

// Helper to extract boxes depending on model output shape
function extractBoxes(output, dims, confThresh) {
    let boxes = [];
    if (dims.length === 3 && dims[1] === 6 && dims[2] === 8400) {
        for (let i = 0; i < 8400; i++) {
            const score0 = output[4 * 8400 + i];
            const score1 = output[5 * 8400 + i];
            const maxScore = Math.max(score0, score1);
            if (maxScore >= confThresh) {
                const classId = score0 > score1 ? 0 : 1;
                const cx = output[0 * 8400 + i];
                const cy = output[1 * 8400 + i];
                const w = output[2 * 8400 + i];
                const h = output[3 * 8400 + i];
                boxes.push({ x1: cx - w/2, y1: cy - h/2, x2: cx + w/2, y2: cy + h/2, score: maxScore, classId });
            }
        }
    } else if (dims.length === 3 && dims[1] === 300 && dims[2] === 6) {
        for (let i = 0; i < 300; i++) {
            const score = output[i * 6 + 4];
            if (score >= confThresh) {
                const classId = Math.round(output[i * 6 + 5]);
                boxes.push({
                    x1: output[i * 6 + 0],
                    y1: output[i * 6 + 1],
                    x2: output[i * 6 + 2],
                    y2: output[i * 6 + 3],
                    score, classId
                });
            }
        }
    }
    return boxes;
}

// Basic Non-Max Suppression algorithm
function nms(boxes, iouThresh) {
    boxes.sort((a, b) => b.score - a.score);
    const result = [];
    while (boxes.length > 0) {
        const best = boxes.shift();
        result.push(best);
        boxes = boxes.filter(box => {
            if (box.classId !== best.classId) return true;
            return iou(best, box) < iouThresh;
        });
    }
    return result;
}

function iou(b1, b2) {
    const x1 = Math.max(b1.x1, b2.x1);
    const y1 = Math.max(b1.y1, b2.y1);
    const x2 = Math.min(b1.x2, b2.x2);
    const y2 = Math.min(b1.y2, b2.y2);
    const interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const b1Area = (b1.x2 - b1.x1) * (b1.y2 - b1.y1);
    const b2Area = (b2.x2 - b2.x1) * (b2.y2 - b2.y1);
    return interArea / (b1Area + b2Area - interArea);
}

// Initial load
initModel(modelSelect.value);
