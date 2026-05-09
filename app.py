"""
Smart Stunting Care – Real-Time Nutritional Status Detection Dashboard
=====================================================================
Classifies children as 'gizi_buruk' or 'normal' using YOLO v8 / v11 / v26
with an optional Lightweight Ensemble (NMS-based voting) mode.

Author  : Senior CV Engineer
Stack   : Streamlit · Ultralytics · OpenCV · NumPy · Pillow
"""

import io
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# ---------------------------------------------------------------------------
# 1. PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Stunting Care",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 2. CONSTANTS
# ---------------------------------------------------------------------------
MODEL_DIR = Path("models")
MODEL_PATHS = {
    "YOLOv8": MODEL_DIR / "yolov8_best_fold_3.pt",
    "YOLO11": MODEL_DIR / "yolov11_best_fold_3.pt",
    "YOLO26": MODEL_DIR / "yolov26_best_fold_3.pt",
}

# Class-index-to-name mapping (aligned with the training label order)
CLASS_NAMES = {0: "gizi_buruk", 1: "normal"}

# Bounding-box colour palette (BGR for OpenCV)
CLASS_COLORS_BGR = {
    "gizi_buruk": (0, 0, 255),     # Red
    "normal":     (0, 200, 0),     # Green
}

# Streamlit-friendly hex colours for metric cards
CLASS_COLORS_HEX = {
    "gizi_buruk": "#FF4B4B",
    "normal":     "#21C354",
}

# ---------------------------------------------------------------------------
# 3. MODEL LOADING – cached so weights are loaded ONCE per session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLO models into memory …")
def load_models() -> dict:
    """Load all three YOLO model weights and return a dict keyed by name.

    Uses @st.cache_resource so the models are initialised only once,
    persisting across reruns and user interactions.
    """
    from ultralytics import YOLO  # deferred import keeps startup clean

    if not MODEL_DIR.exists():
        st.error(
            f"Model directory `{MODEL_DIR}` not found. "
            "Please create it and place the `.pt` weight files inside."
        )
        st.stop()

    models = {}
    for name, path in MODEL_PATHS.items():
        if not path.exists():
            st.error(f"Weight file `{path}` is missing.")
            st.stop()
        models[name] = YOLO(str(path))

    return models


# ---------------------------------------------------------------------------
# 4. CUSTOM BOUNDING-BOX DRAWING (no results.plot())
# ---------------------------------------------------------------------------
def draw_boxes(
    frame: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
) -> np.ndarray:
    """Draw styled bounding boxes on *frame* (BGR image, modified in-place).

    Parameters
    ----------
    frame     : HxWx3 BGR image.
    boxes     : Nx4 array of [x1, y1, x2, y2] pixel coordinates.
    scores    : N-length array of confidence scores.
    class_ids : N-length array of integer class indices.

    Returns
    -------
    The annotated frame.
    """
    for box, score, cid in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        cls_name = CLASS_NAMES.get(int(cid), "unknown")
        colour = CLASS_COLORS_BGR.get(cls_name, (255, 255, 255))

        # Rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)

        # Label background for readability
        label = f"{cls_name} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        cv2.rectangle(
            frame, (x1, y1 - th - baseline - 6), (x1 + tw + 4, y1), colour, -1
        )
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame


# ---------------------------------------------------------------------------
# 5. INFERENCE HELPERS
# ---------------------------------------------------------------------------
def run_single_model(model, frame: np.ndarray, conf: float, iou: float):
    """Run inference on a single YOLO model and return raw detections.

    Returns
    -------
    boxes     : Nx4 numpy array [x1, y1, x2, y2].
    scores    : N numpy array.
    class_ids : N numpy array.
    """
    results = model.predict(frame, conf=conf, iou=iou, verbose=False)
    det = results[0].boxes

    if det is None or len(det) == 0:
        return np.empty((0, 4)), np.array([]), np.array([])

    boxes = det.xyxy.cpu().numpy()
    scores = det.conf.cpu().numpy()
    class_ids = det.cls.cpu().numpy().astype(int)
    return boxes, scores, class_ids


def run_ensemble(models: dict, frame: np.ndarray, conf: float, iou: float):
    """Lightweight Ensemble via NMS-based voting across all three models.

    Pipeline
    --------
    1. Run each model independently on the same frame.
    2. Concatenate all detections (boxes, scores, classes).
    3. Apply OpenCV NMS (`cv2.dnn.NMSBoxes`) across the merged set to
       suppress overlapping boxes.  This is much cheaper than WBF.
    4. For each retained box, average the confidence scores of all
       overlapping source detections that contributed to it.

    Returns
    -------
    boxes, scores, class_ids  – filtered & merged arrays.
    """
    all_boxes, all_scores, all_class_ids = [], [], []

    # Step 1 – Collect detections from every model
    for model in models.values():
        b, s, c = run_single_model(model, frame, conf, iou)
        if len(b) > 0:
            all_boxes.append(b)
            all_scores.append(s)
            all_class_ids.append(c)

    if not all_boxes:
        return np.empty((0, 4)), np.array([]), np.array([])

    # Step 2 – Concatenate
    merged_boxes = np.vstack(all_boxes)
    merged_scores = np.concatenate(all_scores)
    merged_class_ids = np.concatenate(all_class_ids)

    # Convert [x1,y1,x2,y2] → [x,y,w,h] for cv2.dnn.NMSBoxes
    xywh = []
    for b in merged_boxes:
        x1, y1, x2, y2 = b
        xywh.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

    # Step 3 – Cross-model NMS
    indices = cv2.dnn.NMSBoxes(
        xywh,
        merged_scores.tolist(),
        score_threshold=conf,
        nms_threshold=iou,
    )

    if len(indices) == 0:
        return np.empty((0, 4)), np.array([]), np.array([])

    indices = np.array(indices).flatten()

    # Step 4 – Average confidence for overlapping contributors
    # For each kept box, find all raw detections that overlap with it
    # and average their scores for a more stable estimate.
    final_boxes = []
    final_scores = []
    final_classes = []

    for idx in indices:
        anchor = merged_boxes[idx]
        cls = merged_class_ids[idx]

        # Gather overlapping detections of the same class
        contrib_scores = []
        for j in range(len(merged_boxes)):
            if merged_class_ids[j] != cls:
                continue
            iou_val = _compute_iou(anchor, merged_boxes[j])
            if iou_val > 0.3:  # overlap threshold for contribution
                contrib_scores.append(merged_scores[j])

        avg_score = float(np.mean(contrib_scores)) if contrib_scores else float(merged_scores[idx])
        final_boxes.append(anchor)
        final_scores.append(avg_score)
        final_classes.append(cls)

    return (
        np.array(final_boxes),
        np.array(final_scores),
        np.array(final_classes, dtype=int),
    )


def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# 6. FRAME PROCESSING PIPELINE
# ---------------------------------------------------------------------------
def process_frame(
    frame: np.ndarray,
    models: dict,
    engine: str,
    conf: float,
    iou: float,
) -> tuple[np.ndarray, float, int, str, np.ndarray]:
    """Run detection on a single frame and return annotated image + metrics.

    Returns
    -------
    annotated    : BGR annotated frame.
    infer_ms     : Inference time in milliseconds.
    count        : Total objects detected.
    engine_label : Display-friendly engine name.
    class_ids    : Array of detected class indices.
    """
    t0 = time.perf_counter()

    if engine == "Ensemble Mode":
        boxes, scores, class_ids = run_ensemble(models, frame, conf, iou)
        engine_label = "Ensemble (NMS Voting)"
    else:
        model = models[engine]
        boxes, scores, class_ids = run_single_model(model, frame, conf, iou)
        engine_label = engine

    infer_ms = (time.perf_counter() - t0) * 1000.0

    annotated = draw_boxes(frame.copy(), boxes, scores, class_ids)
    count = len(boxes)

    return annotated, infer_ms, count, engine_label, class_ids


def process_single_image(
    frame: np.ndarray,
    models: dict,
    engine: str,
    conf: float,
    iou: float,
    filename: str = "",
) -> tuple[np.ndarray, dict]:
    """Process one image and return annotated frame + a metrics dict.

    The metrics dict contains per-class counts for batch logging.
    """
    annotated, infer_ms, count, engine_label, class_ids = process_frame(
        frame, models, engine, conf, iou
    )

    # Count per-class detections from the already-computed class_ids
    gizi_buruk_count = int(np.sum(class_ids == 0)) if len(class_ids) > 0 else 0
    normal_count = int(np.sum(class_ids == 1)) if len(class_ids) > 0 else 0

    metrics = {
        "Filename": filename,
        "Model_Used": engine_label,
        "Inference_Time_ms": round(infer_ms, 1),
        "Gizi_Buruk_Count": gizi_buruk_count,
        "Normal_Count": normal_count,
    }

    return annotated, metrics


# ---------------------------------------------------------------------------
# 7. MINIMAL CSS – hide default Streamlit branding only
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 8. STREAMLIT SIDEBAR – Control Centre
# ---------------------------------------------------------------------------
def render_sidebar() -> tuple[str, str, float, float]:
    """Build the sidebar controls and return user selections."""
    with st.sidebar:
        st.header("Smart Stunting Care Settings")
        st.divider()

        # Source selection
        st.subheader("Input Source")
        source = st.radio(
            "Select source",
            ["Webcam", "Upload Image"],
            index=1,
            label_visibility="collapsed",
        )

        st.divider()

        # Engine selection
        st.subheader("Detection Engine")
        engine = st.radio(
            "Select engine",
            ["YOLOv8", "YOLO11", "YOLO26", "Ensemble Mode", "Side-by-Side Comparison"],
            index=0,
            label_visibility="collapsed",
        )

        st.divider()

        # Threshold sliders
        st.subheader("Threshold Controls")
        conf = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
        )
        iou = st.slider(
            "IoU Threshold (NMS)",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.05,
        )

        st.divider()

        # Branding footer
        st.caption("Smart Stunting Care v1.0 | Powered by Ultralytics YOLO")

    return source, engine, conf, iou


# ---------------------------------------------------------------------------
# 9. METRIC SCORECARD
# ---------------------------------------------------------------------------
def render_scorecard(engine_label: str, infer_ms: float, count: int):
    """Render metric cards below the visualiser using Streamlit's native st.metric."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Active Engine", value=engine_label)
    with col2:
        st.metric(label="Inference Time", value=f"{infer_ms:.1f} ms")
    with col3:
        st.metric(label="Objects Detected", value=count)


# ---------------------------------------------------------------------------
# 10. MAIN APPLICATION
# ---------------------------------------------------------------------------
def main():
    inject_css()

    # Load models (cached – only once)
    models = load_models()

    # Sidebar controls
    source, engine, conf, iou = render_sidebar()

    # Main panel header
    st.header("Diagnostic Arena")
    st.caption("Real-time nutritional status classification - gizi_buruk & normal")
    st.divider()

    # ----- Upload Image Mode -----
    if source == "Upload Image":
        uploaded_files = st.file_uploader(
            "Upload image(s) for analysis",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True,
        )

        if not uploaded_files:
            st.info("Upload one or more images to begin analysis.")
            return

        # --- Side-by-Side Comparison Mode ---
        if engine == "Side-by-Side Comparison":
            # Process only the first uploaded image for comparison
            uploaded = uploaded_files[0]
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if frame is None:
                st.error("Could not decode the uploaded image.")
                return

            if len(uploaded_files) > 1:
                st.info(
                    "Side-by-Side Comparison uses the first image only. "
                    "Switch to another engine for batch processing."
                )

            st.subheader(f"Comparison: {uploaded.name}")
            cols = st.columns(3)
            model_names = ["YOLOv8", "YOLO11", "YOLO26"]

            for col, model_name in zip(cols, model_names):
                annotated, infer_ms, count, _, _ = process_frame(
                    frame, models, model_name, conf, iou
                )
                with col:
                    st.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        caption=model_name,
                        use_container_width=True,
                    )
                    st.metric("Inference Time", f"{infer_ms:.1f} ms")
                    st.metric("Objects Detected", count)
            return

        # --- Single / Ensemble / Batch Processing ---
        batch_results = []  # collect metrics for CSV export

        for uploaded in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if frame is None:
                st.warning(f"Skipped `{uploaded.name}` - could not decode.")
                continue

            annotated, metrics = process_single_image(
                frame, models, engine, conf, iou, filename=uploaded.name
            )

            # Display each result
            st.subheader(uploaded.name)
            _, img_col, _ = st.columns([1, 3, 1])
            with img_col:
                st.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption="Detection Result",
                    use_container_width=True,
                )
            render_scorecard(
                metrics["Model_Used"],
                metrics["Inference_Time_ms"],
                metrics["Gizi_Buruk_Count"] + metrics["Normal_Count"],
            )
            st.divider()

            batch_results.append(metrics)

        # --- Batch Report & Excel Export ---
        if batch_results:
            st.subheader("Batch Detection Report")
            df = pd.DataFrame(batch_results)
            st.dataframe(df, use_container_width=True)

            # Export as Excel (.xlsx) for proper column formatting
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False, sheet_name="Detection Report")
            buffer.seek(0)

            st.download_button(
                label="Download Detection Report (Excel)",
                data=buffer,
                file_name="detection_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    # ----- Webcam Mode -----
    else:
        # Side-by-Side is not available for Webcam
        if engine == "Side-by-Side Comparison":
            st.warning(
                "Side-by-Side Comparison is only available in Upload Image mode. "
                "Please select a different engine or switch to Upload Image."
            )
            return

        st.info("Allow camera access in your browser when prompted.")

        # Build a video processor class that captures the current settings
        class YOLOVideoProcessor(VideoProcessorBase):
            """Process each webcam frame through the selected YOLO engine.

            streamlit-webrtc streams frames from the user's browser camera
            via WebRTC, so this works on both local and cloud deployments.
            """

            def __init__(self):
                self.engine = engine
                self.conf = conf
                self.iou = iou
                self.result_info = {"engine": "", "infer_ms": 0.0, "count": 0}

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")

                # Run detection
                annotated, infer_ms, count, engine_label, _ = process_frame(
                    img, models, self.engine, self.conf, self.iou
                )

                # Store metrics for display
                self.result_info = {
                    "engine": engine_label,
                    "infer_ms": infer_ms,
                    "count": count,
                }

                return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        # TURN server config for cloud NAT traversal
        rtc_config = {
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }

        ctx = webrtc_streamer(
            key="smart-stunting-webcam",
            video_processor_factory=YOLOVideoProcessor,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": False},
        )

        # Display metrics while streaming
        if ctx.video_processor:
            info = ctx.video_processor.result_info
            render_scorecard(
                info.get("engine", engine),
                info.get("infer_ms", 0.0),
                info.get("count", 0),
            )


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
