# Smart Stunting Care Dashboard

Real-time nutritional status detection dashboard that classifies children as **gizi_buruk** (malnourished) or **normal** using YOLO object detection models.

## Features

- **Multi-Engine Detection** — Switch between YOLOv8, YOLO11, YOLO26, or Ensemble (NMS Voting) mode
- **Side-by-Side Comparison** — Compare all three model outputs on a single image
- **Webcam & Image Upload** — Live feed or batch image processing
- **Batch Processing** — Upload multiple images and export results as Excel (.xlsx)
- **Adjustable Thresholds** — Fine-tune Confidence and IoU (NMS) parameters via sliders

## Tech Stack

Streamlit · Ultralytics YOLO · OpenCV · NumPy · Pandas

## Quick Start

```bash
# Create & activate virtual environment
python -m venv env
env\Scripts\activate        # Windows
# source env/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Project Structure

```
├── app.py                # Main Streamlit application
├── models/               # YOLO weight files (.pt)
│   ├── yolov8_best_fold_3.pt
│   ├── yolov11_best_fold_3.pt
│   └── yolov26_best_fold_3.pt
├── requirements.txt      # Python dependencies
└── README.md
```

## License

This project is for academic and research purposes.
