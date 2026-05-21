import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import io

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Rice Disease Detector",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

    .main { background: #0d1117; color: #e6edf3; }
    .stApp { background: #0d1117; }

    .header-box {
        background: linear-gradient(135deg, #1a2e1a 0%, #0d1f0d 100%);
        border: 1px solid #2ea043;
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .header-box h1 { color: #56d364; font-size: 2rem; margin: 0; }
    .header-box p  { color: #8b949e; margin: 0.5rem 0 0; }

    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        text-align: center;
    }
    .metric-card .label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-card .value { font-size: 1.6rem; font-family: 'IBM Plex Mono', monospace; color: #56d364; font-weight: 600; }

    .detection-row {
        display: flex;
        align-items: center;
        gap: 1rem;
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .det-rank  { font-family: 'IBM Plex Mono', monospace; color: #8b949e; font-size: 0.85rem; min-width: 24px; }
    .det-class { font-weight: 600; color: #e6edf3; flex: 1; }
    .det-conf  { font-family: 'IBM Plex Mono', monospace; color: #56d364; font-size: 0.9rem; }
    .conf-bar  { height: 6px; border-radius: 3px; background: #21262d; flex: 2; overflow: hidden; }
    .conf-fill { height: 100%; border-radius: 3px; background: linear-gradient(90deg, #2ea043, #56d364); }

    .stFileUploader { border: 2px dashed #30363d !important; border-radius: 10px !important; }
    .stButton>button {
        background: #238636 !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.6rem 2rem !important;
        font-family: 'IBM Plex Mono', monospace !important;
        letter-spacing: 0.05em !important;
    }
    .stButton>button:hover { background: #2ea043 !important; }
    .stSpinner > div { border-top-color: #56d364 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🌾 Rice Disease Detector</h1>
    <p>YOLOv8-powered object detection · Upload a rice plant image to identify diseases</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
WEIGHTS_PATH = "best.pt"

@st.cache_resource
def load_model():
    if os.path.exists(WEIGHTS_PATH):
        return YOLO(WEIGHTS_PATH)
    return None

model = load_model()

if model is None:
    st.error(f"⚠️ Model weights not found at `{WEIGHTS_PATH}`. "
             "Ensure `best.pt` is in the same directory as this app.")
    st.stop()

# ─────────────────────────────────────────────
# Sidebar — settings
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Detection Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.10, 0.95, 0.25, 0.05)
    iou_threshold  = st.slider("IoU Threshold (NMS)",  0.10, 0.95, 0.45, 0.05)
    st.markdown("---")
    st.markdown("**Model:** YOLOv8 Custom")
    st.markdown(f"**Classes:** {len(model.names)}")
    st.markdown("**Input:** JPG / JPEG / PNG")

# ─────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload one or more rice plant images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

# ─────────────────────────────────────────────
# Helper — draw boxes manually for fine control
# ─────────────────────────────────────────────
PALETTE = [
    "#FF5252", "#FF9800", "#FFEB3B", "#66BB6A",
    "#26C6DA", "#42A5F5", "#AB47BC", "#EC407A",
]

def draw_detections(image: Image.Image, result) -> Image.Image:
    """Draw bounding boxes with labels and confidence on PIL image."""
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")

    boxes   = result.boxes
    names   = result.names

    try:
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_conf  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font_label = ImageFont.load_default()
        font_conf  = font_label

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cls_id   = int(box.cls[0])
        conf     = float(box.conf[0])
        label    = names[cls_id]
        color    = PALETTE[cls_id % len(PALETTE)]

        # Box fill (semi-transparent)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.rectangle([x1, y1, x2, y2], fill=color + "22")  # ~14% opacity

        # Label background
        text_str = f"{label}  {conf:.0%}"
        try:
            tw, th = draw.textlength(text_str, font=font_label), 18
        except Exception:
            tw, th = len(text_str) * 8, 18

        pad = 4
        draw.rectangle(
            [x1, y1 - th - pad * 2, x1 + int(tw) + pad * 2, y1],
            fill=color,
        )
        draw.text((x1 + pad, y1 - th - pad), text_str, fill="white", font=font_label)

    return img


# ─────────────────────────────────────────────
# Process each uploaded image
# ─────────────────────────────────────────────
if uploaded_files:
    run_btn = st.button("🔍 Run Detection on All Images", use_container_width=True)

    if run_btn:
        for idx, uploaded_file in enumerate(uploaded_files):
            st.markdown(f"---\n### Image {idx + 1}: `{uploaded_file.name}`")

            image = Image.open(uploaded_file).convert("RGB")
            col_orig, col_det = st.columns(2)

            with col_orig:
                st.markdown("**Original**")
                st.image(image, use_container_width=True)

            with st.spinner(f"Running detection on {uploaded_file.name}…"):
                results = model.predict(
                    source=image,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False,
                )
                result = results[0]

            # Annotated image
            annotated = draw_detections(image, result)

            with col_det:
                st.markdown("**Detections**")
                st.image(annotated, use_container_width=True)

            # ── Summary metrics ──────────────────────
            boxes  = result.boxes
            n_det  = len(boxes)
            names  = result.names

            m1, m2, m3 = st.columns(3)
            avg_conf = float(boxes.conf.mean()) if n_det > 0 else 0.0
            n_classes = len(boxes.cls.unique()) if n_det > 0 else 0

            m1.markdown(f"""<div class="metric-card">
                <div class="label">Objects Detected</div>
                <div class="value">{n_det}</div>
            </div>""", unsafe_allow_html=True)

            m2.markdown(f"""<div class="metric-card">
                <div class="label">Unique Classes</div>
                <div class="value">{n_classes}</div>
            </div>""", unsafe_allow_html=True)

            m3.markdown(f"""<div class="metric-card">
                <div class="label">Avg Confidence</div>
                <div class="value">{avg_conf:.0%}</div>
            </div>""", unsafe_allow_html=True)

            # ── Per-detection table ──────────────────
            st.markdown("#### Detected Objects")

            if n_det == 0:
                st.info("No objects detected above the confidence threshold.")
            else:
                # Sort by confidence descending
                sorted_indices = boxes.conf.argsort(descending=True).tolist()

                for rank, i in enumerate(sorted_indices, start=1):
                    cls_id = int(boxes.cls[i])
                    conf   = float(boxes.conf[i])
                    label  = names[cls_id]
                    x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
                    color  = PALETTE[cls_id % len(PALETTE)]
                    pct    = int(conf * 100)

                    st.markdown(f"""
                    <div class="detection-row">
                        <span class="det-rank">#{rank}</span>
                        <span class="det-class" style="color:{color}">{label}</span>
                        <div class="conf-bar">
                            <div class="conf-fill" style="width:{pct}%; background:{color};"></div>
                        </div>
                        <span class="det-conf">{conf:.2%}</span>
                        <span style="font-size:0.78rem;color:#8b949e;font-family:'IBM Plex Mono',monospace;">
                            [{x1},{y1} → {x2},{y2}]
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Download annotated image ─────────────
            buf = io.BytesIO()
            annotated.save(buf, format="PNG")
            st.download_button(
                label=f"⬇ Download annotated image",
                data=buf.getvalue(),
                file_name=f"detected_{uploaded_file.name}",
                mime="image/png",
            )
