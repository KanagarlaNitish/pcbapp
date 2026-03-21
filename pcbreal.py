import streamlit as st
import torch
from PIL import Image, ImageDraw
import pathlib
import pandas as pd
import numpy as np

# -------- WINDOWS POSIXPATH FIX --------
pathlib.PosixPath = pathlib.WindowsPath

# -------- PAGE CONFIG --------
st.set_page_config(
    page_title="AI PCB Inspector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------- CUSTOM UI --------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg,#0B0F14,#0F172A);
}
h1 {color:#00FFAA;}
h2,h3 {color:#E5E7EB;}
.stFileUploader{
border:2px dashed #00FFAA;
border-radius:12px;
}
</style>
""", unsafe_allow_html=True)

# -------- HEADER --------
st.markdown(
"""
# 🧠 **AI PCB DEFECT INSPECTOR**
### Deep Learning–Powered PCB Quality Inspection
"""
)

st.markdown("---")

# -------- SIDEBAR --------
st.sidebar.markdown("## 🔍 Inspection Control")

option = st.sidebar.radio(
    "Select Inspection Stage:",
    ("Bare PCB Inspection", "Soldering Stage Inspection")
)

st.sidebar.markdown("### Models Used")
st.sidebar.write("Bare PCB → bestt.pt")
st.sidebar.write("Soldering → best_win.pt")

# -------- FILE UPLOAD --------
st.markdown("## 📤 Upload Image")

uploaded_file = st.file_uploader(
    "Upload PCB Image",
    type=["jpg","jpeg","png","bmp","webp"]
)

# =====================================
# IMAGE PREPROCESSING
# =====================================
def preprocess_image(uploaded_file):

    image = Image.open(uploaded_file)

    if image.mode != "RGB":
        image = image.convert("RGB")

    max_size = 1600
    if max(image.size) > max_size:
        image.thumbnail((max_size,max_size))

    return image

# =====================================
# LOAD YOLO MODEL
# =====================================
@st.cache_resource
def load_model(model_path):

    model = torch.hub.load(
        r"C:\Users\bambo\yolov5",
        "custom",
        path=model_path,
        source="local"
    )

    model.conf = 0.05

    return model

# =====================================
# SLIDING WINDOW DETECTION (IMPROVED UI)
# =====================================
def tiled_detection(model, image, tile_size=640):

    img = np.array(image)
    h, w = img.shape[:2]

    results_boxes = []
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)

    scale = max(h, w) / 800
    thickness = int(3 * scale)
    font_size = int(12 * scale)

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):

            tile = img[y:y+tile_size, x:x+tile_size]

            if tile.shape[0] == 0 or tile.shape[1] == 0:
                continue

            results = model(tile)
            df = results.pandas().xyxy[0]

            df = df[df['confidence'] > 0.05]

            for _, row in df.iterrows():

                x1 = int(row['xmin']) + x
                y1 = int(row['ymin']) + y
                x2 = int(row['xmax']) + x
                y2 = int(row['ymax']) + y

                # enlarge box slightly
                expand = int(5 * scale)
                x1 -= expand
                y1 -= expand
                x2 += expand
                y2 += expand

                label = f"{row['name']} {row['confidence']:.2f}"

                # draw bounding box
                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline="lime",
                    width=thickness
                )

                # label background
                text_w = len(label) * font_size // 2
                text_h = font_size + 4

                draw.rectangle(
                    [x1, y1 - text_h, x1 + text_w, y1],
                    fill="lime"
                )

                draw.text(
                    (x1 + 2, y1 - text_h),
                    label,
                    fill="black"
                )

                results_boxes.append({
                    "name": row["name"],
                    "confidence": row["confidence"]
                })

    return draw_img, results_boxes

# =====================================

if uploaded_file is not None:

    image = preprocess_image(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📷 Input Image")
        st.image(image, use_container_width=True)

    # -------- MODEL SELECTION --------
    if option == "Bare PCB Inspection":
        model_path = r"C:\Users\bambo\OneDrive\Desktop\Desktop\MiniProject\BarePcb-stage\Work Done\yolov5-(good accuarcy)\bestt.pt"
        st.sidebar.success("Bare PCB Model Active")
    else:
        model_path = r"C:\Users\bambo\yolov5\best_win.pt"
        st.sidebar.success("Soldering Model Active")

    # -------- LOAD MODEL --------
    with st.spinner("Loading AI model..."):
        model = load_model(model_path)

    # -------- RUN DETECTION --------
    with st.spinner("Analyzing image..."):
        output_img, boxes = tiled_detection(model, image)

    with col2:
        st.markdown("### 🔍 Detection Output")
        st.image(output_img, use_container_width=True)
        st.caption("Tip: Zoom into the image to inspect defects clearly")

    st.markdown("---")

    # -------- RESULTS --------
    st.markdown("## 📊 Detection Summary")

    if len(boxes) == 0:
        st.info("No objects detected in this image.")
    else:

        df = pd.DataFrame(boxes)
        df['confidence'] = df['confidence'].round(3)

        st.dataframe(df)

        st.markdown("### 📈 Defect Analytics")

        count_df = df['name'].value_counts().reset_index()
        count_df.columns = ['Defect', 'Count']

        st.dataframe(count_df)

    # -------- DOWNLOAD --------
    st.markdown("## 📥 Download Result")

    output_img.save("pcb_result.jpg")

    with open("pcb_result.jpg","rb") as f:
        st.download_button(
            "Download Annotated Image",
            f,
            "pcb_detection_result.jpg",
            "image/jpeg"
        )