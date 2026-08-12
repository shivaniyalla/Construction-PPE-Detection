import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile

# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="Construction PPE Detection",
    page_icon="🦺",
    layout="wide"
)

# ==============================
# LOAD MODEL
# ==============================

MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ==============================
# TITLE
# ==============================

st.title("🦺 Construction PPE Detection System")

st.write(
    "Upload a construction-site image to detect PPE equipment "
    "and identify potential safety violations."
)

st.divider()

# ==============================
# IMAGE UPLOAD
# ==============================

uploaded_file = st.file_uploader(
    "📷 Upload Construction Image",
    type=["jpg", "jpeg", "png"]
)

# ==============================
# DETECTION
# ==============================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    if st.button("🔍 Detect PPE", type="primary"):

        with st.spinner("Running YOLO detection..."):

            results = model.predict(
                source=image,
                conf=0.30,
                iou=0.50,
                verbose=False
            )

        result = results[0]

        # ==============================
        # ANNOTATED IMAGE
        # ==============================

        annotated_image = result.plot()

        st.subheader("Detection Result")
        st.image(
            annotated_image,
            channels="BGR",
            use_container_width=True
        )

        # ==============================
        # DETECTION SUMMARY
        # ==============================

        st.subheader("📊 Detection Summary")

        detected_classes = []

        if result.boxes is not None:

            for cls in result.boxes.cls:
                class_id = int(cls.item())
                detected_classes.append(model.names[class_id])

        if detected_classes:

            counts = {}

            for name in detected_classes:
                counts[name] = counts.get(name, 0) + 1

            cols = st.columns(min(len(counts), 4))

            for i, (name, count) in enumerate(counts.items()):

                with cols[i % len(cols)]:
                    st.metric(name, count)

        else:

            st.warning("No objects detected.")

        # ==============================
        # SAFETY STATUS
        # ==============================

        violations = [
            "NO-Hardhat",
            "NO-Mask",
            "NO-Safety Vest"
        ]

        detected_violations = [
            name
            for name in detected_classes
            if name in violations
        ]

        st.subheader("🚨 Safety Status")

        if detected_violations:

            st.error("⚠️ SAFETY VIOLATION DETECTED")

            st.write(
                "Detected violations:",
                ", ".join(set(detected_violations))
            )

        else:

            st.success("✅ No PPE violations detected")
