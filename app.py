import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import subprocess
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Construction PPE Detection",
    page_icon="🦺",
    layout="wide"
)

# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "best.pt"


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()

# ============================================================
# TITLE
# ============================================================

st.title("🦺 Construction PPE Detection System")

st.write(
    "Upload a construction-site image or video to detect PPE "
    "equipment and identify potential safety violations."
)

st.divider()

# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("📷 Image Detection")

uploaded_file = st.file_uploader(
    "Upload Construction Image",
    type=["jpg", "jpeg", "png"]
)

# ============================================================
# IMAGE DETECTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Original Image")

    st.image(
        image,
        width=500
    )

    if st.button(
        "🔍 Detect PPE",
        type="primary"
    ):

        with st.spinner("Running YOLO detection..."):

            results = model.predict(
                source=image,
                conf=0.30,
                iou=0.50,
                verbose=False
            )

        result = results[0]

        # ====================================================
        # ANNOTATED IMAGE
        # ====================================================

        annotated_image = result.plot()

        st.subheader("🎯 Detection Result")

        st.image(
            annotated_image,
            channels="BGR",
            width=500
        )

        # ====================================================
        # DETECTION SUMMARY
        # ====================================================

        st.subheader("📊 Detection Summary")

        detected_classes = []

        if result.boxes is not None:

            for cls in result.boxes.cls:

                class_id = int(cls.item())

                detected_classes.append(
                    model.names[class_id]
                )

        if detected_classes:

            counts = {}

            for name in detected_classes:

                counts[name] = (
                    counts.get(name, 0) + 1
                )

            cols = st.columns(
                min(len(counts), 4)
            )

            for i, (name, count) in enumerate(
                counts.items()
            ):

                with cols[i % len(cols)]:

                    st.metric(
                        name,
                        count
                    )

        else:

            st.warning(
                "No objects detected."
            )

        # ====================================================
        # SAFETY STATUS
        # ====================================================

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

            st.error(
                "⚠️ SAFETY VIOLATION DETECTED"
            )

            st.write(
                "Detected violations:",
                ", ".join(
                    set(detected_violations)
                )
            )

        else:

            st.success(
                "✅ No PPE violations detected"
            )


# ============================================================
# VIDEO UPLOAD
# ============================================================

st.divider()

st.subheader("🎥 Video Detection")

uploaded_video = st.file_uploader(
    "Upload Construction Video",
    type=["mp4", "avi", "mov", "mkv"],
    key="video_uploader"
)

# ============================================================
# VIDEO DETECTION
# ============================================================

if uploaded_video is not None:

    st.subheader("🎬 Original Video")

    # Show uploaded video
    st.video(uploaded_video)

    if st.button(
        "🎥 Detect PPE in Video",
        type="primary"
    ):

        with st.spinner(
            "🤖 Processing video... Please wait."
        ):

            # =================================================
            # SAVE UPLOADED VIDEO
            # =================================================

            input_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            input_file.write(
                uploaded_video.read()
            )

            input_file.close()

            # =================================================
            # OPEN VIDEO
            # =================================================

            cap = cv2.VideoCapture(
                input_file.name
            )

            width = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            height = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            if fps <= 0:

                fps = 25

            total_frames = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            # =================================================
            # RAW OUTPUT VIDEO
            # =================================================

            raw_output = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            raw_output_path = raw_output.name

            raw_output.close()

            # =================================================
            # VIDEO WRITER
            # =================================================

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            out = cv2.VideoWriter(
                raw_output_path,
                fourcc,
                fps,
                (width, height)
            )

            # =================================================
            # PROGRESS BAR
            # =================================================

            progress_bar = st.progress(0)

            frame_count = 0

            # =================================================
            # PROCESS VIDEO FRAME BY FRAME
            # =================================================

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:

                    break

                # =============================================
                # YOLO DETECTION
                # =============================================

                results = model.predict(
                    source=frame,
                    conf=0.30,
                    iou=0.50,
                    verbose=False
                )

                # =============================================
                # DRAW DETECTIONS
                # =============================================

                annotated_frame = results[0].plot()

                # =============================================
                # WRITE FRAME
                # =============================================

                out.write(
                    annotated_frame
                )

                frame_count += 1

                # =============================================
                # UPDATE PROGRESS
                # =============================================

                if total_frames > 0:

                    progress = (
                        frame_count /
                        total_frames
                    )

                    progress_bar.progress(
                        min(progress, 1.0)
                    )

            # =================================================
            # RELEASE VIDEO
            # =================================================

            cap.release()

            out.release()

            progress_bar.empty()

            # =================================================
            # CONVERT OUTPUT TO BROWSER-FRIENDLY MP4
            # =================================================

            final_output = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            final_output_path = final_output.name

            final_output.close()

            ffmpeg_command = [
                "ffmpeg",
                "-y",
                "-i",
                raw_output_path,
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                final_output_path
            ]

            conversion = subprocess.run(
                ffmpeg_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # =================================================
            # CHECK CONVERSION
            # =================================================

            if conversion.returncode != 0:

                st.error(
                    "❌ Video conversion failed."
                )

                st.code(
                    conversion.stderr.decode(
                        errors="ignore"
                    )
                )

            else:

                st.success(
                    "✅ Video detection completed!"
                )

                st.subheader(
                    "🎯 Detection Result"
                )

                st.video(
                    final_output_path
                )

            # =================================================
            # CLEANUP
            # =================================================

            try:

                os.remove(
                    input_file.name
                )

                os.remove(
                    raw_output_path
                )

            except:

                pass

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🦺 Construction PPE Detection System • "
    "Powered by YOLO & Streamlit"
)
