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
    page_title="GuardX AI",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# SESSION STATE - PAGE NAVIGATION
# ============================================================

if "show_project" not in st.session_state:
    st.session_state.show_project = False


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Remove default Streamlit top spacing */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    /* Main outer design */
    .main-frame {
        border: 3px solid #111827;
        border-radius: 8px;
        overflow: hidden;
        background: white;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
    }

    /* Header */
    .main-header {
        min-height: 115px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        padding: 20px;
        border-bottom: 3px solid #111827;
        background: #ffffff;
    }

    .main-header h1 {
        margin: 0;
        font-size: 38px;
        font-weight: 800;
        letter-spacing: 1px;
        color: #111827;
    }

    .main-header p {
        margin: 7px 0 0 0;
        font-size: 18px;
        font-weight: 600;
        color: #4b5563;
    }

    /* Left side */
    .left-panel {
        min-height: 600px;
        padding: 35px 28px;
        border-right: 3px solid #111827;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        background: #ffffff;
    }

    .aicw-text {
        font-size: 23px;
        font-weight: 800;
        color: #111827;
        line-height: 1.25;
    }

    .capstone-text {
        margin-top: 18px;
        font-size: 17px;
        font-weight: 600;
        color: #4b5563;
    }

    /* Right title */
    .title-box {
        min-height: 105px;
        padding: 28px 35px;
        border-bottom: 3px solid #111827;
        display: flex;
        align-items: center;
    }

    .title-box h2 {
        margin: 0;
        font-size: 30px;
        font-weight: 800;
        color: #111827;
    }

    /* Description */
    .description-box {
        min-height: 255px;
        padding: 30px 35px;
        border-bottom: 3px solid #111827;
    }

    .section-heading {
        font-size: 20px;
        font-weight: 800;
        margin-bottom: 14px;
        color: #111827;
    }

    .description-text {
        font-size: 16px;
        line-height: 1.75;
        color: #374151;
        text-align: justify;
    }

    /* Bottom area */
    .bottom-left {
        min-height: 205px;
        padding: 28px 35px;
        border-right: 3px solid #111827;
    }

    .bottom-right {
        min-height: 205px;
        padding: 28px 35px;
    }

    .member {
        font-size: 15px;
        line-height: 1.9;
        color: #374151;
    }

    .guide-name {
        font-size: 18px;
        font-weight: 700;
        color: #111827;
        margin-top: 15px;
    }

    .guide-designation {
        font-size: 15px;
        color: #4b5563;
        margin-top: 5px;
    }

    /* Predict button */
    div.stButton > button {
        width: 100%;
        min-height: 52px;
        border-radius: 7px;
        border: 2px solid #111827;
        font-size: 18px;
        font-weight: 800;
        letter-spacing: 0.5px;
    }

    /* Detection page */
    .project-title {
        text-align: center;
        font-size: 38px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 5px;
    }

    .project-subtitle {
        text-align: center;
        font-size: 17px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PAGE 1 - PROJECT INTRODUCTION
# ============================================================

if not st.session_state.show_project:

    # Header
    st.markdown(
        """
        <div class="main-frame">

            <div class="main-header">
                <h1>GUARDX AI</h1>
                <p>AI-Powered Construction PPE Detection System</p>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # Main two-column structure
    left_col, right_col = st.columns(
        [0.37, 0.63],
        gap="small"
    )

    # ========================================================
    # LEFT SIDE
    # ========================================================

    with left_col:

        st.markdown(
            """
            <div class="left-panel">

                <div>
                    <div class="aicw-text">
                        AI Career for Women<br>
                        (AICW)
                    </div>

                    <div class="capstone-text">
                        Capstone Project
                    </div>
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        # PREDICT button
        if st.button(
            "🚀 PREDICT",
            key="predict_button",
            type="primary",
            use_container_width=True
        ):
            st.session_state.show_project = True
            st.rerun()


    # ========================================================
    # RIGHT SIDE
    # ========================================================

    with right_col:

        # TITLE
        st.markdown(
            """
            <div class="title-box">
                <h2>Construction PPE Detection</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        # DESCRIPTION
        st.markdown(
            """
            <div class="description-box">

                <div class="section-heading">
                    DESCRIPTION
                </div>

                <div class="description-text">

                    Construction sites involve high-risk activities where
                    proper Personal Protective Equipment (PPE) is essential
                    for worker safety. However, manually monitoring whether
                    every worker is wearing the required PPE continuously is
                    difficult, time-consuming, and prone to human error.
                    GuardX AI is an AI-powered Construction PPE Detection
                    System designed to automatically identify safety equipment
                    such as hardhats, masks, and safety vests from construction
                    site images and videos. Using YOLO-based object detection,
                    the system detects PPE items and identifies potential
                    safety violations. The solution provides visual detection
                    results, helping improve safety monitoring, reduce manual
                    inspection effort, and support faster identification of
                    unsafe working conditions.

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        # ====================================================
        # BOTTOM SECTION
        # ====================================================

        team_col, guide_col = st.columns(
            [0.55, 0.45],
            gap="small"
        )

        # TEAM MEMBERS
        with team_col:

            st.markdown(
                """
                <div class="bottom-left">

                    <div class="section-heading">
                        TEAM MEMBERS
                    </div>

                    <div class="member">
                        <b>1.</b> Member Name — Email<br>
                        <b>2.</b> Member Name — Email<br>
                        <b>3.</b> Member Name — Email<br>
                        <b>4.</b> Member Name — Email
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        # GUIDE
        with guide_col:

            st.markdown(
                """
                <div class="bottom-right">

                    <div class="section-heading">
                        GUIDE
                    </div>

                    <div class="guide-name">
                        Guide Name
                    </div>

                    <div class="guide-designation">
                        Designation
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# PAGE 2 - ACTUAL PPE DETECTION APPLICATION
# ============================================================

else:

    # ========================================================
    # BACK TO HOME
    # ========================================================

    if st.button(
        "⬅️ Back to Home",
        key="back_home"
    ):
        st.session_state.show_project = False
        st.rerun()


    # ========================================================
    # PROJECT TITLE
    # ========================================================

    st.markdown(
        """
        <div class="project-title">
            🦺 GUARDX AI
        </div>

        <div class="project-subtitle">
            Construction PPE Detection System
        </div>
        """,
        unsafe_allow_html=True
    )


    st.divider()


    # ========================================================
    # LOAD MODEL
    # ========================================================

    MODEL_PATH = "best.pt"


    @st.cache_resource
    def load_model():
        return YOLO(MODEL_PATH)


    try:
        model = load_model()

    except Exception as e:

        st.error(
            "❌ Unable to load the YOLO model."
        )

        st.info(
            "Make sure that 'best.pt' is present in the project folder."
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # DESCRIPTION
    # ========================================================

    st.write(
        "Upload a construction-site image or video to detect PPE "
        "equipment and identify potential safety violations."
    )


    # ========================================================
    # IMAGE UPLOAD
    # ========================================================

    st.divider()

    st.subheader("📷 Image Detection")


    uploaded_file = st.file_uploader(
        "Upload Construction Image",
        type=["jpg", "jpeg", "png"]
    )


    # ========================================================
    # IMAGE DETECTION
    # ========================================================

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.subheader("Original Image")

        st.image(
            image,
            width=500
        )


        if st.button(
            "🔍 Detect PPE",
            type="primary",
            key="image_detect"
        ):

            with st.spinner(
                "Running YOLO detection..."
            ):

                results = model.predict(
                    source=image,
                    conf=0.30,
                    iou=0.50,
                    verbose=False
                )


            result = results[0]


            # =================================================
            # ANNOTATED IMAGE
            # =================================================

            annotated_image = result.plot()

            st.subheader(
                "🎯 Detection Result"
            )

            st.image(
                annotated_image,
                channels="BGR",
                width=500
            )


            # =================================================
            # DETECTION SUMMARY
            # =================================================

            st.subheader(
                "📊 Detection Summary"
            )

            detected_classes = []


            if result.boxes is not None:

                for cls in result.boxes.cls:

                    class_id = int(
                        cls.item()
                    )

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


            # =================================================
            # SAFETY STATUS
            # =================================================

            st.subheader(
                "🚨 Safety Status"
            )


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


    # ========================================================
    # VIDEO UPLOAD
    # ========================================================

    st.divider()

    st.subheader(
        "🎥 Video Detection"
    )


    uploaded_video = st.file_uploader(
        "Upload Construction Video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader"
    )


    # ========================================================
    # VIDEO DETECTION
    # ========================================================

    if uploaded_video is not None:

        st.subheader(
            "🎬 Original Video"
        )

        # Show uploaded video
        st.video(uploaded_video)


        if st.button(
            "🎥 Detect PPE in Video",
            type="primary",
            key="video_detect"
        ):

            with st.spinner(
                "🤖 Processing video... Please wait."
            ):

                # =================================================
                # SAVE UPLOADED VIDEO
                # =================================================

                file_extension = os.path.splitext(
                    uploaded_video.name
                )[1]

                if not file_extension:
                    file_extension = ".mp4"


                input_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=file_extension
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


    # ========================================================
    # FOOTER
    # ========================================================

    st.divider()

    st.caption(
        "🦺 GuardX AI • Construction PPE Detection System • "
        "Powered by YOLO & Streamlit"
    )
