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
    page_title="GuardX-AI",
    page_icon="🦺",
    layout="wide"
)

# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 25px;
    padding-bottom: 30px;
    max-width: 1500px;
}

/* ================= HOME PAGE ================= */

.home-container {
    width: 100%;
    border: 3px solid #111111;
    border-radius: 5px;
    overflow: hidden;
    background: white;
}

/* Header */

.home-header {
    height: 120px;
    border-bottom: 3px solid #111111;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

.home-header-title {
    font-size: 42px;
    font-weight: 800;
    margin: 0;
    color: #111111;
}

.home-header-subtitle {
    font-size: 18px;
    margin-top: 5px;
    color: #444444;
}

/* Main left/right */

.home-body {
    display: grid;
    grid-template-columns: 35% 65%;
}

/* Left */

.left-section {
    min-height: 650px;
    border-right: 3px solid #111111;
    padding: 35px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.aicw {
    font-size: 25px;
    font-weight: 800;
    line-height: 1.3;
}

.capstone {
    margin-top: 15px;
    font-size: 18px;
    font-weight: 600;
}

.predict-area {
    margin-top: auto;
    margin-bottom: 30px;
}

/* Right */

.right-section {
    min-height: 650px;
}

/* Title */

.project-title {
    height: 105px;
    border-bottom: 3px solid #111111;
    display: flex;
    align-items: center;
    padding-left: 35px;
}

.project-title-text {
    font-size: 32px;
    font-weight: 800;
}

/* Description */

.description-section {
    min-height: 285px;
    border-bottom: 3px solid #111111;
    padding: 30px 35px;
}

.section-heading {
    font-size: 21px;
    font-weight: 800;
    margin-bottom: 15px;
}

.description-text {
    font-size: 16px;
    line-height: 1.7;
    color: #333333;
    text-align: justify;
}

/* Bottom */

.bottom-section {
    display: grid;
    grid-template-columns: 55% 45%;
    min-height: 230px;
}

.team-section {
    padding: 30px 35px;
    border-right: 3px solid #111111;
}

.guide-section {
    padding: 30px 35px;
}

.team-member {
    font-size: 15px;
    line-height: 2;
    color: #333333;
}

.guide-name {
    font-size: 18px;
    font-weight: 700;
    margin-top: 15px;
}

.guide-designation {
    font-size: 15px;
    margin-top: 7px;
    color: #444444;
}

/* Streamlit button */

.predict-button button {
    width: 100% !important;
    height: 55px !important;
    border-radius: 6px !important;
    font-size: 19px !important;
    font-weight: 800 !important;
}

/* ================= DETECTION PAGE ================= */

.detection-title {
    text-align: center;
    font-size: 40px;
    font-weight: 800;
    margin-bottom: 5px;
}

.detection-subtitle {
    text-align: center;
    color: #555555;
    font-size: 17px;
    margin-bottom: 25px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PAGE 1 — HOME
# ============================================================

if st.session_state.page == "home":

    st.markdown("""
    <div class="home-container">

        <!-- HEADER -->

        <div class="home-header">

            <div class="home-header-title">
                GuardX-AI
            </div>

            <div class="home-header-subtitle">
                AI-Powered Construction PPE Detection System
            </div>

        </div>


        <!-- BODY -->

        <div class="home-body">

            <!-- LEFT SECTION -->

            <div class="left-section">

                <div>

                    <div class="aicw">
                        AI Career for Women
                        <br>
                        (AICW)
                    </div>

                    <div class="capstone">
                        Capstone Project
                    </div>

                </div>

                <div class="predict-area">

                    <div style="
                        font-size:20px;
                        font-weight:800;
                        margin-bottom:12px;
                    ">
                        Start Detection
                    </div>

                    <!-- PREDICT BUTTON IS ADDED BELOW -->

                </div>

            </div>


            <!-- RIGHT SECTION -->

            <div class="right-section">

                <!-- TITLE -->

                <div class="project-title">

                    <div class="project-title-text">
                        GuardX-AI
                    </div>

                </div>


                <!-- DESCRIPTION -->

                <div class="description-section">

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
                        such as hardhats, masks, and safety vests from
                        construction-site images and videos. Using YOLO-based
                        object detection, the system detects PPE items and
                        identifies potential safety violations. The solution
                        provides visual detection results, helping improve
                        safety monitoring, reduce manual inspection effort,
                        and support faster identification of unsafe working
                        conditions.

                    </div>

                </div>


                <!-- TEAM + GUIDE -->

                <div class="bottom-section">

                    <div class="team-section">

                        <div class="section-heading">
                            TEAM MEMBERS
                        </div>

                        <div class="team-member">

                            <b>1.</b>
                            Y.D.V.Sivani -
                            yallashivani@gmail.com

                            <br>

                            <b>2.</b>
                            V.L.S.Asritha -
                            Asrithavantipalli@gmail.com

                            <br>

                            <b>3.</b>
                            R.Likhitha -
                            likhitharayudu@gmail.com

                            <br>

                            <b>4.</b>
                            S.Poojitha sri -
                            pujithasari@gmail.com

                        </div>

                    </div>


                    <div class="guide-section">

                        <div class="section-heading">
                            GUIDE
                        </div>

                        <div class="guide-name">
                            MD.Abdul Aziz
                        </div>

                        <div class="guide-designation">
                            Trainer, Co-Lead-AICW
                        </div>

                    </div>

                </div>

            </div>

        </div>

    </div>
    """, unsafe_allow_html=True)


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    st.markdown(
        '<div class="predict-button">',
        unsafe_allow_html=True
    )

    if st.button(
        "🚀 PREDICT",
        type="primary",
        key="predict_home",
        use_container_width=True
    ):
        st.session_state.page = "detection"
        st.rerun()

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 2 — PPE DETECTION
# ============================================================

else:

    # ========================================================
    # BACK BUTTON
    # ========================================================

    if st.button(
        "⬅️ Back to Home",
        key="back_home"
    ):
        st.session_state.page = "home"
        st.rerun()


    # ========================================================
    # TITLE
    # ========================================================

    st.markdown(
        """
        <div class="detection-title">
            🦺 GuardX-AI
        </div>

        <div class="detection-subtitle">
            Construction PPE Detection System
        </div>
        """,
        unsafe_allow_html=True
    )


    st.write(
        "Upload a construction-site image or video to detect PPE "
        "equipment and identify potential safety violations."
    )


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
            "Make sure 'best.pt' is present in the same project folder as app.py."
        )

        st.exception(e)

        st.stop()


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
            key="detect_image"
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


    # ============================================================
    # VIDEO UPLOAD
    # ============================================================

    st.divider()

    st.subheader(
        "🎥 Video Detection"
    )


    uploaded_video = st.file_uploader(
        "Upload Construction Video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader"
    )


    # ============================================================
    # VIDEO DETECTION
    # ============================================================

    if uploaded_video is not None:

        st.subheader(
            "🎬 Original Video"
        )


        st.video(
            uploaded_video
        )


        if st.button(
            "🎥 Detect PPE in Video",
            type="primary",
            key="detect_video"
        ):

            with st.spinner(
                "🤖 Processing video... Please wait."
            ):

                # =============================================
                # SAVE UPLOADED VIDEO
                # =============================================

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


                # =============================================
                # OPEN VIDEO
                # =============================================

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


                # =============================================
                # RAW OUTPUT
                # =============================================

                raw_output = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )


                raw_output_path = raw_output.name


                raw_output.close()


                # =============================================
                # VIDEO WRITER
                # =============================================

                fourcc = cv2.VideoWriter_fourcc(
                    *"mp4v"
                )


                out = cv2.VideoWriter(
                    raw_output_path,
                    fourcc,
                    fps,
                    (width, height)
                )


                # =============================================
                # PROGRESS
                # =============================================

                progress_bar = st.progress(0)

                frame_count = 0


                # =============================================
                # FRAME-BY-FRAME DETECTION
                # =============================================

                while cap.isOpened():

                    ret, frame = cap.read()


                    if not ret:

                        break


                    # YOLO detection

                    results = model.predict(
                        source=frame,
                        conf=0.30,
                        iou=0.50,
                        verbose=False
                    )


                    # Draw detections

                    annotated_frame = results[0].plot()


                    # Write frame

                    out.write(
                        annotated_frame
                    )


                    frame_count += 1


                    # Progress

                    if total_frames > 0:

                        progress = (
                            frame_count /
                            total_frames
                        )


                        progress_bar.progress(
                            min(progress, 1.0)
                        )


                # =============================================
                # RELEASE
                # =============================================

                cap.release()

                out.release()

                progress_bar.empty()


                # =============================================
                # FINAL MP4
                # =============================================

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


                # =============================================
                # RESULT
                # =============================================

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


                # =============================================
                # CLEANUP
                # =============================================

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
        "🦺 GuardX-AI • Construction PPE Detection System • "
        "Powered by YOLO & Streamlit"
    )
