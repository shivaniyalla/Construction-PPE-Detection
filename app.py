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

/* ============================================================
   GENERAL
   ============================================================ */

.block-container {
    padding-top: 20px !important;
    padding-bottom: 30px !important;
    padding-left: 25px !important;
    padding-right: 25px !important;
    max-width: 100% !important;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}


/* ============================================================
   HOME MAIN FRAME
   ============================================================ */

.guardx-frame {
    width: 100%;
    min-height: 760px;
    border: 3px solid #111111;
    background: white;
    overflow: hidden;
    box-sizing: border-box;
}


/* ============================================================
   HEADER
   ============================================================ */

.guardx-header {
    height: 115px;
    border-bottom: 3px solid #111111;

    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;

    text-align: center;
    box-sizing: border-box;
}

.guardx-header-title {
    font-size: 40px;
    font-weight: 800;
    color: #111111;
    line-height: 1.1;
}

.guardx-header-subtitle {
    font-size: 17px;
    color: #444444;
    margin-top: 7px;
}


/* ============================================================
   BODY
   ============================================================ */

.guardx-body {
    display: grid;
    grid-template-columns: 34% 66%;
    min-height: 642px;
}


/* ============================================================
   LEFT SECTION
   ============================================================ */

.guardx-left {
    border-right: 3px solid #111111;
    padding: 35px;

    display: flex;
    flex-direction: column;
    justify-content: space-between;

    min-height: 642px;
    box-sizing: border-box;
}

.aicw-title {
    font-size: 25px;
    font-weight: 800;
    color: #111111;
    line-height: 1.3;
}

.capstone-title {
    font-size: 18px;
    font-weight: 600;
    color: #333333;
    margin-top: 18px;
}


/* ============================================================
   RIGHT SECTION
   ============================================================ */

.guardx-right {
    min-width: 0;
}


/* ============================================================
   PROJECT TITLE
   ============================================================ */

.guardx-title-box {
    height: 105px;

    border-bottom: 3px solid #111111;

    display: flex;
    align-items: center;

    padding-left: 35px;

    box-sizing: border-box;
}

.guardx-title {
    font-size: 31px;
    font-weight: 800;
    color: #111111;
}


/* ============================================================
   DESCRIPTION
   ============================================================ */

.guardx-description {
    min-height: 315px;

    border-bottom: 3px solid #111111;

    padding: 30px 35px;

    box-sizing: border-box;
}

.guardx-section-title {
    font-size: 20px;
    font-weight: 800;
    color: #111111;
    margin-bottom: 15px;
}

.guardx-description-text {
    font-size: 16px;
    line-height: 1.7;
    color: #222222;
    text-align: justify;
}


/* ============================================================
   TEAM + GUIDE
   ============================================================ */

.guardx-bottom {
    display: grid;
    grid-template-columns: 55% 45%;
    min-height: 220px;
}

.guardx-team {
    border-right: 3px solid #111111;
    padding: 28px 35px;
    box-sizing: border-box;
}

.guardx-members {
    font-size: 15px;
    line-height: 2;
    color: #222222;
}

.guardx-guide {
    padding: 28px 35px;
    box-sizing: border-box;
}

.guardx-guide-name {
    font-size: 18px;
    font-weight: 700;
    margin-top: 15px;
    color: #111111;
}

.guardx-guide-designation {
    font-size: 15px;
    margin-top: 7px;
    color: #333333;
}


/* ============================================================
   PREDICT BUTTON
   ============================================================ */

.predict-area {
    margin-top: -105px;
    width: 34%;
    padding-left: 35px;
    padding-right: 35px;
    box-sizing: border-box;
    position: relative;
    z-index: 50;
}

.predict-area button {
    width: 100% !important;
    height: 55px !important;

    border: 2px solid #111111 !important;
    border-radius: 6px !important;

    font-size: 18px !important;
    font-weight: 800 !important;
}


/* ============================================================
   DETECTION PAGE
   ============================================================ */

.detection-title {
    text-align: center;
    font-size: 40px;
    font-weight: 800;
    color: #111111;
}

.detection-subtitle {
    text-align: center;
    font-size: 17px;
    color: #555555;
    margin-bottom: 25px;
}


/* ============================================================
   MOBILE / SMALL SCREEN
   ============================================================ */

@media (max-width: 900px) {

    .guardx-body {
        grid-template-columns: 1fr;
    }

    .guardx-left {
        border-right: none;
        border-bottom: 3px solid #111111;
        min-height: 300px;
    }

    .guardx-bottom {
        grid-template-columns: 1fr;
    }

    .guardx-team {
        border-right: none;
        border-bottom: 3px solid #111111;
    }

    .predict-area {
        width: 100%;
        margin-top: -80px;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PAGE 1 — GUARDX-AI HOME PAGE
# ============================================================

if st.session_state.page == "home":

    st.markdown("""
    <div class="guardx-frame">

        <!-- HEADER -->

        <div class="guardx-header">

            <div class="guardx-header-title">
                GuardX-AI
            </div>

            <div class="guardx-header-subtitle">
                AI-Powered Construction PPE Detection System
            </div>

        </div>


        <!-- BODY -->

        <div class="guardx-body">


            <!-- LEFT SECTION -->

            <div class="guardx-left">

                <div>

                    <div class="aicw-title">
                        AI Career for Women
                        <br>
                        (AICW)
                    </div>

                    <div class="capstone-title">
                        Capstone Project
                    </div>

                </div>

            </div>


            <!-- RIGHT SECTION -->

            <div class="guardx-right">


                <!-- TITLE -->

                <div class="guardx-title-box">

                    <div class="guardx-title">
                        GuardX-AI
                    </div>

                </div>


                <!-- DESCRIPTION -->

                <div class="guardx-description">

                    <div class="guardx-section-title">
                        DESCRIPTION
                    </div>

                    <div class="guardx-description-text">

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

                <div class="guardx-bottom">


                    <!-- TEAM MEMBERS -->

                    <div class="guardx-team">

                        <div class="guardx-section-title">
                            TEAM MEMBERS
                        </div>

                        <div class="guardx-members">

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


                    <!-- GUIDE -->

                    <div class="guardx-guide">

                        <div class="guardx-section-title">
                            GUIDE
                        </div>

                        <div class="guardx-guide-name">
                            MD.Abdul Aziz
                        </div>

                        <div class="guardx-guide-designation">
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
        '<div class="predict-area">',
        unsafe_allow_html=True
    )

    if st.button(
        "🚀 PREDICT",
        key="predict_home",
        type="primary",
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
    # BACK TO HOME
    # ========================================================

    if st.button(
        "⬅️ Back to Home",
        key="back_home"
    ):

        st.session_state.page = "home"
        st.rerun()


    # ========================================================
    # PAGE TITLE
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
            "Make sure 'best.pt' is in the same GitHub repository "
            "folder as app.py."
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # IMAGE DETECTION
    # ========================================================

    st.divider()

    st.subheader("📷 Image Detection")


    uploaded_file = st.file_uploader(
        "Upload Construction Image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )


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


            st.subheader(
                "🚨 Safety Status"
            )


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
    # VIDEO DETECTION
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

                video_extension = os.path.splitext(
                    uploaded_video.name
                )[1]


                if not video_extension:

                    video_extension = ".mp4"


                input_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=video_extension
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
                # FRAME BY FRAME DETECTION
                # =============================================

                while cap.isOpened():

                    ret, frame = cap.read()


                    if not ret:

                        break


                    results = model.predict(
                        source=frame,
                        conf=0.30,
                        iou=0.50,
                        verbose=False
                    )


                    annotated_frame = results[0].plot()


                    out.write(
                        annotated_frame
                    )


                    frame_count += 1


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

                except Exception:

                    pass


    # ========================================================
    # FOOTER
    # ========================================================

    st.divider()

    st.caption(
        "🦺 GuardX-AI • Construction PPE Detection System • "
        "Powered by YOLO & Streamlit"
    )
