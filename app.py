import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
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
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = 1

if "result_ready" not in st.session_state:
    st.session_state.result_ready = False

if "result_type" not in st.session_state:
    st.session_state.result_type = None

if "result_data" not in st.session_state:
    st.session_state.result_data = None


# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best.pt"
)

CONF_THRESHOLD = 0.30


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* ==========================================================
   STREAMLIT CLEANUP
   ========================================================== */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.block-container {
    padding-top: 0.8rem !important;
    padding-bottom: 1rem !important;
    padding-left: 1.2rem !important;
    padding-right: 1.2rem !important;
    max-width: 1400px !important;
}


/* ==========================================================
   PAGE 1 - MAIN FRAME
   COMPACT ASPECT RATIO
   ========================================================== */

.guardx-frame {
    width: 100%;
    height: 610px;

    border: 2px solid #111111;
    background: #ffffff;

    overflow: hidden;
    box-sizing: border-box;
}


/* ==========================================================
   HEADER
   ========================================================== */

.guardx-header {
    height: 82px;

    border-bottom: 2px solid #111111;

    display: flex;
    flex-direction: column;

    justify-content: center;
    align-items: center;

    text-align: center;

    box-sizing: border-box;
}

.guardx-header-title {
    font-size: 30px;
    font-weight: 800;
    line-height: 1.05;
    color: #111111;
}

.guardx-header-subtitle {
    font-size: 13px;
    margin-top: 5px;
    color: #444444;
}


/* ==========================================================
   MAIN BODY
   ========================================================== */

.guardx-body {
    height: 526px;

    display: grid;
    grid-template-columns: 29% 71%;
}


/* ==========================================================
   LEFT PANEL
   ========================================================== */

.guardx-left {
    border-right: 2px solid #111111;

    padding: 25px;

    display: flex;
    flex-direction: column;

    justify-content: space-between;

    box-sizing: border-box;
}

.aicw-title {
    font-size: 21px;
    font-weight: 800;

    line-height: 1.25;

    color: #111111;
}

.capstone-title {
    font-size: 15px;
    font-weight: 600;

    margin-top: 12px;

    color: #444444;
}


/* ==========================================================
   RIGHT PANEL
   ========================================================== */

.guardx-right {
    min-width: 0;
}


/* ==========================================================
   PROJECT TITLE
   ========================================================== */

.guardx-project-title {
    height: 72px;

    border-bottom: 2px solid #111111;

    display: flex;
    align-items: center;

    padding: 0 25px;

    box-sizing: border-box;
}

.guardx-project-title-text {
    font-size: 25px;
    font-weight: 800;
    color: #111111;
}


/* ==========================================================
   DESCRIPTION
   ========================================================== */

.guardx-description {
    height: 275px;

    border-bottom: 2px solid #111111;

    padding: 22px 25px;

    box-sizing: border-box;
}

.section-heading {
    font-size: 17px;
    font-weight: 800;

    margin-bottom: 10px;

    color: #111111;
}

.description-text {
    font-size: 13px;
    line-height: 1.55;

    color: #222222;

    text-align: justify;
}


/* ==========================================================
   TEAM + GUIDE
   ========================================================== */

.guardx-bottom {
    height: 179px;

    display: grid;

    grid-template-columns: 58% 42%;
}

.guardx-team {
    border-right: 2px solid #111111;

    padding: 18px 25px;

    box-sizing: border-box;
}

.guardx-guide {
    padding: 18px 25px;

    box-sizing: border-box;
}

.team-members {
    font-size: 11.5px;
    line-height: 1.7;

    color: #222222;
}

.guide-name {
    font-size: 15px;
    font-weight: 700;

    margin-top: 8px;

    color: #111111;
}

.guide-designation {
    font-size: 12px;

    margin-top: 5px;

    color: #444444;
}


/* ==========================================================
   PREDICT BUTTON
   ========================================================== */

.predict-container {
    position: relative;

    width: 29%;

    margin-top: -92px;

    padding-left: 25px;
    padding-right: 25px;

    box-sizing: border-box;

    z-index: 20;
}

.predict-container button {
    height: 45px !important;

    font-size: 15px !important;

    font-weight: 800 !important;

    border-radius: 5px !important;
}


/* ==========================================================
   DETECTION PAGE
   ========================================================== */

.detection-title {
    text-align: center;

    font-size: 32px;

    font-weight: 800;

    color: #111111;

    margin-bottom: 4px;
}

.detection-subtitle {
    text-align: center;

    font-size: 14px;

    color: #555555;

    margin-bottom: 15px;
}


/* ==========================================================
   DETECTION RESULT BOXES
   ========================================================== */

.waiting {
    background: #f8fafc;

    border: 2px dashed #cbd5e1;

    border-radius: 10px;

    padding: 35px 15px;

    text-align: center;

    margin-top: 10px;
}

.waiting h3 {
    color: #64748b;
    font-size: 18px;
}


.safe-result {
    background: #ecfdf5;

    border: 2px solid #86efac;

    border-radius: 10px;

    padding: 22px;

    text-align: center;
}

.safe-result h2 {
    color: #15803d;

    font-size: 25px;
}


.violation-result {
    background: #fef2f2;

    border: 2px solid #fca5a5;

    border-radius: 10px;

    padding: 22px;

    text-align: center;
}

.violation-result h2 {
    color: #dc2626;

    font-size: 25px;
}


.detection-info {
    background: #fff7ed;

    border-left: 4px solid #f97316;

    padding: 13px;

    border-radius: 7px;

    margin-top: 12px;
}

.confidence {
    font-size: 16px;

    font-weight: 700;

    color: #334155;
}


/* ==========================================================
   RESPONSIVE
   ========================================================== */

@media (max-width: 900px) {

    .guardx-frame {
        height: auto;
    }

    .guardx-body {
        height: auto;

        grid-template-columns: 1fr;
    }

    .guardx-left {
        min-height: 220px;

        border-right: none;

        border-bottom: 2px solid #111111;
    }

    .guardx-description {
        height: auto;

        min-height: 260px;
    }

    .guardx-bottom {
        height: auto;

        grid-template-columns: 1fr;
    }

    .guardx-team {
        border-right: none;

        border-bottom: 2px solid #111111;
    }

    .predict-container {
        width: 100%;

        margin-top: -70px;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PAGE 1
# ============================================================

if st.session_state.page == 1:

    # --------------------------------------------------------
    # MAIN PROJECT INTERFACE
    # --------------------------------------------------------

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


            <!-- ============================================= -->
            <!-- LEFT -->
            <!-- ============================================= -->

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


            <!-- ============================================= -->
            <!-- RIGHT -->
            <!-- ============================================= -->

            <div class="guardx-right">


                <!-- PROJECT TITLE -->

                <div class="guardx-project-title">

                    <div class="guardx-project-title-text">
                        GuardX-AI
                    </div>

                </div>


                <!-- DESCRIPTION -->

                <div class="guardx-description">

                    <div class="section-heading">
                        DESCRIPTION
                    </div>

                    <div class="description-text">

                        Construction sites involve high-risk activities
                        where proper Personal Protective Equipment (PPE)
                        is essential for worker safety. However, manually
                        monitoring whether every worker is wearing the
                        required PPE continuously is difficult,
                        time-consuming, and prone to human error.

                        GuardX AI is an AI-powered Construction PPE
                        Detection System designed to automatically
                        identify safety equipment such as hardhats,
                        masks, and safety vests from construction-site
                        images and videos. Using YOLO-based object
                        detection, the system detects PPE items and
                        identifies potential safety violations.

                        The solution provides visual detection results,
                        helping improve safety monitoring, reduce manual
                        inspection effort, and support faster
                        identification of unsafe working conditions.

                    </div>

                </div>


                <!-- TEAM + GUIDE -->

                <div class="guardx-bottom">


                    <!-- TEAM -->

                    <div class="guardx-team">

                        <div class="section-heading">
                            TEAM MEMBERS
                        </div>

                        <div class="team-members">

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


    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    st.markdown(
        '<div class="predict-container">',
        unsafe_allow_html=True
    )

    if st.button(
        "🔍 PREDICT",
        key="predict_button",
        use_container_width=True
    ):

        st.session_state.page = 2

        st.session_state.result_ready = False

        st.session_state.result_type = None

        st.session_state.result_data = None

        st.rerun()

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 2 — PPE DETECTION
# ============================================================

else:

    # --------------------------------------------------------
    # LOAD MODEL ONLY AFTER PREDICT
    # --------------------------------------------------------

    try:

        model = load_model()

    except Exception as e:

        st.error(
            "❌ Trained model could not be loaded."
        )

        st.write(
            "Make sure `best.pt` is present beside `app.py`."
        )

        st.code(str(e))

        st.stop()


    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # BACK BUTTON
    # --------------------------------------------------------

    if st.button(
        "← Back to Project",
        key="back_button"
    ):

        st.session_state.page = 1

        st.session_state.result_ready = False

        st.session_state.result_type = None

        st.session_state.result_data = None

        st.rerun()


    # ========================================================
    # DETECTION COLUMNS
    # ========================================================

    input_col, result_col = st.columns(
        [1, 1],
        gap="large"
    )


    # ========================================================
    # INPUT SECTION
    # ========================================================

    with input_col:

        st.markdown(
            '<div class="section-heading">📥 INPUT</div>',
            unsafe_allow_html=True
        )


        input_type = st.radio(
            "Select Input Type",
            [
                "🖼️ Image",
                "📷 Camera",
                "🎥 Video"
            ],
            horizontal=True
        )


        # ====================================================
        # IMAGE
        # ====================================================

        if input_type == "🖼️ Image":

            uploaded_image = st.file_uploader(
                "Upload Construction Image",
                type=[
                    "jpg",
                    "jpeg",
                    "png"
                ],
                key="image_upload"
            )


            if uploaded_image:

                image = Image.open(
                    uploaded_image
                ).convert("RGB")


                st.image(
                    image,
                    caption="Original Image",
                    use_container_width=True
                )


                if st.button(
                    "🔍 Detect PPE",
                    key="detect_image",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Detecting PPE..."
                    ):

                        result = model.predict(
                            source=np.array(image),
                            conf=CONF_THRESHOLD,
                            verbose=False
                        )[0]


                    detections = []


                    if result.boxes is not None:

                        for box in result.boxes:

                            class_id = int(
                                box.cls[0]
                            )

                            confidence = float(
                                box.conf[0]
                            )

                            class_name = result.names[
                                class_id
                            ]

                            detections.append(
                                (
                                    class_name,
                                    confidence
                                )
                            )


                    # ----------------------------------------
                    # ANNOTATED IMAGE
                    # ----------------------------------------

                    annotated = result.plot()

                    annotated = cv2.cvtColor(
                        annotated,
                        cv2.COLOR_BGR2RGB
                    )


                    # ----------------------------------------
                    # SAFETY VIOLATIONS
                    # ----------------------------------------

                    violation_names = [
                        "NO-Hardhat",
                        "NO-Mask",
                        "NO-Safety Vest"
                    ]


                    violations = [

                        item
                        for item in detections
                        if item[0] in violation_names

                    ]


                    # ----------------------------------------
                    # SAVE RESULT
                    # ----------------------------------------

                    st.session_state.result_ready = True

                    st.session_state.result_data = {
                        "detections": detections,
                        "violations": violations,
                        "image": annotated
                    }


                    if violations:

                        st.session_state.result_type = "violation"

                    else:

                        st.session_state.result_type = "safe"


                    st.rerun()


        # ====================================================
        # CAMERA
        # ====================================================

        elif input_type == "📷 Camera":

            camera_image = st.camera_input(
                "Take construction-site photo"
            )


            if camera_image:

                image = Image.open(
                    camera_image
                ).convert("RGB")


                if st.button(
                    "🔍 Detect PPE",
                    key="detect_camera",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Detecting PPE..."
                    ):

                        result = model.predict(
                            source=np.array(image),
                            conf=CONF_THRESHOLD,
                            verbose=False
                        )[0]


                    detections = []


                    if result.boxes is not None:

                        for box in result.boxes:

                            class_id = int(
                                box.cls[0]
                            )

                            confidence = float(
                                box.conf[0]
                            )

                            class_name = result.names[
                                class_id
                            ]

                            detections.append(
                                (
                                    class_name,
                                    confidence
                                )
                            )


                    annotated = result.plot()


                    annotated = cv2.cvtColor(
                        annotated,
                        cv2.COLOR_BGR2RGB
                    )


                    violation_names = [
                        "NO-Hardhat",
                        "NO-Mask",
                        "NO-Safety Vest"
                    ]


                    violations = [

                        item
                        for item in detections
                        if item[0] in violation_names

                    ]


                    st.session_state.result_ready = True

                    st.session_state.result_data = {
                        "detections": detections,
                        "violations": violations,
                        "image": annotated
                    }


                    if violations:

                        st.session_state.result_type = "violation"

                    else:

                        st.session_state.result_type = "safe"


                    st.rerun()


        # ====================================================
        # VIDEO
        # ====================================================

        elif input_type == "🎥 Video":

            uploaded_video = st.file_uploader(
                "Upload Construction Video",
                type=[
                    "mp4",
                    "avi",
                    "mov",
                    "mkv"
                ],
                key="video_upload"
            )


            if uploaded_video:

                st.video(
                    uploaded_video
                )


                if st.button(
                    "🎥 Detect PPE in Video",
                    key="detect_video",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Processing video... Please wait."
                    ):

                        # ------------------------------------
                        # INPUT VIDEO
                        # ------------------------------------

                        input_temp = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
                        )


                        input_temp.write(
                            uploaded_video.getbuffer()
                        )


                        input_temp.close()


                        # ------------------------------------
                        # VIDEO
                        # ------------------------------------

                        cap = cv2.VideoCapture(
                            input_temp.name
                        )


                        fps = cap.get(
                            cv2.CAP_PROP_FPS
                        )


                        if fps <= 0:

                            fps = 20


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


                        total_frames = int(
                            cap.get(
                                cv2.CAP_PROP_FRAME_COUNT
                            )
                        )


                        # ------------------------------------
                        # OUTPUT
                        # ------------------------------------

                        output_temp = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
                        )


                        output_temp.close()


                        fourcc = cv2.VideoWriter_fourcc(
                            *"mp4v"
                        )


                        writer = cv2.VideoWriter(
                            output_temp.name,
                            fourcc,
                            fps,
                            (width, height)
                        )


                        detected_violations = {}

                        progress = st.progress(0)

                        frame_count = 0


                        # ------------------------------------
                        # PROCESS VIDEO
                        # ------------------------------------

                        while True:

                            ret, frame = cap.read()


                            if not ret:

                                break


                            result = model.predict(
                                source=frame,
                                conf=CONF_THRESHOLD,
                                verbose=False
                            )[0]


                            # --------------------------------
                            # DETECTIONS
                            # --------------------------------

                            if result.boxes is not None:

                                for box in result.boxes:

                                    class_id = int(
                                        box.cls[0]
                                    )

                                    confidence = float(
                                        box.conf[0]
                                    )

                                    class_name = result.names[
                                        class_id
                                    ]


                                    if class_name in [
                                        "NO-Hardhat",
                                        "NO-Mask",
                                        "NO-Safety Vest"
                                    ]:

                                        if (
                                            class_name
                                            not in detected_violations
                                        ):

                                            detected_violations[
                                                class_name
                                            ] = confidence

                                        else:

                                            detected_violations[
                                                class_name
                                            ] = max(
                                                detected_violations[
                                                    class_name
                                                ],
                                                confidence
                                            )


                            # --------------------------------
                            # DRAW BOXES
                            # --------------------------------

                            annotated = result.plot()


                            writer.write(
                                annotated
                            )


                            frame_count += 1


                            if total_frames > 0:

                                progress.progress(
                                    min(
                                        frame_count /
                                        total_frames,
                                        1.0
                                    )
                                )


                        cap.release()

                        writer.release()

                        progress.empty()


                        # ------------------------------------
                        # CONVERT VIDEO
                        # ------------------------------------

                        final_video = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
                        )


                        final_video.close()


                        command = [
                            "ffmpeg",
                            "-y",
                            "-i",
                            output_temp.name,
                            "-vcodec",
                            "libx264",
                            "-pix_fmt",
                            "yuv420p",
                            "-movflags",
                            "+faststart",
                            final_video.name
                        ]


                        conversion = subprocess.run(
                            command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )


                        # ------------------------------------
                        # SAVE RESULT
                        # ------------------------------------

                        st.session_state.result_ready = True


                        st.session_state.result_data = {
                            "video": final_video.name,
                            "violations": detected_violations
                        }


                        if detected_violations:

                            st.session_state.result_type = "video_violation"

                        else:

                            st.session_state.result_type = "video_safe"


                        try:

                            os.remove(
                                input_temp.name
                            )

                            os.remove(
                                output_temp.name
                            )

                        except:

                            pass


                        st.rerun()


    # ========================================================
    # RESULT SECTION
    # ========================================================

    with result_col:

        st.markdown(
            '<div class="section-heading">'
            '🤖 INSPECTION RESULT'
            '</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # WAITING
        # ====================================================

        if not st.session_state.result_ready:

            st.markdown(
                """
                <div class="waiting">

                    <h3>
                        ⏳ WAITING FOR ANALYSIS
                    </h3>

                    <p>
                        Upload an image/video and run
                        PPE detection.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # SAFE IMAGE/CAMERA
        # ====================================================

        elif st.session_state.result_type == "safe":

            data = st.session_state.result_data


            st.markdown(
                """
                <div class="safe-result">

                    <h2>
                        🟢 PPE COMPLIANCE
                    </h2>

                    <p>
                        No PPE safety violations detected.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.image(
                data["image"],
                caption="PPE Detection Result",
                use_container_width=True
            )


            if data["detections"]:

                st.write(
                    "### Detected PPE"
                )


                for name, confidence in data["detections"]:

                    st.write(
                        f"✅ **{name}** — "
                        f"{confidence * 100:.1f}%"
                    )


        # ====================================================
        # VIOLATION IMAGE/CAMERA
        # ====================================================

        elif st.session_state.result_type == "violation":

            data = st.session_state.result_data


            st.markdown(
                """
                <div class="violation-result">

                    <h2>
                        🔴 SAFETY VIOLATION
                    </h2>

                    <p>
                        PPE violation detected.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.image(
                data["image"],
                caption="Detected PPE Violations",
                use_container_width=True
            )


            st.write(
                "### ⚠️ Violations Detected"
            )


            for name, confidence in data["violations"]:

                st.markdown(
                    f"""
                    <div class="detection-info">

                        <b>
                            ⚠️ {name}
                        </b>

                        <br><br>

                        <span class="confidence">

                            Confidence:
                            {confidence * 100:.1f}%

                        </span>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # ====================================================
        # SAFE VIDEO
        # ====================================================

        elif st.session_state.result_type == "video_safe":

            data = st.session_state.result_data


            st.markdown(
                """
                <div class="safe-result">

                    <h2>
                        🟢 PPE COMPLIANCE
                    </h2>

                    <p>
                        No PPE safety violations detected
                        in the processed video.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


            if os.path.exists(data["video"]):

                with open(
                    data["video"],
                    "rb"
                ) as video_file:

                    video_bytes = video_file.read()


                st.video(
                    video_bytes
                )


        # ====================================================
        # VIDEO VIOLATION
        # ====================================================

        elif st.session_state.result_type == "video_violation":

            data = st.session_state.result_data


            st.markdown(
                """
                <div class="violation-result">

                    <h2>
                        🔴 SAFETY VIOLATION
                    </h2>

                    <p>
                        PPE violations detected in video.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.write(
                "### ⚠️ Violations Detected"
            )


            for name, confidence in data["violations"].items():

                st.markdown(
                    f"""
                    <div class="detection-info">

                        <b>
                            ⚠️ {name}
                        </b>

                        <br><br>

                        <span class="confidence">

                            Confidence:
                            {confidence * 100:.1f}%

                        </span>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            if os.path.exists(data["video"]):

                st.write(
                    "### 🎥 Processed Video"
                )


                with open(
                    data["video"],
                    "rb"
                ) as video_file:

                    video_bytes = video_file.read()


                st.video(
                    video_bytes
                )


# ============================================================
# END
# ============================================================
