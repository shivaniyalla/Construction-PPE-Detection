import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import textwrap


# ============================================================
# PAGE CONFIG
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
    st.session_state.page = "home"


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = os.path.join(
        os.path.dirname(__file__),
        "best.pt"
    )

    return YOLO(model_path)


# ============================================================
# CSS
# ============================================================

st.markdown(
    textwrap.dedent("""
    <style>

    #MainMenu {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    .stApp {
        background-color: #f5f6fa;
    }

    .block-container {
        max-width: 1400px !important;
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 7% !important;
        padding-right: 7% !important;
    }


    /* ==============================
       TITLE
       ============================== */

    .main-title {
        text-align: center;
        color: #172b55;
        font-size: 31px;
        font-weight: 800;
        margin-top: -5px;
        margin-bottom: 30px;
    }


    /* ==============================
       AICW
       ============================== */

    .aicw-title {
        color: #172b55;
        font-size: 25px;
        font-weight: 800;
        line-height: 1.55;
    }


    /* ==============================
       CAPSTONE
       ============================== */

    .capstone-title {
        color: #303c52;
        font-size: 22px;
        font-weight: 700;
        margin-top: 42px;
    }


    /* ==============================
       DESCRIPTION
       ============================== */

    .description-title {
        color: #172b55;
        font-size: 23px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .description-text {
        color: #626b78;
        font-size: 15px;
        line-height: 1.65;
        max-width: 850px;
    }


    /* ==============================
       PREDICT BUTTON
       ============================== */

    div.stButton > button {
        height: 44px !important;
        border-radius: 8px !important;
        background-color: white !important;
        border: 1px solid #d8dde5 !important;
        color: #394252 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }


    /* ==============================
       CARDS
       ============================== */

    .info-card {
        background-color: white;
        border: 1px solid #e0e3e8;
        border-radius: 17px;
        padding: 20px 16px;
        min-height: 235px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        box-sizing: border-box;
    }

    .card-title {
        color: #303743;
        font-size: 14px;
        font-weight: 800;
        margin-bottom: 25px;
    }

    .card-content {
        color: #4d5562;
        font-size: 14px;
        line-height: 2.7;
    }

    .guide-name {
        color: #4d5562;
        font-size: 15px;
        margin-bottom: 28px;
    }

    .designation-title {
        color: #303743;
        font-size: 14px;
        font-weight: 800;
        margin-bottom: 20px;
    }

    .designation {
        color: #4d5562;
        font-size: 14px;
        line-height: 1.7;
    }


    /* ==============================
       FOOTER
       ============================== */

    .project-footer {
        text-align: center;
        color: #737b87;
        font-size: 14px;
        margin-top: 35px;
    }


    /* ==============================
       DETECTION PAGE
       ============================== */

    .detect-title {
        text-align: center;
        color: #172b55;
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .detect-subtitle {
        text-align: center;
        color: #626b78;
        font-size: 15px;
        margin-bottom: 20px;
    }

    .waiting-box {
        background-color: white;
        border: 1px dashed #c8ced8;
        border-radius: 12px;
        padding: 60px 20px;
        text-align: center;
        color: #727b88;
    }

    .result-safe {
        background-color: #ecfdf3;
        border: 1px solid #9be3b5;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: #16803c;
    }

    .result-danger {
        background-color: #fff0f0;
        border: 1px solid #f1aaaa;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: #c62828;
    }


    /* ==============================
       MOBILE
       ============================== */

    @media(max-width: 900px) {

        .block-container {
            padding-left: 5% !important;
            padding-right: 5% !important;
        }

        .main-title {
            font-size: 24px;
        }

        .aicw-title {
            font-size: 21px;
        }

        .capstone-title {
            font-size: 19px;
        }

    }

    </style>
    """),
    unsafe_allow_html=True
)


# ============================================================
# PAGE 1 — HOME
# ============================================================

if st.session_state.page == "home":

    # --------------------------------------------------------
    # MAIN TITLE
    # --------------------------------------------------------

    st.markdown(
        textwrap.dedent("""
        <div class="main-title">
            🦺 GuardX-AI – Construction PPE Detection System
        </div>
        """),
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # TOP SECTION
    # --------------------------------------------------------

    left_col, right_col = st.columns(
        [0.36, 0.64],
        gap="large"
    )


    # ========================================================
    # LEFT — AICW + CAPSTONE + PREDICT
    # ========================================================

    with left_col:

        st.markdown(
            textwrap.dedent("""
            <div class="aicw-title">
                AI Career for Women
                <br>
                (AICW)
            </div>

            <div class="capstone-title">
                Capstone Project
            </div>
            """),
            unsafe_allow_html=True
        )


        st.markdown(
            "<div style='height:28px'></div>",
            unsafe_allow_html=True
        )


        if st.button(
            "🔍 PREDICT",
            key="predict_button",
            use_container_width=True
        ):

            st.session_state.page = "predict"

            st.rerun()


    # ========================================================
    # RIGHT — PROJECT DESCRIPTION
    # ========================================================

    with right_col:

        st.markdown(
            textwrap.dedent("""
            <div class="description-title">
                Project Description
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
                identifies potential safety violations.

                The solution provides visual detection results, helping
                improve safety monitoring, reduce manual inspection
                effort, and support faster identification of unsafe
                working conditions.

            </div>
            """),
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # SPACE BEFORE CARDS
    # --------------------------------------------------------

    st.markdown(
        "<div style='height:35px'></div>",
        unsafe_allow_html=True
    )


    # ========================================================
    # BOTTOM THREE CARDS
    # ========================================================

    team_col, gmail_col, guide_col = st.columns(
        [1.25, 1.25, 0.75],
        gap="large"
    )


    # ========================================================
    # TEAM MEMBERS CARD
    # ========================================================

    with team_col:

        st.markdown(
            textwrap.dedent("""
            <div class="info-card">

                <div class="card-title">
                    TEAM MEMBERS
                </div>

                <div class="card-content">

                    1. Y.D.V.Sivani

                    <br>

                    2. V.L.S.Asritha

                    <br>

                    3. R.Likhitha

                    <br>

                    4. S.Poojitha sri

                </div>

            </div>
            """),
            unsafe_allow_html=True
        )


    # ========================================================
    # GMAIL CARD
    # ========================================================

    with gmail_col:

        st.markdown(
            textwrap.dedent("""
            <div class="info-card">

                <div class="card-title">
                    GMAIL
                </div>

                <div class="card-content">

                    yallashivani@gmail.com

                    <br>

                    Asrithavantipalli@gmail.com

                    <br>

                    likhitharayudu@gmail.com

                    <br>

                    pujithasari@gmail.com

                </div>

            </div>
            """),
            unsafe_allow_html=True
        )


    # ========================================================
    # GUIDE CARD
    # ========================================================

    with guide_col:

        st.markdown(
            textwrap.dedent("""
            <div class="info-card">

                <div class="card-title">
                    GUIDE NAME
                </div>

                <div class="guide-name">
                    MD.Abdul Aziz
                </div>

                <div class="designation-title">
                    Designation
                </div>

                <div class="designation">
                    Trainer, Co-Lead-AICW
                </div>

            </div>
            """),
            unsafe_allow_html=True
        )


    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown(
        textwrap.dedent("""
        <div class="project-footer">
            GuardX-AI – Construction PPE Detection System
        </div>
        """),
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 2 — DETECTION
# ============================================================

else:

    st.markdown(
        textwrap.dedent("""
        <div class="detect-title">
            🦺 GuardX-AI
        </div>

        <div class="detect-subtitle">
            AI-Powered Construction PPE Detection System
        </div>
        """),
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # BACK BUTTON
    # --------------------------------------------------------

    if st.button(
        "← Back to Home",
        key="back_button"
    ):

        st.session_state.page = "home"

        st.rerun()


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    try:

        model = load_model()

    except Exception as e:

        st.error(
            "❌ best.pt model load avvaledu."
        )

        st.info(
            "best.pt file app.py tho same folder lo undali."
        )

        st.stop()


    # --------------------------------------------------------
    # INPUT TYPE
    # --------------------------------------------------------

    input_type = st.radio(
        "Select Input",
        [
            "🖼️ Image",
            "📷 Camera",
            "🎥 Video"
        ],
        horizontal=True
    )


    # ========================================================
    # IMAGE DETECTION
    # ========================================================

    if input_type == "🖼️ Image":

        input_col, result_col = st.columns(
            2,
            gap="large"
        )


        with input_col:

            uploaded_image = st.file_uploader(
                "Upload Construction Image",
                type=[
                    "jpg",
                    "jpeg",
                    "png"
                ]
            )


            if uploaded_image:

                image = Image.open(
                    uploaded_image
                ).convert("RGB")


                st.image(
                    image,
                    caption="Input Image",
                    use_container_width=True
                )


                if st.button(
                    "🔍 Detect PPE",
                    key="image_detect",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Detecting PPE..."
                    ):

                        result = model.predict(
                            np.array(image),
                            conf=0.30,
                            verbose=False
                        )[0]


                    annotated = result.plot()


                    annotated = cv2.cvtColor(
                        annotated,
                        cv2.COLOR_BGR2RGB
                    )


                    detections = []


                    if result.boxes is not None:

                        for box in result.boxes:

                            class_id = int(
                                box.cls[0]
                            )

                            confidence = float(
                                box.conf[0]
                            )

                            name = result.names[
                                class_id
                            ]

                            detections.append(
                                (
                                    name,
                                    confidence
                                )
                            )


                    st.session_state.image_result = annotated

                    st.session_state.image_detections = detections


        with result_col:

            st.subheader(
                "🤖 Detection Result"
            )


            if "image_result" not in st.session_state:

                st.markdown(
                    textwrap.dedent("""
                    <div class="waiting-box">

                        <h3>
                            Waiting for Detection
                        </h3>

                        <p>
                            Upload an image and click
                            <b>Detect PPE</b>.
                        </p>

                    </div>
                    """),
                    unsafe_allow_html=True
                )

            else:

                annotated = st.session_state.image_result

                detections = st.session_state.image_detections


                st.image(
                    annotated,
                    caption="GuardX-AI Detection",
                    use_container_width=True
                )


                violations = [
                    x for x in detections
                    if (
                        "no-" in x[0].lower()
                        or
                        "without" in x[0].lower()
                    )
                ]


                if violations:

                    st.markdown(
                        textwrap.dedent("""
                        <div class="result-danger">

                            <h2>
                                🔴 SAFETY VIOLATION
                            </h2>

                            PPE violation detected.

                        </div>
                        """),
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        textwrap.dedent("""
                        <div class="result-safe">

                            <h2>
                                🟢 PPE COMPLIANCE
                            </h2>

                            No safety violation detected.

                        </div>
                        """),
                        unsafe_allow_html=True
                    )


                if detections:

                    st.write(
                        "### Detected Objects"
                    )


                    for name, confidence in detections:

                        st.write(
                            f"**{name}** — "
                            f"{confidence * 100:.1f}%"
                        )


    # ========================================================
    # CAMERA
    # ========================================================

    elif input_type == "📷 Camera":

        camera_image = st.camera_input(
            "Take Construction Site Photo"
        )


        if camera_image:

            image = Image.open(
                camera_image
            ).convert("RGB")


            if st.button(
                "🔍 Detect PPE",
                key="camera_detect",
                use_container_width=True
            ):

                with st.spinner(
                    "Detecting PPE..."
                ):

                    result = model.predict(
                        np.array(image),
                        conf=0.30,
                        verbose=False
                    )[0]


                annotated = result.plot()


                annotated = cv2.cvtColor(
                    annotated,
                    cv2.COLOR_BGR2RGB
                )


                st.image(
                    annotated,
                    caption="GuardX-AI Detection Result",
                    use_container_width=True
                )


                detections = []


                if result.boxes is not None:

                    for box in result.boxes:

                        class_id = int(
                            box.cls[0]
                        )

                        confidence = float(
                            box.conf[0]
                        )

                        name = result.names[
                            class_id
                        ]

                        detections.append(
                            (
                                name,
                                confidence
                            )
                        )


                violations = [
                    x for x in detections
                    if (
                        "no-" in x[0].lower()
                        or
                        "without" in x[0].lower()
                    )
                ]


                if violations:

                    st.error(
                        "🔴 SAFETY VIOLATION DETECTED"
                    )

                else:

                    st.success(
                        "🟢 PPE COMPLIANCE"
                    )


                for name, confidence in detections:

                    st.write(
                        f"**{name}** — "
                        f"{confidence * 100:.1f}%"
                    )


    # ========================================================
    # VIDEO
    # ========================================================

    elif input_type == "🎥 Video":

        uploaded_video = st.file_uploader(
            "Upload Construction Video",
            type=[
                "mp4",
                "avi",
                "mov",
                "mkv"
            ]
        )


        if uploaded_video:

            st.video(
                uploaded_video
            )


            if st.button(
                "🎥 Detect PPE in Video",
                key="video_detect",
                use_container_width=True
            ):

                with st.spinner(
                    "Processing video..."
                ):

                    input_file = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".mp4"
                    )

                    input_file.write(
                        uploaded_video.getbuffer()
                    )

                    input_file.close()


                    cap = cv2.VideoCapture(
                        input_file.name
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


                    output_file = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".mp4"
                    )

                    output_file.close()


                    fourcc = cv2.VideoWriter_fourcc(
                        *"mp4v"
                    )


                    writer = cv2.VideoWriter(
                        output_file.name,
                        fourcc,
                        fps,
                        (width, height)
                    )


                    violation_classes = set()


                    while True:

                        ret, frame = cap.read()


                        if not ret:
                            break


                        result = model.predict(
                            frame,
                            conf=0.30,
                            verbose=False
                        )[0]


                        if result.boxes is not None:

                            for box in result.boxes:

                                class_id = int(
                                    box.cls[0]
                                )

                                name = result.names[
                                    class_id
                                ]


                                if (
                                    "no-" in name.lower()
                                    or
                                    "without" in name.lower()
                                ):

                                    violation_classes.add(
                                        name
                                    )


                        annotated = result.plot()


                        writer.write(
                            annotated
                        )


                    cap.release()

                    writer.release()


                st.success(
                    "✅ Video processing completed."
                )


                if violation_classes:

                    st.error(
                        "🔴 Safety violations detected: "
                        + ", ".join(
                            violation_classes
                        )
                    )

                else:

                    st.success(
                        "🟢 No PPE violations detected."
                    )


                st.video(
                    output_file.name
                )
