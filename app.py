import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os


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
# GLOBAL CSS
# ============================================================

st.markdown("""
<style>

    /* ================================
       APP BACKGROUND
       ================================ */

    .stApp {
        background: #f4f6fa;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 25px;
        padding-bottom: 30px;
        padding-left: 7%;
        padding-right: 7%;
    }


    /* ================================
       HIDE STREAMLIT DEFAULT UI
       ================================ */

    #MainMenu {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* ================================
       MAIN TITLE
       ================================ */

    .main-title {
        color: #17345f !important;
        font-size: 32px !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 32px;
    }


    /* ================================
       AICW
       ================================ */

    .aicw-text {
        color: #17345f !important;
        font-size: 25px !important;
        font-weight: 800 !important;
        line-height: 1.55;
    }


    .capstone-text {
        color: #334155 !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        margin-top: 42px;
    }


    /* ================================
       DESCRIPTION
       ================================ */

    .description-title {
        color: #17345f !important;
        font-size: 24px !important;
        font-weight: 800 !important;
        margin-bottom: 12px;
    }


    .description-box {
        background: #ffffff;
        border: 1px solid #dfe4ec;
        border-radius: 14px;
        padding: 22px;
        color: #374151 !important;
        font-size: 15px;
        line-height: 1.7;
        box-shadow: 0 2px 8px rgba(30, 41, 59, 0.05);
    }


    /* ================================
       STREAMLIT TEXT
       ================================ */

    .stMarkdown,
    .stMarkdown p,
    .stMarkdown li {
        color: #374151;
    }


    /* ================================
       CARD CONTAINERS
       ================================ */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff !important;
        border: 1px solid #dfe4ec !important;
        border-radius: 16px !important;
        box-shadow: 0 2px 10px rgba(30, 41, 59, 0.06);
        padding: 8px !important;
    }


    /* ================================
       CARD HEADINGS
       ================================ */

    .card-heading {
        color: #26364d !important;
        font-size: 15px !important;
        font-weight: 800 !important;
        margin-bottom: 16px;
    }


    /* ================================
       CARD TEXT
       ================================ */

    .card-text {
        color: #4b5563 !important;
        font-size: 14px !important;
        line-height: 2.2 !important;
    }


    /* ================================
       BUTTON
       ================================ */

    div.stButton > button {
        width: 100%;
        height: 45px;
        background: #ffffff !important;
        color: #334155 !important;
        border: 1px solid #d5dce6 !important;
        border-radius: 8px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }

    div.stButton > button:hover {
        border-color: #17345f !important;
        color: #17345f !important;
        background: #f8fafc !important;
    }


    /* ================================
       RADIO
       ================================ */

    div[data-testid="stRadio"] label {
        color: #334155 !important;
        font-weight: 600 !important;
    }


    /* ================================
       UPLOADERS
       ================================ */

    div[data-testid="stFileUploader"] {
        background: #ffffff;
        border-radius: 12px;
    }


    /* ================================
       INPUT LABELS
       ================================ */

    label {
        color: #334155 !important;
    }


    /* ================================
       FOOTER
       ================================ */

    .footer-text {
        color: #6b7280 !important;
        text-align: center;
        font-size: 14px;
        margin-top: 32px;
    }


    /* ================================
       DETECTION PAGE
       ================================ */

    .detect-title {
        color: #17345f !important;
        font-size: 32px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }

    .detect-subtitle {
        color: #64748b !important;
        text-align: center;
        font-size: 15px;
        margin-bottom: 25px;
    }


    /* ================================
       MOBILE
       ================================ */

    @media(max-width: 900px) {

        .block-container {
            padding-left: 5%;
            padding-right: 5%;
        }

        .main-title {
            font-size: 24px !important;
        }

        .aicw-text {
            font-size: 21px !important;
        }

        .capstone-text {
            font-size: 19px !important;
        }

    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = os.path.join(
        os.path.dirname(__file__),
        "best.pt"
    )

    return YOLO(model_path)


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "home":

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    st.markdown(
        '<div class="main-title">'
        '🦺 GuardX-AI – Construction PPE Detection System'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # TOP SECTION
    # --------------------------------------------------------

    left_col, right_col = st.columns(
        [0.38, 0.62],
        gap="large"
    )


    # ========================================================
    # LEFT SECTION
    # ========================================================

    with left_col:

        st.markdown(
            '<div class="aicw-text">'
            'AI Career for Women'
            '<br>'
            '(AICW)'
            '</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            '<div class="capstone-text">'
            'Capstone Project'
            '</div>',
            unsafe_allow_html=True
        )


        st.write("")


        if st.button(
            "🔍  PREDICT",
            key="predict",
            use_container_width=True
        ):

            st.session_state.page = "predict"

            st.rerun()


    # ========================================================
    # RIGHT SECTION
    # ========================================================

    with right_col:

        st.markdown(
            '<div class="description-title">'
            'Project Description'
            '</div>',
            unsafe_allow_html=True
        )


        with st.container(border=True):

            st.markdown(
                """
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
                """
            )


    st.write("")
    st.write("")


    # ========================================================
    # BOTTOM CARDS
    # ========================================================

    team_col, gmail_col, guide_col = st.columns(
        [1.25, 1.25, 0.75],
        gap="large"
    )


    # ========================================================
    # TEAM MEMBERS
    # ========================================================

    with team_col:

        with st.container(border=True):

            st.markdown(
                '<div class="card-heading">'
                'TEAM MEMBERS'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="card-text">

                1. Y.D.V.Sivani<br>
                2. V.L.S.Asritha<br>
                3. R.Likhitha<br>
                4. S.Poojitha sri

                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # GMAIL
    # ========================================================

    with gmail_col:

        with st.container(border=True):

            st.markdown(
                '<div class="card-heading">'
                'GMAIL'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="card-text">

                yallashivani@gmail.com<br>
                Asrithavantipalli@gmail.com<br>
                likhitharayudu@gmail.com<br>
                pujithasari@gmail.com

                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # GUIDE
    # ========================================================

    with guide_col:

        with st.container(border=True):

            st.markdown(
                '<div class="card-heading">'
                'GUIDE NAME'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="card-text">'
                'MD.Abdul Aziz'
                '</div>',
                unsafe_allow_html=True
            )

            st.write("")

            st.markdown(
                '<div class="card-heading">'
                'Designation'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="card-text">'
                'Trainer, Co-Lead-AICW'
                '</div>',
                unsafe_allow_html=True
            )


    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown(
        '<div class="footer-text">'
        'GuardX-AI – Construction PPE Detection System'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

else:

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    st.markdown(
        '<div class="detect-title">'
        '🦺 GuardX-AI'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="detect-subtitle">'
        'AI-Powered Construction PPE Detection System'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # BACK BUTTON
    # --------------------------------------------------------

    if st.button(
        "← Back to Home",
        key="back"
    ):

        st.session_state.page = "home"

        st.rerun()


    st.write("")


    # --------------------------------------------------------
    # LOAD YOLO
    # --------------------------------------------------------

    try:

        model = load_model()

    except Exception:

        st.error(
            "❌ best.pt model load avvaledu."
        )

        st.info(
            "Make sure best.pt is in the same folder as app.py."
        )

        st.stop()


    # --------------------------------------------------------
    # INPUT TYPE
    # --------------------------------------------------------

    input_type = st.radio(
        "Select Input Type",
        [
            "🖼️ Image",
            "📷 Camera",
            "🎥 Video"
        ],
        horizontal=True
    )


    # ========================================================
    # IMAGE
    # ========================================================

    if input_type == "🖼️ Image":

        input_col, result_col = st.columns(
            2,
            gap="large"
        )


        with input_col:

            st.subheader("Upload Image")

            uploaded_image = st.file_uploader(
                "Choose a construction image",
                type=["jpg", "jpeg", "png"]
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


                    st.session_state.result_image = annotated

                    st.session_state.detections = detections


        with result_col:

            st.subheader("Detection Result")


            if "result_image" not in st.session_state:

                st.info(
                    "Upload an image and click Detect PPE."
                )

            else:

                st.image(
                    st.session_state.result_image,
                    caption="GuardX-AI Result",
                    use_container_width=True
                )


                detections = st.session_state.detections


                if detections:

                    st.write("### Detected Objects")


                    for name, confidence in detections:

                        st.write(
                            f"**{name}** — "
                            f"{confidence * 100:.1f}%"
                        )

                else:

                    st.info(
                        "No objects detected."
                    )


    # ========================================================
    # CAMERA
    # ========================================================

    elif input_type == "📷 Camera":

        camera_image = st.camera_input(
            "Take a photo"
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

                        st.write(
                            f"**{name}** — "
                            f"{confidence * 100:.1f}%"
                        )


    # ========================================================
    # VIDEO
    # ========================================================

    elif input_type == "🎥 Video":

        uploaded_video = st.file_uploader(
            "Upload construction video",
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


                    output_path = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".mp4"
                    ).name


                    fourcc = cv2.VideoWriter_fourcc(
                        *"mp4v"
                    )


                    writer = cv2.VideoWriter(
                        output_path,
                        fourcc,
                        fps,
                        (width, height)
                    )


                    while True:

                        ret, frame = cap.read()


                        if not ret:
                            break


                        result = model.predict(
                            frame,
                            conf=0.30,
                            verbose=False
                        )[0]


                        annotated = result.plot()


                        writer.write(
                            annotated
                        )


                    cap.release()

                    writer.release()


                st.success(
                    "✅ Video processing completed."
                )


                st.video(
                    output_path
                )
