import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import json
import firebase_admin
from firebase_admin import credentials, messaging
from streamlit_js_eval import streamlit_js_eval


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

if "fcm_token" not in st.session_state:
    st.session_state.fcm_token = None

if "push_enabled" not in st.session_state:
    st.session_state.push_enabled = False


# ============================================================
# FIREBASE ADMIN INITIALIZATION
# ============================================================

def initialize_firebase():

    try:

        if not firebase_admin._apps:

            firebase_config = dict(
                st.secrets["firebase"]
            )

            cred = credentials.Certificate(
                firebase_config
            )

            firebase_admin.initialize_app(
                cred
            )

        return True

    except Exception as e:

        st.session_state.firebase_error = str(e)

        return False


firebase_ready = initialize_firebase()


# ============================================================
# FIREBASE PUSH NOTIFICATION
# ============================================================

def send_push_notification(
    token,
    title,
    body
):

    if not token:
        return False

    if not firebase_ready:
        return False

    try:

        message = messaging.Message(

            notification=messaging.Notification(
                title=title,
                body=body
            ),

            token=token,

            webpush=messaging.WebpushConfig(

                notification=messaging.WebpushNotification(
                    title=title,
                    body=body,
                    icon="/favicon.ico"
                )
            )
        )

        messaging.send(message)

        return True

    except Exception as e:

        st.session_state.push_error = str(e)

        return False


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
"""
<style>

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

#MainMenu {
    visibility: hidden;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

.main-title {
    color: #17345f !important;
    font-size: 32px !important;
    font-weight: 800 !important;
    text-align: center;
    margin-bottom: 32px;
}

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

.stMarkdown,
.stMarkdown p,
.stMarkdown li {
    color: #374151;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1px solid #dfe4ec !important;
    border-radius: 16px !important;
    box-shadow: 0 2px 10px rgba(30, 41, 59, 0.06);
    padding: 8px !important;
}

.card-heading {
    color: #26364d !important;
    font-size: 15px !important;
    font-weight: 800 !important;
    margin-bottom: 16px;
}

.card-text {
    color: #4b5563 !important;
    font-size: 14px !important;
    line-height: 2.2 !important;
}

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

div[data-testid="stRadio"] label {
    color: #334155 !important;
    font-weight: 600 !important;
}

div[data-testid="stFileUploader"] {
    background: #ffffff;
    border-radius: 12px;
}

label {
    color: #334155 !important;
}

.footer-text {
    color: #6b7280 !important;
    text-align: center;
    font-size: 14px;
    margin-top: 32px;
}

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

.push-card {
    background: #eef6ff;
    border: 1px solid #c9ddf5;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 20px;
}

.push-title {
    color: #17345f;
    font-size: 17px;
    font-weight: 800;
}

.safe-box {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    border-radius: 12px;
    padding: 15px;
    color: #065f46 !important;
    font-weight: 700;
}

.violation-box {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 12px;
    padding: 15px;
    color: #991b1b !important;
    font-weight: 700;
}

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
""",
unsafe_allow_html=True
)


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
# FCM TOKEN SETUP
# ============================================================

def get_fcm_token():

    firebase_config = {
        "apiKey": "AIzaSyDB0kB4le-PFo_AILSxTZG1m7yUsOizrDI",
        "authDomain": "guardx-ai.firebaseapp.com",
        "projectId": "guardx-ai",
        "storageBucket": "guardx-ai.firebasestorage.app",
        "messagingSenderId": "551369972933",
        "appId": "1:551369972933:web:f4e4186a3501ee0ee8e4ac"
    }

    vapid_key = (
        "BGE1N7sXhztv_V_XZ8yxQR9dn74UMgg17UKGmigh"
        "YXZGmW1sGfOCHSAqyIvJZ77GeS2-tnlgBbLVTmAVgJrdQ7M"
    )

    js_code = f"""
    (async () => {{

        try {{

            if (!("Notification" in window)) {{
                return "NOT_SUPPORTED";
            }}

            const permission =
                await Notification.requestPermission();

            if (permission !== "granted") {{
                return "PERMISSION_DENIED";
            }}

            if (!("serviceWorker" in navigator)) {{
                return "SERVICE_WORKER_NOT_SUPPORTED";
            }}

            const registration =
                await navigator.serviceWorker.register(
                    "/firebase-messaging-sw.js"
                );

            const firebaseModule =
                await import(
                    "https://www.gstatic.com/firebasejs/10.13.2/firebase-app.js"
                );

            const messagingModule =
                await import(
                    "https://www.gstatic.com/firebasejs/10.13.2/firebase-messaging.js"
                );

            const app =
                firebaseModule.initializeApp(
                    {json.dumps(firebase_config)}
                );

            const messaging =
                messagingModule.getMessaging(app);

            const token =
                await messagingModule.getToken(
                    messaging,
                    {{
                        vapidKey: "{vapid_key}",
                        serviceWorkerRegistration: registration
                    }}
                );

            return token || "NO_TOKEN";

        }} catch(error) {{

            return "ERROR:" + error.message;

        }}

    }})();
    """

    return streamlit_js_eval(
        js_expressions=js_code,
        want_output=True,
        key="fcm_token_generator"
    )


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "home":

    st.markdown(
        '<div class="main-title">'
        '🦺 GuardX-AI – Construction PPE Detection System'
        '</div>',
        unsafe_allow_html=True
    )

    left_col, right_col = st.columns(
        [0.38, 0.62],
        gap="large"
    )

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

    team_col, gmail_col, guide_col = st.columns(
        [1.25, 1.25, 0.75],
        gap="large"
    )

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

    # ========================================================
    # PUSH NOTIFICATION SETUP
    # ========================================================

    st.markdown(
        """
        <div class="push-card">
            <div class="push-title">
                🔔 Safety Push Notifications
            </div>
            <div>
                Enable notifications to receive a safety alert when
                GuardX-AI detects a PPE violation.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not st.session_state.push_enabled:

        if st.button(
            "🔔 Enable Push Notifications",
            key="enable_push",
            use_container_width=True
        ):

            token_result = get_fcm_token()

            if token_result:

                if isinstance(token_result, str):

                    if token_result.startswith("ERROR:"):

                        st.error(
                            "Push notification setup failed."
                        )

                        st.caption(
                            token_result
                        )

                    elif token_result == "PERMISSION_DENIED":

                        st.warning(
                            "Notification permission was denied. "
                            "Please allow notifications in your browser."
                        )

                    elif token_result == "NOT_SUPPORTED":

                        st.warning(
                            "This browser does not support notifications."
                        )

                    elif token_result == "SERVICE_WORKER_NOT_SUPPORTED":

                        st.warning(
                            "Browser service workers are not supported."
                        )

                    elif token_result == "NO_TOKEN":

                        st.warning(
                            "FCM token was not generated."
                        )

                    else:

                        st.session_state.fcm_token = token_result

                        st.session_state.push_enabled = True

                        st.success(
                            "🔔 Push notifications enabled successfully!"
                        )

                        st.rerun()

    else:

        st.success(
            "🔔 Push notifications are enabled."
        )


    # ========================================================
    # BACK BUTTON
    # ========================================================

    if st.button(
        "← Back to Home",
        key="back"
    ):

        st.session_state.page = "home"

        st.rerun()

    st.write("")


    # ========================================================
    # LOAD YOLO
    # ========================================================

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


    # ========================================================
    # INPUT TYPE
    # ========================================================

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

                        st.write(
                            f"**{name}** — "
                            f"{confidence * 100:.1f}%"
                        )

                # ------------------------------------------------
                # CAMERA VIOLATION ALERT
                # ------------------------------------------------

                violation_names = [
                    "no_helmet",
                    "no_hardhat",
                    "no_mask",
                    "no_vest",
                    "without_helmet",
                    "without_mask",
                    "without_vest"
                ]

                violations = [
                    (name, conf)
                    for name, conf in detections
                    if name.lower().replace(" ", "_")
                    in violation_names
                ]

                if violations:

                    violation_text = ", ".join(
                        [
                            name
                            for name, _ in violations
                        ]
                    )

                    if st.session_state.fcm_token:

                        send_push_notification(
                            st.session_state.fcm_token,
                            "🚨 GuardX-AI Safety Alert",
                            f"PPE violation detected: "
                            f"{violation_text}"
                        )

                    st.markdown(
                        """
                        <div class="violation-box">
                        🚨 PPE VIOLATION DETECTED
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        """
                        <div class="safe-box">
                        🟢 No configured PPE violation detected.
                        </div>
                        """,
                        unsafe_allow_html=True
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

                    video_violations = set()

                    frame_count = 0

                    while True:

                        ret, frame = cap.read()

                        if not ret:
                            break

                        frame_count += 1

                        result = model.predict(
                            frame,
                            conf=0.30,
                            verbose=False
                        )[0]

                        annotated = result.plot()

                        writer.write(
                            annotated
                        )

                        if result.boxes is not None:

                            for box in result.boxes:

                                class_id = int(
                                    box.cls[0]
                                )

                                name = result.names[
                                    class_id
                                ]

                                normalized_name = (
                                    name.lower()
                                    .replace(" ", "_")
                                )

                                violation_names = [
                                    "no_helmet",
                                    "no_hardhat",
                                    "no_mask",
                                    "no_vest",
                                    "without_helmet",
                                    "without_mask",
                                    "without_vest"
                                ]

                                if normalized_name in violation_names:

                                    video_violations.add(
                                        name
                                    )

                    cap.release()
                    writer.release()

                st.success(
                    "✅ Video processing completed."
                )

                st.video(
                    output_path
                )

                if video_violations:

                    violation_text = ", ".join(
                        sorted(video_violations)
                    )

                    st.markdown(
                        f"""
                        <div class="violation-box">
                        🚨 PPE VIOLATION DETECTED<br><br>
                        {violation_text}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if st.session_state.fcm_token:

                        send_push_notification(
                            st.session_state.fcm_token,
                            "🚨 GuardX-AI Safety Alert",
                            f"PPE violation detected in video: "
                            f"{violation_text}"
                        )

                else:

                    st.markdown(
                        """
                        <div class="safe-box">
                        🟢 No configured PPE violation detected
                        in the video.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
