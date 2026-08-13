import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import json
import smtplib
from email.message import EmailMessage
from datetime import datetime

# Firebase
import firebase_admin
from firebase_admin import credentials, messaging

# Browser JS
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

if "firebase_ready" not in st.session_state:
    st.session_state.firebase_ready = False

if "location" not in st.session_state:
    st.session_state.location = None


# ============================================================
# DETECTION SETTINGS
# ============================================================

GENERAL_CONFIDENCE = 0.30

# NO-Safety Vest false detections reduce cheyyadaniki
NO_VEST_CONFIDENCE = 0.75


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

    .safe-box {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-radius: 12px;
        padding: 16px;
        color: #065f46 !important;
        font-weight: 700;
        margin-top: 15px;
    }

    .violation-box {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 12px;
        padding: 18px;
        color: #991b1b !important;
        margin-top: 15px;
    }

    .violation-title {
        font-size: 18px;
        font-weight: 800;
        color: #991b1b !important;
        margin-bottom: 12px;
    }

    .violation-text {
        font-size: 15px;
        font-weight: 600;
        color: #991b1b !important;
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

        .detect-title {
            font-size: 26px;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = os.path.join(
        os.path.dirname(__file__),
        "best.pt"
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"best.pt not found at: {model_path}"
        )

    return YOLO(model_path)


# ============================================================
# FIREBASE WEB CONFIG
# ============================================================

firebase_web_config = {

    "apiKey": "AIzaSyDBk0B4le-PFo_AILSxTZG1m7yUsOizrDI",

    "authDomain": "guardx-ai.firebaseapp.com",

    "projectId": "guardx-ai",

    "storageBucket": "guardx-ai.firebasestorage.app",

    "messagingSenderId": "551369972933",

    "appId": "1:551369972933:web:f4e4186a3501ee0ee8e4ac"
}


# ============================================================
# VAPID KEY
# ============================================================

VAPID_KEY = (
    "BGE1N7sXhztv_V_XZ8yxQR9dn74UMgg17UKGmigh"
    "YXZGmW1sGfOCHSAqyIvJZ77GeS2-tnlgBbLVTmAVgJrdQ7M"
)


# ============================================================
# GET BROWSER LIVE LOCATION
# ============================================================

def get_live_location():

    js_code = """
    (async () => {

        try {

            if (!navigator.geolocation) {
                return "LOCATION_NOT_SUPPORTED";
            }

            const position = await new Promise(
                (resolve, reject) => {

                    navigator.geolocation.getCurrentPosition(
                        resolve,
                        reject,
                        {
                            enableHighAccuracy: true,
                            timeout: 10000,
                            maximumAge: 0
                        }
                    );

                }
            );

            const latitude =
                position.coords.latitude;

            const longitude =
                position.coords.longitude;

            return JSON.stringify({
                latitude: latitude,
                longitude: longitude
            });

        } catch(error) {

            return "LOCATION_ERROR:" + error.message;

        }

    })();
    """

    return streamlit_js_eval(
        js_expressions=js_code,
        want_output=True,
        key="live_location"
    )


# ============================================================
# GET FCM TOKEN
# ============================================================

def get_fcm_token():

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

            const firebaseAppModule =
                await import(
                    "https://www.gstatic.com/firebasejs/10.13.2/firebase-app.js"
                );

            const firebaseMessagingModule =
                await import(
                    "https://www.gstatic.com/firebasejs/10.13.2/firebase-messaging.js"
                );

            const firebaseConfig =
                {json.dumps(firebase_web_config)};

            const app =
                firebaseAppModule.initializeApp(
                    firebaseConfig
                );

            const messaging =
                firebaseMessagingModule.getMessaging(app);

            const token =
                await firebaseMessagingModule.getToken(
                    messaging,
                    {{
                        vapidKey: "{VAPID_KEY}",
                        serviceWorkerRegistration: registration
                    }}
                );

            if (!token) {{
                return "NO_TOKEN";
            }}

            return token;

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
# SEND PUSH NOTIFICATION
# ============================================================

def send_push_notification(token, title, body):

    try:

        if not firebase_ready:
            return False

        message = messaging.Message(

            notification=messaging.Notification(
                title=title,
                body=body
            ),

            token=token
        )

        messaging.send(message)

        return True

    except Exception as e:

        st.session_state.push_error = str(e)

        return False


# ============================================================
# SEND GMAIL SAFETY ALERT
# ============================================================

def send_email_alert(
    violations,
    source="image",
    location_data=None
):

    try:

        sender = st.secrets["EMAIL_SENDER"]
        password = st.secrets["EMAIL_PASSWORD"]
        receiver = st.secrets["EMAIL_RECEIVER"]

        # ----------------------------------------------------
        # DATE / TIME
        # ----------------------------------------------------

        now = datetime.now()

        date_text = now.strftime(
            "%d-%b-%Y"
        )

        time_text = now.strftime(
            "%I:%M:%S %p"
        )

        # ----------------------------------------------------
        # VIOLATIONS
        # ----------------------------------------------------

        violation_text = ", ".join(
            sorted(
                set(
                    name
                    for name, confidence
                    in violations
                )
            )
        )

        # ----------------------------------------------------
        # LOCATION
        # ----------------------------------------------------

        if location_data:

            latitude = location_data["latitude"]
            longitude = location_data["longitude"]

            maps_link = (
                f"https://www.google.com/maps/"
                f"search/?api=1&query="
                f"{latitude},{longitude}"
            )

            location_text = (
                f"Latitude  : {latitude}\n"
                f"Longitude : {longitude}\n"
                f"Google Maps: {maps_link}"
            )

        else:

            fixed_location = st.secrets.get(
                "LOCATION",
                "Location unavailable"
            )

            location_text = (
                f"Location : {fixed_location}\n"
                f"Google Maps : Location unavailable"
            )

        # ----------------------------------------------------
        # EMAIL SUBJECT
        # ----------------------------------------------------

        subject = (
            f"🚨 GuardX-AI Safety Alert | "
            f"{violation_text}"
        )

        # ----------------------------------------------------
        # EMAIL BODY
        # ----------------------------------------------------

        body = f"""
GUARDX-AI
AI-POWERED CONSTRUCTION PPE DETECTION SYSTEM
================================================

🚨 PPE SAFETY VIOLATION DETECTED

GuardX-AI has detected a potential PPE violation.

------------------------------------------------
VIOLATION DETAILS
------------------------------------------------

Violation        : {violation_text}
Detection Source : {source}
Date             : {date_text}
Time             : {time_text}

------------------------------------------------
LOCATION
------------------------------------------------

{location_text}

------------------------------------------------
SAFETY ACTION REQUIRED
------------------------------------------------

Please immediately verify the detected worker.

Ensure that the required Personal Protective
Equipment (PPE) is being worn correctly.

This alert indicates a potential unsafe
working condition and should be verified
immediately.

------------------------------------------------

GuardX-AI
Construction PPE Safety Monitoring System

This is an automatically generated safety alert.
"""

        # ----------------------------------------------------
        # CREATE EMAIL
        # ----------------------------------------------------

        message = EmailMessage()

        message["From"] = sender
        message["To"] = receiver
        message["Subject"] = subject

        message.set_content(body)

        # ----------------------------------------------------
        # SEND THROUGH GMAIL
        # ----------------------------------------------------

        with smtplib.SMTP(
            "smtp.gmail.com",
            587
        ) as server:

            server.starttls()

            server.login(
                sender,
                password
            )

            server.send_message(
                message
            )

        return True

    except Exception as e:

        st.session_state.email_error = str(e)

        return False


# ============================================================
# EXTRACT DETECTIONS
# ============================================================

def extract_detections(result):

    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:

        class_id = int(
            box.cls[0]
        )

        confidence = float(
            box.conf[0]
        )

        name = str(
            result.names[class_id]
        )

        detections.append(
            (
                name,
                confidence
            )
        )

    return detections


# ============================================================
# NORMALIZE CLASS NAME
# ============================================================

def normalize_class_name(name):

    normalized = str(
        name
    ).lower().strip()

    normalized = normalized.replace(
        "_",
        "-"
    )

    normalized = " ".join(
        normalized.split()
    )

    return normalized


# ============================================================
# CHECK PPE VIOLATIONS
# ============================================================

def get_violations(detections):

    """
    GuardX-AI classes:

    0 -> Hardhat
    1 -> NO-Hardhat
    2 -> Mask
    3 -> NO-Mask
    4 -> Safety Vest
    5 -> NO-Safety Vest

    IMPORTANT:

    Class 4 has ZERO annotations in dataset.

    Therefore we DO NOT rely on positive
    Safety Vest detection.

    NO-Safety Vest is considered a violation
    only when confidence >= 0.75.
    """

    violations = []

    for name, confidence in detections:

        normalized_name = normalize_class_name(
            name
        )

        # --------------------------------------------
        # NO HARDHAT
        # --------------------------------------------

        if normalized_name in {
            "no-hardhat",
            "no hardhat"
        }:

            violations.append(
                (
                    name,
                    confidence
                )
            )

        # --------------------------------------------
        # NO MASK
        # --------------------------------------------

        elif normalized_name in {
            "no-mask",
            "no mask"
        }:

            violations.append(
                (
                    name,
                    confidence
                )
            )

        # --------------------------------------------
        # NO SAFETY VEST
        # --------------------------------------------

        elif normalized_name in {
            "no-safety-vest",
            "no safety vest"
        }:

            if confidence >= NO_VEST_CONFIDENCE:

                violations.append(
                    (
                        name,
                        confidence
                    )
                )

    return violations


# ============================================================
# DISPLAY VIOLATION BOX
# ============================================================

def display_violation_box(violations):

    if not violations:
        return

    violation_names = []

    for name, confidence in violations:

        if name not in violation_names:

            violation_names.append(
                name
            )

    violation_text = ", ".join(
        violation_names
    )

    st.markdown(
        f"""
        <div class="violation-box">

            <div class="violation-title">
                🚨 PPE VIOLATION DETECTED
            </div>

            <div class="violation-text">
                Violations:
                <strong>{violation_text}</strong>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DISPLAY SAFE BOX
# ============================================================

def display_safe_box():

    st.markdown(
        """
        <div class="safe-box">
            🟢 SAFE — No PPE violation detected.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HANDLE VIOLATION ALERT
# ============================================================

def handle_violation_alert(
    violations,
    source="image"
):

    if not violations:
        return

    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    location_data = None

    try:

        location_result = get_live_location()

        if location_result:

            if isinstance(
                location_result,
                str
            ):

                if location_result.startswith(
                    "{"
                ):

                    location_data = json.loads(
                        location_result
                    )

                    st.session_state.location = (
                        location_data
                    )

    except Exception:

        location_data = None

    # --------------------------------------------------------
    # VIOLATION TEXT
    # --------------------------------------------------------

    violation_text = ", ".join(
        sorted(
            set(
                name
                for name, confidence
                in violations
            )
        )
    )

    # --------------------------------------------------------
    # PUSH NOTIFICATION
    # --------------------------------------------------------

    title = "🚨 GuardX-AI Safety Alert"

    body = (
        f"PPE violation detected: "
        f"{violation_text}"
    )

    if st.session_state.fcm_token:

        success = send_push_notification(
            st.session_state.fcm_token,
            title,
            body
        )

        if success:

            st.success(
                "📲 Safety push notification sent!"
            )

    # --------------------------------------------------------
    # GMAIL ALERT
    # --------------------------------------------------------

    email_success = send_email_alert(
        violations,
        source,
        location_data
    )

    if email_success:

        st.success(
            "📧 Safety alert sent to Gmail!"
        )

    else:

        st.warning(
            "⚠️ Violation detected, but Gmail alert "
            "could not be sent."
        )

        if "email_error" in st.session_state:

            st.caption(
                st.session_state.email_error
            )


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "home":

    st.markdown(
        """
        <div class="main-title">
            🦺 GuardX-AI – Construction PPE Detection System
        </div>
        """,
        unsafe_allow_html=True
    )

    left_col, right_col = st.columns(
        [0.38, 0.62],
        gap="large"
    )

    with left_col:

        st.markdown(
            """
            <div class="aicw-text">
                AI Career for Women
                <br>
                (AICW)
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="capstone-text">
                Capstone Project
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        if st.button(
            "🔍  PREDICT",
            key="home_predict_button",
            use_container_width=True
        ):

            st.session_state.page = "predict"

            st.rerun()

    with right_col:

        st.markdown(
            """
            <div class="description-title">
                Project Description
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.container(border=True):

            st.write(
                "Construction sites involve high-risk activities "
                "where proper Personal Protective Equipment (PPE) "
                "is essential for worker safety. However, manually "
                "monitoring PPE continuously is difficult and "
                "time-consuming."
            )

            st.write(
                "GuardX-AI is an AI-powered Construction PPE "
                "Detection System that uses YOLO object detection "
                "to identify PPE violations from construction-site "
                "images, camera captures, and videos."
            )

            st.write(
                "When a potential PPE violation is detected, "
                "GuardX-AI provides an on-screen warning and "
                "can send a safety alert through push notification "
                "and Gmail."
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
                """
                <div class="card-heading">
                    TEAM MEMBERS
                </div>

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
                """
                <div class="card-heading">
                    GMAIL
                </div>

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
                """
                <div class="card-heading">
                    GUIDE NAME
                </div>

                <div class="card-text">
                    MD.Abdul Aziz
                </div>

                <br>

                <div class="card-heading">
                    DESIGNATION
                </div>

                <div class="card-text">
                    Trainer, Co-Lead-AICW
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        """
        <div class="footer-text">
            GuardX-AI – Construction PPE Detection System
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

else:

    st.markdown(
        """
        <div class="detect-title">
            🦺 GuardX-AI
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="detect-subtitle">
            AI-Powered Construction PPE Detection System
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    try:

        model = load_model()

    except Exception as e:

        st.error(
            "❌ best.pt model load avvaledu."
        )

        st.info(
            "Make sure best.pt is in the same folder as app.py."
        )

        st.caption(
            str(e)
        )

        st.stop()

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    with st.container(border=True):

        st.markdown(
            """
            <div class="card-heading">
                🤖 GuardX-AI Detection Model
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write(
            "Configured PPE classes:"
        )

        st.markdown(
            """
            🪖 **Hardhat**  
            🚨 **NO-Hardhat**  
            😷 **Mask**  
            🚨 **NO-Mask**  
            🦺 **Safety Vest**  
            🚨 **NO-Safety Vest**
            """
        )

        st.caption(
            f"General confidence: {GENERAL_CONFIDENCE:.2f} | "
            f"NO-Safety Vest confidence: {NO_VEST_CONFIDENCE:.2f}"
        )

    st.write("")

    # ========================================================
    # PUSH NOTIFICATION
    # ========================================================

    with st.container(border=True):

        st.markdown(
            "### 🔔 Safety Push Notifications"
        )

        st.write(
            "Enable notifications to receive a safety alert "
            "when GuardX-AI detects a PPE violation."
        )

        if not st.session_state.push_enabled:

            if st.button(
                "🔔 Enable Push Notifications",
                key="prediction_push_enable",
                use_container_width=True
            ):

                token_result = get_fcm_token()

                if token_result:

                    if isinstance(
                        token_result,
                        str
                    ):

                        if token_result.startswith(
                            "ERROR:"
                        ):

                            st.error(
                                "❌ Push notification setup failed."
                            )

                            st.caption(
                                token_result
                            )

                        elif token_result == "PERMISSION_DENIED":

                            st.warning(
                                "🔕 Notification permission denied."
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

                            st.session_state.fcm_token = (
                                token_result
                            )

                            st.session_state.push_enabled = (
                                True
                            )

                            st.success(
                                "🔔 Push notifications enabled successfully!"
                            )

                            st.rerun()

                else:

                    st.warning(
                        "Waiting for notification permission/token..."
                    )

        else:

            st.success(
                "🔔 Push notifications are enabled."
            )

    st.write("")

    # ========================================================
    # LOCATION
    # ========================================================

    with st.container(border=True):

        st.markdown(
            "### 📍 Live Location"
        )

        st.write(
            "When a violation occurs, GuardX-AI can request "
            "your browser's current location and include it "
            "in the Gmail safety alert."
        )

        if st.button(
            "📍 Allow Location Access",
            key="location_button",
            use_container_width=True
        ):

            location_result = get_live_location()

            if location_result:

                if isinstance(
                    location_result,
                    str
                ):

                    if location_result.startswith("{"):

                        try:

                            location_data = json.loads(
                                location_result
                            )

                            st.session_state.location = (
                                location_data
                            )

                            st.success(
                                "📍 Live location access enabled."
                            )

                            st.write(
                                f"Latitude: "
                                f"{location_data['latitude']}"
                            )

                            st.write(
                                f"Longitude: "
                                f"{location_data['longitude']}"
                            )

                        except Exception:

                            st.warning(
                                "Location data could not be read."
                            )

                    elif location_result.startswith(
                        "LOCATION_ERROR:"
                    ):

                        st.warning(
                            "Location permission was denied "
                            "or location could not be obtained."
                        )

                    elif location_result == (
                        "LOCATION_NOT_SUPPORTED"
                    ):

                        st.warning(
                            "This browser does not support location."
                        )

        if st.session_state.location:

            latitude = st.session_state.location[
                "latitude"
            ]

            longitude = st.session_state.location[
                "longitude"
            ]

            maps_link = (
                f"https://www.google.com/maps/"
                f"search/?api=1&query="
                f"{latitude},{longitude}"
            )

            st.success(
                "📍 Location ready for safety alerts."
            )

            st.markdown(
                f"[🗺️ Open current location in Google Maps]({maps_link})"
            )

    st.write("")

    # ========================================================
    # BACK
    # ========================================================

    if st.button(
        "← Back to Home",
        key="prediction_back_button",
        use_container_width=True
    ):

        st.session_state.page = "home"

        st.rerun()

    st.write("")

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
        horizontal=True,
        key="input_type_selector"
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

            st.subheader(
                "Upload Image"
            )

            uploaded_image = st.file_uploader(
                "Choose a construction image",
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
                    caption="Input Image",
                    use_container_width=True
                )

                if st.button(
                    "🔍 Detect PPE",
                    key="image_detect_button",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Detecting PPE..."
                    ):

                        result = model.predict(
                            np.array(image),
                            conf=GENERAL_CONFIDENCE,
                            verbose=False
                        )[0]

                    annotated = result.plot()

                    annotated = cv2.cvtColor(
                        annotated,
                        cv2.COLOR_BGR2RGB
                    )

                    detections = extract_detections(
                        result
                    )

                    violations = get_violations(
                        detections
                    )

                    st.session_state.result_image = (
                        annotated
                    )

                    st.session_state.detections = (
                        detections
                    )

                    st.session_state.image_violations = (
                        violations
                    )

                    if violations:

                        handle_violation_alert(
                            violations,
                            "image"
                        )

        with result_col:

            st.subheader(
                "Detection Result"
            )

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

                detections = (
                    st.session_state.detections
                )

                violations = (
                    st.session_state.get(
                        "image_violations",
                        []
                    )
                )

                if violations:

                    display_violation_box(
                        violations
                    )

                else:

                    display_safe_box()

                if detections:

                    st.write(
                        "### Detected Objects"
                    )

                    for name, confidence in detections:

                        # Low confidence NO-Vest
                        # UI lo kuda chupinchakudadhu.
                        normalized = normalize_class_name(
                            name
                        )

                        if (
                            normalized in {
                                "no-safety-vest",
                                "no safety vest"
                            }
                            and
                            confidence < NO_VEST_CONFIDENCE
                        ):
                            continue

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
            "Take a photo",
            key="camera_input"
        )

        if camera_image:

            image = Image.open(
                camera_image
            ).convert("RGB")

            if st.button(
                "🔍 Detect PPE",
                key="camera_detect_button",
                use_container_width=True
            ):

                with st.spinner(
                    "Detecting PPE..."
                ):

                    result = model.predict(
                        np.array(image),
                        conf=GENERAL_CONFIDENCE,
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

                detections = extract_detections(
                    result
                )

                violations = get_violations(
                    detections
                )

                if violations:

                    display_violation_box(
                        violations
                    )

                    handle_violation_alert(
                        violations,
                        "camera image"
                    )

                else:

                    display_safe_box()

                if detections:

                    st.write(
                        "### Detected Objects"
                    )

                    for name, confidence in detections:

                        normalized = normalize_class_name(
                            name
                        )

                        if (
                            normalized in {
                                "no-safety-vest",
                                "no safety vest"
                            }
                            and
                            confidence < NO_VEST_CONFIDENCE
                        ):
                            continue

                        st.write(
                            f"**{name}** — "
                            f"{confidence * 100:.1f}%"
                        )

                else:

                    st.info(
                        "No objects detected."
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
            ],
            key="video_upload"
        )

        if uploaded_video:

            st.video(
                uploaded_video
            )

            if st.button(
                "🎥 Detect PPE in Video",
                key="video_detect_button",
                use_container_width=True
            ):

                with st.spinner(
                    "Processing video... Please wait."
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

                    output_path = output_file.name

                    output_file.close()

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

                    while True:

                        ret, frame = cap.read()

                        if not ret:
                            break

                        result = model.predict(
                            frame,
                            conf=GENERAL_CONFIDENCE,
                            verbose=False
                        )[0]

                        annotated = result.plot()

                        writer.write(
                            annotated
                        )

                        frame_detections = (
                            extract_detections(
                                result
                            )
                        )

                        frame_violations = (
                            get_violations(
                                frame_detections
                            )
                        )

                        for name, confidence in frame_violations:

                            video_violations.add(
                                name
                            )

                    cap.release()

                    writer.release()

                    try:

                        os.remove(
                            input_file.name
                        )

                    except:

                        pass

                st.success(
                    "✅ Video processing completed."
                )

                st.video(
                    output_path
                )

                if video_violations:

                    video_violation_list = [
                        (
                            violation,
                            1.0
                        )
                        for violation
                        in video_violations
                    ]

                    display_violation_box(
                        video_violation_list
                    )

                    handle_violation_alert(
                        video_violation_list,
                        "video"
                    )

                else:

                    display_safe_box()
