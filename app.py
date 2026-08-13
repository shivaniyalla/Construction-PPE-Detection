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
from textwrap import dedent

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

defaults = {
    "page": "home",
    "fcm_token": None,
    "push_enabled": False,
    "firebase_ready": False,
    "location": None,
    "email_error": None,
    "push_error": None,
    "result_image": None,
    "detections": [],
    "image_violations": [],
    "video_output": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# DETECTION SETTINGS
# ============================================================

GENERAL_CONFIDENCE = 0.30
NO_VEST_CONFIDENCE = 0.75


# ============================================================
# HTML RENDER HELPER
# IMPORTANT:
# dedent removes leading spaces so HTML will NOT appear
# as plain code/text on the page.
# ============================================================

def render_html(content):
    st.markdown(
        dedent(content),
        unsafe_allow_html=True
    )


# ============================================================
# FIREBASE ADMIN
# ============================================================

def initialize_firebase():

    try:

        if firebase_admin._apps:
            return True

        if "firebase" not in st.secrets:
            return False

        firebase_config = dict(
            st.secrets["firebase"]
        )

        if "private_key" in firebase_config:
            firebase_config["private_key"] = (
                firebase_config["private_key"]
                .replace("\\n", "\n")
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
# GLOBAL CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(124,58,237,0.18),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 15%,
            rgba(6,182,212,0.13),
            transparent 28%
        ),
        radial-gradient(
            circle at 50% 100%,
            rgba(236,72,153,0.08),
            transparent 35%
        ),
        #090b16;

    color: #f8fafc;
}

.block-container {
    max-width: 1400px;
    padding-top: 30px;
    padding-bottom: 50px;
    padding-left: 5%;
    padding-right: 5%;
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


/* =========================================================
   HOME HERO
========================================================= */

.home-hero {
    text-align: center;
    padding: 18px 20px 30px 20px;
}

.hero-badge {
    display: inline-block;

    padding: 7px 15px;

    border-radius: 30px;

    background: rgba(139,92,246,0.14);

    border:
        1px solid rgba(167,139,250,0.35);

    color: #c4b5fd;

    font-size: 13px;

    font-weight: 700;

    margin-bottom: 15px;
}

.hero-title {
    font-size: 56px;

    line-height: 1.1;

    font-weight: 900;

    letter-spacing: -2px;

    background:
        linear-gradient(
            90deg,
            #ffffff,
            #c4b5fd,
            #67e8f9,
            #f9a8d4
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #cbd5e1;

    font-size: 19px;

    font-weight: 600;

    margin-top: 10px;
}


/* =========================================================
   HOME CARDS
========================================================= */

.home-card {
    background:
        linear-gradient(
            145deg,
            rgba(30,41,59,0.88),
            rgba(15,23,42,0.78)
        );

    border:
        1px solid rgba(148,163,184,0.17);

    border-radius: 24px;

    padding: 28px;

    box-shadow:
        0 18px 50px rgba(0,0,0,0.28),
        inset 0 1px 0 rgba(255,255,255,0.04);

    min-height: 250px;
}

.home-card:hover {
    border-color:
        rgba(139,92,246,0.40);
}


/* =========================================================
   AICW
========================================================= */

.aicw-card {
    min-height: 330px;

    display: flex;

    flex-direction: column;

    justify-content: center;
}

.aicw-icon {
    font-size: 58px;

    margin-bottom: 15px;
}

.aicw-title {
    font-size: 30px;

    font-weight: 900;

    color: #ffffff;
}

.aicw-subtitle {
    color: #a78bfa;

    font-size: 18px;

    font-weight: 700;

    margin-top: 8px;
}

.capstone {
    margin-top: 22px;

    padding: 9px 15px;

    border-radius: 30px;

    background:
        rgba(34,211,238,0.10);

    border:
        1px solid rgba(34,211,238,0.25);

    color: #67e8f9;

    font-weight: 700;

    display: inline-block;
}


/* =========================================================
   ABOUT
========================================================= */

.about-title {
    color: #ffffff;

    font-size: 20px;

    font-weight: 850;

    margin-bottom: 14px;
}

.about-text {
    color: #aab5c7;

    font-size: 14px;

    line-height: 1.8;
}


/* =========================================================
   TEAM
========================================================= */

.team-heading {
    color: #f8fafc;

    font-size: 22px;

    font-weight: 900;

    margin: 12px 0 16px 0;
}

.team-card {
    background:
        linear-gradient(
            145deg,
            rgba(30,41,59,0.82),
            rgba(15,23,42,0.72)
        );

    border:
        1px solid rgba(148,163,184,0.15);

    border-radius: 20px;

    padding: 20px;

    min-height: 165px;

    box-shadow:
        0 12px 35px rgba(0,0,0,0.20);
}

.avatar {
    width: 56px;

    height: 56px;

    border-radius: 50%;

    display: flex;

    align-items: center;

    justify-content: center;

    font-size: 28px;

    background:
        linear-gradient(
            135deg,
            #7c3aed,
            #06b6d4
        );

    box-shadow:
        0 8px 25px rgba(124,58,237,0.30);

    margin-bottom: 12px;
}

.member-name {
    color: #f8fafc;

    font-weight: 800;

    font-size: 15px;
}

.member-role {
    color: #94a3b8;

    font-size: 12px;

    margin-top: 4px;
}


/* =========================================================
   GUIDE
========================================================= */

.guide-card {
    background:
        linear-gradient(
            145deg,
            rgba(30,41,59,0.88),
            rgba(15,23,42,0.78)
        );

    border:
        1px solid rgba(167,139,250,0.20);

    border-radius: 22px;

    padding: 20px 24px;

    box-shadow:
        0 12px 35px rgba(0,0,0,0.22);
}


/* =========================================================
   BUTTONS
========================================================= */

div.stButton > button {
    width: 100%;

    min-height: 48px;

    border-radius: 13px !important;

    background:
        linear-gradient(
            135deg,
            #7c3aed,
            #2563eb
        ) !important;

    color: white !important;

    border:
        1px solid rgba(167,139,250,0.4) !important;

    font-size: 14px !important;

    font-weight: 800 !important;

    box-shadow:
        0 8px 25px rgba(124,58,237,0.25);

    transition: 0.25s ease;
}

div.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 12px 35px rgba(124,58,237,0.40);

    border-color:
        #a78bfa !important;
}


/* =========================================================
   RADIO
========================================================= */

div[data-testid="stRadio"] {
    background:
        rgba(15,23,42,0.65);

    border:
        1px solid rgba(148,163,184,0.14);

    padding: 12px 18px;

    border-radius: 16px;
}

div[data-testid="stRadio"] label {
    color: #cbd5e1 !important;

    font-weight: 700 !important;
}


/* =========================================================
   FILE UPLOADER
========================================================= */

div[data-testid="stFileUploader"] {
    background:
        rgba(15,23,42,0.65);

    border:
        1px dashed rgba(139,92,246,0.45);

    border-radius: 16px;

    padding: 10px;
}


/* =========================================================
   STREAMLIT CONTAINERS
========================================================= */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background:
        linear-gradient(
            145deg,
            rgba(30,41,59,0.80),
            rgba(15,23,42,0.72)
        ) !important;

    border:
        1px solid rgba(148,163,184,0.14) !important;

    border-radius: 20px !important;

    box-shadow:
        0 15px 40px rgba(0,0,0,0.22);

    padding: 10px !important;
}


/* =========================================================
   TEXT
========================================================= */

.stMarkdown,
.stMarkdown p,
.stMarkdown li {
    color: #cbd5e1;
}

label {
    color: #cbd5e1 !important;
}


/* =========================================================
   CARD TITLE
========================================================= */

.card-title {
    color: #ffffff;

    font-size: 20px;

    font-weight: 900;

    margin-bottom: 12px;
}


/* =========================================================
   DETECTION
========================================================= */

.detect-title {
    text-align: center;

    font-size: 44px;

    font-weight: 900;

    background:
        linear-gradient(
            90deg,
            #c4b5fd,
            #67e8f9,
            #f9a8d4
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}

.detect-subtitle {
    text-align: center;

    color: #94a3b8;

    font-size: 15px;

    margin-bottom: 28px;
}


/* =========================================================
   SAFE
========================================================= */

.safe-box {
    margin-top: 16px;

    padding: 18px;

    border-radius: 16px;

    background:
        rgba(16,185,129,0.10);

    border:
        1px solid rgba(52,211,153,0.35);

    color: #6ee7b7 !important;

    font-weight: 800;
}


/* =========================================================
   VIOLATION
========================================================= */

.violation-box {
    margin-top: 16px;

    padding: 20px;

    border-radius: 17px;

    background:
        linear-gradient(
            135deg,
            rgba(239,68,68,0.14),
            rgba(127,29,29,0.12)
        );

    border:
        1px solid rgba(248,113,113,0.42);

    box-shadow:
        0 10px 30px rgba(239,68,68,0.12);
}

.violation-title {
    color: #fca5a5 !important;

    font-size: 18px;

    font-weight: 900;
}

.violation-text {
    color: #fecaca !important;

    font-size: 14px;

    font-weight: 700;

    margin-top: 8px;
}


/* =========================================================
   ALERT
========================================================= */

.alert-card {
    padding: 16px;

    border-radius: 15px;

    background:
        rgba(59,130,246,0.08);

    border:
        1px solid rgba(96,165,250,0.22);

    color: #bfdbfe;
}


/* =========================================================
   FOOTER
========================================================= */

.footer {
    text-align: center;

    color: #64748b;

    font-size: 13px;

    padding: 35px 0 10px 0;
}


/* =========================================================
   MOBILE
========================================================= */

@media(max-width: 900px) {

    .hero-title {
        font-size: 38px;
    }

    .detect-title {
        font-size: 30px;
    }

    .block-container {
        padding-left: 4%;
        padding-right: 4%;
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

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"best.pt not found: {model_path}"
        )

    return YOLO(model_path)


# ============================================================
# NORMALIZE CLASS NAME
# ============================================================

def normalize_class_name(name):

    name = str(name).lower().strip()

    name = name.replace("_", "-")

    name = " ".join(name.split())

    return name


# ============================================================
# IS VIOLATION
# ============================================================

def is_violation(name, confidence):

    normalized = normalize_class_name(name)

    if normalized in {
        "no-hardhat",
        "no hardhat"
    }:
        return True

    if normalized in {
        "no-mask",
        "no mask"
    }:
        return True

    if normalized in {
        "no-safety-vest",
        "no safety vest"
    }:

        return confidence >= NO_VEST_CONFIDENCE

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

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        detections.append({

            "name": name,

            "confidence": confidence,

            "box": (
                x1,
                y1,
                x2,
                y2
            ),

            "violation": is_violation(
                name,
                confidence
            )
        })

    return detections


# ============================================================
# GET VIOLATIONS
# ============================================================

def get_violations(detections):

    violations = []

    for detection in detections:

        if detection["violation"]:

            violations.append(
                (
                    detection["name"],
                    detection["confidence"]
                )
            )

    return violations


# ============================================================
# CUSTOM BOXES
# ============================================================

def draw_custom_boxes(
    frame,
    detections
):

    output = frame.copy()

    for detection in detections:

        name = detection["name"]

        confidence = detection["confidence"]

        x1, y1, x2, y2 = detection["box"]

        violation = detection["violation"]

        if violation:

            color = (
                40,
                40,
                255
            )

            label_color = (
                40,
                40,
                255
            )

        else:

            color = (
                0,
                220,
                190
            )

            label_color = (
                0,
                220,
                190
            )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        label = (
            f"{name} "
            f"{confidence * 100:.1f}%"
        )

        font = cv2.FONT_HERSHEY_SIMPLEX

        font_scale = 0.55

        thickness = 2

        (
            tw,
            th
        ), _ = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness
        )

        label_y = max(
            y1,
            th + 10
        )

        cv2.rectangle(
            output,
            (
                x1,
                label_y - th - 10
            ),
            (
                x1 + tw + 10,
                label_y
            ),
            label_color,
            -1
        )

        cv2.putText(
            output,
            label,
            (
                x1 + 5,
                label_y - 5
            ),
            font,
            font_scale,
            (
                255,
                255,
                255
            ),
            thickness,
            cv2.LINE_AA
        )

    return output


# ============================================================
# LIVE LOCATION
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

            return JSON.stringify({
                latitude: position.coords.latitude,
                longitude: position.coords.longitude
            });

        } catch(error) {

            return "LOCATION_ERROR:" + error.message;

        }

    })();
    """

    return streamlit_js_eval(
        js_expressions=js_code,
        want_output=True,
        key="location_request"
    )


# ============================================================
# FCM TOKEN
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

            const appModule =
                await import(
                    "https://www.gstatic.com/firebasejs/10.13.2/firebase-app.js"
                );

            const messagingModule =
                await import(
                    "https://www.gstatic.com/firebasejs/10.13.2/firebase-messaging.js"
                );

            const firebaseConfig =
                {json.dumps(firebase_web_config)};

            const app =
                appModule.initializeApp(
                    firebaseConfig
                );

            const messaging =
                messagingModule.getMessaging(app);

            const token =
                await messagingModule.getToken(
                    messaging,
                    {{
                        vapidKey: "{VAPID_KEY}",
                        serviceWorkerRegistration:
                            registration
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
# PUSH NOTIFICATION
# ============================================================

def send_push_notification(
    token,
    title,
    body
):

    try:

        if not firebase_ready:
            return False

        message = messaging.Message(

            notification=
                messaging.Notification(
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
# EMAIL ALERT
# ============================================================

def send_email_alert(
    violations,
    source="image",
    location_data=None
):

    try:

        required = [
            "EMAIL_SENDER",
            "EMAIL_PASSWORD",
            "EMAIL_RECEIVER"
        ]

        missing = [
            key
            for key in required
            if key not in st.secrets
        ]

        if missing:

            st.session_state.email_error = (
                "Missing Streamlit secrets: "
                + ", ".join(missing)
            )

            return False

        sender = st.secrets[
            "EMAIL_SENDER"
        ]

        password = st.secrets[
            "EMAIL_PASSWORD"
        ]

        receiver = st.secrets[
            "EMAIL_RECEIVER"
        ]

        now = datetime.now()

        date_text = now.strftime(
            "%d-%b-%Y"
        )

        time_text = now.strftime(
            "%I:%M:%S %p"
        )

        violation_text = ", ".join(
            sorted(
                set(
                    name
                    for name, confidence
                    in violations
                )
            )
        )

        if location_data:

            latitude = location_data[
                "latitude"
            ]

            longitude = location_data[
                "longitude"
            ]

            maps_link = (
                "https://www.google.com/maps/"
                "search/?api=1&query="
                f"{latitude},{longitude}"
            )

            location_text = (
                f"Latitude  : {latitude}\n"
                f"Longitude : {longitude}\n"
                f"Google Maps: {maps_link}"
            )

        else:

            location_text = (
                "Location : Not available"
            )

        subject = (
            f"🚨 GuardX-AI Safety Alert | "
            f"{violation_text}"
        )

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
ACTION REQUIRED
------------------------------------------------

Please immediately verify the detected worker.

Ensure that the required Personal Protective
Equipment is being worn correctly.

This alert indicates a potential unsafe
working condition and should be verified.

------------------------------------------------

GuardX-AI
Construction PPE Safety Monitoring System

Automatically generated safety alert.
"""

        message = EmailMessage()

        message["From"] = sender

        message["To"] = receiver

        message["Subject"] = subject

        message.set_content(body)

        with smtplib.SMTP(
            "smtp.gmail.com",
            587,
            timeout=20
        ) as server:

            server.ehlo()

            server.starttls()

            server.ehlo()

            server.login(
                sender,
                password
            )

            server.send_message(
                message
            )

        st.session_state.email_error = None

        return True

    except Exception as e:

        st.session_state.email_error = str(e)

        return False


# ============================================================
# ALERT HANDLER
# ============================================================

def handle_violation_alert(
    violations,
    source="image"
):

    if not violations:
        return

    location_data = st.session_state.get(
        "location"
    )

    if not location_data:

        try:

            location_result = (
                get_live_location()
            )

            if (
                isinstance(
                    location_result,
                    str
                )
                and location_result.startswith(
                    "{"
                )
            ):

                location_data = json.loads(
                    location_result
                )

                st.session_state.location = (
                    location_data
                )

        except Exception:

            location_data = None

    violation_text = ", ".join(
        sorted(
            set(
                name
                for name, confidence
                in violations
            )
        )
    )

    # PUSH
    if st.session_state.fcm_token:

        push_success = (
            send_push_notification(
                st.session_state.fcm_token,
                "🚨 GuardX-AI Safety Alert",
                (
                    "PPE violation detected: "
                    + violation_text
                )
            )
        )

        if push_success:

            st.success(
                "📲 Push safety alert sent!"
            )

        else:

            st.warning(
                "⚠️ Push notification failed."
            )

    # EMAIL
    email_success = send_email_alert(
        violations,
        source,
        location_data
    )

    if email_success:

        st.success(
            "📧 Gmail safety alert sent successfully!"
        )

    else:

        st.error(
            "❌ Gmail alert could not be sent."
        )

        if st.session_state.email_error:

            st.caption(
                "Email error: "
                + st.session_state.email_error
            )


# ============================================================
# DISPLAY VIOLATION
# ============================================================

def display_violation_box(
    violations
):

    if not violations:
        return

    names = []

    for name, confidence in violations:

        if name not in names:

            names.append(name)

    text = ", ".join(names)

    render_html(
        f"""
        <div class="violation-box">

            <div class="violation-title">
                🚨 PPE VIOLATION DETECTED
            </div>

            <div class="violation-text">
                Detected:
                <strong>{text}</strong>
            </div>

            <div class="violation-text">
                🔴 Red bounding box indicates
                a safety violation.
            </div>

        </div>
        """
    )


# ============================================================
# SAFE BOX
# ============================================================

def display_safe_box():

    render_html(
        """
        <div class="safe-box">
            🟢 SAFE — No PPE violation detected.
        </div>
        """
    )


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "home":

    # ========================================================
    # HERO
    # ========================================================

    render_html(
        """
        <div class="home-hero">

            <div class="hero-badge">
                ✨ AI-POWERED SAFETY MONITORING
            </div>

            <div class="hero-title">
                🦺 GuardX-AI
            </div>

            <div class="hero-subtitle">
                AI-Powered Construction Safety Monitoring
            </div>

        </div>
        """
    )


    # ========================================================
    # TOP CARDS
    # ========================================================

    left_col, right_col = st.columns(
        [0.40, 0.60],
        gap="large"
    )


    # ========================================================
    # AICW CARD
    # ========================================================

    with left_col:

        render_html(
            """
            <div class="home-card aicw-card">

                <div class="aicw-icon">
                    👩🏻‍💻
                </div>

                <div class="aicw-title">
                    AI Career for Women
                </div>

                <div class="aicw-subtitle">
                    AICW • Capstone Project
                </div>

                <div class="capstone">
                    🚀 Intelligent Safety • Real-Time Detection
                </div>

            </div>
            """
        )

        st.write("")

        if st.button(
            "🔍 PREDICT",
            key="home_predict_button"
        ):

            st.session_state.page = (
                "predict"
            )

            st.rerun()


    # ========================================================
    # ABOUT CARD
    # ========================================================

    with right_col:

        render_html(
            """
            <div class="home-card">

                <div class="about-title">
                    🛡️ What is GuardX-AI?
                </div>

                <div class="about-text">

                    Construction sites involve high-risk
                    activities where Personal Protective
                    Equipment is essential for worker safety.

                    <br><br>

                    GuardX-AI uses YOLO-based computer vision
                    to automatically identify PPE violations
                    from images, camera captures and videos.

                    <br><br>

                    🚨 Unsafe conditions are highlighted
                    with red bounding boxes and safety alerts
                    can be delivered through Gmail and browser
                    push notifications.

                </div>

            </div>
            """
        )


    st.write("")
    st.write("")


    # ========================================================
    # TEAM HEADING
    # ========================================================

    render_html(
        """
        <div class="team-heading">
            👩🏻‍💻 Our Team
        </div>
        """
    )


    # ========================================================
    # TEAM CARDS
    # ========================================================

    c1, c2, c3, c4 = st.columns(
        4,
        gap="medium"
    )

    members = [

        (
            "👩🏻‍💻",
            "Y.D.V.Sivani",
            "AI / ML"
        ),

        (
            "👩🏻‍💻",
            "V.L.S.Asritha",
            "AI / ML"
        ),

        (
            "👩🏻‍💻",
            "R.Likhitha",
            "Development"
        ),

        (
            "👩🏻‍💻",
            "S.Poojitha Sri",
            "Development"
        )
    ]

    for col, member in zip(
        [c1, c2, c3, c4],
        members
    ):

        with col:

            render_html(
                f"""
                <div class="team-card">

                    <div class="avatar">
                        {member[0]}
                    </div>

                    <div class="member-name">
                        {member[1]}
                    </div>

                    <div class="member-role">
                        {member[2]}
                    </div>

                </div>
                """
            )


    st.write("")


    # ========================================================
    # GUIDE CARD
    # ========================================================

    render_html(
        """
        <div class="guide-card">

            <div style="
                display:flex;
                align-items:center;
                gap:18px;
            ">

                <div class="avatar"
                     style="margin:0;">
                    👨🏻‍🏫
                </div>

                <div>

                    <div class="member-name"
                         style="font-size:18px;">
                        MD. Abdul Aziz
                    </div>

                    <div class="member-role"
                         style="font-size:13px;">
                        Trainer • Co-Lead – AICW
                    </div>

                </div>

            </div>

        </div>
        """
    )


    # ========================================================
    # FOOTER
    # ========================================================

    render_html(
        """
        <div class="footer">
            🦺 GuardX-AI • Building Safer Construction Sites with AI
        </div>
        """
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

else:

    render_html(
        """
        <div class="detect-title">
            🦺 GuardX-AI
        </div>

        <div class="detect-subtitle">
            AI-Powered Construction Safety Monitoring
            • Visual Alerts • Email Safety Alerts
        </div>
        """
    )


    # ========================================================
    # MODEL
    # ========================================================

    try:

        model = load_model()

    except Exception as e:

        st.error(
            "❌ best.pt model load avvaledu."
        )

        st.info(
            "best.pt file app.py same folder lo undali."
        )

        st.caption(str(e))

        st.stop()


    # ========================================================
    # MODEL CARD
    # ========================================================

    with st.container(border=True):

        render_html(
            """
            <div class="card-title">
                🤖 GuardX-AI Vision Model
            </div>
            """
        )

        st.markdown(
            """
            🪖 Hardhat &nbsp;&nbsp;
            🚨 NO-Hardhat &nbsp;&nbsp;
            😷 Mask &nbsp;&nbsp;
            🚨 NO-Mask &nbsp;&nbsp;
            🦺 Safety Vest &nbsp;&nbsp;
            🚨 NO-Safety Vest
            """
        )

        st.caption(
            f"Detection confidence: "
            f"{GENERAL_CONFIDENCE:.2f} "
            f"| NO-Safety Vest: "
            f"{NO_VEST_CONFIDENCE:.2f}"
        )


    st.write("")


    # ========================================================
    # ALERT CARDS
    # ========================================================

    alert1, alert2 = st.columns(
        2,
        gap="large"
    )


    # ========================================================
    # PUSH
    # ========================================================

    with alert1:

        with st.container(border=True):

            st.markdown(
                "### 🔔 Push Alerts"
            )

            st.write(
                "Receive browser/mobile safety "
                "notifications when a violation "
                "is detected."
            )

            if not st.session_state.push_enabled:

                if st.button(
                    "🔔 Enable Push Notifications",
                    key="prediction_push_enable"
                ):

                    token_result = (
                        get_fcm_token()
                    )

                    if (
                        isinstance(
                            token_result,
                            str
                        )
                        and token_result
                        and not token_result.startswith(
                            (
                                "ERROR:",
                                "PERMISSION_DENIED",
                                "NOT_SUPPORTED",
                                "SERVICE_WORKER_NOT_SUPPORTED",
                                "NO_TOKEN"
                            )
                        )
                    ):

                        st.session_state.fcm_token = (
                            token_result
                        )

                        st.session_state.push_enabled = (
                            True
                        )

                        st.success(
                            "🔔 Push notifications enabled!"
                        )

                        st.rerun()

                    elif token_result == (
                        "PERMISSION_DENIED"
                    ):

                        st.warning(
                            "🔕 Browser notification "
                            "permission denied."
                        )

                    elif token_result == "NO_TOKEN":

                        st.warning(
                            "FCM token generate avvaledu."
                        )

                    elif (
                        isinstance(
                            token_result,
                            str
                        )
                        and token_result.startswith(
                            "ERROR:"
                        )
                    ):

                        st.error(
                            "❌ Push setup failed."
                        )

                        st.caption(
                            token_result
                        )

            else:

                st.success(
                    "🟢 Push notifications are enabled."
                )


    # ========================================================
    # EMAIL
    # ========================================================

    with alert2:

        with st.container(border=True):

            st.markdown(
                "### 📧 Gmail Safety Alerts"
            )

            st.write(
                "Every detected violation can generate "
                "an automatic Gmail safety alert."
            )

            if (
                "EMAIL_SENDER" in st.secrets
                and
                "EMAIL_PASSWORD" in st.secrets
                and
                "EMAIL_RECEIVER" in st.secrets
            ):

                st.success(
                    "🟢 Gmail is configured."
                )

            else:

                st.warning(
                    "🟡 Gmail is not configured yet."
                )

                st.caption(
                    "Add EMAIL_SENDER, EMAIL_PASSWORD "
                    "and EMAIL_RECEIVER in Streamlit Secrets."
                )


    st.write("")


    # ========================================================
    # LOCATION
    # ========================================================

    with st.container(border=True):

        st.markdown(
            "### 📍 Safety Location"
        )

        st.write(
            "Allow location access if you want the "
            "safety email to contain the current "
            "Google Maps location."
        )

        if st.button(
            "📍 Enable Live Location",
            key="location_button"
        ):

            location_result = (
                get_live_location()
            )

            if (
                isinstance(
                    location_result,
                    str
                )
                and location_result.startswith(
                    "{"
                )
            ):

                try:

                    location_data = json.loads(
                        location_result
                    )

                    st.session_state.location = (
                        location_data
                    )

                    st.success(
                        "📍 Location successfully enabled."
                    )

                except Exception:

                    st.warning(
                        "Location data could not be read."
                    )

            else:

                st.warning(
                    "Location permission was not available."
                )

        if st.session_state.location:

            latitude = st.session_state.location[
                "latitude"
            ]

            longitude = st.session_state.location[
                "longitude"
            ]

            maps_link = (
                "https://www.google.com/maps/"
                "search/?api=1&query="
                f"{latitude},{longitude}"
            )

            st.success(
                f"📍 {latitude:.6f}, "
                f"{longitude:.6f}"
            )

            st.markdown(
                f"[🗺️ Open Location in Google Maps]"
                f"({maps_link})"
            )


    st.write("")


    # ========================================================
    # BACK
    # ========================================================

    if st.button(
        "← Back to Home",
        key="prediction_back_button"
    ):

        st.session_state.page = "home"

        st.rerun()


    st.write("")


    # ========================================================
    # INPUT SELECTOR
    # ========================================================

    input_type = st.radio(
        "Choose Detection Mode",
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

            with st.container(border=True):

                st.markdown(
                    "### 📸 Upload Construction Image"
                )

                uploaded_image = (
                    st.file_uploader(
                        "Choose image",
                        type=[
                            "jpg",
                            "jpeg",
                            "png"
                        ],
                        key="image_upload"
                    )
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
                        key="image_detect_button"
                    ):

                        with st.spinner(
                            "🔎 AI is analysing the image..."
                        ):

                            result = model.predict(
                                np.array(image),
                                conf=GENERAL_CONFIDENCE,
                                verbose=False
                            )[0]

                        frame = np.array(
                            image
                        )

                        detections = (
                            extract_detections(
                                result
                            )
                        )

                        annotated = (
                            draw_custom_boxes(
                                frame,
                                detections
                            )
                        )

                        st.session_state.result_image = (
                            annotated
                        )

                        st.session_state.detections = (
                            detections
                        )

                        violations = (
                            get_violations(
                                detections
                            )
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

            with st.container(border=True):

                st.markdown(
                    "### 🎯 Detection Result"
                )

                if (
                    st.session_state.result_image
                    is None
                ):

                    st.info(
                        "Upload an image and click "
                        "Detect PPE."
                    )

                else:

                    st.image(
                        st.session_state.result_image,
                        caption="GuardX-AI AI Result",
                        use_container_width=True
                    )

                    violations = (
                        st.session_state.image_violations
                    )

                    if violations:

                        display_violation_box(
                            violations
                        )

                    else:

                        display_safe_box()

                    detections = (
                        st.session_state.detections
                    )

                    if detections:

                        st.markdown(
                            "### 🔎 Detected Objects"
                        )

                        for detection in detections:

                            name = detection[
                                "name"
                            ]

                            confidence = detection[
                                "confidence"
                            ]

                            violation = detection[
                                "violation"
                            ]

                            if violation:

                                st.markdown(
                                    f"""
                                    🔴 **{name}**
                                    — {confidence * 100:.1f}%
                                    """
                                )

                            else:

                                st.markdown(
                                    f"""
                                    🟢 **{name}**
                                    — {confidence * 100:.1f}%
                                    """
                                )


    # ========================================================
    # CAMERA
    # ========================================================

    elif input_type == "📷 Camera":

        with st.container(border=True):

            st.markdown(
                "### 📷 Live Camera Capture"
            )

            camera_image = st.camera_input(
                "Take a photo",
                key="camera_input"
            )

            if camera_image:

                image = Image.open(
                    camera_image
                ).convert("RGB")

                if st.button(
                    "🔍 Analyse Camera Image",
                    key="camera_detect_button"
                ):

                    with st.spinner(
                        "Analysing..."
                    ):

                        result = model.predict(
                            np.array(image),
                            conf=GENERAL_CONFIDENCE,
                            verbose=False
                        )[0]

                    frame = np.array(
                        image
                    )

                    detections = (
                        extract_detections(
                            result
                        )
                    )

                    annotated = (
                        draw_custom_boxes(
                            frame,
                            detections
                        )
                    )

                    st.image(
                        annotated,
                        caption="GuardX-AI Camera Result",
                        use_container_width=True
                    )

                    violations = (
                        get_violations(
                            detections
                        )
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

                        st.markdown(
                            "### 🔎 Detected Objects"
                        )

                        for detection in detections:

                            name = detection[
                                "name"
                            ]

                            confidence = detection[
                                "confidence"
                            ]

                            if detection[
                                "violation"
                            ]:

                                st.markdown(
                                    f"🔴 **{name}** "
                                    f"— {confidence * 100:.1f}%"
                                )

                            else:

                                st.markdown(
                                    f"🟢 **{name}** "
                                    f"— {confidence * 100:.1f}%"
                                )


    # ========================================================
    # VIDEO
    # ========================================================

    elif input_type == "🎥 Video":

        with st.container(border=True):

            st.markdown(
                "### 🎥 Construction Video Analysis"
            )

            uploaded_video = (
                st.file_uploader(
                    "Upload video",
                    type=[
                        "mp4",
                        "avi",
                        "mov",
                        "mkv"
                    ],
                    key="video_upload"
                )
            )

            if uploaded_video:

                st.video(
                    uploaded_video
                )

                if st.button(
                    "🎥 Analyse Entire Video",
                    key="video_detect_button"
                ):

                    with st.spinner(
                        "🎬 Processing video... Please wait."
                    ):

                        input_file = (
                            tempfile.NamedTemporaryFile(
                                delete=False,
                                suffix=".mp4"
                            )
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

                        output_file = (
                            tempfile.NamedTemporaryFile(
                                delete=False,
                                suffix=".mp4"
                            )
                        )

                        output_path = (
                            output_file.name
                        )

                        output_file.close()

                        fourcc = (
                            cv2.VideoWriter_fourcc(
                                *"mp4v"
                            )
                        )

                        writer = cv2.VideoWriter(
                            output_path,
                            fourcc,
                            fps,
                            (
                                width,
                                height
                            )
                        )

                        video_violations = set()

                        while True:

                            ret, frame = (
                                cap.read()
                            )

                            if not ret:
                                break

                            result = model.predict(
                                frame,
                                conf=GENERAL_CONFIDENCE,
                                verbose=False
                            )[0]

                            detections = (
                                extract_detections(
                                    result
                                )
                            )

                            annotated = (
                                draw_custom_boxes(
                                    frame,
                                    detections
                                )
                            )

                            writer.write(
                                annotated
                            )

                            frame_violations = (
                                get_violations(
                                    detections
                                )
                            )

                            for (
                                name,
                                confidence
                            ) in frame_violations:

                                video_violations.add(
                                    name
                                )

                        cap.release()

                        writer.release()

                        try:

                            os.remove(
                                input_file.name
                            )

                        except Exception:

                            pass

                    st.success(
                        "✅ Video processing completed!"
                    )

                    st.video(
                        output_path
                    )

                    if video_violations:

                        video_violation_list = [

                            (
                                name,
                                1.0
                            )

                            for name
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


    # ========================================================
    # FOOTER
    # ========================================================

    render_html(
        """
        <div class="footer">
            🦺 GuardX-AI • AI-Powered Construction Safety
        </div>
        """
    )
