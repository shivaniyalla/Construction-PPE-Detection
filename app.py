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

# ============================================================
# OPTIONAL FIREBASE
# ============================================================

try:
    import firebase_admin
    from firebase_admin import credentials, messaging
    FIREBASE_AVAILABLE = True
except Exception:
    FIREBASE_AVAILABLE = False

# ============================================================
# BROWSER JS
# ============================================================

try:
    from streamlit_js_eval import streamlit_js_eval
    JS_AVAILABLE = True
except Exception:
    JS_AVAILABLE = False


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
    "firebase_error": None,
    "result_image": None,
    "detections": [],
    "image_violations": [],
    "camera_result": None,
    "camera_detections": [],
    "camera_violations": [],
    "video_result": None,
    "video_violations": []
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# DETECTION SETTINGS
# ============================================================

GENERAL_CONFIDENCE = 0.30

# NO-Safety Vest is more sensitive in your dataset,
# so we use a higher confidence threshold.
NO_VEST_CONFIDENCE = 0.75


# ============================================================
# FIREBASE ADMIN INITIALIZATION
# ============================================================

def initialize_firebase():

    if not FIREBASE_AVAILABLE:
        return False

    try:

        if firebase_admin._apps:
            return True

        if "firebase" not in st.secrets:
            st.session_state.firebase_error = (
                "Firebase credentials are missing in Streamlit secrets."
            )
            return False

        firebase_config = dict(st.secrets["firebase"])

        cred = credentials.Certificate(firebase_config)

        firebase_admin.initialize_app(cred)

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
# GLOBAL DARK UI
# ============================================================

st.markdown(
    """
<style>

/* =========================================================
   GLOBAL
========================================================= */

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(37, 99, 235, 0.20),
            transparent 28%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(124, 58, 237, 0.18),
            transparent 30%
        ),
        radial-gradient(
            circle at 50% 100%,
            rgba(6, 182, 212, 0.10),
            transparent 30%
        ),
        #070b16 !important;

    color: #f8fafc !important;
}

.block-container {
    max-width: 1450px;
    padding-top: 35px;
    padding-bottom: 40px;
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
   ALL TEXT
========================================================= */

.stMarkdown,
.stMarkdown p,
.stMarkdown li,
.stText,
label,
p,
span {
    color: #e5e7eb;
}


/* =========================================================
   MAIN TITLE
========================================================= */

.main-title {
    text-align: center;
    font-size: 38px !important;
    font-weight: 900 !important;

    background:
        linear-gradient(
            90deg,
            #60a5fa,
            #a78bfa,
            #22d3ee,
            #60a5fa
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    background-size: 300% auto;

    animation: gradientMove 5s linear infinite;

    margin-bottom: 35px;
}

@keyframes gradientMove {

    0% {
        background-position: 0% center;
    }

    50% {
        background-position: 100% center;
    }

    100% {
        background-position: 0% center;
    }

}


/* =========================================================
   HOME LEFT
========================================================= */

.aicw-text {
    color: #93c5fd !important;
    font-size: 28px !important;
    font-weight: 900 !important;
    line-height: 1.5;
}

.capstone-text {
    color: #c4b5fd !important;
    font-size: 23px !important;
    font-weight: 800 !important;
    margin-top: 40px;
}


/* =========================================================
   DESCRIPTION
========================================================= */

.description-title {
    color: #67e8f9 !important;
    font-size: 25px !important;
    font-weight: 900 !important;
    margin-bottom: 15px;
}


/* =========================================================
   STREAMLIT CONTAINERS / CARDS
========================================================= */

div[data-testid="stVerticalBlockBorderWrapper"] {

    background:
        linear-gradient(
            145deg,
            rgba(20, 29, 52, 0.94),
            rgba(11, 17, 32, 0.94)
        ) !important;

    border: 1px solid rgba(96, 165, 250, 0.20) !important;

    border-radius: 20px !important;

    box-shadow:
        0 10px 40px rgba(0, 0, 0, 0.35),
        inset 0 1px 0 rgba(255,255,255,0.04);

    padding: 16px !important;

    backdrop-filter: blur(18px);

}


/* =========================================================
   CARD TEXT
========================================================= */

.card-heading {
    color: #93c5fd !important;
    font-size: 16px !important;
    font-weight: 900 !important;
    letter-spacing: 0.5px;
    margin-bottom: 15px;
}

.card-text {
    color: #cbd5e1 !important;
    font-size: 14px !important;
    line-height: 2 !important;
}


/* =========================================================
   BUTTONS
========================================================= */

div.stButton > button {

    width: 100%;

    min-height: 48px;

    border-radius: 12px !important;

    border: 1px solid rgba(96,165,250,0.30) !important;

    background:
        linear-gradient(
            135deg,
            rgba(37,99,235,0.22),
            rgba(124,58,237,0.22)
        ) !important;

    color: #f8fafc !important;

    font-size: 14px !important;

    font-weight: 800 !important;

    transition:
        all 0.25s ease !important;

    box-shadow:
        0 5px 20px rgba(37,99,235,0.10);

}

div.stButton > button:hover {

    transform: translateY(-2px);

    border-color: #60a5fa !important;

    background:
        linear-gradient(
            135deg,
            rgba(37,99,235,0.40),
            rgba(124,58,237,0.40)
        ) !important;

    box-shadow:
        0 8px 30px rgba(59,130,246,0.25);

}


/* =========================================================
   RADIO
========================================================= */

div[data-testid="stRadio"] label {

    color: #cbd5e1 !important;

    font-weight: 700 !important;

}


/* =========================================================
   FILE UPLOADER
========================================================= */

div[data-testid="stFileUploader"] {

    background:
        rgba(15,23,42,0.75) !important;

    border:
        1px solid rgba(96,165,250,0.18) !important;

    border-radius: 15px !important;

    padding: 8px;

}

div[data-testid="stFileUploader"] section {

    background:
        rgba(15,23,42,0.50) !important;

    border-radius: 12px !important;

}


/* =========================================================
   INPUT LABELS
========================================================= */

label {
    color: #cbd5e1 !important;
}


/* =========================================================
   TITLES
========================================================= */

.detect-title {

    text-align: center;

    font-size: 40px;

    font-weight: 900;

    background:
        linear-gradient(
            90deg,
            #60a5fa,
            #a78bfa,
            #22d3ee
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;

    margin-bottom: 4px;

}

.detect-subtitle {

    color: #94a3b8 !important;

    text-align: center;

    font-size: 15px;

    margin-bottom: 30px;

}


/* =========================================================
   SAFE BOX
========================================================= */

.safe-box {

    background:
        linear-gradient(
            135deg,
            rgba(16,185,129,0.18),
            rgba(6,78,59,0.22)
        );

    border:
        1px solid rgba(52,211,153,0.45);

    border-radius: 16px;

    padding: 18px;

    color: #6ee7b7 !important;

    font-weight: 800;

    margin-top: 18px;

    box-shadow:
        0 8px 30px rgba(16,185,129,0.10);

}


/* =========================================================
   VIOLATION BOX
========================================================= */

.violation-box {

    background:
        linear-gradient(
            135deg,
            rgba(239,68,68,0.18),
            rgba(127,29,29,0.22)
        );

    border:
        1px solid rgba(248,113,113,0.55);

    border-radius: 16px;

    padding: 20px;

    margin-top: 18px;

    box-shadow:
        0 8px 35px rgba(239,68,68,0.13);

    animation: alertPulse 2s ease-in-out infinite;

}

@keyframes alertPulse {

    0% {
        box-shadow:
            0 8px 35px rgba(239,68,68,0.10);
    }

    50% {
        box-shadow:
            0 8px 45px rgba(239,68,68,0.28);
    }

    100% {
        box-shadow:
            0 8px 35px rgba(239,68,68,0.10);
    }

}

.violation-title {

    font-size: 19px;

    font-weight: 900;

    color: #fca5a5 !important;

    margin-bottom: 12px;

}

.violation-text {

    font-size: 15px;

    font-weight: 700;

    color: #fecaca !important;

}


/* =========================================================
   DETECTION OBJECT CARD
========================================================= */

.object-card {

    background:
        linear-gradient(
            135deg,
            rgba(30,41,59,0.75),
            rgba(15,23,42,0.75)
        );

    border:
        1px solid rgba(148,163,184,0.14);

    border-radius: 13px;

    padding: 12px 15px;

    margin: 7px 0;

    display: flex;

    justify-content: space-between;

    align-items: center;

}

.object-name {

    color: #e2e8f0 !important;

    font-weight: 800;

}

.object-confidence {

    color: #67e8f9 !important;

    font-weight: 900;

}


/* =========================================================
   MODEL CARD
========================================================= */

.model-header {

    color: #a78bfa !important;

    font-size: 17px;

    font-weight: 900;

}

.class-pill {

    display: inline-block;

    background:
        rgba(96,165,250,0.12);

    border:
        1px solid rgba(96,165,250,0.20);

    border-radius: 999px;

    padding: 7px 12px;

    margin: 4px;

    color: #bfdbfe !important;

    font-size: 13px;

    font-weight: 700;

}


/* =========================================================
   ALERT STATUS
========================================================= */

.alert-status {

    border-radius: 12px;

    padding: 12px 15px;

    margin-top: 10px;

    background:
        rgba(59,130,246,0.10);

    border:
        1px solid rgba(59,130,246,0.20);

    color: #bfdbfe !important;

    font-weight: 700;

}


/* =========================================================
   FOOTER
========================================================= */

.footer-text {

    color: #64748b !important;

    text-align: center;

    font-size: 13px;

    margin-top: 35px;

}


/* =========================================================
   STREAMLIT ALERT TEXT
========================================================= */

div[data-testid="stAlert"] {

    border-radius: 12px !important;

}


/* =========================================================
   IMAGE
========================================================= */

img {

    border-radius: 14px !important;

}


/* =========================================================
   MOBILE
========================================================= */

@media(max-width: 900px) {

    .block-container {

        padding-left: 4%;

        padding-right: 4%;

    }

    .main-title {

        font-size: 27px !important;

    }

    .detect-title {

        font-size: 30px;

    }

    .aicw-text {

        font-size: 23px !important;

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
        os.path.dirname(os.path.abspath(__file__)),
        "best.pt"
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"best.pt not found at: {model_path}"
        )

    return YOLO(model_path)


# ============================================================
# GET LIVE LOCATION
# ============================================================

def get_live_location():

    if not JS_AVAILABLE:
        return None

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

    try:

        return streamlit_js_eval(
            js_expressions=js_code,
            want_output=True,
            key="location_" + str(datetime.now().timestamp())
        )

    except Exception:
        return None


# ============================================================
# GET FCM TOKEN
# ============================================================

def get_fcm_token():

    if not JS_AVAILABLE:
        return "ERROR: streamlit-js-eval is not installed."

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

    try:

        return streamlit_js_eval(
            js_expressions=js_code,
            want_output=True,
            key="fcm_token_generator"
        )

    except Exception as e:

        return "ERROR:" + str(e)


# ============================================================
# SEND PUSH NOTIFICATION
# ============================================================

def send_push_notification(token, title, body):

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

            token=token
        )

        messaging.send(message)

        return True

    except Exception as e:

        st.session_state.push_error = str(e)

        return False


# ============================================================
# SEND GMAIL ALERT
# ============================================================

def send_email_alert(
    violations,
    source="image",
    location_data=None
):

    try:

        # ----------------------------------------------------
        # CHECK EMAIL SECRETS
        # ----------------------------------------------------

        required_keys = [
            "EMAIL_SENDER",
            "EMAIL_PASSWORD",
            "EMAIL_RECEIVER"
        ]

        missing_keys = [
            key
            for key in required_keys
            if key not in st.secrets
        ]

        if missing_keys:

            raise Exception(
                "Missing Streamlit secrets: "
                + ", ".join(missing_keys)
            )

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
                    str(name)
                    for name, confidence
                    in violations
                )
            )
        )

        # ----------------------------------------------------
        # LOCATION
        # ----------------------------------------------------

        if location_data:

            latitude = location_data.get(
                "latitude"
            )

            longitude = location_data.get(
                "longitude"
            )

            maps_link = (
                "https://www.google.com/maps/"
                "?api=1&query="
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
                f"Location: {fixed_location}\n"
                f"Google Maps: Location unavailable"
            )

        # ----------------------------------------------------
        # SUBJECT
        # ----------------------------------------------------

        subject = (
            f"🚨 GuardX-AI Safety Alert | "
            f"{violation_text}"
        )

        # ----------------------------------------------------
        # BODY
        # ----------------------------------------------------

        body = f"""
GUARDX-AI
AI-POWERED CONSTRUCTION PPE DETECTION SYSTEM
================================================

🚨 PPE SAFETY VIOLATION DETECTED

GuardX-AI has detected a potential PPE safety
violation.

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
        # EMAIL MESSAGE
        # ----------------------------------------------------

        message = EmailMessage()

        message["From"] = sender
        message["To"] = receiver
        message["Subject"] = subject

        message.set_content(body)

        # ----------------------------------------------------
        # GMAIL SMTP
        # ----------------------------------------------------

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

        try:

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

        except Exception:
            continue

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
# PPE VIOLATIONS
# ============================================================

def get_violations(detections):

    violations = []

    for name, confidence in detections:

        normalized_name = normalize_class_name(
            name
        )

        # ----------------------------------------------------
        # NO HARDHAT
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # NO MASK
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # NO SAFETY VEST
        # ----------------------------------------------------

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

        clean_name = str(
            name
        ).strip()

        if clean_name not in violation_names:

            violation_names.append(
                clean_name
            )

    violation_text = ", ".join(
        violation_names
    )

    # IMPORTANT:
    # HTML starts from column 0.
    # This prevents Streamlit from showing HTML as CODE.
    html = f"""
<div class="violation-box">
<div class="violation-title">
🚨 PPE VIOLATION DETECTED
</div>
<div class="violation-text">
Violations: <strong>{violation_text}</strong>
</div>
</div>
"""

    st.markdown(
        html,
        unsafe_allow_html=True
    )


# ============================================================
# DISPLAY SAFE BOX
# ============================================================

def display_safe_box():

    html = """
<div class="safe-box">
🟢 SAFE — No PPE violation detected.
</div>
"""

    st.markdown(
        html,
        unsafe_allow_html=True
    )


# ============================================================
# DISPLAY DETECTIONS
# ============================================================

def display_detections(detections):

    if not detections:

        st.info(
            "No objects detected."
        )

        return

    st.markdown(
        "### 🔎 Detected Objects"
    )

    for name, confidence in detections:

        normalized = normalize_class_name(
            name
        )

        # Hide low-confidence NO-Vest
        if (
            normalized in {
                "no-safety-vest",
                "no safety vest"
            }
            and
            confidence < NO_VEST_CONFIDENCE
        ):
            continue

        html = f"""
<div class="object-card">
<span class="object-name">
{name}
</span>
<span class="object-confidence">
{confidence * 100:.1f}%
</span>
</div>
"""

        st.markdown(
            html,
            unsafe_allow_html=True
        )


# ============================================================
# HANDLE ALERTS
# ============================================================

def handle_violation_alert(
    violations,
    source="image"
):

    if not violations:
        return

    # --------------------------------------------------------
    # GET LOCATION
    # --------------------------------------------------------

    location_data = st.session_state.get(
        "location"
    )

    # If location is already available,
    # don't ask again.
    if not location_data:

        try:

            location_result = get_live_location()

            if (
                location_result
                and isinstance(
                    location_result,
                    str
                )
                and location_result.startswith("{")
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
                str(name)
                for name, confidence
                in violations
            )
        )
    )

    # --------------------------------------------------------
    # PUSH
    # --------------------------------------------------------

    push_sent = False

    if st.session_state.fcm_token:

        push_sent = send_push_notification(

            st.session_state.fcm_token,

            "🚨 GuardX-AI Safety Alert",

            f"PPE violation detected: "
            f"{violation_text}"
        )

    # --------------------------------------------------------
    # EMAIL
    # --------------------------------------------------------

    email_sent = send_email_alert(
        violations,
        source,
        location_data
    )

    # --------------------------------------------------------
    # ALERT STATUS UI
    # --------------------------------------------------------

    if push_sent:

        st.success(
            "📲 Safety push notification sent!"
        )

    elif st.session_state.push_enabled:

        st.warning(
            "⚠️ Violation detected, but push notification failed."
        )

        if st.session_state.push_error:

            st.caption(
                "Push error: "
                + str(
                    st.session_state.push_error
                )
            )

    if email_sent:

        st.success(
            "📧 Gmail safety alert sent successfully!"
        )

    else:

        st.error(
            "❌ Violation detected, but Gmail alert could not be sent."
        )

        if st.session_state.email_error:

            st.caption(
                "Email error: "
                + str(
                    st.session_state.email_error
                )
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

    # --------------------------------------------------------
    # LEFT
    # --------------------------------------------------------

    with left_col:

        st.markdown(
            """
<div class="aicw-text">
AI Career for Women<br>
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

    # --------------------------------------------------------
    # RIGHT
    # --------------------------------------------------------

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
                "is essential for worker safety. Manual monitoring "
                "of PPE continuously is difficult and time-consuming."
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

    # --------------------------------------------------------
    # BOTTOM CARDS
    # --------------------------------------------------------

    team_col, gmail_col, guide_col = st.columns(
        [1.25, 1.25, 0.75],
        gap="large"
    )

    # TEAM

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

    # GMAIL

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

    # GUIDE

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
GuardX-AI – AI-Powered Construction PPE Safety Monitoring
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
AI-Powered Construction PPE Detection & Safety Monitoring
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
    # MODEL CARD
    # ========================================================

    with st.container(border=True):

        st.markdown(
            """
<div class="model-header">
🤖 GuardX-AI Detection Model
</div>
""",
            unsafe_allow_html=True
        )

        st.write(
            "Detected PPE categories"
        )

        classes = [
            ("🪖", "Hardhat"),
            ("🚨", "NO-Hardhat"),
            ("😷", "Mask"),
            ("🚨", "NO-Mask"),
            ("🦺", "Safety Vest"),
            ("🚨", "NO-Safety Vest")
        ]

        pills = ""

        for icon, name in classes:

            pills += (
                f'<span class="class-pill">'
                f'{icon} {name}'
                f'</span>'
            )

        st.markdown(
            pills,
            unsafe_allow_html=True
        )

        st.caption(
            f"General confidence: {GENERAL_CONFIDENCE:.2f} "
            f"| NO-Safety Vest: {NO_VEST_CONFIDENCE:.2f}"
        )

    st.write("")

    # ========================================================
    # PUSH NOTIFICATIONS
    # ========================================================

    with st.container(border=True):

        st.markdown(
            """
<div class="model-header">
🔔 Safety Push Notifications
</div>
""",
            unsafe_allow_html=True
        )

        st.write(
            "Enable browser notifications to receive an "
            "instant safety alert when a PPE violation is detected."
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

                            st.session_state.push_enabled = True

                            st.success(
                                "🔔 Push notifications enabled!"
                            )

                            st.rerun()

                else:

                    st.warning(
                        "Waiting for notification permission..."
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
            """
<div class="model-header">
📍 Live Safety Location
</div>
""",
            unsafe_allow_html=True
        )

        st.write(
            "Allow location access so the Gmail safety alert "
            "can include the current construction-site location."
        )

        if st.button(
            "📍 Allow Location Access",
            key="location_button",
            use_container_width=True
        ):

            location_result = get_live_location()

            if location_result:

                if (
                    isinstance(
                        location_result,
                        str
                    )
                    and
                    location_result.startswith("{")
                ):

                    try:

                        location_data = json.loads(
                            location_result
                        )

                        st.session_state.location = (
                            location_data
                        )

                        st.success(
                            "📍 Live location captured successfully."
                        )

                    except Exception:

                        st.warning(
                            "Location data could not be read."
                        )

                elif location_result == (
                    "LOCATION_NOT_SUPPORTED"
                ):

                    st.warning(
                        "This browser does not support location."
                    )

                elif str(
                    location_result
                ).startswith(
                    "LOCATION_ERROR:"
                ):

                    st.warning(
                        "Location permission was denied "
                        "or location could not be obtained."
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
                "?api=1&query="
                f"{latitude},{longitude}"
            )

            st.success(
                f"📍 Location ready: "
                f"{latitude:.6f}, {longitude:.6f}"
            )

            st.markdown(
                f"[🗺️ Open Current Location in Google Maps]({maps_link})"
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

        # ----------------------------------------------------
        # INPUT
        # ----------------------------------------------------

        with input_col:

            st.markdown(
                "### 📤 Upload Image"
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
                        "🤖 GuardX-AI is analyzing the image..."
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

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        with result_col:

            st.markdown(
                "### 🎯 Detection Result"
            )

            if (
                st.session_state.result_image
                is None
            ):

                st.info(
                    "Upload an image and click "
                    "**Detect PPE** to see the result."
                )

            else:

                st.image(
                    st.session_state.result_image,
                    caption="GuardX-AI Detection Result",
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

                display_detections(
                    detections
                )


    # ========================================================
    # CAMERA
    # ========================================================

    elif input_type == "📷 Camera":

        st.markdown(
            "### 📷 Camera PPE Detection"
        )

        camera_image = st.camera_input(
            "Take a photo of the construction site",
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
                    "🤖 Analyzing camera image..."
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

                st.session_state.camera_result = (
                    annotated
                )

                st.session_state.camera_detections = (
                    detections
                )

                st.session_state.camera_violations = (
                    violations
                )

                st.image(
                    annotated,
                    caption="GuardX-AI Camera Result",
                    use_container_width=True
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

                display_detections(
                    detections
                )


    # ========================================================
    # VIDEO
    # ========================================================

    elif input_type == "🎥 Video":

        st.markdown(
            "### 🎥 Video PPE Detection"
        )

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
                    "🤖 GuardX-AI is processing the video..."
                ):

                    input_path = None
                    output_path = None

                    try:

                        # ------------------------------------------------
                        # INPUT VIDEO
                        # ------------------------------------------------

                        input_file = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
                        )

                        input_file.write(
                            uploaded_video.getbuffer()
                        )

                        input_file.close()

                        input_path = input_file.name

                        # ------------------------------------------------
                        # OPEN VIDEO
                        # ------------------------------------------------

                        cap = cv2.VideoCapture(
                            input_path
                        )

                        if not cap.isOpened():

                            raise Exception(
                                "Unable to open uploaded video."
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

                        # ------------------------------------------------
                        # OUTPUT
                        # ------------------------------------------------

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

                        if not writer.isOpened():

                            cap.release()

                            raise Exception(
                                "Unable to create output video."
                            )

                        # ------------------------------------------------
                        # VIOLATIONS
                        # ------------------------------------------------

                        video_violations = set()

                        # ------------------------------------------------
                        # FRAME LOOP
                        # ------------------------------------------------

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

                        # ------------------------------------------------
                        # STORE RESULT
                        # ------------------------------------------------

                        st.session_state.video_result = (
                            output_path
                        )

                        st.session_state.video_violations = (
                            sorted(
                                video_violations
                            )
                        )

                    except Exception as e:

                        st.error(
                            "❌ Video processing failed."
                        )

                        st.caption(
                            str(e)
                        )

                        try:
                            cap.release()
                        except:
                            pass

                        try:
                            writer.release()
                        except:
                            pass

                    finally:

                        if input_path:

                            try:
                                os.remove(
                                    input_path
                                )
                            except:
                                pass

                # ----------------------------------------------------
                # RESULT
                # ----------------------------------------------------

                if st.session_state.video_result:

                    st.success(
                        "✅ Video processing completed successfully."
                    )

                    st.video(
                        st.session_state.video_result
                    )

                    video_violations = (
                        st.session_state.video_violations
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


# ============================================================
# END
# ============================================================
