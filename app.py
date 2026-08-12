import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os


st.set_page_config(
    page_title="GuardX-AI",
    page_icon="🦺",
    layout="wide"
)


# Session State
if "page" not in st.session_state:
    st.session_state.page = "home"


# ============================================================
# PAGE 1
# ============================================================

if st.session_state.page == "home":

    st.markdown(
        """
        <style>

        .stApp {
            background-color: #f5f6fa;
        }

        .block-container {
            max-width: 1400px;
            padding-top: 20px;
            padding-left: 7%;
            padding-right: 7%;
        }

        .main-title {
            text-align: center;
            color: #172b55;
            font-size: 31px;
            font-weight: 800;
            margin-bottom: 35px;
        }

        .aicw {
            color: #172b55;
            font-size: 25px;
            font-weight: 800;
            line-height: 1.6;
        }

        .capstone {
            color: #303c52;
            font-size: 22px;
            font-weight: 700;
            margin-top: 40px;
        }

        .description-title {
            color: #172b55;
            font-size: 23px;
            font-weight: 800;
            margin-bottom: 12px;
        }

        .description {
            color: #626b78;
            font-size: 15px;
            line-height: 1.65;
        }

        .card {
            background: white;
            border: 1px solid #e0e3e8;
            border-radius: 17px;
            padding: 20px;
            min-height: 235px;
        }

        .card-title {
            color: #303743;
            font-size: 14px;
            font-weight: 800;
            margin-bottom: 20px;
        }

        .card-text {
            color: #4d5562;
            font-size: 14px;
            line-height: 2.5;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


    # TITLE

    st.markdown(
        """
        <div class="main-title">
            🦺 GuardX-AI – Construction PPE Detection System
        </div>
        """,
        unsafe_allow_html=True
    )


    # TOP SECTION

    left, right = st.columns(
        [0.36, 0.64],
        gap="large"
    )


    # LEFT SIDE

    with left:

        st.markdown(
            """
            <div class="aicw">
                AI Career for Women
                <br>
                (AICW)
            </div>

            <div class="capstone">
                Capstone Project
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        if st.button(
            "🔍 PREDICT",
            use_container_width=True
        ):

            st.session_state.page = "predict"
            st.rerun()


    # RIGHT SIDE

    with right:

        st.markdown(
            """
            <div class="description-title">
                Project Description
            </div>

            <div class="description">

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
            """,
            unsafe_allow_html=True
        )


    st.write("")
    st.write("")


    # BOTTOM CARDS

    team, gmail, guide = st.columns(
        [1.25, 1.25, 0.75],
        gap="large"
    )


    # TEAM MEMBERS

    with team:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    TEAM MEMBERS
                </div>

                <div class="card-text">

                    1. Y.D.V.Sivani
                    <br>
                    2. V.L.S.Asritha
                    <br>
                    3. R.Likhitha
                    <br>
                    4. S.Poojitha sri

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # GMAIL

    with gmail:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    GMAIL
                </div>

                <div class="card-text">

                    yallashivani@gmail.com
                    <br>
                    Asrithavantipalli@gmail.com
                    <br>
                    likhitharayudu@gmail.com
                    <br>
                    pujithasari@gmail.com

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # GUIDE

    with guide:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    GUIDE NAME
                </div>

                <div class="card-text">

                    MD.Abdul Aziz

                    <br><br>

                    <b>Designation</b>

                    <br><br>

                    Trainer, Co-Lead-AICW

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )
