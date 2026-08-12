import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import subprocess
import os

st.set_page_config(
    page_title="GuardX-AI",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "home"


# ------------------------------------------------------------
# CSS — FRIEND PROJECT STYLE
# ------------------------------------------------------------

st.markdown("""
<style>

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
    max-width: 1400px !important;
    padding-top: 0.5rem !important;
    padding-bottom: 0.5rem !important;
    padding-left: 7% !important;
    padding-right: 7% !important;
}


/* ============================================
   MAIN BACKGROUND
   ============================================ */

.stApp {
    background: #f5f6fa;
}


/* ============================================
   TOP TITLE
   ============================================ */

.main-title {
    text-align: center;

    font-size: 31px;
    font-weight: 800;

    color: #182b55;

    margin-top: -8px;
    margin-bottom: 28px;
}


/* ============================================
   TOP CONTENT
   ============================================ */

.top-section {
    display: grid;

    grid-template-columns: 36% 64%;

    column-gap: 35px;

    margin-bottom: 28px;
}


/* ============================================
   LEFT CONTENT
   ============================================ */

.left-content {
    padding-top: 5px;
}

.aicw {
    font-size: 25px;

    font-weight: 800;

    line-height: 1.55;

    color: #182b55;
}

.capstone {
    font-size: 22px;

    font-weight: 700;

    color: #26354f;

    margin-top: 42px;
}


/* ============================================
   PREDICT BUTTON
   ============================================ */

.predict-button {
    margin-top: 36px;
}

.predict-button button {
    height: 44px !important;

    border-radius: 8px !important;

    border: 1px solid #d7dce5 !important;

    background: white !important;

    color: #364152 !important;

    font-size: 14px !important;

    font-weight: 500 !important;
}

.predict-button button:hover {
    border-color: #b9c2d0 !important;
}


/* ============================================
   DESCRIPTION
   ============================================ */

.description-title {
    font-size: 23px;

    font-weight: 800;

    color: #182b55;

    margin-bottom: 10px;
}

.description-text {
    font-size: 15px;

    line-height: 1.65;

    color: #5f6673;

    max-width: 850px;
}


/* ============================================
   BOTTOM CARDS
   ============================================ */

.cards {
    display: grid;

    grid-template-columns: 1.25fr 1.25fr 0.75fr;

    gap: 34px;

    margin-top: 42px;
}


/* ============================================
   CARD
   ============================================ */

.card {
    background: white;

    border: 1px solid #e0e3e8;

    border-radius: 17px;

    padding: 20px 16px;

    min-height: 235px;

    box-sizing: border-box;

    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}


/* ============================================
   CARD HEADING
   ============================================ */

.card-heading {
    font-size: 14px;

    font-weight: 800;

    color: #303743;

    margin-bottom: 28px;
}


/* ============================================
   CARD TEXT
   ============================================ */

.card-text {
    font-size: 14px;

    color: #4d5562;

    line-height: 2.9;
}


/* ============================================
   GUIDE
   ============================================ */

.guide-name {
    font-size: 15px;

    color: #4d5562;

    margin-bottom: 32px;
}

.designation-title {
    font-size: 14px;

    font-weight: 800;

    color: #303743;

    margin-bottom: 27px;
}

.designation {
    font-size: 14px;

    color: #4d5562;

    line-height: 1.8;
}


/* ============================================
   FOOTER PROJECT NAME
   ============================================ */

.project-footer {
    text-align: center;

    font-size: 14px;

    color: #737b87;

    margin-top: 36px;

    margin-bottom: 5px;
}


/* ============================================
   MOBILE
   ============================================ */

@media (max-width: 900px) {

    .block-container {
        padding-left: 5% !important;
        padding-right: 5% !important;
    }

    .top-section {
        grid-template-columns: 1fr;
    }

    .cards {
        grid-template-columns: 1fr;
    }

}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PAGE 1 — HOME
# ============================================================

if st.session_state.page == "home":

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="main-title">
            🦺 GuardX-AI – Construction PPE Detection System
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # TOP SECTION
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="top-section">

            <div class="left-content">

                <div class="aicw">
                    AI Career for Women
                    <br>
                    (AICW)
                </div>

                <div class="capstone">
                    Capstone Project
                </div>

            </div>


            <div class="right-content">

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
                    identifies potential safety violations. The solution
                    provides visual detection results, helping improve safety
                    monitoring, reduce manual inspection effort, and support
                    faster identification of unsafe working conditions.

                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    st.markdown(
        '<div class="predict-button">',
        unsafe_allow_html=True
    )

    if st.button(
        "🔍 PREDICT",
        key="predict",
        use_container_width=False
    ):

        st.session_state.page = "predict"

        st.rerun()

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # CARDS
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="cards">


            <!-- TEAM MEMBERS -->

            <div class="card">

                <div class="card-heading">
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


            <!-- GMAIL -->

            <div class="card">

                <div class="card-heading">
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


            <!-- GUIDE -->

            <div class="card">

                <div class="card-heading">
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


        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="project-footer">
            GuardX-AI – Construction PPE Detection System
        </div>
        """,
        unsafe_allow_html=True
    )
