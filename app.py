# ============================================================
# PAGE 1 — EXACT REFERENCE STYLE
# ============================================================

if st.session_state.page == "home":

    # -----------------------------
    # HOME PAGE HTML + CSS
    # -----------------------------

    st.markdown("""
    <style>

    /* REMOVE DEFAULT STREAMLIT SPACE */
    .block-container {
        padding-top: 20px !important;
        padding-left: 25px !important;
        padding-right: 25px !important;
        max-width: 100% !important;
    }

    /* MAIN FRAME */
    .guardx-frame {
        width: 100%;
        min-height: 760px;
        border: 3px solid #000000;
        background: #ffffff;
        display: grid;
        grid-template-rows: 115px 1fr;
        overflow: hidden;
        box-sizing: border-box;
    }

    /* HEADER */
    .guardx-header {
        border-bottom: 3px solid #000000;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }

    .guardx-header-title {
        font-size: 40px;
        font-weight: 800;
        color: #000000;
        line-height: 1.1;
    }

    .guardx-header-subtitle {
        font-size: 17px;
        font-weight: 500;
        color: #333333;
        margin-top: 7px;
    }

    /* BODY */
    .guardx-body {
        display: grid;
        grid-template-columns: 34% 66%;
        min-height: 645px;
    }

    /* LEFT PANEL */
    .guardx-left {
        border-right: 3px solid #000000;
        padding: 35px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
    }

    .aicw-title {
        font-size: 25px;
        font-weight: 800;
        color: #000000;
        line-height: 1.3;
    }

    .capstone-title {
        font-size: 18px;
        font-weight: 600;
        color: #333333;
        margin-top: 18px;
    }

    /* PREDICT AREA */
    .predict-area {
        width: 100%;
        margin-top: auto;
    }

    .predict-label {
        font-size: 18px;
        font-weight: 700;
        color: #000000;
        margin-bottom: 10px;
    }

    .predict-btn {
        width: 100%;
        height: 55px;
        border: 2px solid #000000;
        border-radius: 6px;
        background: #000000;
        color: #ffffff;
        font-size: 18px;
        font-weight: 800;
        cursor: pointer;
        text-align: center;
        line-height: 55px;
        box-sizing: border-box;
    }

    /* RIGHT PANEL */
    .guardx-right {
        display: grid;
        grid-template-rows: 105px 1fr 220px;
        min-width: 0;
    }

    /* TITLE BOX */
    .guardx-title-box {
        border-bottom: 3px solid #000000;
        display: flex;
        align-items: center;
        padding: 0 35px;
    }

    .guardx-title {
        font-size: 30px;
        font-weight: 800;
        color: #000000;
    }

    /* DESCRIPTION BOX */
    .guardx-description {
        border-bottom: 3px solid #000000;
        padding: 30px 35px;
        box-sizing: border-box;
    }

    .guardx-section-title {
        font-size: 20px;
        font-weight: 800;
        color: #000000;
        margin-bottom: 15px;
    }

    .guardx-description-text {
        font-size: 16px;
        line-height: 1.7;
        color: #222222;
        text-align: justify;
    }

    /* BOTTOM */
    .guardx-bottom {
        display: grid;
        grid-template-columns: 55% 45%;
    }

    /* TEAM */
    .guardx-team {
        border-right: 3px solid #000000;
        padding: 28px 35px;
        box-sizing: border-box;
    }

    .guardx-members {
        font-size: 15px;
        line-height: 2;
        color: #222222;
    }

    /* GUIDE */
    .guardx-guide {
        padding: 28px 35px;
        box-sizing: border-box;
    }

    .guardx-guide-name {
        font-size: 18px;
        font-weight: 700;
        color: #000000;
        margin-top: 15px;
    }

    .guardx-guide-designation {
        font-size: 15px;
        color: #333333;
        margin-top: 7px;
    }

    /* RESPONSIVE */
    @media (max-width: 900px) {

        .guardx-body {
            grid-template-columns: 1fr;
        }

        .guardx-left {
            border-right: none;
            border-bottom: 3px solid #000000;
            min-height: 300px;
        }

        .guardx-right {
            grid-template-rows: auto auto auto;
        }

        .guardx-bottom {
            grid-template-columns: 1fr;
        }

        .guardx-team {
            border-right: none;
            border-bottom: 3px solid #000000;
        }
    }

    </style>


    <div class="guardx-frame">

        <!-- ========================================== -->
        <!-- HEADER -->
        <!-- ========================================== -->

        <div class="guardx-header">

            <div class="guardx-header-title">
                GuardX-AI
            </div>

            <div class="guardx-header-subtitle">
                AI-Powered Construction PPE Detection System
            </div>

        </div>


        <!-- ========================================== -->
        <!-- BODY -->
        <!-- ========================================== -->

        <div class="guardx-body">


            <!-- ====================================== -->
            <!-- LEFT SIDE -->
            <!-- ====================================== -->

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


                <!-- PREDICT -->

                <div class="predict-area">

                    <div class="predict-label">
                        Start Detection
                    </div>

                </div>

            </div>


            <!-- ====================================== -->
            <!-- RIGHT SIDE -->
            <!-- ====================================== -->

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


                <!-- ================================= -->
                <!-- TEAM + GUIDE -->
                <!-- ================================= -->

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
    # REAL STREAMLIT PREDICT BUTTON
    # ========================================================

    st.markdown("""
    <style>

    /* Position Streamlit button over the left panel */
    div[data-testid="stButton"] {
        position: relative;
        margin-top: -115px;
        margin-left: 25px;
        width: calc(34% - 50px);
        z-index: 10;
    }

    div[data-testid="stButton"] button {
        height: 55px;
        border: 2px solid #000000;
        border-radius: 6px;
        font-size: 18px;
        font-weight: 800;
    }

    </style>
    """, unsafe_allow_html=True)


    if st.button(
        "🚀 PREDICT",
        key="predict_home",
        type="primary"
    ):

        st.session_state.page = "detection"
        st.rerun()
