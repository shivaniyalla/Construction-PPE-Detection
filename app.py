# ============================================================
# PAGE 1 — HOME
# ============================================================

if st.session_state.page == "home":

    # ---------------- TITLE ----------------

    st.markdown(
        "<h1 style='text-align:center; color:#172b55;'>"
        "🦺 GuardX-AI – Construction PPE Detection System"
        "</h1>",
        unsafe_allow_html=True
    )

    st.write("")

    # ---------------- TOP SECTION ----------------

    left_col, right_col = st.columns([0.36, 0.64], gap="large")

    # ---------------- LEFT ----------------

    with left_col:

        st.markdown(
            """
            <div style="
                color:#172b55;
                font-size:25px;
                font-weight:800;
                line-height:1.55;
            ">
                AI Career for Women
                <br>
                (AICW)
            </div>

            <div style="
                color:#303c52;
                font-size:22px;
                font-weight:700;
                margin-top:42px;
            ">
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

    # ---------------- PROJECT DESCRIPTION ----------------

    with right_col:

        st.markdown(
            """
            <div style="
                color:#172b55;
                font-size:23px;
                font-weight:800;
                margin-bottom:12px;
            ">
                Project Description
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style="
                color:#626b78;
                font-size:15px;
                line-height:1.65;
            ">
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

    # ========================================================
    # BOTTOM CARDS
    # ========================================================

    team_col, gmail_col, guide_col = st.columns(
        [1.25, 1.25, 0.75],
        gap="large"
    )

    # ---------------- TEAM MEMBERS ----------------

    with team_col:

        st.markdown(
            """
            <div style="
                background:white;
                border:1px solid #e0e3e8;
                border-radius:17px;
                padding:20px;
                min-height:235px;
            ">

                <div style="
                    color:#303743;
                    font-size:14px;
                    font-weight:800;
                    margin-bottom:20px;
                ">
                    TEAM MEMBERS
                </div>

                <div style="
                    color:#4d5562;
                    font-size:14px;
                    line-height:2.7;
                ">
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

    # ---------------- GMAIL ----------------

    with gmail_col:

        st.markdown(
            """
            <div style="
                background:white;
                border:1px solid #e0e3e8;
                border-radius:17px;
                padding:20px;
                min-height:235px;
            ">

                <div style="
                    color:#303743;
                    font-size:14px;
                    font-weight:800;
                    margin-bottom:20px;
                ">
                    GMAIL
                </div>

                <div style="
                    color:#4d5562;
                    font-size:14px;
                    line-height:2.7;
                ">
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

    # ---------------- GUIDE ----------------

    with guide_col:

        st.markdown(
            """
            <div style="
                background:white;
                border:1px solid #e0e3e8;
                border-radius:17px;
                padding:20px;
                min-height:235px;
            ">

                <div style="
                    color:#303743;
                    font-size:14px;
                    font-weight:800;
                    margin-bottom:25px;
                ">
                    GUIDE NAME
                </div>

                <div style="
                    color:#4d5562;
                    font-size:15px;
                    margin-bottom:30px;
                ">
                    MD.Abdul Aziz
                </div>

                <div style="
                    color:#303743;
                    font-size:14px;
                    font-weight:800;
                    margin-bottom:20px;
                ">
                    Designation
                </div>

                <div style="
                    color:#4d5562;
                    font-size:14px;
                    line-height:1.7;
                ">
                    Trainer, Co-Lead-AICW
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------------- FOOTER ----------------

    st.markdown(
        """
        <div style="
            text-align:center;
            color:#737b87;
            font-size:14px;
            margin-top:35px;
        ">
            GuardX-AI – Construction PPE Detection System
        </div>
        """,
        unsafe_allow_html=True
    )
