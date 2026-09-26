import streamlit as st
import pandas as pd
import joblib
import plotly.express as px


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    df = pd.read_csv("data/Telco-Customer-Churn.csv")

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df = df.dropna()

    return df


@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")


df = load_data()
model = load_model()


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 1rem;
        color: #8b949e;
        margin-top: 5px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .metric-card {
        border: 1px solid rgba(128, 128, 128, 0.20);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 10px;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #8b949e;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-top: 5px;
    }

    .insight-card {
        border: 1px solid rgba(128, 128, 128, 0.20);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("📊 Churn Analytics")

    st.caption(
        "Customer behavior analysis and machine learning"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Customer Analysis",
            "Model Performance",
            "Churn Prediction"
        ]
    )

    st.divider()

    st.caption("Machine Learning Models")

    st.write("• Logistic Regression")
    st.write("• Random Forest")

    st.divider()

    st.caption("Dataset")

    st.write("IBM Telco Customer Churn")


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">Customer Churn Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Analyze customer behavior, identify churn patterns,
    and predict customer churn using machine learning.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# OVERVIEW PAGE
# =========================================================

if page == "Overview":

    st.markdown(
        '<div class="section-title">Overview</div>',
        unsafe_allow_html=True
    )

    total_customers = len(df)

    churned_customers = (
        df["Churn"] == "Yes"
    ).sum()

    churn_rate = (
        churned_customers / total_customers
    ) * 100

    avg_monthly = df["MonthlyCharges"].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Total Customers
                </div>
                <div class="metric-value">
                    {total_customers:,}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Churned Customers
                </div>
                <div class="metric-value">
                    {churned_customers:,}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Churn Rate
                </div>
                <div class="metric-value">
                    {churn_rate:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Avg. Monthly Charges
                </div>
                <div class="metric-value">
                    ${avg_monthly:.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    chart1, chart2 = st.columns(2)

    # Churn Distribution
    churn_counts = (
        df["Churn"]
        .value_counts()
        .reset_index()
    )

    churn_counts.columns = [
        "Churn",
        "Customers"
    ]

    fig_churn = px.pie(
        churn_counts,
        names="Churn",
        values="Customers",
        hole=0.55,
        title="Customer Churn Distribution"
    )

    fig_churn.update_layout(
        legend_title_text="Churn"
    )

    chart1.plotly_chart(
        fig_churn,
        use_container_width=True
    )

    # Contract Churn Rate
    contract_rate = (
        df.groupby("Contract")["Churn"]
        .apply(
            lambda x: (
                x == "Yes"
            ).mean() * 100
        )
        .reset_index(name="Churn Rate")
    )

    fig_contract_rate = px.bar(
        contract_rate,
        x="Contract",
        y="Churn Rate",
        title="Churn Rate by Contract Type",
        text_auto=".1f"
    )

    fig_contract_rate.update_layout(
        yaxis_title="Churn Rate (%)",
        xaxis_title=""
    )

    chart2.plotly_chart(
        fig_contract_rate,
        use_container_width=True
    )

    # -------------------------
    # KEY INSIGHTS
    # -------------------------

    st.subheader("Key Insights")

    month_rate = contract_rate.loc[
        contract_rate["Contract"] == "Month-to-month",
        "Churn Rate"
    ].iloc[0]

    two_year_rate = contract_rate.loc[
        contract_rate["Contract"] == "Two year",
        "Churn Rate"
    ].iloc[0]

    churn_monthly = df.loc[
        df["Churn"] == "Yes",
        "MonthlyCharges"
    ].mean()

    stay_monthly = df.loc[
        df["Churn"] == "No",
        "MonthlyCharges"
    ].mean()

    i1, i2, i3 = st.columns(3)

    with i1:
        st.markdown(
            f"""
            <div class="insight-card">
                <b>Month-to-Month Risk</b><br><br>
                {month_rate:.1f}% of month-to-month
                customers churn in this dataset.
            </div>
            """,
            unsafe_allow_html=True
        )

    with i2:
        st.markdown(
            f"""
            <div class="insight-card">
                <b>Long-Term Contracts</b><br><br>
                Two-year customers have a churn rate
                of only {two_year_rate:.1f}%.
            </div>
            """,
            unsafe_allow_html=True
        )

    with i3:
        st.markdown(
            f"""
            <div class="insight-card">
                <b>Monthly Charges</b><br><br>
                Churned customers pay ${churn_monthly:.2f}
                per month on average versus
                ${stay_monthly:.2f} for retained customers.
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# CUSTOMER ANALYSIS PAGE
# =========================================================

elif page == "Customer Analysis":

    st.markdown(
        '<div class="section-title">Customer Analysis</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Explore how customer characteristics relate to churn."
    )

    col1, col2 = st.columns(2)

    # Contract
    contract_data = (
        df.groupby(["Contract", "Churn"])
        .size()
        .reset_index(name="Customers")
    )

    fig_contract = px.bar(
        contract_data,
        x="Contract",
        y="Customers",
        color="Churn",
        barmode="group",
        title="Churn by Contract Type"
    )

    col1.plotly_chart(
        fig_contract,
        use_container_width=True
    )

    # Internet Service
    internet_data = (
        df.groupby(["InternetService", "Churn"])
        .size()
        .reset_index(name="Customers")
    )

    fig_internet = px.bar(
        internet_data,
        x="InternetService",
        y="Customers",
        color="Churn",
        barmode="group",
        title="Churn by Internet Service"
    )

    col2.plotly_chart(
        fig_internet,
        use_container_width=True
    )

    col3, col4 = st.columns(2)

    # Tenure
    fig_tenure = px.box(
        df,
        x="Churn",
        y="tenure",
        color="Churn",
        title="Customer Tenure by Churn"
    )

    col3.plotly_chart(
        fig_tenure,
        use_container_width=True
    )

    # Monthly Charges
    fig_monthly = px.box(
        df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        title="Monthly Charges by Churn"
    )

    col4.plotly_chart(
        fig_monthly,
        use_container_width=True
    )

    # Payment Method
    payment_data = (
        df.groupby(["PaymentMethod", "Churn"])
        .size()
        .reset_index(name="Customers")
    )

    fig_payment = px.bar(
        payment_data,
        x="PaymentMethod",
        y="Customers",
        color="Churn",
        barmode="group",
        title="Churn by Payment Method"
    )

    fig_payment.update_layout(
        xaxis_tickangle=-20
    )

    st.plotly_chart(
        fig_payment,
        use_container_width=True
    )


# =========================================================
# MODEL PERFORMANCE PAGE
# =========================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="section-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Comparison of the two machine learning models "
        "evaluated on the held-out test set."
    )

    model_results = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest"
        ],

        "Accuracy": [
            0.8038,
            0.7910
        ],

        "Precision": [
            0.6485,
            0.6379
        ],

        "Recall": [
            0.5722,
            0.4947
        ],

        "F1 Score": [
            0.6080,
            0.5572
        ],

        "ROC AUC": [
            0.8359,
            0.8348
        ]
    })

    display_results = model_results.copy()

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC"
    ]:
        display_results[column] = (
            display_results[column]
            .map(lambda x: f"{x:.4f}")
        )

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )

    st.write("")

    metric = st.selectbox(
        "Compare models by metric",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC AUC"
        ]
    )

    fig_model = px.bar(
        model_results,
        x="Model",
        y=metric,
        text_auto=".4f",
        title=f"{metric} Comparison"
    )

    fig_model.update_yaxes(
        range=[0, 1]
    )

    st.plotly_chart(
        fig_model,
        use_container_width=True
    )

    st.success(
        "Logistic Regression achieved the highest "
        "F1 score and is used as the final prediction model."
    )


# =========================================================
# CHURN PREDICTION PAGE
# =========================================================

elif page == "Churn Prediction":

    st.markdown(
        '<div class="section-title">Churn Prediction</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter the customer's information to estimate "
        "their probability of churn."
    )

    # -------------------------
    # CUSTOMER PROFILE
    # -------------------------

    st.subheader("1. Customer Profile")

    p1, p2, p3 = st.columns(3)

    with p1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )

    with p2:

        partner = st.selectbox(
            "Partner",
            ["No", "Yes"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["No", "Yes"]
        )

    with p3:

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

    st.divider()

    # -------------------------
    # SERVICES
    # -------------------------

    st.subheader("2. Services")

    s1, s2, s3 = st.columns(3)

    with s1:

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "No",
                "Yes",
                "No phone service"
            ]
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        online_security = st.selectbox(
            "Online Security",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

    with s2:

        online_backup = st.selectbox(
            "Online Backup",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        device_protection = st.selectbox(
            "Device Protection",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        tech_support = st.selectbox(
            "Tech Support",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

    with s3:

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

    st.divider()

    # -------------------------
    # BILLING
    # -------------------------

    st.subheader("3. Contract & Billing")

    b1, b2, b3 = st.columns(3)

    with b1:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

    with b2:

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            value=70.0
        )

    with b3:

        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=840.0
        )

    st.write("")

    # -------------------------
    # PREDICTION BUTTON
    # -------------------------

    if st.button(
        "Predict Customer Churn",
        type="primary",
        use_container_width=True
    ):

        customer = pd.DataFrame({

            "gender": [gender],

            "SeniorCitizen": [
                1 if senior == "Yes" else 0
            ],

            "Partner": [partner],

            "Dependents": [dependents],

            "tenure": [tenure],

            "PhoneService": [
                phone_service
            ],

            "MultipleLines": [
                multiple_lines
            ],

            "InternetService": [
                internet_service
            ],

            "OnlineSecurity": [
                online_security
            ],

            "OnlineBackup": [
                online_backup
            ],

            "DeviceProtection": [
                device_protection
            ],

            "TechSupport": [
                tech_support
            ],

            "StreamingTV": [
                streaming_tv
            ],

            "StreamingMovies": [
                streaming_movies
            ],

            "Contract": [
                contract
            ],

            "PaperlessBilling": [
                paperless
            ],

            "PaymentMethod": [
                payment_method
            ],

            "MonthlyCharges": [
                monthly_charges
            ],

            "TotalCharges": [
                total_charges
            ]
        })

        prediction = model.predict(
            customer
        )[0]

        probability = model.predict_proba(
            customer
        )[0][1]

        st.divider()

        st.subheader("Prediction Result")

        r1, r2 = st.columns(2)

        with r1:

            st.metric(
                "Churn Probability",
                f"{probability * 100:.1f}%"
            )

        with r2:

            if prediction == 1:

                st.error(
                    "⚠️ HIGH CHURN RISK"
                )

            else:

                st.success(
                    "✓ LOW CHURN RISK"
                )

        st.progress(
            float(probability)
        )

        if prediction == 1:

            st.write(
                "This customer has been classified as "
                "**likely to churn** by the Logistic "
                "Regression model."
            )

        else:

            st.write(
                "This customer has been classified as "
                "**likely to stay** by the Logistic "
                "Regression model."
            )