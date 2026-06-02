import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
data = joblib.load("model.pkl")

model = data["model"]
features = data["features"]
mae = data["mae"]

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("About")

st.sidebar.markdown(f"""
### Model Information

**Algorithm**
Random Forest Regressor

**Dataset**
Kaggle House Prices Dataset

**Features Used**
{len(features)}

**Model MAE**
${mae:,.2f}

**Developer**
Mico Yumul
""")

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("🏠 House Price Predictor")

st.caption(
    "Predict residential property prices using machine learning"
)

st.divider()

# --------------------------------------------------
# DASHBOARD CARDS
# --------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        "Random Forest"
    )

with col2:
    st.metric(
        "Features",
        len(features)
    )

with col3:
    st.metric(
        "MAE",
        f"${mae:,.0f}"
    )

st.divider()

# --------------------------------------------------
# PROPERTY INPUTS
# --------------------------------------------------
st.header("Property Information")

left, right = st.columns(2)

with left:

    lot_area = st.number_input(
        "Lot Area (sq ft)",
        min_value=0,
        value=0
    )

    living_area = st.number_input(
        "Living Area (sq ft)",
        min_value=0,
        value=0
    )

    year_built = st.number_input(
        "Year Built",
        min_value=0,
        value=0
    )

    year_remod = st.number_input(
        "Year Remodeled",
        min_value=0,
        value=0
    )

    overall_qual = st.slider(
        "Overall Quality (1-10)",
        0,
        10,
        0
    )

    overall_cond = st.slider(
        "Overall Condition (1-10)",
        0,
        10,
        0
    )

with right:

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=0,
        value=0
    )

    full_bath = st.number_input(
        "Full Bathrooms",
        min_value=0,
        value=0
    )

    half_bath = st.number_input(
        "Half Bathrooms",
        min_value=0,
        value=0
    )

    rooms = st.number_input(
        "Total Rooms",
        min_value=0,
        value=0
    )

    garage_cars = st.number_input(
        "Garage Capacity (cars)",
        min_value=0,
        value=0
    )

    garage_area = st.number_input(
        "Garage Area (sq ft)",
        min_value=0,
        value=0
    )

st.divider()

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if st.button(
    "🔮 Predict House Price",
    use_container_width=True
):

    if lot_area == 0 or living_area == 0:

        st.error(
            "Please enter at least the Lot Area and Living Area."
        )

    else:

        house = pd.DataFrame([{
            "LotArea": lot_area,
            "OverallQual": overall_qual,
            "OverallCond": overall_cond,
            "YearBuilt": year_built,
            "YearRemodAdd": year_remod,
            "BedroomAbvGr": bedrooms,
            "FullBath": full_bath,
            "HalfBath": half_bath,
            "TotRmsAbvGrd": rooms,
            "GarageCars": garage_cars,
            "GarageArea": garage_area,
            "GrLivArea": living_area
        }])

        prediction = model.predict(house)

        st.markdown("## Estimated House Price")

        st.success(
            f"${prediction[0]:,.2f}"
        )

        st.subheader("Property Summary")

        st.dataframe(
            house,
            use_container_width=True
        )

st.divider()

# --------------------------------------------------
# MODEL INSIGHTS
# --------------------------------------------------
st.header("Model Insights")

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=True
)

fig = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Feature Importance"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.subheader("Feature Rankings")

st.dataframe(
    importance_df.sort_values(
        by="Importance",
        ascending=False
    ),
    use_container_width=True
)

st.divider()

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.caption(
    "Built with Streamlit, Scikit-learn, Pandas, Joblib, and Plotly"
)