
import streamlit as st

st.set_page_config(page_title="CFC Performance Insights", page_icon="⚽", layout="wide")

st.title("⚽ Chelsea FC Performance Insights")
st.markdown("Welcome to the all-in-one dashboard to analyze player workload, recovery, and performance metrics.")

st.markdown("""
### 📈 Available Dashboards:
- **[Load & Injury Risk](./Load_Injury_Risk)**
- **[Recovery Status](./Recovery_Status)**
- **[Physical Capability](./Physical_Capability)**
- **[Injury Risk Prediction](./Injury_Risk_Prediction)**
""")
