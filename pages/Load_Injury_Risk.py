import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import load_all_data, calculate_injury_risk
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("📉 Team Load Demand & Injury Risk Analysis")

# Load data
gps_data, _, _, _ = load_all_data()
gps_data['injury_risk'] = gps_data.apply(lambda row: calculate_injury_risk(row, gps_data), axis=1)

# Sidebar filters
st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(gps_data['date'].min(), gps_data['date'].max())
)

# Filter data based on date selection
filtered_data = gps_data[
    (gps_data['date'].dt.date >= date_range[0]) &
    (gps_data['date'].dt.date <= date_range[1])
]

# Calculate daily team metrics
daily_metrics = filtered_data.groupby('date').agg({
    'distance': 'sum',
    'accel_decel_over_2_5': 'sum',
    'peak_speed': 'max',
    'injury_risk': 'mean'
}).reset_index()

# Main metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Average Daily Team Distance", f"{daily_metrics['distance'].mean():.1f} km")
with col2:
    st.metric("Team Peak Speed", f"{daily_metrics['peak_speed'].max():.1f} km/h")
with col3:
    st.metric("Average Team Injury Risk", f"{daily_metrics['injury_risk'].mean():.2%}")
with col4:
    st.metric("High Risk Days", f"{len(daily_metrics[daily_metrics['injury_risk'] > 0.7])}")

# Create tabs for different visualizations
tab1, tab2 = st.tabs(["Load Trends", "Risk Analysis"])

with tab1:
    # Load trends visualization
    fig_load = go.Figure()
    fig_load.add_trace(go.Scatter(
        x=daily_metrics['date'],
        y=daily_metrics['distance'],
        mode='lines+markers',
        name='Team Distance'
    ))
    
    fig_load.update_layout(
        title="Daily Team Load Trends",
        xaxis_title="Date",
        yaxis_title="Total Distance (km)",
        height=500
    )
    st.plotly_chart(fig_load, use_container_width=True)

with tab2:
    # Risk analysis visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig_risk = px.scatter(
            daily_metrics,
            x='distance',
            y='accel_decel_over_2_5',
            size='injury_risk',
            color='injury_risk',
            color_continuous_scale='RdYlGn_r',
            title='Team Load vs. Acceleration/Deceleration',
            labels={'distance': 'Total Distance (km)',
                   'accel_decel_over_2_5': 'Total Accel/Decel Count'}
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        fig_heatmap = px.density_heatmap(
            daily_metrics,
            x='distance',
            y='peak_speed',
            title='Team Load Distribution',
            labels={'distance': 'Total Distance (km)',
                   'peak_speed': 'Peak Speed (km/h)'}
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

# Risk alerts
st.subheader("⚠️ Team Risk Alerts")
high_risk_days = daily_metrics[daily_metrics['injury_risk'] > 0.7]
if not high_risk_days.empty:
    for _, row in high_risk_days.iterrows():
        st.warning(
            f"High team injury risk ({row['injury_risk']:.2%}) detected on "
            f"{row['date'].strftime('%Y-%m-%d')}. Total Distance: {row['distance']:.1f}km, "
            f"Peak Speed: {row['peak_speed']:.1f}km/h"
        )
else:
    st.success("No high-risk days detected in the selected period.")

# Additional insights
st.subheader("📊 Team Insights")
col1, col2 = st.columns(2)

with col1:
    st.write("**Load Patterns**")
    load_trend = np.polyfit(range(len(daily_metrics)), daily_metrics['distance'], 1)[0]
    if load_trend > 0:
        st.info(f"📈 Upward trend in team load ({load_trend:.2f} km/day)")
    else:
        st.info(f"📉 Downward trend in team load ({abs(load_trend):.2f} km/day)")

with col2:
    st.write("**Risk Factors**")
    # Calculate correlations and handle NaN values
    risk_factors = daily_metrics[['distance', 'accel_decel_over_2_5', 'peak_speed', 'injury_risk']].corr()['injury_risk']
    risk_factors = risk_factors.dropna()  # Remove any NaN values
    
    if not risk_factors.empty:
        # Find the dominant factor excluding 'injury_risk' itself
        risk_factors_without_self = risk_factors[risk_factors.index != 'injury_risk']
        if not risk_factors_without_self.empty:
            dominant_factor = risk_factors_without_self.abs().idxmax()
            correlation_value = risk_factors_without_self[dominant_factor]
            st.info(f"Dominant team risk factor: {dominant_factor} (correlation: {correlation_value:.2f})")
        else:
            st.info("No significant risk factors found in the data.")
    else:
        st.info("Insufficient data to calculate risk factors.")

# Weekly summary
st.subheader("📅 Weekly Summary")
weekly_metrics = daily_metrics.set_index('date').resample('W').agg({
    'distance': 'sum',
    'accel_decel_over_2_5': 'sum',
    'peak_speed': 'max',
    'injury_risk': 'mean'
}).reset_index()

fig_weekly = go.Figure()
fig_weekly.add_trace(go.Bar(
    x=weekly_metrics['date'],
    y=weekly_metrics['distance'],
    name='Weekly Distance'
))
fig_weekly.add_trace(go.Scatter(
    x=weekly_metrics['date'],
    y=weekly_metrics['injury_risk'],
    name='Average Risk',
    yaxis='y2'
))

fig_weekly.update_layout(
    title="Weekly Team Performance",
    xaxis_title="Week",
    yaxis_title="Total Distance (km)",
    yaxis2=dict(
        title="Average Injury Risk",
        overlaying='y',
        side='right'
    ),
    barmode='group'
)
st.plotly_chart(fig_weekly, use_container_width=True)
