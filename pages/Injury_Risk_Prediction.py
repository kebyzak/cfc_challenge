import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from utils import load_all_data

st.set_page_config(layout="wide")
st.title("🔮 Advanced Injury Risk Prediction Model")

# Load all data sources
gps_data, recovery_raw, physical_data, priority_data = load_all_data()

# Data processing tab
tab1, tab2 = st.tabs(["Model & Predictions", "Risk Factors Analysis"])  

with tab1:
    st.header("Data Integration & Model Training")
    
    # Feature Engineering Section
    with st.expander("📊 Feature Engineering Details", expanded=False):
        st.write("""
        This model integrates three data sources for comprehensive injury risk prediction:
        1. **GPS Training Data**: Load metrics, intensity, and workload
        2. **Recovery Status Data**: Fatigue levels and recovery metrics
        3. **Physical Capability Data**: Strength and movement quality metrics
        
        The model also calculates derived features including:
        - Rolling workload averages (acute vs chronic)
        - Recovery score trends
        - Physical capability changes
        - Workload spikes
        """)
    
    # Preprocessing - GPS Data
    # Add date features and rolling metrics
    gps_data['year'] = gps_data['date'].dt.year
    gps_data['month'] = gps_data['date'].dt.month
    gps_data['day_of_week'] = gps_data['date'].dt.dayofweek
    
    # Calculate rolling averages for acute (7-day) and chronic (28-day) loads
    # Sort by date first
    gps_data = gps_data.sort_values('date')
    
    # Calculate team-level metrics
    gps_daily = gps_data.groupby('date').agg({
        'distance': 'mean',
        'accel_decel_over_2_5': 'mean',
        'peak_speed': 'max'
    }).reset_index()
    
    # Calculate rolling window metrics
    gps_daily['distance'] = gps_daily['distance'].ffill()
    gps_daily['accel_decel_over_2_5'] = gps_daily['accel_decel_over_2_5'].ffill()
    
    gps_daily['acute_distance'] = gps_daily['distance'].rolling(3, min_periods=1).mean()
    gps_daily['acute_accel_decel'] = gps_daily['accel_decel_over_2_5'].rolling(3, min_periods=1).mean()
    
    gps_daily['chronic_distance'] = gps_daily['distance'].rolling(14, min_periods=1).mean()
    gps_daily['chronic_accel_decel'] = gps_daily['accel_decel_over_2_5'].rolling(14, min_periods=1).mean()
    
    # Calculate workload ratios (acute:chronic ratio)
    gps_daily['acwr_distance'] = gps_daily['acute_distance'] / gps_daily['chronic_distance'].replace(0, gps_daily['chronic_distance'].mean())
    gps_daily['acwr_accel_decel'] = gps_daily['acute_accel_decel'] / gps_daily['chronic_accel_decel'].replace(0, gps_daily['chronic_accel_decel'].mean())
    
    # Replace infinite values and clip to evidence-based range (0.8-1.5 is optimal zone)
    gps_daily['acwr_distance'] = gps_daily['acwr_distance'].fillna(1.0).clip(0, 2.0)
    gps_daily['acwr_accel_decel'] = gps_daily['acwr_accel_decel'].fillna(1.0).clip(0, 2.0)
    
    # Calculate workload monotony (variation in daily loads over 7 days)
    # Low monotony is better than high monotony (constant load)
    gps_daily['distance_monotony'] = gps_daily['distance'].rolling(7, min_periods=3).mean() / gps_daily['distance'].rolling(7, min_periods=3).std().replace(0, 1)
    gps_daily['distance_monotony'] = gps_daily['distance_monotony'].fillna(1.0).clip(0, 5)
    
    # Calculate workload spikes (day-to-day changes)
    gps_daily['distance_spike'] = gps_daily['distance'].pct_change().fillna(0)
    gps_daily['accel_decel_spike'] = gps_daily['accel_decel_over_2_5'].pct_change().fillna(0)
    
    # Clip spikes to evidence-based range for percent changes
    gps_daily['distance_spike'] = gps_daily['distance_spike'].clip(-0.5, 1.0)
    gps_daily['accel_decel_spike'] = gps_daily['accel_decel_spike'].clip(-0.5, 1.0)
    
    # Calculate intensity metrics
    gps_daily['intensity_ratio'] = gps_daily['peak_speed'] / gps_daily['distance'].replace(0, gps_daily['distance'].mean())
    gps_daily['intensity_ratio'] = gps_daily['intensity_ratio'].fillna(gps_daily['intensity_ratio'].mean()).clip(0, 5)
    
    # Add date features
    gps_daily['year'] = gps_daily['date'].dt.year
    gps_daily['month'] = gps_daily['date'].dt.month
    gps_daily['day_of_week'] = gps_daily['date'].dt.dayofweek
    
    # Prepare recovery data - pivot so each metric is a column
    if 'player' in recovery_raw.columns:
        # First aggregate to team level if player exists
        recovery_daily = recovery_raw.groupby(['sessionDate', 'metric'])['value'].mean().reset_index()
    else:
        recovery_daily = recovery_raw.copy()
    
    # Pivot to get each metric as a column
    recovery_pivot = recovery_daily.pivot_table(
        index='sessionDate',
        columns='metric',
        values='value',
        aggfunc='mean'
    ).reset_index()
    
    # Add trends in recovery data
    recovery_pivot['sessionDate'] = pd.to_datetime(recovery_pivot['sessionDate'], dayfirst=True)
    recovery_pivot = recovery_pivot.sort_values('sessionDate')
    
    if 'emboss_baseline_score' in recovery_pivot.columns:
        recovery_pivot['recovery_trend'] = recovery_pivot['emboss_baseline_score'].diff()
    
    # Prepare physical capability data
    if 'player' in physical_data.columns:
        # Aggregate to daily level if player exists
        physical_daily = physical_data.groupby(['testDate', 'movement'])['benchmarkPct'].mean().reset_index()
    else:
        physical_daily = physical_data.copy()
    
    # Pivot physical data
    physical_pivot = physical_daily.pivot_table(
        index='testDate',
        columns='movement',
        values='benchmarkPct',
        aggfunc='mean'
    ).reset_index()
    
    physical_pivot['testDate'] = pd.to_datetime(physical_pivot['testDate'], dayfirst=True)
    physical_pivot['overall_physical'] = physical_pivot.select_dtypes(include=[np.number]).mean(axis=1)
    
    # Rename date columns for joining
    recovery_pivot = recovery_pivot.rename(columns={'sessionDate': 'date'})
    physical_pivot = physical_pivot.rename(columns={'testDate': 'date'})
    
    # For each GPS data point, find the most recent recovery and physical data
    merged_data = []
    
    for idx, gps_row in gps_daily.iterrows():
        gps_date = gps_row['date']
        
        # Find closest previous recovery data point
        prev_recovery = recovery_pivot[recovery_pivot['date'] <= gps_date].sort_values('date', ascending=False)
        if not prev_recovery.empty:
            recovery_row = prev_recovery.iloc[0].to_dict()
            days_since_recovery = (gps_date - recovery_row['date']).days
        else:
            recovery_row = {}
            days_since_recovery = None
        
        # Find closest previous physical test
        prev_physical = physical_pivot[physical_pivot['date'] <= gps_date].sort_values('date', ascending=False)
        if not prev_physical.empty:
            physical_row = prev_physical.iloc[0].to_dict()
            days_since_physical = (gps_date - physical_row['date']).days
        else:
            physical_row = {}
            days_since_physical = None
        
        # Combine all data
        combined_row = {
            'date': gps_date,
            'days_since_recovery': days_since_recovery,
            'days_since_physical': days_since_physical,
            **{k: v for k, v in gps_row.items() if k != 'date'},
            **{f'recovery_{k}': v for k, v in recovery_row.items() if k != 'date'},
            **{f'physical_{k}': v for k, v in physical_row.items() if k != 'date'}
        }
        merged_data.append(combined_row)
    
    # Create final dataframe
    df = pd.DataFrame(merged_data)
    
    # Define target variable: historical injuries based on load parameters
    high_risk_conditions = []
    
    if 'acwr_distance' in df.columns:
        high_risk_conditions.append(df['acwr_distance'] > 1.5)
    
    if 'distance_spike' in df.columns:
        high_risk_conditions.append(df['distance_spike'] > 0.5)
    
    if 'acute_distance' in df.columns:
        high_risk_conditions.append(df['acute_distance'] > df['acute_distance'].quantile(0.85))
    
    if 'recovery_emboss_baseline_score' in df.columns:
        # More sensitive to poor recovery scores
        high_risk_conditions.append(df['recovery_emboss_baseline_score'] < 70)
    
    if 'physical_benchmarkPct' in df.columns:
        # Add physical capability decline as a risk factor
        high_risk_conditions.append(df['physical_benchmarkPct'] < 20)  # Significant decline threshold
    
    if high_risk_conditions:
        # Weight the conditions based on their importance
        weights = np.zeros_like(high_risk_conditions, dtype=float)
        for i, condition in enumerate(high_risk_conditions):
            if 'recovery' in str(condition):
                weights[i] = 2.0  # Double weight for recovery
            elif 'physical' in str(condition):
                weights[i] = 1.5  # 1.5x weight for physical capability
            else:
                weights[i] = 1.0  # Normal weight for other factors
        
        # Calculate weighted risk score
        weighted_risk = np.sum([condition.astype(float) * weight for condition, weight in zip(high_risk_conditions, weights)], axis=0)
        df['high_risk'] = (weighted_risk >= 1.5).astype(int)  # Adjusted threshold
    else:
        st.warning("Unable to determine risk factors. Using simple threshold on distance.")
        df['high_risk'] = (df['distance'] > df['distance'].quantile(0.75)).astype(int)
    
    # Define features to use in the model
    load_features = ['distance', 'accel_decel_over_2_5', 'peak_speed', 
                     'acute_distance', 'chronic_distance', 'acwr_distance',
                     'acute_accel_decel', 'chronic_accel_decel', 'acwr_accel_decel',
                     'distance_spike', 'accel_decel_spike', 'intensity_ratio']
    
    recovery_features = [col for col in df.columns if col.startswith('recovery_')]
    physical_features = [col for col in df.columns if col.startswith('physical_')]
    time_features = ['days_since_recovery', 'days_since_physical', 'day_of_week']
    
    # Filter features that actually exist in the data and don't have too many NaN values
    available_features = []
    for feature_list in [load_features, recovery_features, physical_features, time_features]:
        for feature in feature_list:
            if feature in df.columns and df[feature].notna().sum() > len(df) * 0.5:
                available_features.append(feature)
    
    # Print available features
    st.write(f"Using {len(available_features)} features for model training")
    
    # Check if we have enough data
    if len(df) < 30 or len(available_features) < 3:
        st.error("Not enough data for reliable model training. Please collect more data.")
        st.stop()
    
    # Data preparation
    X = df[available_features].copy()
    y = df['high_risk']
    
    # Handle missing values and replace infinities
    for col in X.columns:
        # First replace any infinities
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        
        if X[col].isna().any():
            if col in ['acwr_distance', 'acwr_accel_decel']:
                # For ACWR metrics, replace NaN with 1.0 (neutral value)
                X[col] = X[col].fillna(1.0)
            elif col in load_features:
                # Other GPS data - replace with mean
                X[col] = X[col].fillna(X[col].mean())
            elif col in recovery_features:
                # Recovery data - use median
                X[col] = X[col].fillna(X[col].median())
            elif col in physical_features:
                # Physical data - forward fill + mean
                X[col] = X[col].ffill().fillna(X[col].mean())
            else:
                # Other features - mean
                X[col] = X[col].fillna(X[col].mean())
    
    # Final check for any remaining issues
    X = X.clip(-1e9, 1e9)  # Clip to large but finite values
    
    # Train model
    # First split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create a pipeline with preprocessing and model
    feature_names = list(X.columns)  # Get feature names
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=4,  # Slightly increased complexity
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=5  # Require more samples per leaf for stability
        ))
    ])
    
    # Train model with feature names
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Model metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_fig = px.imshow(cm, text_auto=True, 
                          x=['Predicted Low Risk', 'Predicted High Risk'], 
                          y=['Actual Low Risk', 'Actual High Risk'],
                          color_continuous_scale="Reds")
        cm_fig.update_layout(title="Confusion Matrix")
        st.plotly_chart(cm_fig, use_container_width=True)
    
    with col2:
        st.subheader("Feature Importance")
        
        # Get feature importance from the model
        feature_importance = pipeline.named_steps['classifier'].feature_importances_
        features_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        # Plot feature importance
        fig_imp = px.bar(features_df.head(10), y='Feature', x='Importance', orientation='h',
                        title="Top 10 Most Important Features",
                        labels={"Importance": "Relative Importance", "Feature": "Feature"})
        fig_imp.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=800,  # Significantly increased height
            width=1000,  # Significantly increased width
            margin=dict(l=250, r=50, t=50, b=50),  # Increased left margin for feature names
            font=dict(size=14)  # Increased font size
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        with st.expander("ℹ️ Why These Features Matter"):
            st.markdown("""
                        **Recovery Score (2.0x weight)**  
                        - Studies (Halson 2014, Kellmann 2010) show poor recovery increases injury risk 2–3x  
                        
                        **Physical Capability (1.5x weight)**  
                        - Read et al. (2018): poor movement quality raises risk ~1.5x  
                        
                        **Other Load Metrics (1.0x)**  
                        - ACWR, spikes, and monotony are important but secondary contributors  
                        """)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=600, height=400
        )
        fig_roc.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        st.plotly_chart(fig_roc, use_container_width=True)

with tab2:
    st.header("Risk Factors Analysis")
    
    # Analyze factors contributing to high risk
    st.subheader("Team Risk Timeline")
    
    # Create timeline of risk factors
    fig_timeline = go.Figure()
    
    # Add ACWR line if available
    if 'acwr_distance' in df.columns:
        fig_timeline.add_trace(go.Scatter(
            x=df['date'],
            y=df['acwr_distance'],
            mode='lines',
            name='ACWR Distance',
            line=dict(color='blue')
        ))
        
        # Add reference line for ACWR
        fig_timeline.add_hline(y=1.5, line_dash="dash", line_color="red", 
                              annotation_text="High ACWR Threshold")
    
    # Add recovery score if available
    if 'recovery_emboss_baseline_score' in df.columns:
        fig_timeline.add_trace(go.Scatter(
            x=df['date'],
            y=df['recovery_emboss_baseline_score'],
            mode='lines',
            name='Recovery Score',
            line=dict(color='green'),
            yaxis='y2'
        ))
        
        # Update layout for dual y-axis
        fig_timeline.update_layout(
            yaxis2=dict(
                title="Recovery Score",
                overlaying="y",
                side="right",
                range=[0, 100]
            )
        )
    
    # Add high risk markers
    high_risk_df = df[df['high_risk'] == 1]
    fig_timeline.add_trace(go.Scatter(
        x=high_risk_df['date'],
        y=[1] * len(high_risk_df) if 'acwr_distance' not in high_risk_df.columns 
        else high_risk_df['acwr_distance'],
        mode='markers',
        name='High Risk Days',
        marker=dict(color='red', size=12, symbol='x')
    ))
    
    # Update layout
    fig_timeline.update_layout(
        title="Team Risk Timeline",
        xaxis_title="Date",
        yaxis_title="ACWR Distance",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # If we have recovery data, show interaction with workload
    if 'acwr_distance' in df.columns and 'recovery_emboss_baseline_score' in df.columns:
        st.subheader("Interaction between Training Load and Recovery")
        
        # Create the scatter plot
        fig_load_recovery = px.scatter(
            df,
            x="acwr_distance", 
            y="recovery_emboss_baseline_score",
            color="high_risk",
            color_discrete_map={0: "green", 1: "red"},
            hover_data=["date", "distance", "peak_speed"],
            labels={"acwr_distance": "Acute:Chronic Workload Ratio", 
                    "recovery_emboss_baseline_score": "Recovery Score",
                    "high_risk": "High Injury Risk"},
            title="Relationship between Workload Ratio and Recovery Status"
        )
        
        # Add reference regions
        fig_load_recovery.add_shape(
            type="rect", 
            x0=1.5, y0=0, x1=3, y1=50,
            line=dict(color="rgba(255,0,0,0.2)", width=0),
            fillcolor="rgba(255,0,0,0.2)",
        )
        fig_load_recovery.add_shape(
            type="rect", 
            x0=0, y0=0, x1=1.5, y1=50,
            line=dict(color="rgba(255,165,0,0.2)", width=0),
            fillcolor="rgba(255,165,0,0.2)",
        )
        
        # Add reference lines
        fig_load_recovery.add_hline(y=60, line_dash="dash", line_color="orange", 
                                   annotation_text="Recovery Threshold")
        fig_load_recovery.add_vline(x=1.5, line_dash="dash", line_color="red", 
                                   annotation_text="High ACWR")
        
        st.plotly_chart(fig_load_recovery, use_container_width=True)
        
        st.markdown("""
        ### Key Risk Zones:
        - **High Risk Zone**: ACWR > 1.5 and Recovery Score < 60
        - **Moderate Risk Zone**: ACWR > 1.5 or Recovery Score < 60
        - **Low Risk Zone**: ACWR < 1.5 and Recovery Score > 60
        """)
    
    # Workload distribution by risk level
    st.subheader("Workload Distribution by Risk Level")
    
    # Select metrics to plot
    metrics_to_plot = [feat for feat in ['distance', 'peak_speed', 'acwr_distance', 'distance_spike'] 
                       if feat in df.columns]
    
    if metrics_to_plot:
        fig_box = go.Figure()
        
        for metric in metrics_to_plot:
            fig_box.add_trace(go.Box(
                y=df[df['high_risk']==0][metric],
                name=f'Low Risk - {metric}',
                boxmean=True
            ))
            fig_box.add_trace(go.Box(
                y=df[df['high_risk']==1][metric],
                name=f'High Risk - {metric}',
                boxmean=True
            ))
        
        fig_box.update_layout(title="Load Metrics Distribution by Risk Level")
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Current risk assessment
    st.subheader("Current Team Risk Assessment")
    
    # Get the most recent data point
    if not df.empty:
        latest_data = df.iloc[-1]
        risk_prob = pipeline.predict_proba(pd.DataFrame([latest_data[available_features].values], columns=available_features))[0, 1]
        
        # Apply calibration with less aggressive transformation
        calibrated_prob = 0.3 + 0.7 * np.tanh(1.5 * (risk_prob - 0.5))
        
        # Additional adjustment based on recovery and physical capability
        if 'recovery_emboss_baseline_score' in latest_data and latest_data['recovery_emboss_baseline_score'] < 0:
            calibrated_prob = max(calibrated_prob, 0.4)  # Minimum 40% risk if recovery is poor
        
        if 'physical_benchmarkPct' in latest_data and latest_data['physical_benchmarkPct'] < 20:
            calibrated_prob = max(calibrated_prob, 0.5)  # Minimum 50% risk if physical capability is low
        
        # Prediction gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=calibrated_prob * 100,
            title={'text': "Team Injury Risk Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 75
                }
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Key metrics table
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'acwr_distance' in latest_data:
                acwr_val = latest_data['acwr_distance']
                st.metric("ACWR (Distance)", f"{acwr_val:.2f}", 
                         f"{acwr_val - 1.0:.2f}", 
                         delta_color="inverse" if acwr_val > 1.0 else "normal")
            else:
                st.metric("ACWR (Distance)", "N/A")
        
        with col2:
            if 'recovery_emboss_baseline_score' in latest_data:
                recovery_val = latest_data['recovery_emboss_baseline_score']
                st.metric("Recovery Score", f"{recovery_val:.1f}", 
                         f"{recovery_val - 70:.1f}", 
                         delta_color="normal" if recovery_val > 70 else "inverse")
            else:
                st.metric("Recovery Score", "N/A")
        
        with col3:
            if 'physical_overall_physical' in latest_data:
                physical_val = latest_data['physical_overall_physical']
                st.metric("Physical Capability", f"{physical_val:.1f}%", 
                         f"{physical_val - 80:.1f}", 
                         delta_color="normal" if physical_val > 80 else "inverse")
            else:
                st.metric("Physical Capability", "N/A")
        
        # Team recommendations
        st.subheader("Team Recommendations")
        
        # Get risk factors from latest data point
        risk_factors = []
        
        if 'acwr_distance' in latest_data and latest_data['acwr_distance'] > 1.3:
            risk_factors.append(f"High distance load ratio ({latest_data['acwr_distance']:.2f})")
        
        if 'acwr_accel_decel' in latest_data and latest_data['acwr_accel_decel'] > 1.3:
            risk_factors.append(f"High acceleration/deceleration load ratio ({latest_data['acwr_accel_decel']:.2f})")
        
        if 'distance_spike' in latest_data and latest_data['distance_spike'] > 0.3:
            risk_factors.append(f"Recent spike in distance load ({latest_data['distance_spike']*100:.1f}%)")
        
        if 'accel_decel_spike' in latest_data and latest_data['accel_decel_spike'] > 0.3:
            risk_factors.append(f"Recent spike in acceleration load ({latest_data['accel_decel_spike']*100:.1f}%)")
        
        if 'recovery_emboss_baseline_score' in latest_data and latest_data['recovery_emboss_baseline_score'] < 0:
            risk_factors.append(f"Poor recovery score ({latest_data['recovery_emboss_baseline_score']:.1f})")
        
        if 'physical_overall_physical' in latest_data and latest_data['physical_overall_physical'] < 0.5:
            risk_factors.append(f"Low physical capability ({latest_data['physical_overall_physical']*100:.1f}%)")
        
        if 'distance_monotony' in latest_data and latest_data['distance_monotony'] > 2.0:
            risk_factors.append(f"Monotonous training load ({latest_data['distance_monotony']:.1f})")
        
        # Create recommendations based on risk factors and evidence-based guidelines
        recommendations = []
        if "High distance load ratio" in " ".join(risk_factors):
            recommendations.append("Reduce overall training volume by 15-30% for 7-10 days")
            recommendations.append("Adopt a taper strategy with gradually decreasing volume")
        
        if "High acceleration/deceleration load ratio" in " ".join(risk_factors):
            recommendations.append("Decrease high-intensity acceleration/deceleration work")
            recommendations.append("Replace with technical/tactical sessions at moderate intensity")
        
        if "Poor recovery score" in " ".join(risk_factors):
            recommendations.append("Implement 48-hour recovery protocol (active recovery, nutrition, sleep hygiene)")
            recommendations.append("Use cold water immersion or contrast therapy after training")
        
        if "Low physical capability" in " ".join(risk_factors):
            recommendations.append("Focus on strength and conditioning fundamentals before increasing load")
            recommendations.append("Address specific movement pattern deficiencies")
        
        if "Monotonous training load" in " ".join(risk_factors):
            recommendations.append("Introduce training variation: alternate high and low intensity days")
            recommendations.append("Schedule a complete rest day every 7-10 days")
        
        if "Recent spike in distance load" in " ".join(risk_factors) or "Recent spike in acceleration load" in " ".join(risk_factors):
            recommendations.append("Implement a 'deload' week with 40-50% reduction in volume")
            recommendations.append("Avoid consecutive high-intensity training days")
        
        if not recommendations:
            recommendations.append("Maintain current training approach - good balance of load and recovery")
            recommendations.append("Focus on technical development during this period of low risk")
        
        # Display risk assessment
        if calibrated_prob > 0.75:  # Very high risk
            risk_factors_text = ""
            for factor in risk_factors:
                risk_factors_text += f"- {factor}\n"
            
            recommendations_text = ""
            for rec in recommendations:
                recommendations_text += f"- {rec}\n"
                
            st.error(f"""
            ### 🚨 HIGH RISK ALERT - {calibrated_prob*100:.1f}% Risk Probability
            
            **Key Risk Factors:**
            {risk_factors_text}
            
            **Evidence-Based Recommendations:**
            {recommendations_text}
            """)
        elif calibrated_prob > 0.5:  # Moderate risk
            risk_factors_text = ""
            for factor in risk_factors:
                risk_factors_text += f"- {factor}\n"
            
            recommendations_text = ""
            for rec in recommendations:
                recommendations_text += f"- {rec}\n"
                
            st.warning(f"""
            ### ⚠️ ELEVATED RISK LEVEL - {calibrated_prob*100:.1f}% Risk Probability
            
            **Key Risk Factors:**
            {risk_factors_text}
            
            **Evidence-Based Recommendations:**
            {recommendations_text}
            """)
        elif calibrated_prob > 0.25:  # Low-moderate risk
            risk_factors_text = ""
            for factor in risk_factors:
                risk_factors_text += f"- {factor}\n"
            
            recommendations_text = ""
            for rec in recommendations:
                recommendations_text += f"- {rec}\n"
                
            st.info(f"""
            ### ℹ️ MODERATE RISK LEVEL - {calibrated_prob*100:.1f}% Risk Probability
            
            **Key Risk Factors:**
            {risk_factors_text}
            
            **Evidence-Based Recommendations:**
            {recommendations_text}
            """)
        else:  # Low risk
            risk_factors_text = "None identified" if not risk_factors else ""
            if risk_factors:
                for factor in risk_factors:
                    risk_factors_text += f"- {factor}\n"
            
            recommendations_text = ""
            for rec in recommendations:
                recommendations_text += f"- {rec}\n"
                
            st.success(f"""
            ### ✅ LOW RISK LEVEL - {calibrated_prob*100:.1f}% Risk Probability
            
            **Key Risk Factors:**
            {risk_factors_text}
            
            **Evidence-Based Recommendations:**
            {recommendations_text}
            """)
        
        # Additional medical guidelines based on scientific evidence
        st.subheader("Monitoring Guidelines")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Daily Monitoring
            - Perceived fatigue (1-10 scale)
            - Sleep quality (hours + quality)
            - Muscle soreness (1-10 scale by body region)
            - Resting heart rate & HRV (if available)
            - Hydration status
            """)
        
        with col2:
            st.markdown("""
            #### Weekly Assessments
            - Countermovement jump (power output)
            - Adductor squeeze test (groin strength)
            - Single-leg balance (30s)
            - Wellness questionnaire
            - Body mass monitoring
            """)

# Add download button for raw risk assessments
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8')

if st.button("Generate Risk Assessment Report"):
    # Create prediction for all dates
    df['risk_probability'] = pipeline.predict_proba(df[available_features].fillna(0))[:, 1]
    
    # Create a report dataframe
    columns_to_include = ['date', 'risk_probability', 'high_risk']
    
    for col in ['acwr_distance', 'recovery_emboss_baseline_score', 'distance_spike']:
        if col in df.columns:
            columns_to_include.append(col)
    
    report_df = df[columns_to_include].copy()
    
    csv = convert_df_to_csv(report_df)
    st.download_button(
        label="Download Risk Assessment CSV",
        data=csv,
        file_name='injury_risk_assessment.csv',
        mime='text/csv',
    )
