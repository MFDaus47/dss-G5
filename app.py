import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime, timedelta

# ==============================
# LOAD MODELS & METADATA
# ==============================
@st.cache_resource
def load_models():
    try:
        with open("model/lr.pkl", "rb") as f:
            lr = pickle.load(f)
        with open("model/rf.pkl", "rb") as f:
            rf = pickle.load(f)
        with open("model/gb.pkl", "rb") as f:
            gb = pickle.load(f)
        with open("model/ensemble_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        return lr, rf, gb, meta
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run `train_model.py` first.")
        st.stop()

lr, rf, gb, meta = load_models()
weights = meta["weights"]
mean_vol = meta["mean_volume"]
std_vol = meta["std_volume"]
min_vol = meta.get("min_volume", 0)
max_vol = meta.get("max_volume", 100)
data_type = meta.get("data_type", "unknown")
percentiles = meta.get("percentiles", {})

# ==============================
# ENSEMBLE PREDICTION FUNCTION
# ==============================
def ensemble_predict(X):
    """Weighted ensemble prediction"""
    pred_lr = lr.predict(X)[0]
    pred_rf = rf.predict(X)[0]
    pred_gb = gb.predict(X)[0]
    
    ensemble_pred = (
        weights["lr"] * pred_lr +
        weights["rf"] * pred_rf +
        weights["gb"] * pred_gb
    )
    return ensemble_pred, {"lr": pred_lr, "rf": pred_rf, "gb": pred_gb}

# ==============================
# ITERATIVE 24-HOUR FORECASTING
# ==============================
def forecast_24_hours(initial_lags, adjustments=None):
    """
    Forecast next 24 hours using iterative prediction
    """
    if adjustments is None:
        adjustments = {"multiplier": 1.0}
    
    history = list(initial_lags)
    predictions = []
    individual_preds = []
    
    for hour in range(24):
        X = pd.DataFrame([[history[-1], history[-2], history[-3], history[-4]]],
                        columns=["Lag01", "Lag02", "Lag03", "Lag04"])
        
        pred, individual = ensemble_predict(X)
        
        # Apply environmental adjustments
        adjusted_pred = pred * adjustments["multiplier"]
        
        # Ensure predictions stay within reasonable bounds
        adjusted_pred = np.clip(adjusted_pred, min_vol, max_vol)
        
        predictions.append(adjusted_pred)
        individual_preds.append(individual)
        
        history.append(adjusted_pred)
    
    return predictions, individual_preds

# ==============================
# STAFFING CALCULATION
# ==============================
def calculate_staffing_needs(predictions, base_staff, traffic_to_customer_ratio=1.0):
    """
    Calculate recommended staffing based on traffic predictions
    
    Args:
        predictions: 24-hour traffic predictions
        base_staff: Baseline staffing level
        traffic_to_customer_ratio: Ratio of traffic volume to customer visits
                                   (e.g., 0.01 = 1% of passing traffic becomes customers)
    """
    # Normalize predictions to z-scores
    z_scores = [(p - mean_vol) / std_vol for p in predictions]
    
    staffing_plan = []
    demand_levels = []
    expected_customers = []
    
    for hour, (pred, z_score) in enumerate(zip(predictions, z_scores)):
        # Determine demand level and multiplier based on z-score
        if z_score > 1.5:
            demand = "Very High"
            multiplier = 1.6
        elif z_score > 0.5:
            demand = "High"
            multiplier = 1.3
        elif z_score > -0.5:
            demand = "Medium"
            multiplier = 1.0
        elif z_score > -1.5:
            demand = "Low"
            multiplier = 0.8
        else:
            demand = "Very Low"
            multiplier = 0.6
        
        recommended = math.ceil(base_staff * multiplier)
        recommended = max(1, recommended)
        
        # Estimate customer flow
        customers = pred * traffic_to_customer_ratio
        
        staffing_plan.append(recommended)
        demand_levels.append(demand)
        expected_customers.append(customers)
    
    return staffing_plan, demand_levels, expected_customers

# ==============================
# UI HEADER
# ==============================
st.title("Regression-Powered Traffic Forecasting DSS")
st.write("""
This system uses **machine learning ensemble models** to forecast traffic patterns 
for the next 24 hours and recommend optimal staffing levels based on predicted customer flow.
""")

# Dataset info banner
if data_type == "raw_counts":
    st.info(f"📊 **Dataset**: Real vehicle counts | Range: {min_vol:.0f} - {max_vol:.0f} vehicles/hour | Samples: {meta.get('dataset_samples', 'N/A')}")
elif data_type == "normalized":
    st.info(f"📊 **Dataset**: Normalized traffic index (0-100 scale) | Samples: {meta.get('dataset_samples', 'N/A')}")
else:
    st.info(f"📊 **Dataset**: {meta.get('dataset_samples', 'N/A')} samples | Range: {min_vol:.0f} - {max_vol:.0f}")

# Show model performance
with st.expander("📊 Model Performance Metrics"):
    st.write("**Training Performance:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Linear Regression", f"RMSE: {meta['rmse']['lr']:.2f}")
    with col2:
        st.metric("Random Forest", f"RMSE: {meta['rmse']['rf']:.2f}")
    with col3:
        st.metric("Gradient Boosting", f"RMSE: {meta['rmse']['gb']:.2f}")
    
    st.write("**Ensemble Configuration:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"LR Weight: **{weights['lr']:.1%}**")
    with col2:
        st.write(f"RF Weight: **{weights['rf']:.1%}**")
    with col3:
        st.write(f"GB Weight: **{weights['gb']:.1%}**")
    
    if "ensemble_performance" in meta:
        st.write("**Ensemble Test Performance:**")
        perf = meta["ensemble_performance"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RMSE", f"{perf['rmse']:.2f}")
        with col2:
            st.metric("MAE", f"{perf['mae']:.2f}")
        with col3:
            st.metric("MAPE", f"{perf.get('mape', 0):.2f}%")

# ==============================
# USER INPUTS
# ==============================
st.subheader("📅 Forecast Configuration")

forecast_date = st.date_input(
    "Forecast Date", 
    datetime.now() + timedelta(days=1),
    help="Select the date you want to forecast for"
)

st.subheader("📊 Recent Business Activity")

activity_level = st.selectbox(
    "Provide the **most recent 4 hours** of traffic data?",
    ["Very Quiet", "Quiet", "Normal", "Busy", "Very Busy"]
)

activity_multiplier = {
    "Very Quiet": 0.4,
    "Quiet": 0.6,
    "Normal": 1.0,
    "Busy": 1.3,
    "Very Busy": 1.6
}[activity_level]

base_level = mean_vol * activity_multiplier

initial_lags = [
    base_level * 0.9,
    base_level * 0.95,
    base_level,
    base_level * 1.05
]


st.subheader("🌍 Environmental Conditions")

col1, col2 = st.columns(2)
with col1:
    road_type = st.selectbox(
        "Road Type", 
        ["Urban Arterial", "Highway"],
        help="Highway traffic typically 15-20% higher"
    )
with col2:
    condition = st.selectbox(
        "Special Conditions", 
        ["Normal", "Rainy Weather", "Minor Incident", "Major Incident", "Special Event"],
        help="Adjust predictions based on expected conditions"
    )

# Calculate adjustment multiplier
adjustment_multiplier = 1.0
if road_type == "Highway":
    adjustment_multiplier *= 1.15
if condition == "Rainy Weather":
    adjustment_multiplier *= 1.1
elif condition == "Minor Incident":
    adjustment_multiplier *= 1.25
elif condition == "Major Incident":
    adjustment_multiplier *= 1.5
elif condition == "Special Event":
    adjustment_multiplier *= 1.4

st.info(f"📊 Total traffic adjustment: **{adjustment_multiplier:.2f}x** (Road: {road_type}, Condition: {condition})")

st.subheader("👥 Business Configuration")

col1, col2 = st.columns(2)
with col1:
    base_staff = st.slider(
        "Baseline staffing level",
        min_value=1,
        max_value=50,
        value=5,
        help="Number of staff you typically schedule on a normal day"
    )

with col2:
    if data_type == "raw_counts":
        traffic_ratio = st.slider(
            "Traffic-to-Customer Conversion %",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="What % of passing traffic becomes customers? (e.g., 1% = 100 vehicles → 1 customer)"
        ) / 100
    else:
        traffic_ratio = st.slider(
            "Traffic Index Sensitivity",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="How sensitive is customer flow to traffic index changes?"
        )

# ==============================
# RUN FORECAST
# ==============================
if st.button("🔮 Generate 24-Hour Forecast", type="primary"):
    
    with st.spinner("Running AI forecast models..."):
        adjustments = {"multiplier": adjustment_multiplier}
        
        predictions, individual_preds = forecast_24_hours(initial_lags, adjustments)
        staffing_plan, demand_levels, expected_customers = calculate_staffing_needs(
            predictions, base_staff, traffic_ratio
        )
    
    st.success("✅ Forecast complete!")
    
    # ==============================
    # VISUALIZATION
    # ==============================
    st.subheader("📈 24-Hour Traffic & Staffing Forecast")
    
    hours = list(range(24))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    
    # Traffic predictions
    ax1.plot(hours, predictions, marker='o', linewidth=2.5, markersize=7, color='#1f77b4', label='Forecast')
    ax1.fill_between(hours, predictions, alpha=0.2, color='#1f77b4')
    ax1.axhline(y=mean_vol, color='gray', linestyle='--', alpha=0.6, linewidth=2, label='Historical Mean')
    ax1.axhline(y=mean_vol + std_vol, color='red', linestyle=':', alpha=0.5, linewidth=1.5, label='±1 Std Dev')
    ax1.axhline(y=mean_vol - std_vol, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
    
    # Add initial values
    init_hours = [-4, -3, -2, -1]
    ax1.plot(init_hours, initial_lags, 'go-', markersize=8, linewidth=2, label='Initial Input', alpha=0.7)
    
    if data_type == "raw_counts":
        ax1.set_ylabel('Traffic Volume (vehicles/hour)', fontsize=12, fontweight='bold')
    else:
        ax1.set_ylabel('Traffic Volume Index', fontsize=12, fontweight='bold')
    
    ax1.set_title('Predicted Traffic Pattern', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([min_vol * 0.9, max_vol * 1.1])
    
    # Staffing recommendations
    colors = ['#d62728' if d in ['Very High', 'High'] else 
              '#ff7f0e' if d == 'Medium' else 
              '#2ca02c' for d in demand_levels]
    
    ax2.bar(hours, staffing_plan, color=colors, alpha=0.75, edgecolor='black', linewidth=1.2)
    ax2.axhline(y=base_staff, color='blue', linestyle='--', linewidth=2.5, label='Baseline Staff', alpha=0.8)
    ax2.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Recommended Staff', fontsize=12, fontweight='bold')
    ax2.set_title('Dynamic Staffing Recommendations', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(hours)
    ax2.set_ylim([0, max(staffing_plan) * 1.2])
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # ==============================
    # KEY INSIGHTS
    # ==============================
    st.subheader("🎯 Key Insights")
    
    peak_hour = int(np.argmax(predictions))
    peak_value = predictions[peak_hour]
    peak_staff = staffing_plan[peak_hour]
    
    low_hour = int(np.argmin(predictions))
    low_value = predictions[low_hour]
    low_staff = staffing_plan[low_hour]
    
    total_staff_hours = sum(staffing_plan)
    baseline_staff_hours = base_staff * 24
    efficiency_change = ((total_staff_hours - baseline_staff_hours) / baseline_staff_hours) * 100
    
    avg_traffic = np.mean(predictions)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Peak Hour", 
            f"{peak_hour}:00",
            f"Traffic: {peak_value:.0f}"
        )
    
    with col2:
        st.metric(
            "Peak Staffing", 
            f"{peak_staff} staff",
            f"{demand_levels[peak_hour]}"
        )
    
    with col3:
        st.metric(
            "Avg Traffic", 
            f"{avg_traffic:.0f}",
            f"{((avg_traffic - mean_vol)/mean_vol*100):+.1f}% vs historical"
        )
    
    with col4:
        st.metric(
            "Total Staff-Hours", 
            f"{total_staff_hours}",
            f"{efficiency_change:+.1f}% vs baseline"
        )
    
    # ==============================
    # DETAILED HOURLY BREAKDOWN
    # ==============================
    with st.expander("📋 Detailed Hourly Breakdown"):
        breakdown_df = pd.DataFrame({
            'Hour': [f"{h:02d}:00" for h in hours],
            'Traffic Forecast': [f"{p:.0f}" for p in predictions],
            'Expected Customers': [f"{c:.1f}" for c in expected_customers] if data_type == "raw_counts" else ["-"] * 24,
            'Demand Level': demand_levels,
            'Recommended Staff': staffing_plan,
            'vs. Baseline': [f"{s - base_staff:+d}" for s in staffing_plan]
        })
        
        # Remove customer column if not applicable
        if data_type != "raw_counts":
            breakdown_df = breakdown_df.drop('Expected Customers', axis=1)
        
        st.dataframe(breakdown_df, use_container_width=True, height=400)
    
    # ==============================
    # ACTIONABLE RECOMMENDATIONS
    # ==============================
    st.subheader("💡 Actionable Recommendations")
    
    high_demand_hours = [h for h, d in enumerate(demand_levels) if d in ['Very High', 'High']]
    low_demand_hours = [h for h, d in enumerate(demand_levels) if d in ['Very Low', 'Low']]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**🔴 High Traffic Periods** ({len(high_demand_hours)} hours)")
        if high_demand_hours:
            st.write(f"• **Hours**: {', '.join([f'{h}:00' for h in high_demand_hours])}")
            st.write(f"• **Peak**: {peak_hour}:00 with {peak_staff} staff needed")
            st.write(f"• Schedule senior/experienced staff")
            st.write(f"• Prepare for {peak_value:.0f} traffic volume")
        else:
            st.write("• No high-traffic periods predicted")
    
    with col2:
        st.write(f"**🟢 Low Traffic Periods** ({len(low_demand_hours)} hours)")
        if low_demand_hours:
            st.write(f"• **Hours**: {', '.join([f'{h}:00' for h in low_demand_hours])}")
            st.write(f"• **Minimum**: {min(staffing_plan)} staff required")
            st.write(f"• Good for breaks, training, maintenance")
            st.write(f"• Expected traffic: {low_value:.0f}")
        else:
            st.write("• Consistent demand throughout the day")
    
    # Cost estimation
    with st.expander("💰 Cost-Benefit Analysis"):
        hourly_wage = st.number_input("Average hourly wage (RM)", value=15.0, step=1.0, min_value=1.0)
        
        baseline_cost = baseline_staff_hours * hourly_wage
        optimized_cost = total_staff_hours * hourly_wage
        savings = baseline_cost - optimized_cost
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Baseline Cost (Fixed)", f"RM {baseline_cost:.2f}", 
                     f"{base_staff} staff × 24 hours")
        with col2:
            st.metric("Optimized Cost (Dynamic)", f"RM {optimized_cost:.2f}",
                     f"{total_staff_hours} total staff-hours")
        
        if savings > 0:
            st.success(f"💰 **Potential savings: RM {savings:.2f}** ({(savings/baseline_cost)*100:.1f}% reduction)")
            st.write(f"Monthly savings estimate: **RM {savings * 30:.2f}**")
        else:
            st.warning(f"⚠️ **Additional investment: RM {abs(savings):.2f}** to handle expected high demand")
            st.write("This investment may be worthwhile to:")
            st.write("• Improve customer service during peak hours")
            st.write("• Reduce wait times and increase satisfaction")
            st.write("• Capture additional revenue opportunities")
    
    # ==============================
    # MODEL INSIGHTS
    # ==============================
    with st.expander("🤖 Model Prediction Details"):
        st.write("**How the models voted (first 6 hours):**")
        
        model_comparison = []
        for hour in range(min(6, 24)):
            ind = individual_preds[hour]
            model_comparison.append({
                'Hour': f"{hour}:00",
                'Linear Reg': f"{ind['lr']:.1f}",
                'Random Forest': f"{ind['rf']:.1f}",
                'Gradient Boost': f"{ind['gb']:.1f}",
                'Ensemble': f"{predictions[hour]:.1f}"
            })
        
        st.dataframe(pd.DataFrame(model_comparison), use_container_width=True)
        
        st.write(f"""
        **Ensemble Method**: Weighted average based on test performance
        - Models with lower error get higher weight
        - Combines strengths of linear, tree-based, and boosting approaches
        """)
    
    # ==============================
    # CONFIDENCE & LIMITATIONS
    # ==============================
    with st.expander("ℹ️ About This Forecast"):
        st.write(f"""
        **Data & Model Information:**
        - Dataset: {meta.get('dataset_samples', 'N/A')} hourly observations
        - Data type: {data_type.replace('_', ' ').title()}
        - Value range: {min_vol:.0f} - {max_vol:.0f}
        - Ensemble RMSE: {meta.get('ensemble_performance', {}).get('rmse', 'N/A'):.2f}
        - Ensemble R²: {meta.get('ensemble_performance', {}).get('r2', 'N/A'):.3f}
        
        **Current Forecast Settings:**
        - Recent business activity level: {activity_level}
        - Derived initial traffic level based on historical patterns
        - Environmental multiplier: {adjustment_multiplier:.2f}x
        - Road type: {road_type}
        - Special conditions: {condition}
        
        **Important Limitations:**
        - Predictions become less certain further into the 24-hour window
        - Iterative forecasting can accumulate errors over time
        - Unexpected events (major accidents, closures) not accounted for
        - Model trained on historical data; future patterns may differ
        - Environmental adjustments are estimates, not data-driven
        
        **Best Practices:**
        - Monitor actual conditions during the day
        - Be prepared to adjust staffing in real-time
        - Use as decision support, not sole decision-maker
        - Validate predictions against reality to improve over time
        - Consider maintaining a buffer for unexpected situations
        """)

# ==============================
# FOOTER
# ==============================
st.divider()
st.caption(f"Model trained on {meta.get('dataset_samples', 'N/A')} samples | Last updated: {datetime.now().strftime('%Y-%m-%d')}")