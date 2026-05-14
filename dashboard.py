# dashboard.py
# Run using:
#     streamlit run dashboard.py

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="Smart Traffic Dashboard",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 AI Smart Traffic Management Dashboard")
st.markdown(
    """
    Real-time analytics dashboard for the YOLOv8 + OpenCV + Arduino based
    adaptive traffic signal control system.
    """
)

# ==========================================================
# FILE SETTINGS
# ==========================================================
LOG_FILE = "traffic_log.csv"

# ==========================================================
# AUTO REFRESH
# ==========================================================
# Refresh every 5 seconds
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=5000, key="refresh")

# ==========================================================
# LOAD DATA
# ==========================================================
log_path = Path(LOG_FILE)

if not log_path.exists():
    st.warning(
        "Log file `traffic_log.csv` not found.\n\n"
        "Run your main Python traffic script first so that it creates the log file."
    )
    st.stop()

try:
    df = pd.read_csv(LOG_FILE)
except Exception as e:
    st.error(f"Error reading CSV file: {e}")
    st.stop()

if df.empty:
    st.warning("The log file exists but contains no data.")
    st.stop()

# Convert timestamp column if present
if "Timestamp" in df.columns:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

# ==========================================================
# REQUIRED COLUMNS
# ==========================================================
required_columns = [
    "Road A Count",
    "Road B Count",
    "Road A Score",
    "Road B Score",
    "Active Road",
    "Green Time",
    "Phase",
    "Emergency"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    st.error(
        "Missing required columns in traffic_log.csv:\n\n"
        + "\n".join(f"- {col}" for col in missing)
    )
    st.stop()

# Latest record
latest = df.iloc[-1]

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.header("Dashboard Controls")

# Handle very small CSV files
total_rows = len(df)

if total_rows <= 2:
    num_rows = total_rows
else:
    num_rows = st.sidebar.slider(
        "Rows to analyze",
        min_value=2,
        max_value=total_rows,
        value=min(200, total_rows),
        step=1
    )

filtered_df = df.tail(num_rows)

# ==========================================================
# KPI CARDS
# ==========================================================
st.subheader("📌 Live System Status")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Road A Count", int(latest["Road A Count"]))
col2.metric("Road B Count", int(latest["Road B Count"]))
col3.metric("Active Road", str(latest["Active Road"]))
col4.metric("Green Time (s)", int(latest["Green Time"]))
col5.metric("Emergency", str(latest["Emergency"]))

col6, col7, col8 = st.columns(3)
col6.metric("Road A Score", round(float(latest["Road A Score"]), 2))
col7.metric("Road B Score", round(float(latest["Road B Score"]), 2))
col8.metric("Phase", str(latest["Phase"]))

# ==========================================================
# TRAFFIC COUNT TREND
# ==========================================================
st.subheader("📈 Vehicle Count Trend")

if "Timestamp" in filtered_df.columns:
    fig_counts = px.line(
        filtered_df,
        x="Timestamp",
        y=["Road A Count", "Road B Count"],
        markers=True,
        title="Vehicle Counts Over Time"
    )
else:
    fig_counts = px.line(
        filtered_df.reset_index(),
        x="index",
        y=["Road A Count", "Road B Count"],
        markers=True,
        title="Vehicle Counts Over Time"
    )

st.plotly_chart(fig_counts, use_container_width=True)

# ==========================================================
# WEIGHTED SCORE TREND
# ==========================================================
st.subheader("⚖️ Weighted Traffic Score Trend")

if "Timestamp" in filtered_df.columns:
    fig_scores = px.line(
        filtered_df,
        x="Timestamp",
        y=["Road A Score", "Road B Score"],
        markers=True,
        title="Weighted Traffic Scores Over Time"
    )
else:
    fig_scores = px.line(
        filtered_df.reset_index(),
        x="index",
        y=["Road A Score", "Road B Score"],
        markers=True,
        title="Weighted Traffic Scores Over Time"
    )

st.plotly_chart(fig_scores, use_container_width=True)

# ==========================================================
# ACTIVE ROAD DISTRIBUTION
# ==========================================================
st.subheader("🚦 Signal Allocation Distribution")

road_counts = filtered_df["Active Road"].value_counts().reset_index()
road_counts.columns = ["Road", "Cycles"]

fig_pie = px.pie(
    road_counts,
    names="Road",
    values="Cycles",
    title="Percentage of Cycles Assigned to Each Road"
)

st.plotly_chart(fig_pie, use_container_width=True)

# ==========================================================
# GREEN TIME DISTRIBUTION
# ==========================================================
st.subheader("⏱️ Green Time Distribution")

fig_green = px.histogram(
    filtered_df,
    x="Green Time",
    nbins=10,
    title="Distribution of Green Signal Durations"
)

st.plotly_chart(fig_green, use_container_width=True)

# ==========================================================
# EMERGENCY EVENT HISTORY
# ==========================================================
st.subheader("🚨 Emergency Event History")

if "Emergency" in filtered_df.columns:
    # Keep only emergency rows
    emergency_rows = filtered_df[
        filtered_df["Emergency"].astype(str).str.upper().isin(["ON", "TRUE", "1"])
    ].copy()

    if emergency_rows.empty:
        st.success("No emergency events recorded.")
    else:
        # Mark the start of each new emergency event
        emergency_rows["New Event"] = (
            emergency_rows["Active Road"]
            != emergency_rows["Active Road"].shift()
        )

        # First row is always a new event
        emergency_rows.iloc[0, emergency_rows.columns.get_loc("New Event")] = True

        # Keep only rows where a new emergency started
        event_starts = emergency_rows[emergency_rows["New Event"]].copy()

        # Assign event numbers
        event_starts["Event No"] = range(1, len(event_starts) + 1)

        # Create readable description
        event_starts["Description"] = event_starts.apply(
            lambda row: f"{row['Event No']}. Emergency at Road {row['Active Road']}",
            axis=1
        )

        # Show each event
        for desc in event_starts["Description"]:
            st.error(desc)

        # Optional detailed table
        with st.expander("View Emergency Event Details"):
            st.dataframe(
                event_starts[
                    ["Event No", "Timestamp", "Active Road", "Description"]
                ],
                use_container_width=True
            )

# ==========================================================
# SUMMARY STATISTICS
# ==========================================================
st.subheader("📊 Summary Statistics")

summary = pd.DataFrame({
    "Metric": [
        "Average Road A Count",
        "Average Road B Count",
        "Average Road A Score",
        "Average Road B Score",
        "Average Green Time",
        "Maximum Green Time",
        "Minimum Green Time",
        "Total Cycles"
    ],
    "Value": [
        round(filtered_df["Road A Count"].mean(), 2),
        round(filtered_df["Road B Count"].mean(), 2),
        round(filtered_df["Road A Score"].mean(), 2),
        round(filtered_df["Road B Score"].mean(), 2),
        round(filtered_df["Green Time"].mean(), 2),
        int(filtered_df["Green Time"].max()),
        int(filtered_df["Green Time"].min()),
        len(filtered_df)
    ]
})

st.dataframe(summary, use_container_width=True)

# ==========================================================
# RAW DATA
# ==========================================================
st.subheader("🗂️ Raw Log Data")
st.dataframe(filtered_df, use_container_width=True)

# ==========================================================
# DOWNLOAD CSV
# ==========================================================
st.download_button(
    label="⬇️ Download Traffic Log CSV",
    data=df.to_csv(index=False),
    file_name="traffic_log.csv",
    mime="text/csv"
)