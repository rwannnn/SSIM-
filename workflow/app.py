# app.py
import streamlit as st
import cv2
import numpy as np
import wf2_mod as script
from datetime import datetime
import time

st.set_page_config(page_title="SSIM - Site Safety Intelligent Monitor", layout="wide")

# --- Session State ---
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "thread" not in st.session_state:
    st.session_state.thread = None

# --- UI ---
st.title("⬡ Site Safety Intelligent Monitor (SSIM)")
st.markdown("### Building Safety With Intelligent Eyes")

# Start/End Button
def toggle_system():
    if not st.session_state.is_running:
        st.session_state.is_running = True
        st.session_state.thread = script.run_main_threaded()
    else:
        st.session_state.is_running = False
        script.stop_main()

st.button("🟢 START" if not st.session_state.is_running else "🔴 STOP", on_click=toggle_system)

# --- Layout ---
col1, col2 = st.columns([2, 1])

# Camera Feed
with col1:
    st.subheader("Live Camera Feed")
    frame_placeholder = st.empty()

# Logs
with col2:
    st.subheader("System Activity Log")
    log_placeholder = st.empty()

# --- Initialize Log History ---
if "log_history" not in st.session_state:
    st.session_state.log_history = []

# --- Stream Updates Loop ---
while True:
    # Live frame
    frame = script.get_latest_frame()
    if frame is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    # Logs
    new_logs = script.get_latest_logs()
    if new_logs:
        # Append new logs to session history
        st.session_state.log_history.extend(new_logs)

        # Limit stored lines to 500 for performance
        if len(st.session_state.log_history) > 500:
            st.session_state.log_history = st.session_state.log_history[-500:]

        # Render scrollable box
        log_html = """
        <div style="height:500px;overflow-y:auto;font-size:1.1em;
                    background:#111;padding:10px;border-radius:8px;">
        """
        for line in st.session_state.log_history:
            time_str = datetime.now().strftime("%H:%M:%S")
            log_html += f"<div style='color:#FFC107;'><b>[{time_str}]</b> {line}</div>"
        log_html += "</div>"

        log_placeholder.markdown(log_html, unsafe_allow_html=True)

    if not st.session_state.is_running:
        break

    time.sleep(0.1)
    
