# app.py
import streamlit as st
import os
import pandas as pd
from datetime import datetime
import numpy as np

# Import project logic
from record import record_user_audio
from extraction import extract_features
from testing import compare_user_voices

# ---------------- Settings ----------------
DATA_DIR = "data"
ADMIN_LOG = os.path.join("admin_logs", "overall_attendance.csv")
ADMIN_PASSWORD = "admin123"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(ADMIN_LOG), exist_ok=True) if os.path.dirname(ADMIN_LOG) else None

# ---------------- Helper functions ----------------
def log_attendance(user_id, decision, similarity, spoof_score):
    """Log user attendance in both user and admin logs."""
    username = user_id
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    user_log_dir = os.path.join(DATA_DIR, user_id, "logs")
    os.makedirs(user_log_dir, exist_ok=True)
    user_log_path = os.path.join(user_log_dir, "attendance.csv")

    # Log for user
    with open(user_log_path, "a") as f:
        f.write(f"{timestamp},{user_id},{decision},{similarity:.3f},{spoof_score:.2f}\n")

    # Log for admin
    with open(ADMIN_LOG, "a") as f:
        f.write(f"{timestamp},{user_id},{decision},{similarity:.3f},{spoof_score:.2f}\n")

def get_user_log(user_id):
    path = os.path.join(DATA_DIR, user_id, "logs", "attendance.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, header=None, names=["Timestamp", "UserID", "Decision", "Similarity", "SpoofScore"])
        return df
    return None

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="veriVoice Attendance", page_icon="🎙️", layout="centered")
st.title("🎙️ veriVoice — Voice-Based Attendance System")

mode = st.radio("Choose mode", ["Sign Up (New)", "Sign In (Existing)", "Admin"], index=1)

# ---------------- SIGN UP ----------------
if mode == "Sign Up (New)":
    st.header("🆕 Sign Up — Register Template Voice")
    user_id = st.text_input("Enter a new User ID", "")
    if st.button("🎤 Record Template Voice"):
        if not user_id.strip():
            st.error("Please enter a User ID")
        else:
            record_user_audio(user_id, mode="template")
            st.success("Template voice recorded successfully!")

    if st.button("🧠 Extract Template Features"):
        if not user_id.strip():
            st.error("Enter User ID first")
        else:
            extract_features(user_id, mode="template")
            st.success("Template features extracted and saved successfully!")

# ---------------- SIGN IN ----------------
elif mode == "Sign In (Existing)":
    st.header("👤 Sign In — Mark Attendance")
    user_id = st.text_input("Enter your User ID", "")
    if st.button("🎙️ Record Test Voice"):
        if not user_id.strip():
            st.error("Please enter User ID")
        else:
            record_user_audio(user_id, mode="test")
            st.success("Test voice recorded successfully!")

    if st.button("✅ Test and Mark Attendance"):
        if not user_id.strip():
            st.error("Please enter User ID")
        else:
            extract_features(user_id, mode="test")
            similarity, spoof_score, decision = compare_user_voices(user_id)
            st.write(f"**Similarity:** {similarity:.3f}")
            st.write(f"**Spoof Score:** {spoof_score:.2f}")
            if decision == "Present":
                st.success("✅ Marked PRESENT")
            else:
                st.error("❌ Marked ABSENT")
            log_attendance(user_id, decision, similarity, spoof_score)

    st.markdown("---")
    st.subheader("📅 View My Attendance")
    view_user = st.text_input("Enter your User ID again to view history", "")
    if st.button("📊 Show My Attendance"):
        df = get_user_log(view_user)
        if df is not None:
            st.dataframe(df)
        else:
            st.info("No attendance logs found for this user yet.")

# ---------------- ADMIN ----------------
elif mode == "Admin":
    st.header("🛠️ Admin Dashboard")
    pwd = st.text_input("Enter admin password", type="password")
    if pwd != ADMIN_PASSWORD:
        if pwd:
            st.error("Incorrect password")
        else:
            st.info("Enter admin password to access logs.")
    else:
        st.success("Admin authenticated ✅")
        if os.path.exists(ADMIN_LOG):
            df = pd.read_csv(ADMIN_LOG, header=None, names=["Timestamp", "UserID", "Decision", "Similarity", "SpoofScore"])
            st.dataframe(df)
            st.markdown("---")
            st.subheader("Summary")
            summary = df.groupby("Decision").size().reset_index(name="Count")
            st.table(summary)
        else:
            st.info("No attendance logs yet.")
