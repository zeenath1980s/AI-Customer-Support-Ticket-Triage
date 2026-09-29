import streamlit as st
import pandas as pd
import joblib
import re
import os

# Load models from GitHub repository
category_model = joblib.load("category_model.pkl")
urgency_model = joblib.load("urgency_model.pkl")
category_vectorizer = joblib.load("category_vectorizer.pkl")
urgency_vectorizer = joblib.load("urgency_vectorizer.pkl")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


queue_map = {
    "billing": "Billing Team",
    "technical": "Technical Support Team",
    "account": "Account Support Team",
    "product": "Product Support Team"
}


st.set_page_config(
    page_title="AI Customer Support Ticket Triage",
    page_icon="🎫",
    layout="wide"
)

st.title("🎫 AI Customer Support Ticket Triage")
st.write("AI-based support ticket classification and urgency prediction.")

ticket = st.text_area(
    "Enter customer support ticket:",
    placeholder="Example: My payment failed during checkout"
)

if st.button("🔍 Analyze Ticket"):

    if ticket.strip() == "":
        st.warning("Please enter a support ticket.")
    else:
        cleaned = clean_text(ticket)

        # Category prediction
        category_input = category_vectorizer.transform([cleaned])
        category = category_model.predict(category_input)[0]
        category_prob = max(
            category_model.predict_proba(category_input)[0]
        )

        # Urgency prediction
        urgency_input = urgency_vectorizer.transform([cleaned])
        urgency = urgency_model.predict(urgency_input)[0]
        urgency_prob = max(
            urgency_model.predict_proba(urgency_input)[0]
        )

        confidence = min(category_prob, urgency_prob)

        if confidence < 0.60:
            status = "⚠️ Human Review Required"
        else:
            status = "✅ Automatically Routed"

        queue = queue_map.get(category, "General Support Team")

        st.subheader("Prediction Result")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Category", category)

        with col2:
            st.metric("Urgency", urgency)

        with col3:
            st.metric("Queue", queue)

        with col4:
            st.metric("Confidence", f"{confidence:.1%}")

        st.info(status)

        # Save prediction history
        history_file = "prediction_history.csv"

        new_row = pd.DataFrame([{
            "Ticket": ticket,
            "Category": category,
            "Urgency": urgency,
            "Queue": queue,
            "Confidence": confidence,
            "Status": status
        }])

        if os.path.exists(history_file):
            history = pd.read_csv(history_file)
            history = pd.concat(
                [history, new_row],
                ignore_index=True
            )
        else:
            history = new_row

        history.to_csv(history_file, index=False)

st.divider()

st.subheader("📊 Prediction History")

if os.path.exists("prediction_history.csv"):
    history = pd.read_csv("prediction_history.csv")
    st.dataframe(history, use_container_width=True)

    st.subheader("Dashboard Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Tickets", len(history))

    with col2:
        st.metric(
            "High Urgency",
            len(history[history["Urgency"] == "high"])
        )

    with col3:
        st.metric(
            "Human Review",
            len(
                history[
                    history["Status"].str.contains("Human Review")
                ]
            )
        )
else:
    st.write("No predictions yet.")

st.caption("AI Customer Support Ticket Triage — Prototype")
