
import streamlit as st
import pandas as pd
import joblib
import os
import re
from datetime import datetime

project_path = "/content/drive/MyDrive/AI_Customer_Support_Ticket_Triage"

category_model = joblib.load(os.path.join(project_path, "category_model.pkl"))
urgency_model = joblib.load(os.path.join(project_path, "urgency_model.pkl"))

category_vectorizer = joblib.load(
    os.path.join(project_path, "category_vectorizer.pkl")
)

urgency_vectorizer = joblib.load(
    os.path.join(project_path, "urgency_vectorizer.pkl")
)

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
    page_title="AI Customer Support Triage",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Customer Support Ticket Triage")
st.write("AI-based ticket classification, urgency prediction and support routing.")

st.divider()

st.subheader("🎫 Enter Customer Ticket")

ticket = st.text_area(
    "Customer Ticket",
    placeholder="Example: My payment failed during checkout"
)

if st.button("🔍 Analyze Ticket", use_container_width=True):

    if not ticket.strip():
        st.warning("Please enter a customer ticket.")

    else:
        cleaned = clean_text(ticket)

        category_input = category_vectorizer.transform([cleaned])
        category = category_model.predict(category_input)[0]
        category_confidence = max(
            category_model.predict_proba(category_input)[0]
        )

        urgency_input = urgency_vectorizer.transform([cleaned])
        urgency = urgency_model.predict(urgency_input)[0]
        urgency_confidence = max(
            urgency_model.predict_proba(urgency_input)[0]
        )

        confidence = min(
            category_confidence,
            urgency_confidence
        )

        confidence_percent = confidence * 100

        queue = queue_map.get(
            category,
            "General Support Team"
        )

        if confidence < 0.60:
            status = "Human Review Required"
        else:
            status = "Automatically Routed"

        st.divider()
        st.subheader("📊 Prediction Result")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Category", category.title())
        c2.metric("Urgency", urgency.title())
        c3.metric("Confidence", f"{confidence_percent:.2f}%")
        c4.metric("Queue", queue)

        if confidence < 0.60:
            st.warning("⚠️ Human Review Required")
        else:
            st.success("✅ Automatically Routed")

        result = pd.DataFrame({
            "Ticket": [ticket],
            "Category": [category.title()],
            "Urgency": [urgency.title()],
            "Queue": [queue],
            "Confidence": [f"{confidence_percent:.2f}%"],
            "Status": [status]
        })

        st.subheader("📋 Current Prediction")

        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True
        )

        history_file = os.path.join(
            project_path,
            "prediction_history.csv"
        )

        history_data = pd.DataFrame({
            "Time": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "Ticket": [ticket],
            "Category": [category.title()],
            "Urgency": [urgency.title()],
            "Queue": [queue],
            "Confidence": [f"{confidence_percent:.2f}%"],
            "Status": [status]
        })

        if os.path.exists(history_file):
            history_data.to_csv(
                history_file,
                mode="a",
                header=False,
                index=False
            )
        else:
            history_data.to_csv(
                history_file,
                index=False
            )

        st.success("Prediction saved to history! ✅")

st.divider()

st.subheader("📚 Prediction History")

history_file = os.path.join(
    project_path,
    "prediction_history.csv"
)

if os.path.exists(history_file):

    history = pd.read_csv(history_file)

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📈 Dashboard Statistics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Tickets", len(history))

    c2.metric(
        "Billing",
        (history["Category"] == "Billing").sum()
    )

    c3.metric(
        "Technical",
        (history["Category"] == "Technical").sum()
    )

    c4.metric(
        "Human Reviews",
        (history["Status"] == "Human Review Required").sum()
    )

else:
    st.info("No prediction history yet.")

st.divider()

st.caption(
    "AI Customer Support Ticket Triage | TF-IDF + Logistic Regression"
)
