import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open('/Users/rubaka/idaourproject/addiction_model_pkl', 'rb') as file:
    model = pickle.load(file)

# Mapping from options to values
likert_map = {
    "Never": 0,
    "Rarely": 1,
    "Sometimes": 2,
    "Often": 3,
    "Always": 4,
    "Less than 1 hour": 0,
    "1–2 hours": 1,
    "3–4 hours": 2,
    "5–6 hours": 3,
    "More than 6 hours": 4,
    "Rarely (Once or twice a day)": 0,
    "Occasionally (Every few hours)": 1,
    "Frequently (Once an hour)": 2,
    "Very Frequently (Every 30 minutes)": 3,
    "Constantly (Every few minutes)": 4,
    "Not at all": 0,
    "Slightly": 1,
    "Moderately": 2,
    "Strongly": 3,
    "Extremely": 4,
    "Not difficult at all": 0,
    "Slightly difficult": 1,
    "Moderately difficult": 2,
    "Very difficult": 3
}

st.set_page_config(page_title="Social Media Addiction Predictor", layout="wide")
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>Social Media Addiction Predictor</h1>", unsafe_allow_html=True)
st.write("### Please answer the following 12 questions:")

questions = [
    ("How many hours do you spend on social media daily?", ["Less than 1 hour", "1–2 hours", "3–4 hours", "5–6 hours", "More than 6 hours"]),
    ("How often do you check your phone for social media notifications?", ["Rarely (Once or twice a day)", "Occasionally (Every few hours)", "Frequently (Once an hour)", "Very Frequently (Every 30 minutes)", "Constantly (Every few minutes)"]),
    ("How strongly do you feel the fear of missing out on social media updates?", ["Not at all", "Slightly", "Moderately", "Strongly", "Extremely"]),
    ("How much do your friends and peers influence your social media usage?", ["Not at all", "Slightly", "Moderately", "Strongly", "Extremely"]),
    ("How many social media platforms do you actively use?", ["1", "2–3", "4–5", "6–7", "More than 7"]),
    ("How often do you use social media before sleeping?", ["Never", "Rarely", "Sometimes", "Often", "Always"]),
    ("How difficult do you find it to stay away from social media?", ["Not difficult at all", "Slightly difficult", "Moderately difficult", "Very difficult", "Extremely difficult"]),
    ("How much do features like reels, auto-play, and infinite scrolling impact your social media usage?", ["Not at all", "Slightly", "Moderately", "Strongly", "Extremely"]),
    ("How actively do you follow social media trends and influencers?", ["Not at all", "Slightly", "Moderately", "Strongly", "Extremely"]),
    ("How often do you use social media to pass time when bored?", ["Never", "Rarely", "Sometimes", "Often", "Always"]),
    ("How often do you open social media apps without planning to?", ["Never", "Rarely", "Sometimes", "Often", "Always"]),
    ("How often do you use social media to escape real-life stress or problems?", ["Never", "Rarely", "Sometimes", "Often", "Always"]),
]

# Create 4 rows of 3 columns
responses = []
for row in range(4):
    cols = st.columns(3)
    for col in range(3):
        index = row * 3 + col
        q_text, options = questions[index]
        answer = cols[col].selectbox(f"{index+1}. {q_text}", options)
        score = likert_map.get(answer, 0)
        responses.append(score)

# Predict on click
if st.button("🧠 Predict Addiction Level"):
    df_input = pd.DataFrame([responses], columns=[
        "Hours Spent",
        "Notifications Check",
        "FOMO Rating",
        "Influence Usage Rating",
        "Platform Count",
        "Surf Before Sleep",
        "Social Media Urge",
        "Features Impact",
        "Follow Trends",
        "Boredom Scrolling",
        "Unplanned App Usage",
        "Stress Escape Usage"
    ])
    prediction = model.predict(df_input)[0]
    st.success(f" You are classified as: *{prediction}*")