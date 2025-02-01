import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(csv_file):
    return pd.read_csv(csv_file)

def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def compute_embeddings(model, texts):
    return model.encode(texts)

def get_best_response(query, model, pattern_embeddings, data, threshold=0.5):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, pattern_embeddings)[0]
    max_sim = np.max(similarities)
    best_idx = np.argmax(similarities)
    if max_sim >= threshold:
        return data.iloc[best_idx]["response"]
    else:
        return "I'm sorry, I don't have information on that topic. Could you please rephrase your question?"

data = load_data("solar_data.csv")
model = load_model()
pattern_list = data["pattern"].tolist()
pattern_embeddings = compute_embeddings(model, pattern_list)

st.title("☀️Solar Panel Chatbot")

if "history_list" not in st.session_state:
    st.session_state.history_list = []
if "selected_question" not in st.session_state:
    st.session_state.selected_question = "-- Select a Question --"

def process_dropdown():
    if st.session_state.selected_question != "-- Select a Question --":
        response = get_best_response(st.session_state.selected_question, model, pattern_embeddings, data)
        st.session_state.history_list.append({"user": st.session_state.selected_question, "bot": response})
        st.session_state.selected_question = "-- Select a Question --"

with st.form(key="query_form", clear_on_submit=True):
    user_input = st.text_input("Enter your question about solar panels:", key="user_query")
    submit_button = st.form_submit_button(label="Submit")

if submit_button and user_input:
    response = get_best_response(user_input, model, pattern_embeddings, data)
    st.session_state.history_list.append({"user": user_input, "bot": response})

st.selectbox("Or select a predefined question:", options=["-- Select a Question --"] + pattern_list, key="selected_question", on_change=process_dropdown)

for chat in st.session_state.history_list:
    st.markdown(f"**User:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")
