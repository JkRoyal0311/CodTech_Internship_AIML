import streamlit as st
from transformers import pipeline

# Page setup
st.set_page_config(page_title="Text Summarizer", page_icon="🧩", layout="centered")

st.title("🧠 Text Summarizer App")
st.write("Paste or type your text below, and get a summarized version instantly.")

# Load model (only once)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# Input box
user_input = st.text_area("Enter text to summarize", height=200, placeholder="Paste a long paragraph or article here...")

if st.button("Summarize"):
    if user_input.strip():
        with st.spinner("Summarizing..."):
            summary = summarizer(user_input, max_length=130, min_length=30, do_sample=False)
            st.subheader("✨ Summary:")
            st.write(summary[0]['summary_text'])
    else:
        st.warning("Please enter some text first.")
