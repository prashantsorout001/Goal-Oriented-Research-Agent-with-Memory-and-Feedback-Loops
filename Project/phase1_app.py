# app.py
import streamlit as st
import requests

API_KEY = "API_KEY"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
rtyui
st.title("Goal-Oriented Research Agent (Phase 1)")
st.write("Using GPT-5 via OpenRouter API")

user_input = st.text_area("Ask me anything:", "")

if st.button("Generate Answer"):
    if user_input.strip() == "":
        st.warning("Please enter a question.")
    else:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "openai/gpt-5",   # GPT-5 model on OpenRouter
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_input}
            ],
            "max_tokens": 1000  # ✅ limit tokens for free credits
        }

        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            data = response.json()

            if "choices" in data:
                answer = data["choices"][0]["message"]["content"]
                st.success(answer)
            else:
                st.error(f"Error: {data}")
        except Exception as e:
            st.error(f"Request failed: {e}")

