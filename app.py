import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Health Assistant", page_icon="🩺")
st.title("🩺 AI Health Assistant")
st.caption("General health guidance only. Not medical advice.")

@st.cache_resource
def load_model():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=200
    )

model = load_model()

SYSTEM_PROMPT = """
You are a safe health assistant.
Give only general health advice.
Do not diagnose diseases.
Do not prescribe medicines.
Always suggest consulting a doctor.
"""

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a health question")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    prompt = SYSTEM_PROMPT + "\nUser: " + user_input
    response = model(prompt)[0]["generated_text"]

    answer = response

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
