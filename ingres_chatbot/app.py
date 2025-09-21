import streamlit as st
import ollama
from utils.llm_helper import get_llm_response
from utils.data_helper import load_data, query_by_district
from utils.voice_helper import recognize_speech_from_mic, convert_text_to_audio
import re
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

st.set_page_config(page_title="INGRES AI ChatBot", page_icon="💧", layout="wide", initial_sidebar_state="expanded")

def initial_model_setup():
    model_name = "qwen2:7b"
    try:
        model_list = ollama.list().get("models", [])
        if model_name not in [m.get("name") for m in model_list]:
            with st.spinner("Performing first-time setup..."):
                ollama.pull(model_name)
            st.success(f"Model '{model_name}' is ready!")
        st.session_state.model_setup_complete = True
    except Exception as e:
        st.error(f"Failed to connect to Ollama. Please ensure Ollama is running. Error: {e}")
        st.session_state.model_setup_complete = False

if "model_setup_complete" not in st.session_state:
    st.session_state.model_setup_complete = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "new_prompt_to_process" not in st.session_state:
    st.session_state.new_prompt_to_process = False

LANGUAGE_MAPPING = {"English": {"llm_lang": "English", "tts_lang": "en"}, "हिंदी (Hindi)": {"llm_lang": "Hindi", "tts_lang": "hi"}, "తెలుగు (Telugu)": {"llm_lang": "Telugu", "tts_lang": "te"}}
with st.sidebar:
    st.header("INGRES AI ChatBot")
    st.markdown("A Smart India Hackathon 2025 Project")
    st.divider()
    selected_language = st.selectbox("Choose a language", list(LANGUAGE_MAPPING.keys()))

st.title("💧 INGRES AI ChatBot")

if not st.session_state.model_setup_complete:
    initial_model_setup()
    if st.button("Continue to Chat"):
        st.rerun()
else:
    def process_prompt(prompt):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.new_prompt_to_process = True
        st.rerun()

    if not st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": {"text": "Welcome! How can I help?"}})

    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # ... (rendering logic as before) ...
            content = message.get("content", {})
            if isinstance(content, dict):
                st.markdown(content.get("text", ""))
            else:
                st.markdown(content)


    if st.session_state.new_prompt_to_process:
        st.session_state.new_prompt_to_process = False
        prompt = st.session_state.messages[-1]["content"]
        with st.spinner("Thinking..."):
            llm_lang = LANGUAGE_MAPPING[selected_language]["llm_lang"]
            if "data for" in prompt.lower():
                district_name = prompt.lower().split("data for")[-1].strip()
                df = load_data("ingres_chatbot/data/groundwater_data.csv")
                data_record = query_by_district(df, district_name)
                if data_record is not None:
                    response_text = f"Data for {data_record['district']}: Level is {data_record['groundwater_level_meters']}m."
                    response_content = {"text": response_text}
                else:
                    response_content = {"text": f"No data for {district_name}."}
            else:
                system_prompt = f"You are a helpful assistant. Please respond exclusively in {llm_lang}."
                full_prompt = f"{system_prompt}\n\nUser question: {prompt}"
                response_text = get_llm_response(full_prompt)
                response_content = {"text": response_text}
        st.session_state.messages.append({"role": "assistant", "content": response_content})
        st.rerun()

    prompt_input = st.chat_input("Ask me anything...")
    if prompt_input:
        process_prompt(prompt_input)
