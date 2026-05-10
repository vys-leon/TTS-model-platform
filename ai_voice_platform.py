import streamlit as st
import numpy as np
from huggingface_hub import login
import os
from dotenv import load_dotenv
import time
from transformers import logging
import tempfile
import soundfile as sf
import io
from models.f5_tts_russian import f5_tts_russian

@st.cache_resource
def hf_login():
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
    else:
        st.error(
            """
                 Error: invalid hugging face authentification!
            """
        )

@st.cache_resource
def load_model(model_name):
    model = None
    if model_name == "F5_TTS_RUSSIAN":
        model = f5_tts_russian()
        model.load_model()
    else:
        pass
    return model


@st.cache_resource
def generate_audio(_model, text, reference_audio, reference_text):
    output_audio, sample_rate = None, None
    if _model.NAME == "F5_TTS_RUSSIAN":
        ref = (
            reference_audio,
            reference_text
        )
        output_audio, sample_rate = _model.generate(text, ref)
    else:
        pass
    return output_audio, sample_rate

logging.set_verbosity_error()
model_instances = {
    "F5_TTS_RUSSIAN" : None, 
    "xTTS" : None, 
    "Maya1": None 
}

# audiogptplat.com
st.set_page_config(page_title="AI voice platform", page_icon="🎙️")
st.header("🎙️ AI voice platform")
st.markdown(
    """
    Добро пожаловать на наш сайт с ИИ моделями для генерации голоса! 
    
    На этом сайте вы можете сгенерировать ИИ-голос с помощью любой выбранной 
    вами модели. Просто выберите нужную вам модель из списка ниже, введите текст, который вы хотите озвучить,
    и нажмите кнопку "Сгенерировать".

    Вы можете попробовать также нашу функцию клонирования голоса! Для этого нужен .wav файл с записью нужного 
    голоса длительностью не более 12 секунд, а также его точная транскрипция: озвученный в аудио файле текст 
    вместе со знаками пунктуации.

    P.S. На данный момент:
    - работает только модель F5_TTS_RUSSIAN
    - сервис работает только с аудио формата wav
    - можно генерировать аудио только на русском языке
"""
)

# chose model, type text, upload voice for cloning reference
chosen_model = st.selectbox(
    label="Аудио модель", 
    options=model_instances.keys(),
    index=None,
    placeholder="Выберите аудио модель..."
)
input_text = st.text_input(label="Текст для озвучивания", placeholder="Введите текст для озвучивания...")
ref_audio, ref_sr = sf.read("data/ref_ru.wav")
with open("data/ref_ru.txt", 'r') as f_in:
    reference_text = f_in.readline()
voice_cloning = st.checkbox("Озвучить текст своим голосом?")
if voice_cloning:
    uploaded_audio = st.file_uploader("Референсное аудио для клонирования голоса")
    if uploaded_audio is not None:
        bytes_audio = uploaded_audio.getvalue()
        ref_audio, ref_sr = sf.read(io.BytesIO(bytes_audio))
    reference_text = st.text_input(label="Текст референсного аудио", placeholder="Введите текст, произнесённый в референсном аудио...")
gen_flag = st.button("Сгенерировать")
audio, sr = None, None
if gen_flag:
    if None not in [chosen_model, input_text] or \
    (voice_cloning and None not in [chosen_model, input_text, uploaded_audio, reference_text]):
        with st.status("Генерируем аудио..."):
            st.write("Ищем модель...")
            hf_login()
            st.write("Загружаем модель...")
            model_instances[chosen_model] = load_model(chosen_model)
            st.write("Генерируем аудио...")
            with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
                sf.write(tmp.name, ref_audio, ref_sr)
                audio, sr = generate_audio(model_instances[chosen_model], input_text, tmp.name, reference_text)
            st.write("Готово!")
        
        if audio is not None:
            st.audio(data=audio, sample_rate=sr)
        else:
            st.write("Что-то пошло не так во время генерации аудио: попробуйте ещё раз.")
        st.button("Попробовать ещё раз")
    else:
        st.write("Вы ввели недостаточно данных для работы модели!")