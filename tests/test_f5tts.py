from huggingface_hub import login
import os
from dotenv import load_dotenv
import soundfile as sf
from models.f5_tts_russian import f5_tts_russian

def hf_login():
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
    else:
        raise ValueError("Error: invalid hugging face authentification!")

def load_model(model_name):
    model = None
    if model_name == "F5_TTS_RUSSIAN":
        model = f5_tts_russian()
        model.load_model()
    else:
        pass
    return model


def generate_audio(model, text, reference_audio, reference_text):
    output_audio, sample_rate = None, None
    if model.NAME == "F5_TTS_RUSSIAN":
        ref = (
            reference_audio,
            reference_text
        )
        output_audio, sample_rate = model.generate(text, ref)
    else:
        pass
    return output_audio, sample_rate

model_instances = {
    "F5_TTS_RUSSIAN" : None, 
    "xTTS" : None, 
    "Maya1": None 
}

chosen_model = "F5_TTS_RUSSIAN"
input_text = "Делу время, потехе - час."
uploaded_audio = "data/ref_ru.wav"
reference_text = "Шла Саша по шоссе и сосала сушку."
gen_flag = True
audio, sr = None, None
if gen_flag:
    if None not in [chosen_model, input_text, uploaded_audio]:
        print("Ищем модель...")
        hf_login()
        print("Загружаем модель...")
        model_instances[chosen_model] = load_model(chosen_model)
        print("Генерируем аудио...")
        audio, sr = generate_audio(model_instances[chosen_model], input_text, uploaded_audio, reference_text)
        print("Готово!")
        
        if audio is not None:
            sf.write(file="data/ru_out.wav", data=audio, samplerate=sr)
        else:
            print("Что-то пошло не так во время генерации аудио: попробуйте ещё раз.")
    else:
        print("Вы ввели недостаточно данных для работы модели!")