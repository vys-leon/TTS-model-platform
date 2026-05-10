from ruaccent import RUAccent
from cached_path import cached_path
from f5_tts.infer.utils_infer import (
  infer_process,
  load_model,
  load_vocoder,
  preprocess_ref_audio_text,
)
from f5_tts.model import DiT
from f5_tts.model.utils import seed_everything
import os

class f5_tts_russian:
    def __init__(self):
        seed_everything(4442)
        
        self.NAME = "F5_TTS_RUSSIAN"
        self.DEVICE = 'cpu'
        self.WEIGHTS_PATH = 'hf://Misha24-10/F5-TTS_RUSSIAN/F5TTS_v1_Base_v4_winter/model_212000.safetensors'
        self.VOCAB_PATH = 'hf://Misha24-10/F5-TTS_RUSSIAN/F5TTS_v1_Base/vocab.txt'
        self.ACCENT_DICT = {
            "реке": "р+еке",
        }
    
    def load_model(self):
        self.vocoder = load_vocoder(device=self.DEVICE)
        ckpt_path = str(cached_path(self.WEIGHTS_PATH))
        vocab_path = str(cached_path(self.VOCAB_PATH))
        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
        self.model_obj = load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path, device=self.DEVICE)
        
        os.makedirs('./ruaccent_cache', exist_ok=True)
        os.environ['RUACCENT_CACHE_DIR'] = './ruaccent_cache'
        self.accentizer = RUAccent()
        self.accentizer.load(
            omograph_model_size='turbo3.1',
            use_dictionary=True,
            tiny_mode=False,
            custom_dict=self.ACCENT_DICT
        )

    def generate(self, text, ref):
        ref_file, ref_text = preprocess_ref_audio_text(ref[0], ref[1])
        gen_text = self.accentizer.process_all(text) + ' '


        wav, sr, _ = infer_process(
            ref_file,
            ref_text,
            gen_text,
            self.model_obj,
            self.vocoder,
            cross_fade_duration=0.15,
            nfe_step=64,
            speed=1,
            device=self.DEVICE,
        )

        return wav, sr