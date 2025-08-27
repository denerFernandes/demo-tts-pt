#!/usr/bin/env python3
"""
Servidor F5-TTS Otimizado para GPU
Versão com suporte CUDA e otimizações de memória - CORRIGIDO
"""

import os
import sys
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional
import time
import gc
import random

import torch
import torchaudio
import librosa
import numpy as np
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import soundfile as sf
from huggingface_hub import snapshot_download
import tqdm

# Imports from f5-tts example
from cached_path import cached_path
from f5_tts.infer.utils_infer import (
    hop_length,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
    target_sample_rate,
)
from f5_tts.model import DiT
from f5_tts.model.utils import seed_everything


# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# F5TTS Class from user's example
class F5TTS:
    def __init__(
        self,
        model_type="F5-TTS",
        ckpt_file="",
        vocab_file="",
        ode_method="euler",
        use_ema=True,
        vocoder_name="vocos",
        local_path=None,
        device=None,
    ):
        # Initialize parameters
        self.final_wave = None
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.seed = -1
        self.mel_spec_type = vocoder_name

        # Set device
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Load models
        self.load_vocoder_model(vocoder_name, local_path)
        self.load_ema_model(model_type, ckpt_file, vocoder_name, vocab_file, ode_method, use_ema)

    def load_vocoder_model(self, vocoder_name, local_path):
        self.vocoder = load_vocoder(vocoder_name, local_path is not None, local_path, self.device)

    def load_ema_model(self, model_type, ckpt_file, mel_spec_type, vocab_file, ode_method, use_ema):
        if model_type == "F5-TTS":
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            model_cls = DiT
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.ema_model = load_model(
            model_cls, model_cfg, ckpt_file, mel_spec_type, vocab_file, ode_method, use_ema, self.device
        )

    def infer(
        self,
        ref_file,
        ref_text,
        gen_text,
        show_info=print,
        progress=tqdm,
        target_rms=0.1,
        cross_fade_duration=0.15,
        sway_sampling_coef=-1,
        cfg_strength=2,
        nfe_step=32,
        speed=1.0,
        fix_duration=None,
        remove_silence=False,
        file_wave=None,
        file_spect=None,
        seed=-1,
    ):
        if seed == -1:
            seed = random.randint(0, sys.maxsize)
        seed_everything(seed)
        self.seed = seed

        ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text, device=self.device)

        wav, sr, spect = infer_process(
            ref_file,
            ref_text,
            gen_text,
            self.ema_model,
            self.vocoder,
            self.mel_spec_type,
            show_info=show_info,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=self.device,
        )

        if file_wave is not None:
            sf.write(file_wave, wav, self.target_sample_rate)
            if remove_silence:
                remove_silence_for_generated_wav(file_wave)

        return wav, sr, spect

class GPUOptimizedF5TTSServer:
    def __init__(self):
        self.model_name = "Tharyck/multispeaker-ptbr-f5tts"
        self.setup_device()
        self.f5tts = None
        self.model_dir = "/app/models"
        
        logger.info(f"🚀 Inicializando servidor F5-TTS GPU-otimizado")
        logger.info(f"📱 Dispositivo: {self.device}")
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs("/app/temp", exist_ok=True)
        
        self.download_model()
    
    def setup_device(self):
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"🎮 GPU detectada: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            logger.warning("🖥️  GPU não disponível - usando CPU")

    def download_model(self):
        try:
            logger.info(f"📥 Iniciando download do modelo: {self.model_name}")
            if not os.listdir(self.model_dir):
                snapshot_download(repo_id=self.model_name, local_dir=self.model_dir, local_dir_use_symlinks=False, resume_download=True)
                logger.info(f"✅ Download completo em: {self.model_dir}")
            else:
                logger.info(f"📂 Modelo já existe em {self.model_dir}, pulando download.")
            self.load_model()
        except Exception as e:
            logger.error(f"❌ Erro crítico no download: {e}")
            self.f5tts = None

    def load_model(self):
        try:
            logger.info("🔄 Carregando modelo F5-TTS...")
            self.clear_gpu_memory()
            
            ckpt_file = os.path.join(self.model_dir, 'model_last.safetensors')
            vocab_file = os.path.join(self.model_dir, 'vocab.txt')

            if not os.path.exists(ckpt_file):
                raise FileNotFoundError(f"Arquivo de checkpoint não encontrado: {ckpt_file}")
            if not os.path.exists(vocab_file):
                raise FileNotFoundError(f"Arquivo de vocabulário não encontrado: {vocab_file}")

            self.f5tts = F5TTS(
                ckpt_file=ckpt_file,
                vocab_file=vocab_file,
                device=self.device
            )
            
            logger.info("✅ Modelo F5-TTS carregado com sucesso!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro crítico ao carregar modelo: {e}")
            self.f5tts = None
            return False

    def clear_gpu_memory(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("🧹 Memória GPU limpa")

    @torch.inference_mode()
    def synthesize_voice_gpu(self, text: str, reference_audio_path: Optional[str] = None) -> np.ndarray:
        try:
            logger.info(f"🎤 Sintetizando: '{text[:50]}...'")
            
            if self.f5tts is None:
                raise Exception("Modelo não está carregado")
            
            self.clear_gpu_memory()

            wav, sr, spect = self.f5tts.infer(
                ref_file=reference_audio_path,
                ref_text="", # Let whisper transcribe
                gen_text=text,
                remove_silence=True,
            )
            
            logger.info("✅ Síntese concluída com sucesso!")
            return wav, sr
            
        except Exception as e:
            logger.error(f"❌ Erro na síntese: {e}")
            raise

# Criar app Flask
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# Inicializar servidor
tts_server = GPUOptimizedF5TTSServer()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'device': tts_server.device,
        'model_loaded': tts_server.f5tts is not None,
    })

@app.route('/synthesize', methods=['POST'])
def synthesize():
    try:
        text = request.form.get('text', '').strip()
        if not text:
            return jsonify({'error': 'Texto é obrigatório'}), 400
        
        logger.info(f"📝 Novo request: {text[:50]}...")
        
        reference_audio_path = None
        temp_path = None

        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file.filename:
                filename = secure_filename(audio_file.filename)
                temp_path = f"/app/temp/{int(time.time())}_{filename}"
                audio_file.save(temp_path)
                reference_audio_path = temp_path
                logger.info(f"🎵 Áudio de referência salvo em: {reference_audio_path}")

        if reference_audio_path is None:
            return jsonify({'error': 'Áudio de referência é obrigatório para clonagem de voz.'}), 400

        synthesized_audio, sample_rate = tts_server.synthesize_voice_gpu(text, reference_audio_path)
        
        output_path = f"/app/temp/output_{int(time.time())}.wav"
        sf.write(output_path, synthesized_audio, sample_rate)
        
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        return send_file(
            output_path,
            as_attachment=True,
            download_name='synthesized_voice.wav',
            mimetype='audio/wav'
        )
        
    except Exception as e:
        logger.error(f"❌ Erro interno: {e}")
        return jsonify({'error': f'Erro interno: {str(e)}'}), 500

@app.route('/', methods=['GET'])
def index():
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>F5-TTS Server</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
            .container {{ background: #f5f5f5; padding: 20px; border-radius: 10px; margin: 20px 0; }}
            textarea {{ width: 100%; height: 100px; margin: 10px 0; }}
            input[type="file"] {{ margin: 10px 0; }}
            button {{ background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }}
            button:hover {{ background: #0056b3; }}
            .status {{ padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .success {{ background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }}
            .error {{ background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }}
            audio {{ width: 100%; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>🎙️ F5-TTS Voice Cloning Server</h1>
        
        <div class="container">
            <h3>🎤 Síntese de Voz</h3>
            <form id="ttsForm" enctype="multipart/form-data">
                <label>📝 Texto:</label><br>
                <textarea id="text" placeholder="Digite o texto para sintetizar..." required></textarea><br>
                
                <label>🎵 Áudio de Referência (para clonagem):</label><br>
                <input type="file" id="audio" accept="audio/*" required><br>
                
                <button type="submit">🚀 Sintetizar</button>
            </form>
            
            <div id="result"></div>
        </div>
        
        <script>
        document.getElementById('ttsForm').onsubmit = async (e) => {{
            e.preventDefault();
            
            const formData = new FormData();
            formData.append('text', document.getElementById('text').value);
            
            const audioFile = document.getElementById('audio').files[0];
            if (audioFile) {{
                formData.append('audio', audioFile);
            }} else {{
                alert("Por favor, envie um áudio de referência.");
                return;
            }}
            
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = '<div class="status">🔄 Processando...</div>';
            
            try {{
                const response = await fetch('/synthesize', {{
                    method: 'POST',
                    body: formData
                }});
                
                if (response.ok) {{
                    const blob = await response.blob();
                    const url = URL.createObjectURL(blob);
                    
                    resultDiv.innerHTML = `
                        <div class="status success">✅ Síntese concluída!</div>
                        <audio controls>
                            <source src="${{url}}" type="audio/wav">
                        </audio>
                        <br>
                        <a href="${{url}}" download="synthesized_voice.wav">
                            <button>📥 Download</button>
                        </a>
                    `;
                }} else {{
                    const error = await response.json();
                    resultDiv.innerHTML = `<div class="status error">❌ Erro: ${{error.error}}</div>`;
                }}
            }} catch (error) {{
                resultDiv.innerHTML = `<div class="status error">❌ Erro de conexão: ${{error}}</div>`;
            }}
        }};
        </script>
    </body>
    </html>
    '''

if __name__ == '__main__':
    logger.info("🌐 Iniciando servidor...")
    app.run(host='0.0.0.0', port=8000, debug=False)
