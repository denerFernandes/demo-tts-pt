#!/usr/bin/env python3
"""
Servidor F5-TTS Otimizado para GPU
Versão com suporte CUDA e otimizações de memória
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

import torch
import torchaudio
import librosa
import numpy as np
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import soundfile as sf
from huggingface_hub import hf_hub_download, snapshot_download

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPUOptimizedF5TTSServer:
    def __init__(self):
        self.model_name = "Tharyck/multispeaker-ptbr-f5tts"
        self.setup_device()
        self.sample_rate = 22050
        self.model = None
        self.model_dir = "/app/models"
        
        logger.info(f"🚀 Inicializando servidor F5-TTS GPU-otimizado")
        logger.info(f"📱 Dispositivo: {self.device}")
        
        # Configurar otimizações CUDA
        self.setup_cuda_optimizations()
        
        # Criar diretórios necessários
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs("/app/temp", exist_ok=True)
        
        # Baixar modelo na inicialização
        self.load_model()
    
    def setup_device(self):
        """Configurar dispositivo com verificações detalhadas"""
        if torch.cuda.is_available():
            self.device = "cuda"
            self.gpu_count = torch.cuda.device_count()
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            logger.info(f"🎮 GPU detectada: {self.gpu_name}")
            logger.info(f"📊 Memória GPU: {self.gpu_memory:.1f}GB")
            logger.info(f"🔢 GPUs disponíveis: {self.gpu_count}")
            
            # Verificar se há memória suficiente
            if self.gpu_memory < 4.0:
                logger.warning(f"⚠️  GPU com pouca memória ({self.gpu_memory:.1f}GB). Recomendado: 6GB+")
            
        else:
            self.device = "cpu"
            logger.warning("🖥️  GPU não disponível - usando CPU")
            logger.info("💡 Para usar GPU:")
            logger.info("   1. Instale drivers NVIDIA")
            logger.info("   2. Configure nvidia-container-toolkit") 
            logger.info("   3. Use docker-compose-gpu.yml")
    
    def setup_cuda_optimizations(self):
        """Configurar otimizações CUDA"""
        if self.device == "cuda":
            # Otimizações de memória
            torch.cuda.empty_cache()
            
            # Configurar mixed precision se disponível
            if hasattr(torch.cuda, 'amp'):
                self.use_mixed_precision = True
                logger.info("⚡ Mixed precision habilitada")
            else:
                self.use_mixed_precision = False
            
            # Configurar cuDNN para performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Configurar alocador de memória
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            
            logger.info("🔧 Otimizações CUDA configuradas")
    
    def get_gpu_memory_info(self):
        """Obter informações de memória GPU"""
        if self.device == "cuda":
            try:
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                total = self.gpu_memory
                
                return {
                    'allocated_gb': round(allocated, 2),
                    'reserved_gb': round(reserved, 2),
                    'total_gb': round(total, 2),
                    'free_gb': round(total - reserved, 2),
                    'usage_percent': round((reserved / total) * 100, 1)
                }
            except Exception as e:
                logger.warning(f"Não foi possível obter informações da memória da GPU: {e}")
                return None
        return None
    
    def clear_gpu_memory(self):
        """Limpar memória GPU"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("🧹 Memória GPU limpa")
    
    
    
    def load_model(self):
        """Carrega o modelo F5-TTS com otimizações GPU"""
        try:
            logger.info("🔄 Carregando modelo...")
            
            # Limpar memória antes de carregar
            self.clear_gpu_memory()
            
            # IMPLEMENTAÇÃO REAL: Descomente quando F5-TTS estiver disponível
            from f5_tts import F5TTS
            
            # Configurar dtype baseado na GPU
            if self.device == "cuda":
                if self.use_mixed_precision:
                    torch_dtype = torch.float16
                    logger.info("🎯 Usando float16 para economia de memória")
                else:
                    torch_dtype = torch.float32
            else:
                torch_dtype = torch.float32
            
            # Carregar modelo
            self.model = F5TTS.from_pretrained(
                self.model_dir,
                torch_dtype=torch_dtype,
                device_map=self.device
            )
            
            # Otimizar modelo para inferência
            if self.device == "cuda":
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("⚡ Modelo compilado com torch.compile")
            
            # Para demonstração, usar placeholder
            # self.model = "placeholder_model_gpu"
            
            # Log de memória após carregamento
            if self.model and self.device == "cuda":
                memory_info = self.get_gpu_memory_info()
                if memory_info:
                    logger.info(f"📊 Memória GPU após carregamento: {memory_info['usage_percent']}% usada")
            
            if self.model:
                logger.info("✅ Modelo carregado com sucesso!")
            else:
                logger.error("❌ Modelo não pode ser carregado.")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            logger.error(f"Arquivos no diretório do modelo: {os.listdir(self.model_dir)}")
            self.model = None
    
    def preprocess_audio_gpu(self, audio_file: str) -> torch.Tensor:
        """Pré-processa áudio de referência com GPU"""
        try:
            # Carregar áudio
            audio, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            # Converter para tensor e enviar para GPU
            audio_tensor = torch.FloatTensor(audio)
            
            if self.device == "cuda":
                audio_tensor = audio_tensor.cuda()
            
            # Normalizar
            audio_tensor = audio_tensor / torch.max(torch.abs(audio_tensor))
            
            # Remover silêncio (usando torch operations para GPU)
            # Implementação simplificada - usar librosa.effects.trim em produção
            
            # Garantir duração mínima/máxima
            min_samples = int(1.0 * self.sample_rate)  # 1 segundo mínimo
            max_samples = int(30.0 * self.sample_rate)  # 30 segundos máximo
            
            if audio_tensor.shape[0] < min_samples:
                # Repetir áudio se muito curto
                repeats = int(np.ceil(min_samples / audio_tensor.shape[0]))
                audio_tensor = audio_tensor.repeat(repeats)[:min_samples]
            
            if audio_tensor.shape[0] > max_samples:
                # Truncar se muito longo
                audio_tensor = audio_tensor[:max_samples]
            
            logger.info(f"🎵 Áudio processado: {audio_tensor.shape[0]/self.sample_rate:.2f}s")
            return audio_tensor
            
        except Exception as e:
            logger.error(f"❌ Erro no pré-processamento GPU: {e}")
            raise
    
    @torch.inference_mode()  # Otimização para inferência
    def synthesize_voice_gpu(self, text: str, reference_audio: Optional[torch.Tensor] = None) -> np.ndarray:
        """Sintetiza voz com otimizações GPU"""
        try:
            logger.info(f"🎤 Sintetizando com GPU: '{text[:50]}...'")
            
            # Limpar memória antes da síntese
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            if self.model and self.model != "placeholder_model_gpu":
                # IMPLEMENTAÇÃO REAL: Descomente quando F5-TTS estiver disponível
                with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                    if reference_audio is not None:
                        # Clonagem de voz
                        audio = self.model.synthesize_with_reference(
                            text=text,
                            reference_audio=reference_audio,
                            sample_rate=self.sample_rate
                        )
                    else:
                        # Síntese normal
                        audio = self.model.synthesize(
                            text=text,
                            sample_rate=self.sample_rate
                        )
                
                # Converter para CPU para salvar
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()
                
                return audio
            
        except Exception as e:
            logger.error(f"❌ Erro na síntese GPU: {e}")
            raise

# Criar app Flask
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# Inicializar servidor GPU-otimizado
tts_server = GPUOptimizedF5TTSServer()

@app.route('/health', methods=['GET'])
def health_check():
    """Verificação de saúde com info GPU"""
    gpu_info = tts_server.get_gpu_memory_info()
    
    response = {
        'status': 'healthy',
        'device': tts_server.device,
        'model_loaded': tts_server.model is not None,
        'version': '1.0.0-gpu'
    }
    
    if gpu_info:
        response['gpu'] = {
            'name': tts_server.gpu_name,
            'memory': gpu_info,
            'count': tts_server.gpu_count
        }
    
    return jsonify(response)

@app.route('/gpu-status', methods=['GET'])
def gpu_status():
    """Status detalhado da GPU"""
    if tts_server.device != "cuda":
        return jsonify({'error': 'GPU não disponível'}), 400
    
    return jsonify({
        'gpu_name': tts_server.gpu_name,
        'memory': tts_server.get_gpu_memory_info(),
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__,
        'mixed_precision': tts_server.use_mixed_precision
    })

@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Limpar cache GPU"""
    tts_server.clear_gpu_memory()
    return jsonify({'status': 'Cache GPU limpo'})

@app.route('/synthesize', methods=['POST'])
def synthesize():
    """Endpoint para síntese de voz GPU-otimizada"""
    try:
        # Log memória antes do processamento
        if tts_server.device == "cuda":
            mem_before = tts_server.get_gpu_memory_info()
            logger.info(f"💾 Memória GPU antes: {mem_before['usage_percent']}%")
        
        # Verificar texto
        text = request.form.get('text', '').strip()
        if not text:
            return jsonify({'error': 'Texto é obrigatório'}), 400
        
        logger.info(f"📝 Novo request GPU: {text[:50]}...")
        
        reference_audio = None
        
        # Processar áudio de referência
        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file.filename:
                filename = secure_filename(audio_file.filename)
                temp_path = f"/app/temp/{int(time.time())}_{filename}"
                audio_file.save(temp_path)
                
                logger.info(f"🎵 Processando áudio com GPU: {filename}")
                
                try:
                    reference_audio = tts_server.preprocess_audio_gpu(temp_path)
                except Exception as e:
                    logger.error(f"Erro no áudio de referência: {e}")
                    return jsonify({'error': f'Erro no áudio: {str(e)}'}), 400
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        
        # Sintetizar com GPU
        try:
            synthesized_audio = tts_server.synthesize_voice_gpu(text, reference_audio)
        except Exception as e:
            logger.error(f"Erro na síntese GPU: {e}")
            return jsonify({'error': f'Erro na síntese: {str(e)}'}), 500
        
        # Salvar resultado
        output_path = f"/app/temp/output_{int(time.time())}.wav"
        sf.write(output_path, synthesized_audio, tts_server.sample_rate)
        
        # Log memória após processamento
        if tts_server.device == "cuda":
            mem_after = tts_server.get_gpu_memory_info()
            logger.info(f"💾 Memória GPU depois: {mem_after['usage_percent']}%")
        
        logger.info("✅ Síntese GPU concluída com sucesso!")
        
        return send_file(
            output_path,
            as_attachment=True,
            download_name='synthesized_voice_gpu.wav',
            mimetype='audio/wav'
        )
        
    except Exception as e:
        logger.error(f"❌ Erro interno GPU: {e}")
        return jsonify({'error': f'Erro interno: {str(e)}'}), 500

@app.route('/clone', methods=['POST'])
def clone_voice():
    """Endpoint específico para clonagem de voz"""
    return synthesize()  # Redirecionar para endpoint principal

@app.route('/', methods=['GET'])
def index():
    """Página inicial com informações GPU"""
    gpu_info = ""
    if tts_server.device == "cuda":
        memory = tts_server.get_gpu_memory_info()
        gpu_info = f"""
        <div class="container">
            <h3>🎮 Status da GPU</h3>
            <p><strong>GPU:</strong> {tts_server.gpu_name}</p>
            <p><strong>Memória:</strong> {memory['usage_percent']}% usada ({memory['allocated_gb']:.1f}GB / {memory['total_gb']:.1f}GB)</p>
            <p><strong>CUDA:</strong> {torch.version.cuda}</p>
            <p><strong>Mixed Precision:</strong> {'✅' if tts_server.use_mixed_precision else '❌'}</p>
            <button onclick="clearGPUCache()">🧹 Limpar Cache GPU</button>
        </div>
        """
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>F5-TTS GPU Server</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
            .container {{ background: #f5f5f5; padding: 20px; border-radius: 10px; margin: 20px 0; }}
            .gpu-container {{ background: #e3f2fd; border-left: 4px solid #2196f3; }}
            textarea {{ width: 100%; height: 100px; margin: 10px 0; }}
            input[type="file"] {{ margin: 10px 0; }}
            button {{ background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }}
            button:hover {{ background: #0056b3; }}
            .status {{ padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .success {{ background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }}
            .error {{ background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }}
            audio {{ width: 100%; margin: 10px 0; }}
            .gpu-badge {{ background: #28a745; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; }}
        </style>
    </head>
    <body>
        <h1>🎙️ F5-TTS Voice Cloning Server <span class="gpu-badge">GPU</span></h1>
        
        {gpu_info}
        
        <div class="container">
            <h3>📊 Status do Servidor</h3>
            <p id="status">Carregando...</p>
        </div>
        
        <div class="container">
            <h3>🎤 Síntese de Voz GPU-Acelerada</h3>
            <form id="ttsForm" enctype="multipart/form-data">
                <label>📝 Texto:</label><br>
                <textarea id="text" placeholder="Digite o texto para sintetizar..." required></textarea><br>
                
                <label>🎵 Áudio de Referência (opcional para clonagem):</label><br>
                <input type="file" id="audio" accept="audio/*"><br>
                
                <button type="submit">🚀 Sintetizar com GPU</button>
            </form>
            
            <div id="result"></div>
        </div>
        
        <script>
        // Verificar status do servidor
        fetch('/health')
            .then(r => r.json())
            .then(data => {{
                let gpu_info = '';
                if (data.gpu) {{
                    gpu_info = `<br><strong>GPU:</strong> ${{data.gpu.name}}<br><strong>Memória GPU:</strong> ${{data.gpu.memory.usage_percent}}%`;
                }}
                document.getElementById('status').innerHTML = `
                    <strong>Status:</strong> ${{data.status}}<br>
                    <strong>Dispositivo:</strong> ${{data.device}}<br>
                    <strong>Modelo:</strong> ${{data.model_loaded ? '✅ Carregado' : '❌ Não carregado'}}<br>
                    <strong>Versão:</strong> ${{data.version}}${{gpu_info}}
                `;
            }});
        
        // Limpar cache GPU
        function clearGPUCache() {{
            fetch('/clear-cache', {{method: 'POST'}})
                .then(r => r.json())
                .then(data => alert('✅ Cache GPU limpo!'))
                .catch(e => alert('❌ Erro ao limpar cache'));
        }}
        
        // Formulário de síntese
        document.getElementById('ttsForm').onsubmit = async (e) => {{
            e.preventDefault();
            
            const formData = new FormData();
            formData.append('text', document.getElementById('text').value);
            
            const audioFile = document.getElementById('audio').files[0];
            if (audioFile) {{
                formData.append('audio', audioFile);
            }}
            
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = '<div class="status">🔄 Processando com GPU...</div>';
            
            try {{
                const response = await fetch('/synthesize', {{
                    method: 'POST',
                    body: formData
                }});
                
                if (response.ok) {{
                    const blob = await response.blob();
                    const url = URL.createObjectURL(blob);
                    
                    resultDiv.innerHTML = `
                        <div class="status success">✅ Síntese GPU concluída!</div>
                        <audio controls>
                            <source src="${{url}}" type="audio/wav">
                        </audio>
                        <br>
                        <a href="${{url}}" download="voice_cloned_gpu.wav">
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
    logger.info("🌐 Iniciando servidor GPU...")
    app.run(host='0.0.0.0', port=8000, debug=False)