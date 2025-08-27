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
        self.download_model()
    
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
    
    def download_model(self):
        """Download automático completo do modelo do HuggingFace"""
        try:
            logger.info(f"📥 Iniciando download automático do modelo: {self.model_name}")
            
            # Verificar se o modelo já existe e tem arquivos
            if os.path.exists(self.model_dir) and os.listdir(self.model_dir):
                logger.info(f"📂 Modelo já existe em {self.model_dir}")
                existing_files = os.listdir(self.model_dir)
                logger.info(f"📋 Arquivos existentes: {existing_files}")
                
                # Verificar se tem arquivos essenciais
                essential_patterns = ['.pt', '.pth', '.bin', '.safetensors', 'config.json']
                has_model_file = any(
                    any(pattern in f.lower() for pattern in essential_patterns)
                    for f in existing_files
                )
                
                if has_model_file:
                    logger.info("✅ Arquivos de modelo encontrados, pulando download")
                    self.load_model()
                    return
                else:
                    logger.info("⚠️  Arquivos essenciais não encontrados, baixando novamente...")
            
            # Método 1: Download completo via snapshot_download (recomendado)
            try:
                logger.info("🔄 Tentando download completo via snapshot...")
                
                # Configurações do download
                download_config = {
                    'repo_id': self.model_name,
                    'local_dir': self.model_dir,
                    'local_dir_use_symlinks': False,
                    'resume_download': True,  # Continuar downloads interrompidos
                }
                
                # Adicionar token se necessário (para modelos privados)
                try:
                    from huggingface_hub import HfApi
                    api = HfApi()
                    # Tentar obter info do repositório para verificar se é público
                    repo_info = api.repo_info(repo_id=self.model_name)
                    logger.info(f"📊 Repositório encontrado: {repo_info.id}")
                    if hasattr(repo_info, 'private') and repo_info.private:
                        logger.warning("🔐 Repositório privado detectado - pode precisar de token")
                except Exception as api_error:
                    logger.warning(f"⚠️  Não foi possível verificar repositório: {api_error}")
                
                # Executar download
                model_path = snapshot_download(**download_config)
                logger.info(f"✅ Download completo realizado em: {model_path}")
                
                # Listar todos os arquivos baixados
                if os.path.exists(self.model_dir):
                    all_files = []
                    for root, dirs, files in os.walk(self.model_dir):
                        for file in files:
                            rel_path = os.path.relpath(os.path.join(root, file), self.model_dir)
                            all_files.append(rel_path)
                    
                    logger.info(f"📦 Total de arquivos baixados: {len(all_files)}")
                    logger.info(f"📋 Estrutura do modelo:")
                    for file in sorted(all_files):
                        file_size = os.path.getsize(os.path.join(self.model_dir, file))
                        size_mb = file_size / (1024 * 1024)
                        logger.info(f"   📄 {file} ({size_mb:.1f} MB)")
                
                self.load_model()
                return
                
            except Exception as snapshot_error:
                logger.warning(f"⚠️  Erro no download via snapshot: {snapshot_error}")
                logger.info("🔄 Tentando método alternativo...")
            
            # Método 2: Download via API do HuggingFace para descobrir arquivos
            try:
                logger.info("🔍 Descobrindo arquivos disponíveis no repositório...")
                
                from huggingface_hub import HfApi, hf_hub_url
                api = HfApi()
                
                # Obter lista de arquivos no repositório
                try:
                    repo_files = api.list_repo_files(repo_id=self.model_name)
                    logger.info(f"📋 Encontrados {len(repo_files)} arquivos no repositório:")
                    
                    # Categorizar arquivos por tipo
                    model_files = []
                    config_files = []
                    other_files = []
                    
                    for file_path in repo_files:
                        file_lower = file_path.lower()
                        if any(ext in file_lower for ext in ['.pt', '.pth', '.bin', '.safetensors']):
                            model_files.append(file_path)
                        elif any(name in file_lower for name in ['config', 'vocab', 'tokenizer']):
                            config_files.append(file_path)
                        else:
                            other_files.append(file_path)
                    
                    logger.info(f"🎯 Arquivos de modelo: {model_files}")
                    logger.info(f"⚙️  Arquivos de config: {config_files}")
                    logger.info(f"📄 Outros arquivos: {other_files[:10]}...")  # Mostrar apenas os primeiros 10
                    
                    # Baixar todos os arquivos automaticamente
                    total_files = len(repo_files)
                    downloaded_files = 0
                    failed_files = []
                    
                    logger.info(f"🚀 Iniciando download de {total_files} arquivos...")
                    
                    for i, file_path in enumerate(repo_files, 1):
                        try:
                            logger.info(f"📥 [{i}/{total_files}] Baixando: {file_path}")
                            
                            downloaded_path = hf_hub_download(
                                repo_id=self.model_name,
                                filename=file_path,
                                local_dir=self.model_dir,
                                local_dir_use_symlinks=False,
                                resume_download=True
                            )
                            
                            file_size = os.path.getsize(downloaded_path)
                            size_mb = file_size / (1024 * 1024)
                            logger.info(f"✅ [{i}/{total_files}] Concluído: {file_path} ({size_mb:.1f} MB)")
                            downloaded_files += 1
                            
                        except Exception as file_error:
                            logger.warning(f"❌ [{i}/{total_files}] Falha: {file_path} - {file_error}")
                            failed_files.append((file_path, str(file_error)))
                            continue
                    
                    logger.info(f"📊 Resultado do download:")
                    logger.info(f"   ✅ Sucesso: {downloaded_files}/{total_files} arquivos")
                    logger.info(f"   ❌ Falhas: {len(failed_files)}/{total_files} arquivos")
                    
                    if failed_files:
                        logger.warning("⚠️  Arquivos que falharam:")
                        for file_path, error in failed_files:
                            logger.warning(f"   📄 {file_path}: {error}")
                    
                    if downloaded_files > 0:
                        logger.info("✅ Download automático concluído com sucesso!")
                        self.load_model()
                        return
                    else:
                        raise Exception("Nenhum arquivo foi baixado com sucesso")
                        
                except Exception as api_error:
                    logger.error(f"❌ Erro ao acessar API do HuggingFace: {api_error}")
                    raise
                    
            except Exception as api_method_error:
                logger.error(f"❌ Erro no método via API: {api_method_error}")
                logger.info("🔄 Tentando método de fallback...")
            
            # Método 3: Fallback - tentar baixar arquivos comuns mesmo sem lista
            logger.info("🆘 Usando método de fallback para arquivos comuns...")
            
            common_files = [
                # Arquivos de modelo
                "pytorch_model.bin",
                "model.safetensors", 
                "model.pt",
                "model.pth",
                "checkpoint.pt",
                "best_model.pt",
                
                # Arquivos de configuração
                "config.json",
                "model_config.json",
                "training_args.json",
                
                # Arquivos de tokenizer
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "vocab.txt",
                "special_tokens_map.json",
                
                # Outros arquivos importantes
                "README.md",
                ".gitattributes",
                "requirements.txt"
            ]
            
            fallback_downloaded = 0
            for filename in common_files:
                try:
                    logger.info(f"🔍 Tentando baixar: {filename}")
                    file_path = hf_hub_download(
                        repo_id=self.model_name,
                        filename=filename,
                        local_dir=self.model_dir,
                        local_dir_use_symlinks=False
                    )
                    logger.info(f"✅ Fallback sucesso: {filename}")
                    fallback_downloaded += 1
                except Exception as fallback_error:
                    logger.debug(f"   ❌ {filename} não encontrado: {fallback_error}")
                    continue
            
            if fallback_downloaded > 0:
                logger.info(f"✅ Fallback baixou {fallback_downloaded} arquivos")
            else:
                logger.warning("⚠️  Nenhum arquivo foi baixado via fallback")
            
            # Tentar carregar independente do resultado
            self.load_model()
            
        except Exception as e:
            logger.error(f"❌ Erro crítico no download: {e}")
            logger.error(f"📂 Diretório atual: {self.model_dir}")
            logger.error(f"📋 Conteúdo: {os.listdir(self.model_dir) if os.path.exists(self.model_dir) else 'Diretório não existe'}")
            logger.warning("🔄 Continuando sem modelo - verifique conectividade e permissões")
            self.model = None
    
    def load_model(self):
        """Carrega o modelo F5-TTS com otimizações GPU"""
        try:
            logger.info("🔄 Carregando modelo F5-TTS...")
            
            # Limpar memória antes de carregar
            self.clear_gpu_memory()
            
            # MÉTODO 1: Tentar carregar F5-TTS oficial
            try:
                logger.info("📦 Tentando importar F5-TTS...")
                
                # Verificar se F5-TTS está instalado
                try:
                    import f5_tts
                    from f5_tts import F5TTS
                    logger.info("✅ F5-TTS importado com sucesso")
                    
                    # Configurar dtype baseado na GPU
                    if self.device == "cuda":
                        if self.use_mixed_precision:
                            torch_dtype = torch.float16
                            logger.info("🎯 Usando float16 para economia de memória")
                        else:
                            torch_dtype = torch.float32
                    else:
                        torch_dtype = torch.float32
                    
                    # Tentar carregar modelo
                    logger.info(f"🔄 Carregando de: {self.model_dir}")
                    self.model = F5TTS.from_pretrained(
                        self.model_dir,
                        torch_dtype=torch_dtype,
                        device_map=self.device if self.device == "cuda" else "auto"
                    )
                    
                    # Otimizar modelo para inferência
                    if self.device == "cuda" and hasattr(torch, 'compile'):
                        try:
                            self.model = torch.compile(self.model, mode="reduce-overhead")
                            logger.info("⚡ Modelo compilado com torch.compile")
                        except Exception as compile_error:
                            logger.warning(f"Não foi possível compilar modelo: {compile_error}")
                    
                    logger.info("✅ Modelo F5-TTS carregado com sucesso!")
                    
                except ImportError as import_error:
                    logger.error(f"❌ F5-TTS não encontrado: {import_error}")
                    logger.info("💡 Para instalar F5-TTS:")
                    logger.info("   pip install f5-tts")
                    logger.info("   ou")
                    logger.info("   pip install git+https://github.com/SWivid/F5-TTS.git")
                    raise
                
            except Exception as f5_error:
                logger.warning(f"⚠️  Erro ao carregar F5-TTS oficial: {f5_error}")
                
                # MÉTODO 2: Tentar carregar modelo genérico do HuggingFace
                logger.info("🔄 Tentando carregar como modelo genérico...")
                try:
                    from transformers import AutoModel, AutoTokenizer
                    
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
                    self.model = AutoModel.from_pretrained(
                        self.model_dir,
                        torch_dtype=torch.float16 if self.device == "cuda" and self.use_mixed_precision else torch.float32,
                        device_map=self.device if self.device == "cuda" else "auto"
                    )
                    
                    if self.device == "cuda":
                        self.model = self.model.cuda()
                    
                    logger.info("✅ Modelo genérico carregado com sucesso!")
                    
                except Exception as generic_error:
                    logger.error(f"❌ Erro ao carregar modelo genérico: {generic_error}")
                    
                    # MÉTODO 3: Carregar modelo PyTorch puro
                    logger.info("🔄 Tentando carregar como PyTorch puro...")
                    try:
                        model_files = [f for f in os.listdir(self.model_dir) if f.endswith(('.pt', '.pth', '.bin'))]
                        if model_files:
                            model_path = os.path.join(self.model_dir, model_files[0])
                            logger.info(f"📂 Carregando {model_path}")
                            
                            self.model = torch.load(
                                model_path, 
                                map_location=self.device,
                                weights_only=False
                            )
                            
                            if hasattr(self.model, 'eval'):
                                self.model.eval()
                            
                            logger.info("✅ Modelo PyTorch carregado com sucesso!")
                        else:
                            logger.error("❌ Nenhum arquivo de modelo encontrado")
                            self.model = None
                            
                    except Exception as pytorch_error:
                        logger.error(f"❌ Erro ao carregar modelo PyTorch: {pytorch_error}")
                        self.model = None
            
            # Log de memória após carregamento
            if self.model and self.device == "cuda":
                memory_info = self.get_gpu_memory_info()
                if memory_info:
                    logger.info(f"📊 Memória GPU após carregamento: {memory_info['usage_percent']}% usada")
            
            # Status final
            if self.model is not None:
                logger.info("✅ Modelo carregado e pronto para uso!")
                return True
            else:
                logger.error("❌ Não foi possível carregar o modelo")
                return False
            
        except Exception as e:
            logger.error(f"❌ Erro crítico ao carregar modelo: {e}")
            logger.error(f"📂 Conteúdo do diretório: {os.listdir(self.model_dir) if os.path.exists(self.model_dir) else 'Diretório não existe'}")
            self.model = None
            return False
    
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
            
            if self.model is None:
                logger.error("❌ Modelo não carregado")
                raise Exception("Modelo não está carregado")
            
            # Limpar memória antes da síntese
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Verificar tipo de modelo e sintetizar
            try:
                # Para F5-TTS oficial
                if hasattr(self.model, 'synthesize') or hasattr(self.model, 'infer'):
                    logger.info("🎯 Usando método de síntese do F5-TTS")
                    
                    with torch.cuda.amp.autocast(enabled=self.use_mixed_precision and self.device == "cuda"):
                        if reference_audio is not None and hasattr(self.model, 'synthesize_with_reference'):
                            # Clonagem de voz
                            audio = self.model.synthesize_with_reference(
                                text=text,
                                reference_audio=reference_audio,
                                sample_rate=self.sample_rate
                            )
                        elif hasattr(self.model, 'synthesize'):
                            # Síntese normal
                            audio = self.model.synthesize(
                                text=text,
                                sample_rate=self.sample_rate
                            )
                        elif hasattr(self.model, 'infer'):
                            # Método alternativo
                            audio = self.model.infer(
                                text=text,
                                ref_audio=reference_audio,
                                speed=1.0
                            )
                        else:
                            raise Exception("Método de síntese não encontrado no modelo")
                
                # Para modelos genéricos - implementação básica
                else:
                    logger.info("🎯 Usando síntese genérica")
                    # Aqui você implementaria a lógica específica do seu modelo
                    # Por enquanto, gerar áudio sintético para teste
                    duration = len(text) * 0.1  # ~0.1s por caractere
                    samples = int(duration * self.sample_rate)
                    audio = np.sin(2 * np.pi * 440 * np.arange(samples) / self.sample_rate) * 0.3
                    logger.warning("⚠️  Usando áudio sintético para teste - implemente lógica específica do modelo")
                
                # Converter para CPU para salvar
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()
                
                # Garantir formato correto
                if audio.ndim > 1:
                    audio = audio.squeeze()
                
                logger.info("✅ Síntese concluída com sucesso!")
                return audio
                
            except Exception as synthesis_error:
                logger.error(f"❌ Erro na síntese: {synthesis_error}")
                raise
            
        except Exception as e:
            logger.error(f"❌ Erro na síntese GPU: {e}")
            raise

# Resto do código Flask permanece o mesmo...
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
        'version': '1.0.1-gpu-fixed'
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

@app.route('/model-info', methods=['GET'])
def model_info():
    """Informações detalhadas do modelo"""
    info = {
        'model_loaded': tts_server.model is not None,
        'model_type': type(tts_server.model).__name__ if tts_server.model else None,
        'model_dir': tts_server.model_dir,
        'files_in_dir': os.listdir(tts_server.model_dir) if os.path.exists(tts_server.model_dir) else [],
        'device': tts_server.device
    }
    
    if tts_server.model:
        try:
            info['model_parameters'] = sum(p.numel() for p in tts_server.model.parameters()) if hasattr(tts_server.model, 'parameters') else 'N/A'
        except:
            info['model_parameters'] = 'N/A'
    
    return jsonify(info)

@app.route('/synthesize', methods=['POST'])
def synthesize():
    """Endpoint para síntese de voz GPU-otimizada"""
    try:
        # Log memória antes do processamento
        if tts_server.device == "cuda":
            mem_before = tts_server.get_gpu_memory_info()
            logger.info(f"💾 Memória GPU antes: {mem_before['usage_percent']}%" if mem_before else "💾 Memória GPU: informação não disponível")
        
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
            logger.info(f"💾 Memória GPU depois: {mem_after['usage_percent']}%" if mem_after else "💾 Memória GPU: informação não disponível")
        
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
        if memory:
            gpu_info = f"""
            <div class="container gpu-container">
                <h3>🎮 Status da GPU</h3>
                <p><strong>GPU:</strong> {tts_server.gpu_name}</p>
                <p><strong>Memória:</strong> {memory['usage_percent']}% usada ({memory['allocated_gb']:.1f}GB / {memory['total_gb']:.1f}GB)</p>
                <p><strong>CUDA:</strong> {torch.version.cuda}</p>
                <p><strong>Mixed Precision:</strong> {'✅' if tts_server.use_mixed_precision else '❌'}</p>
                <button onclick="clearGPUCache()">🧹 Limpar Cache GPU</button>
                <button onclick="showModelInfo()">📋 Info do Modelo</button>
            </div>
            """
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>F5-TTS GPU Server - Fixed</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
            .container {{ background: #f5f5f5; padding: 20px; border-radius: 10px; margin: 20px 0; }}
            .gpu-container {{ background: #e3f2fd; border-left: 4px solid #2196f3; }}
            .success-container {{ background: #d4edda; border-left: 4px solid #28a745; }}
            .error-container {{ background: #f8d7da; border-left: 4px solid #dc3545; }}
            textarea {{ width: 100%; height: 100px; margin: 10px 0; }}
            input[type="file"] {{ margin: 10px 0; }}
            button {{ background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }}
            button:hover {{ background: #0056b3; }}
            .status {{ padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .success {{ background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }}
            .error {{ background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }}
            audio {{ width: 100%; margin: 10px 0; }}
            .gpu-badge {{ background: #28a745; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; }}
            .fixed-badge {{ background: #ffc107; color: #212529; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; }}
            pre {{ background: #f8f9fa; padding: 10px; border-radius: 5px; overflow-x: auto; }}
        </style>
    </head>
    <body>
        <h1>🎙️ F5-TTS Voice Cloning Server 
            <span class="gpu-badge">GPU</span>
            <span class="fixed-badge">FIXED</span>
        </h1>
        
        {gpu_info}
        
        <div class="container">
            <h3>📊 Status do Servidor</h3>
            <p id="status">Carregando...</p>
            <button onclick="refreshStatus()">🔄 Atualizar Status</button>
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
        
        <div class="container">
            <h3>🔧 Diagnóstico</h3>
            <button onclick="showModelInfo()">📋 Informações do Modelo</button>
            <div id="modelInfo"></div>
        </div>
        
        <script>
        // Verificar status do servidor
        function refreshStatus() {{
            fetch('/health')
                .then(r => r.json())
                .then(data => {{
                    let gpu_info = '';
                    if (data.gpu) {{
                        gpu_info = `<br><strong>GPU:</strong> ${{data.gpu.name}}<br><strong>Memória GPU:</strong> ${{data.gpu.memory.usage_percent}}%`;
                    }}
                    
                    const statusElement = document.getElementById('status');
                    const modelStatus = data.model_loaded ? '✅ Carregado' : '❌ Não carregado';
                    const statusClass = data.model_loaded ? 'success-container' : 'error-container';
                    
                    statusElement.innerHTML = `
                        <strong>Status:</strong> ${{data.status}}<br>
                        <strong>Dispositivo:</strong> ${{data.device}}<br>
                        <strong>Modelo:</strong> ${{modelStatus}}<br>
                        <strong>Versão:</strong> ${{data.version}}${{gpu_info}}
                    `;
                    
                    statusElement.className = 'container ' + statusClass;
                }})
                .catch(e => {{
                    document.getElementById('status').innerHTML = `❌ Erro ao conectar: ${{e}}`;
                }});
        }}
        
        // Carregar status inicial
        refreshStatus();
        
        // Informações do modelo
        function showModelInfo() {{
            fetch('/model-info')
                .then(r => r.json())
                .then(data => {{
                    const info = `
                        <h4>📋 Informações do Modelo</h4>
                        <pre>${{JSON.stringify(data, null, 2)}}</pre>
                    `;
                    document.getElementById('modelInfo').innerHTML = info;
                }})
                .catch(e => {{
                    document.getElementById('modelInfo').innerHTML = `❌ Erro: ${{e}}`;
                }});
        }}
        
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
    logger.info("🌐 Iniciando servidor GPU corrigido...")
    app.run(host='0.0.0.0', port=8000, debug=False)