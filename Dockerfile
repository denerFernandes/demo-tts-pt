# Dockerfile.gpu - Versão com suporte NVIDIA GPU
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Metadados
LABEL maintainer="F5-TTS GPU Server"
LABEL description="Servidor F5-TTS com suporte NVIDIA GPU"
LABEL version="1.0-gpu"

# Variáveis de ambiente
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HUB_CACHE=/app/cache

# Instalar Python e dependências do sistema
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    libportaudio2 \
    portaudio19-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Criar link simbólico para python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Criar diretórios
WORKDIR /app
RUN mkdir -p /app/models /app/temp /app/cache

# Copiar requirements
COPY requirements.txt requirements.txt

# Instalar dependências Python
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt && \
    echo "Installing F5-TTS..." && \
    pip install --no-cache-dir --force-reinstall --no-deps git+https://github.com/SWivid/F5-TTS.git && \
    echo "F5-TTS installation complete." && \
    echo "Listing f5_tts library contents:" && \
    pip show f5-tts && \
    ls -l $(pip show f5-tts | grep Location | awk '{print $2}')/f5_tts && \
    rm -rf /root/.cache/pip

# Verificar permissões e conectividade
RUN ls -ld /app && ls -ld /app/models
RUN pip check



# Copiar código da aplicação
COPY server.py server.py

# Criar usuário não-root (mas manter acesso GPU)
RUN useradd -m -u 1000 -G video appuser && \
    chown -R appuser:appuser /app

# Script de verificação GPU
COPY <<EOF /app/check_gpu.py
#!/usr/bin/env python3
import torch
import sys

print("🔍 Verificando suporte GPU...")
print(f"PyTorch versão: {torch.__version__}")
print(f"CUDA disponível: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA versão: {torch.version.cuda}")
    print(f"Dispositivos GPU: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}")
        print(f"    Memória: {props.total_memory / (1024**3):.1f}GB")
        print(f"    Compute: {props.major}.{props.minor}")
else:
    print("❌ CUDA não está disponível!")
    print("Verifique:")
    print("1. Driver NVIDIA instalado")
    print("2. Docker com runtime nvidia")
    print("3. nvidia-container-toolkit instalado")
    sys.exit(1)

print("✅ GPU configurada corretamente!")
EOF

RUN chmod +x /app/check_gpu.py

USER appuser

# Expor porta
EXPOSE 8000

# Health check com verificação GPU
HEALTHCHECK --interval=30s --timeout=30s --start-period=90s --retries=3 \
    CMD python /app/check_gpu.py && curl -f http://localhost:8000/health || exit 1

# Verificar GPU na inicialização e iniciar servidor
CMD ["python", "server.py"]