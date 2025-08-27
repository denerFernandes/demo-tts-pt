#!/bin/bash

# Sair em caso de erro
set -e

# Verificar se o modelo existe e pode ser carregado
python <<EOL
import os
import sys
import torch
from f5_tts import F5TTS

model_dir = "/app/models"

print("Verificando a existência do modelo...")
if not os.path.exists(os.path.join(model_dir, "config.json")):
    print("Erro: Arquivo de configuração do modelo não encontrado.")
    sys.exit(1)

print("Tentando carregar o modelo...")
try:
    F5TTS.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Modelo carregado com sucesso para verificação.")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    sys.exit(1)

EOL

# Se a verificação for bem-sucedida, iniciar o servidor
exec python server.py
