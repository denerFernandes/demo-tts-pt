#!/bin/bash

# Sair em caso de erro
set -e

# Verificar se o modelo existe
if [ ! -f "/app/models/config.json" ]; then
    echo "Erro: Arquivo de configuração do modelo não encontrado."
    exit 1
fi

# Se a verificação for bem-sucedida, iniciar o servidor
exec python server.py