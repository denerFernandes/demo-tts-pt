
import os
from huggingface_hub import snapshot_download

model_name = "Tharyck/multispeaker-ptbr-f5tts"
model_dir = "/app/models"

if __name__ == "__main__":
    print(f"Baixando modelo: {model_name}")
    snapshot_download(
        repo_id=model_name,
        local_dir=model_dir,
        local_dir_use_symlinks=False
    )
    print("Modelo baixado com sucesso!")
