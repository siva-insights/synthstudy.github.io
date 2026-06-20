from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = "http://localhost:11434"


@app.get("/")
def home():
    return {"status": "OLSEDG backend is running"}


@app.get("/check-model/{model_name}")
def check_model(model_name: str):
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        response.raise_for_status()

        models = response.json().get("models", [])
        installed_models = [m["name"] for m in models]

        is_installed = model_name in installed_models

        return {
            "ollama_running": True,
            "model": model_name,
            "model_installed": is_installed,
            "ready_to_use": is_installed,
            "installed_models": installed_models
        }

    except requests.exceptions.RequestException:
        return {
            "ollama_running": False,
            "model": model_name,
            "model_installed": False,
            "ready_to_use": False,
            "message": "Ollama is not running. Start it using: ollama serve"
        }
