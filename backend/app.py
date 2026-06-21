from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI(title="OLSEDG Helper")

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
    return {
        "helper_running": True,
        "message": "OLSEDG Helper is running."
    }


@app.get("/health")
def health_check():
    return {
        "helper_running": True,
        "message": "OLSEDG Helper is ready."
    }


@app.get("/check-ollama")
def check_ollama():
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        response.raise_for_status()

        models = response.json().get("models", [])
        installed_models = [model["name"] for model in models]

        return {
            "helper_running": True,
            "ollama_running": True,
            "installed_models": installed_models,
            "message": "Ollama is running."
        }

    except requests.exceptions.RequestException:
        return {
            "helper_running": True,
            "ollama_running": False,
            "installed_models": [],
            "message": "Ollama is not running. Please start Ollama first."
        }


@app.get("/check-model/{model_name}")
def check_model(model_name: str):
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        response.raise_for_status()

        models = response.json().get("models", [])
        installed_models = [model["name"] for model in models]

        model_installed = model_name in installed_models

        return {
            "helper_running": True,
            "ollama_running": True,
            "model": model_name,
            "model_installed": model_installed,
            "ready_to_use": model_installed,
            "installed_models": installed_models,
            "message": (
                f"{model_name} is installed and ready to use."
                if model_installed
                else f"{model_name} is not installed locally."
            )
        }

    except requests.exceptions.RequestException:
        return {
            "helper_running": True,
            "ollama_running": False,
            "model": model_name,
            "model_installed": False,
            "ready_to_use": False,
            "installed_models": [],
            "message": "Ollama is not running. Please start Ollama first."
        }
