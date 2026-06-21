import subprocess
import requests
import threading
import webbrowser
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

OLLAMA_URL = "http://localhost:11434"

app = FastAPI(title="OLSEDG Helper")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def is_ollama_running():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def start_ollama():
    if is_ollama_running():
        return True

    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        for _ in range(10):
            time.sleep(1)
            if is_ollama_running():
                return True

        return False

    except FileNotFoundError:
        return False


@app.get("/")
def home():
    return {
        "helper_running": True,
        "message": "OLSEDG Helper is running."
    }


@app.get("/health")
def health():
    return {
        "helper_running": True,
        "message": "OLSEDG Helper is ready."
    }


@app.get("/check-ollama")
def check_ollama():
    running = start_ollama()

    if not running:
        return {
            "helper_running": True,
            "ollama_running": False,
            "message": "Ollama is not installed or could not be started."
        }

    r = requests.get(f"{OLLAMA_URL}/api/tags")
    models = r.json().get("models", [])
    installed_models = [m["name"] for m in models]

    return {
        "helper_running": True,
        "ollama_running": True,
        "installed_models": installed_models,
        "message": "Ollama is running."
    }


@app.get("/check-model/{model_name}")
def check_model(model_name: str):
    running = start_ollama()

    if not running:
        return {
            "helper_running": True,
            "ollama_running": False,
            "model_installed": False,
            "ready_to_use": False,
            "message": "Ollama is not installed or could not be started."
        }

    r = requests.get(f"{OLLAMA_URL}/api/tags")
    models = r.json().get("models", [])
    installed_models = [m["name"] for m in models]

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


def open_browser():
    time.sleep(1)
    webbrowser.open("http://localhost:8000/health")


if __name__ == "__main__":
    threading.Thread(target=open_browser).start()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_config=None)
