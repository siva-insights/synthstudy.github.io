import os
import re
import uuid
import random
import requests
import pandas as pd
from datasets import load_dataset
from docx import Document
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
from threading import Thread, Lock
import uvicorn

OLLAMA_URL = "http://localhost:11434"

OUTPUT_DIR = Path.home() / "OLSEDG_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="OLSEDG Helper")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

JOBS = {}
JOBS_LOCK = Lock()


def update_job(job_id, **kwargs):
    with JOBS_LOCK:
        JOBS[job_id].update(kwargs)


class Condition(BaseModel):
    condition_number: int
    stimuli: str


class Question(BaseModel):
    question_number: int
    question_text: str
    scale_points: list[str]


class GenerateRequest(BaseModel):
    study_name: str
    model_name: str
    temperature: float
    sample_count_per_condition: int
    conditions: list[Condition]
    questions: list[Question]


@app.get("/health")
def health():
    return {"helper_running": True, "message": "OLSEDG Helper is running"}


@app.get("/check-model/{model_name}")
def check_model(model_name: str):
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        response.raise_for_status()

        models = response.json().get("models", [])
        installed_models = [m["name"] for m in models]
        model_installed = model_name in installed_models

        return {
            "helper_running": True,
            "ollama_running": True,
            "model": model_name,
            "model_installed": model_installed,
            "ready_to_use": model_installed,
            "installed_models": installed_models,
        }

    except requests.exceptions.RequestException:
        return {
            "helper_running": True,
            "ollama_running": False,
            "model_installed": False,
            "ready_to_use": False,
            "message": "Ollama is not running.",
        }


def load_personas(total_needed: int):
    ds = load_dataset("LLM-Digital-Twin/Twin-2K-500", "full_persona")

    if hasattr(ds, "keys"):
        split_name = list(ds.keys())[0]
        df = ds[split_name].to_pandas()
    else:
        df = ds.to_pandas()

    df_small = df[["pid", "persona_summary"]].dropna().copy()

    replace = total_needed > len(df_small)
    df_sample = df_small.sample(
        n=total_needed,
        replace=replace,
        random_state=random.randint(1, 999999)
    ).reset_index(drop=True)

    return df_sample


def build_prompt(persona, stimuli, questions):
    question_blocks = []

    for q in questions:
        min_code = 1
        max_code = len(q.scale_points)

        scale_text = "\n".join(
            [f"{i + 1} = {label}" for i, label in enumerate(q.scale_points)]
        )

        question_blocks.append(
            f"""
Q{q.question_number}: {q.question_text}
Allowed response codes: {min_code} to {max_code}
Response scale:
{scale_text}
"""
        )

    questions_text = "\n".join(question_blocks)

    prompt = f"""
You are simulating one synthetic survey respondent.

Your task:
1. Read the respondent persona.
2. Read the experimental stimulus.
3. Answer the survey questions from this respondent's perspective.
4. Wherever a placeholder such as {{Q1}}, {{Q2}}, etc. appears in the stimulus, treat that as the point where the corresponding question is asked.

Respondent persona:
{persona}

Experimental stimulus:
{stimuli}

Survey questions:
{questions_text}

Important rules:
- Choose only allowed option codes for single-choice questions.
- Each answer must be an integer within the allowed response-code range for that question.
- Do not choose values below the minimum code or above the maximum code.
- For all questions, return only an integer.
- Do not include explanations.
- Do not include markdown.
- Do not repeat the questions.
- Return answers only in this format:
Q1=?
Q2=?
Q3=?
"""
    return prompt


def call_ollama(model_name, prompt, temperature):
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        },
        timeout=300
    )

    response.raise_for_status()
    return response.json().get("response", "")


def parse_answers(raw_text, questions):
    answers = {}

    for q in questions:
        q_num = q.question_number
        max_code = len(q.scale_points)

        pattern = rf"Q{q_num}\s*=\s*(\d+)"
        match = re.search(pattern, raw_text, re.IGNORECASE)

        if match:
            value = int(match.group(1))
            if 1 <= value <= max_code:
                answers[f"Q{q_num}"] = value
            else:
                answers[f"Q{q_num}"] = ""
        else:
            answers[f"Q{q_num}"] = ""

    return answers


def create_docx(data, filepath):
    doc = Document()

    doc.add_heading("OLSEDG Study Inputs", level=1)

    doc.add_heading("Study Information", level=2)
    doc.add_paragraph(f"Study Name: {data.study_name}")
    doc.add_paragraph(f"Model: {data.model_name}")
    doc.add_paragraph(f"Temperature: {data.temperature}")
    doc.add_paragraph(f"Samples per Condition: {data.sample_count_per_condition}")
    doc.add_paragraph(f"Number of Conditions: {len(data.conditions)}")

    doc.add_heading("Conditions and Stimuli", level=2)
    for c in data.conditions:
        doc.add_heading(f"Condition {c.condition_number}", level=3)
        doc.add_paragraph(c.stimuli)

    doc.add_heading("Questions and Response Scales", level=2)
    for q in data.questions:
        doc.add_heading(f"Q{q.question_number}: {q.question_text}", level=3)
        for i, label in enumerate(q.scale_points, start=1):
            doc.add_paragraph(f"{i} = {label}")

    doc.save(filepath)


def run_generation_job(job_id: str, data: GenerateRequest):
    try:
        total_needed = data.sample_count_per_condition * len(data.conditions)

        update_job(
            job_id,
            status="loading_personas",
            message="Loading respondent personas...",
            completed=0,
            total=total_needed
        )

        df_personas = load_personas(total_needed)

        condition_numbers = []
        for c in data.conditions:
            condition_numbers.extend([c.condition_number] * data.sample_count_per_condition)

        random.shuffle(condition_numbers)

        rows = []
        condition_lookup = {c.condition_number: c.stimuli for c in data.conditions}

        update_job(
            job_id,
            status="generating",
            message="Generating synthetic responses..."
        )

        for i in range(total_needed):
            respondent_id = i + 1
            condition_number = condition_numbers[i]
            stimuli = condition_lookup[condition_number]

            pid = df_personas.loc[i, "pid"]
            persona = df_personas.loc[i, "persona_summary"]

            prompt = build_prompt(persona, stimuli, data.questions)
            raw_response = call_ollama(data.model_name, prompt, data.temperature)
            answers = parse_answers(raw_response, data.questions)

            row = {
                "respondent_id": respondent_id,
                "pid": pid,
                "persona_summary": persona,
                "condition": condition_number,
            }

            row.update(answers)
            rows.append(row)

            update_job(
                job_id,
                completed=i + 1,
                message=f"{i + 1}/{total_needed} respondents completed"
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = data.model_name.replace(":", "-").replace("/", "-")
        num_conditions = len(data.conditions)
        samples_per_condition = data.sample_count_per_condition

        csv_filename = f"OLSEDG_{safe_model_name}_{num_conditions}cond_{samples_per_condition}ppc_{timestamp}.csv"
        docx_filename = f"OLSEDG_INPUTS_{safe_model_name}_{num_conditions}cond_{samples_per_condition}ppc_{timestamp}.docx"

        csv_path = OUTPUT_DIR / csv_filename
        docx_path = OUTPUT_DIR / docx_filename

        pd.DataFrame(rows).to_csv(csv_path, index=False)
        create_docx(data, docx_path)

        update_job(
            job_id,
            status="complete",
            message="Generation complete",
            completed=total_needed,
            csv_url=f"http://localhost:8000/outputs/{csv_filename}",
            docx_url=f"http://localhost:8000/outputs/{docx_filename}"
        )

    except Exception as e:
        update_job(
            job_id,
            status="error",
            message=str(e)
        )


@app.post("/generate")
def generate(data: GenerateRequest):
    total_needed = data.sample_count_per_condition * len(data.conditions)
    job_id = str(uuid.uuid4())

    JOBS[job_id] = {
        "job_id": job_id,
        "status": "started",
        "message": "Generation started",
        "completed": 0,
        "total": total_needed,
        "csv_url": None,
        "docx_url": None
    }

    thread = Thread(target=run_generation_job, args=(job_id, data), daemon=True)
    thread.start()

    return {
        "success": True,
        "job_id": job_id,
        "total_responses": total_needed
    }


@app.get("/progress/{job_id}")
def get_progress(job_id: str):
    if job_id not in JOBS:
        return {
            "success": False,
            "message": "Job not found"
        }

    return {
        "success": True,
        **JOBS[job_id]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
