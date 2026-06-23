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
import json
import time

OLLAMA_URL = "http://localhost:11434"

OUTPUT_DIR = Path.home() / "OLSEDG_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_FILE = OUTPUT_DIR / "generation_history.json"

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

def estimate_num_ctx(prompt: str) -> int:
    prompt_words = len(str(prompt).split())
    num_ctx = (prompt_words + 500 + 3061) * 2
    return int(num_ctx)
    
def load_history():
    if not HISTORY_FILE.exists():
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history_entry(entry):
    history = load_history()
    history.append(entry)

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def get_average_seconds_per_respondent(model_name):
    history = load_history()
    model_rows = [h for h in history if h.get("model_name") == model_name]

    if not model_rows:
        return None

    values = [h["seconds_taken"] for h in model_rows if h.get("seconds_taken")]
    if not values:
        return None

    return sum(values) / len(values)
    
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

@app.get("/estimate-time/{model_name}/{total_respondents}")
def estimate_time(model_name: str, total_respondents: int):
    avg_seconds = get_average_seconds_per_respondent(model_name)

    if avg_seconds is None:
        return {
            "success": True,
            "has_history": False,
            "message": "No prior timing history available for this model."
        }

    estimated_seconds = avg_seconds * total_respondents
    estimated_minutes = round(estimated_seconds / 60, 1)

    return {
        "success": True,
        "has_history": True,
        "model_name": model_name,
        "average_seconds_per_respondent": round(avg_seconds, 2),
        "total_respondents": total_respondents,
        "estimated_seconds": round(estimated_seconds, 1),
        "estimated_minutes": estimated_minutes,
        "message": f"Estimated generation time: approximately {estimated_minutes} minutes."
    }

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
    question_lookup = {}

    for q in questions:
        min_code = 1
        max_code = len(q.scale_points)

        scale_text = "\n".join(
            [f"{i + 1} = {label}" for i, label in enumerate(q.scale_points)]
        )

        embedded_question = f"""
[Q{q.question_number}]
Question: {q.question_text}
Allowed response codes: {min_code} to {max_code}
Response scale:
{scale_text}
[/Q{q.question_number}]
""".strip()

        question_lookup[q.question_number] = embedded_question

    embedded_stimuli = stimuli

    for q in questions:
        placeholder = f"{{Q{q.question_number}}}"
        embedded_stimuli = embedded_stimuli.replace(
            placeholder,
            question_lookup[q.question_number]
        )

    # If the user forgot to include placeholders, append unanswered questions at the end.
    missing_questions = []

    for q in questions:
        placeholder = f"{{Q{q.question_number}}}"
        if placeholder not in stimuli:
            missing_questions.append(question_lookup[q.question_number])

    if missing_questions:
        embedded_stimuli += """

Questions not embedded in the study materials:
""" + "\n\n".join(missing_questions)

    answer_template = "\n".join(
        [f"Q{q.question_number}=?" for q in questions]
    )

    prompt = f"""
You are simulating one synthetic survey respondent.

Your task:
1. Read the respondent persona.
2. Read the study materials exactly as a survey participant would see them.
3. Answer each embedded survey question from this respondent's perspective.
4. Use the respondent persona, the study materials, and the response scale for each question when choosing answers.

Respondent persona:
{persona}

Study materials with embedded questions:
{embedded_stimuli}

Important rules:
- Choose only allowed option codes for each question.
- Each answer must be an integer within the allowed response-code range for that question.
- Do not choose values below the minimum code or above the maximum code.
- Return only the final answer values.
- Do not include explanations.
- Do not include markdown.
- Do not repeat the questions.
- Return answers only in this format:
{answer_template}
""".strip()

    return prompt


def call_ollama(model_name, prompt, temperature):
    num_ctx = estimate_num_ctx(prompt)

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx
            }
        },
        timeout=900
    )

    response.raise_for_status()
    return response.json().get("response", "")


def parse_answers(raw_text, questions):
    answers = {}
    invalid_questions = []

    for q in questions:
        q_num = q.question_number
        max_code = len(q.scale_points)

        # Allows normal numbers and negative numbers, so invalid values are still captured
        pattern = rf"Q{q_num}\s*=\s*(-?\d+)"
        match = re.search(pattern, raw_text, re.IGNORECASE)

        if match:
            value = int(match.group(1))

            # Store the value even if invalid
            answers[f"Q{q_num}"] = value

            # But mark it invalid if outside the allowed range
            if not (1 <= value <= max_code):
                invalid_questions.append(f"Q{q_num}")
        else:
            # No answer found
            answers[f"Q{q_num}"] = ""
            invalid_questions.append(f"Q{q_num}")

    validation = "valid" if not invalid_questions else "invalid"

    return answers, validation, invalid_questions
    
def get_valid_response_with_retries(model_name, prompt, temperature, questions, max_retries=5):
    last_raw_response = ""
    last_answers = {}
    last_validation = "invalid"
    last_invalid_questions = []

    for attempt in range(1, max_retries + 1):
        raw_response = call_ollama(model_name, prompt, temperature)
        answers, validation, invalid_questions = parse_answers(raw_response, questions)

        if validation == "valid":
            return answers, validation, invalid_questions, attempt, raw_response

        last_raw_response = raw_response
        last_answers = answers
        last_validation = validation
        last_invalid_questions = invalid_questions

    return last_answers, last_validation, last_invalid_questions, max_retries, last_raw_response

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

        condition_lookup = {c.condition_number: c.stimuli for c in data.conditions}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = data.model_name.replace(":", "-").replace("/", "-")
        num_conditions = len(data.conditions)
        samples_per_condition = data.sample_count_per_condition

        csv_filename = f"OLSEDG_{safe_model_name}_{num_conditions}cond_{samples_per_condition}ppc_{timestamp}.csv"
        docx_filename = f"OLSEDG_INPUTS_{safe_model_name}_{num_conditions}cond_{samples_per_condition}ppc_{timestamp}.docx"

        csv_path = OUTPUT_DIR / csv_filename
        docx_path = OUTPUT_DIR / docx_filename

        question_columns = [f"Q{q.question_number}" for q in data.questions]
        columns = [
            "respondent_id",
            "pid",
            "persona_summary",
            "condition",
            "condition_stimuli",
            "model_name",
            "temperature",
            "validation",
            "invalid_questions",
            "retry_count",
            "seconds_taken",
            "prompt_words",
            "num_ctx_used",
            "raw_response"
        ] + question_columns
        pd.DataFrame(columns=columns).to_csv(csv_path, index=False)

        update_job(
            job_id,
            status="generating",
            message=f"Generating synthetic responses. Output is being saved continuously to {csv_path}",
            completed=0,
            total=total_needed,
            percent=0,
            csv_url=f"http://localhost:8000/outputs/{csv_filename}",
            docx_url=None
        )

        job_start_time = time.time()

        for i in range(total_needed):
            respondent_id = i + 1
            condition_number = condition_numbers[i]
            stimuli = condition_lookup[condition_number]

            pid = df_personas.loc[i, "pid"]
            persona = df_personas.loc[i, "persona_summary"]

            prompt = build_prompt(persona, stimuli, data.questions)

            start_time = time.time()

            answers, validation, invalid_questions, retry_count, raw_response = get_valid_response_with_retries(
                data.model_name,
                prompt,
                data.temperature,
                data.questions,
                max_retries=5
            )

            end_time = time.time()
            seconds_taken = round(end_time - start_time, 3)

            save_history_entry({
                "timestamp": datetime.now().isoformat(),
                "model_name": data.model_name,
                "seconds_taken": seconds_taken,
                "respondent_id": respondent_id,
                "condition": condition_number
            })

            prompt_words = len(str(prompt).split())
            num_ctx_used = estimate_num_ctx(prompt)
                        
            row = {
                "respondent_id": respondent_id,
                "pid": pid,
                "persona_summary": persona,
                "condition": condition_number,
                "condition_stimuli": stimuli,
                "model_name": data.model_name,
                "temperature": data.temperature,
                "validation": validation,
                "invalid_questions": ",".join(invalid_questions),
                "retry_count": retry_count,
                "seconds_taken": seconds_taken,
                "prompt_words": prompt_words,
                "num_ctx_used": num_ctx_used,
                "raw_response": str(raw_response).replace("\n", " | "),
            }
            
            # Safety check: if answers accidentally comes as a tuple, take only the answers dictionary
            if isinstance(answers, tuple):
                answers = answers[0]
            
            # Safety check: if answers is still not a dictionary, stop with a clear error
            if not isinstance(answers, dict):
                raise ValueError(f"answers should be a dictionary, but got {type(answers)}: {answers}")
            
            # Add each question answer manually instead of using row.update()
            for q in data.questions:
                q_col = f"Q{q.question_number}"
                row[q_col] = answers.get(q_col, "")
            pd.DataFrame([row]).to_csv(
                csv_path,
                mode="a",
                header=False,
                index=False
            )

            completed = i + 1
            pending = total_needed - completed
            percent = round((completed / total_needed) * 100, 1)

            elapsed_seconds = time.time() - job_start_time
            avg_seconds_current_run = elapsed_seconds / completed
            estimated_remaining_seconds = avg_seconds_current_run * pending
            estimated_total_seconds = elapsed_seconds + estimated_remaining_seconds

            update_job(
                job_id,
                completed=completed,
                total=total_needed,
                pending=pending,
                percent=percent,
                elapsed_seconds=round(elapsed_seconds, 1),
                average_seconds_per_respondent=round(avg_seconds_current_run, 2),
                estimated_remaining_seconds=round(estimated_remaining_seconds, 1),
                estimated_total_seconds=round(estimated_total_seconds, 1),
                message=f"{completed}/{total_needed} respondents completed. {pending} remaining."
            )

        create_docx(data, docx_path)

        update_job(
            job_id,
            status="complete",
            message="Generation complete",
            completed=total_needed,
            total=total_needed,
            pending=0,
            percent=100,
            estimated_remaining_seconds=0,
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
