import os
import re
import sys
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
from typing import Literal, Optional

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
    condition_name: Optional[str] = None
    stimuli: str


class Question(BaseModel):
    question_number: int
    question_text: str
    scale_points: list[str]
    scale_start: int = 1
    scale_type: str = "discrete"
    max_words: int = 0


class PersonaRecord(BaseModel):
    pid: str
    persona: str


class GenerateRequest(BaseModel):
    study_name: str
    model_provider: Literal["local", "openai", "gemini", "anthropic"] = "local"
    model_name: str
    temperature: float
    sample_count_per_condition: int
    conditions: list[Condition]
    questions: list[Question]
    generic_instruction: Optional[str] = None
    persona_source: Literal["default", "custom", "none"] = "default"
    custom_personas: Optional[list[PersonaRecord]] = None


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


def sample_personas(df_small: pd.DataFrame, total_needed: int):
    replace = total_needed > len(df_small)
    return df_small.sample(
        n=total_needed,
        replace=replace,
        random_state=random.randint(1, 999999)
    ).reset_index(drop=True)


def load_personas(
    total_needed: int,
    custom_personas: Optional[list[PersonaRecord]] = None,
    use_personas: bool = True
):
    if not use_personas:
        return pd.DataFrame([
            {"pid": f"no_persona_{i + 1}", "persona_summary": ""}
            for i in range(total_needed)
        ])

    if custom_personas:
        df_small = pd.DataFrame([
            {"pid": p.pid, "persona_summary": p.persona}
            for p in custom_personas
        ]).dropna()

        df_small["persona_summary"] = df_small["persona_summary"].astype(str).str.strip()
        df_small = df_small[df_small["persona_summary"] != ""]

        if df_small.empty:
            raise ValueError("Custom persona file must include at least one non-empty persona.")

        return sample_personas(df_small, total_needed)

    ds = load_dataset("LLM-Digital-Twin/Twin-2K-500", "full_persona")

    if hasattr(ds, "keys"):
        split_name = list(ds.keys())[0]
        df = ds[split_name].to_pandas()
    else:
        df = ds.to_pandas()

    df_small = df[["pid", "persona_summary"]].dropna().copy()
    return sample_personas(df_small, total_needed)


def build_prompt(
    persona,
    stimuli,
    questions,
    generic_instruction: Optional[str] = None,
    include_persona: bool = True
):
    question_lookup = {}

    for q in questions:
        if q.scale_type == "text":
            max_words_str = f"{q.max_words} words" if q.max_words > 0 else "unlimited"
            embedded_question = f"""
[Q{q.question_number}]
Question: {q.question_text}
Response type: Text
Max words: {max_words_str}
Please provide a text response within the allowed word count.
[/Q{q.question_number}]
""".strip()
        else:
            min_code = q.scale_start
            max_code = q.scale_start + len(q.scale_points) - 1

            scale_text = "\n".join(
                [f"{q.scale_start + i} = {label}" for i, label in enumerate(q.scale_points)]
            )

            if q.scale_type == "continuous":
                range_instruction = (
                    f"Allowed response range: {min_code} to {max_code}\n"
                    f"Please select a number in the range."
                )
            else:
                range_instruction = (
                    f"Allowed response codes: {min_code} to {max_code}\n"
                    f"Please select an integer in the range."
                )

            response_type_label = "Continuous" if q.scale_type == "continuous" else "Discrete"
            embedded_question = f"""
[Q{q.question_number}]
Question: {q.question_text}
Response type: {response_type_label}
{range_instruction}
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

    answer_template = "\n".join(
        [f'Q{q.question_number}="?"' if q.scale_type == "text" else f"Q{q.question_number}=?"
         for q in questions]
    )

    default_prompt_template = """
You are simulating one synthetic survey respondent.

Your task:
1. Read the respondent persona.
2. Read the study materials exactly as a survey participant would see them.
3. Answer each embedded survey question from this respondent's perspective.
4. Use the respondent persona, the study materials, response scale, and scale type for each question when choosing answers.

Respondent persona:
{persona}

Study materials with embedded questions:
{embedded_stimuli}

Important rules:
* Use only the allowed response codes or response format for each question.
* Match each answer to the question type:
    * Discrete scale: use an integer.
    * Continuous scale: use a number/float.
    * Text response: write text within the allowed word limit.
* Do not use values below the minimum code or above the maximum code.
* Return only the final responses. Do not include explanations, markdown, or repeat questions.
* Return answers only in this format: {answer_template}
""".strip()

    no_persona_prompt_template = """
You are simulating one synthetic survey respondent.

Your task:
1. Read the study materials exactly as a survey participant would see them.
2. Answer each embedded survey question from a participant's perspective.
3. Use the respondent persona, the study materials, response scale, and scale type for each question when choosing answers.

Study materials with embedded questions:
{embedded_stimuli}

Important rules:
* Use only the allowed response codes or response format for each question.
* Match each answer to the question type:
    * Discrete scale: use an integer.
    * Continuous scale: use a number/float.
    * Text response: write text within the allowed word limit.
* Do not use values below the minimum code or above the maximum code.
* Return only the final responses. Do not include explanations, markdown, or repeat questions.
* Return answers only in this format: {answer_template}
""".strip()

    prompt_template = (generic_instruction or "").strip()

    if not prompt_template:
        prompt_template = default_prompt_template if include_persona else no_persona_prompt_template

    if not include_persona:
        prompt_template = re.sub(
            r"\n?Respondent persona:\s*\n\{persona\}\s*\n",
            "\n",
            prompt_template
        )
        prompt_template = re.sub(
            r"\n?\d+\.\s*Read the respondent persona\.\s*\n?",
            "\n",
            prompt_template
        )
        prompt_template = re.sub(r"\n2\. ", "\n1. ", prompt_template)
        prompt_template = re.sub(r"\n3\. ", "\n2. ", prompt_template)
        prompt_template = re.sub(r"\n4\. ", "\n3. ", prompt_template)
        prompt_template = prompt_template.replace("this respondent's perspective", "a participant's perspective")

    prompt = prompt_template.replace("{persona}", str(persona) if include_persona else "")
    prompt = prompt.replace("{embedded_stimuli}", str(embedded_stimuli))
    prompt = prompt.replace("{answer_template}", str(answer_template))

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

        if q.scale_type == "text":
            match = re.search(rf'Q{q_num}\s*=\s*"([^"]*)"', raw_text, re.IGNORECASE)
            if not match:
                match = re.search(rf"Q{q_num}\s*=\s*(.+)", raw_text, re.IGNORECASE)
            if match:
                answers[f"Q{q_num}"] = match.group(1).strip()
            else:
                answers[f"Q{q_num}"] = ""
                invalid_questions.append(f"Q{q_num}")
        else:
            min_code = q.scale_start
            max_code = q.scale_start + len(q.scale_points) - 1
            is_continuous = q.scale_type == "continuous"

            pattern = rf"Q{q_num}\s*=\s*(-?\d+\.?\d*)"
            match = re.search(pattern, raw_text, re.IGNORECASE)

            if match:
                value = float(match.group(1)) if is_continuous else int(match.group(1))
                answers[f"Q{q_num}"] = value

                if not (min_code <= value <= max_code):
                    invalid_questions.append(f"Q{q_num}")
            else:
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
        for i, label in enumerate(q.scale_points, start=q.scale_start):
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

        use_personas = data.persona_source != "none"
        custom_personas = data.custom_personas if data.persona_source == "custom" else None
        df_personas = load_personas(total_needed, custom_personas, use_personas)

        condition_numbers = []
        for c in data.conditions:
            condition_numbers.extend([c.condition_number] * data.sample_count_per_condition)

        random.shuffle(condition_numbers)

        condition_lookup = {c.condition_number: c.stimuli for c in data.conditions}
        condition_name_lookup = {
            c.condition_number: c.condition_name or f"Condition {c.condition_number}"
            for c in data.conditions
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = data.model_name.replace(":", "-").replace("/", "-")
        num_conditions = len(data.conditions)
        samples_per_condition = data.sample_count_per_condition

        csv_filename = f"OLSEDG_{total_needed}samples_{timestamp}.csv"
        docx_filename = f"OLSEDG_INPUTS_{safe_model_name}_{num_conditions}cond_{samples_per_condition}ppc_{timestamp}.docx"

        csv_path = OUTPUT_DIR / csv_filename
        docx_path = OUTPUT_DIR / docx_filename

        question_columns = [f"Q{q.question_number}" for q in data.questions]
        columns = [
            "respondent_id",
            "study_name",
            "pid",
            "persona_summary",
            "condition",
            "condition_name",
            "condition_stimuli",
            "model_name",
            "temperature",
            "validation",
            "invalid_questions",
            "retry_count",
            "seconds_taken",
            "prompt",
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

            # Only the questions actually embedded ({Qn} present) in this condition's
            # stimuli are shown to the model, so only those should be expected back.
            embedded_questions = [
                q for q in data.questions if f"{{Q{q.question_number}}}" in stimuli
            ]

            pid = df_personas.loc[i, "pid"]
            persona = df_personas.loc[i, "persona_summary"]

            prompt = build_prompt(
                persona,
                stimuli,
                embedded_questions,
                data.generic_instruction,
                include_persona=use_personas
            )

            start_time = time.time()

            answers, validation, invalid_questions, retry_count, raw_response = get_valid_response_with_retries(
                data.model_name,
                prompt,
                data.temperature,
                embedded_questions,
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
                "study_name": data.study_name,
                "pid": pid,
                "persona_summary": persona,
                "condition": condition_number,
                "condition_name": condition_name_lookup[condition_number],
                "condition_stimuli": stimuli,
                "model_name": data.model_name,
                "temperature": data.temperature,
                "validation": validation,
                "invalid_questions": ",".join(invalid_questions),
                "retry_count": retry_count,
                "seconds_taken": seconds_taken,
                "prompt": prompt,
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
        

@app.get("/preview-personas/{sample_count}")
def preview_personas(sample_count: int):
    if sample_count < 1:
        sample_count = 1

    df_personas = load_personas(sample_count, use_personas=True)

    return {
        "success": True,
        "personas": [
            {
                "pid": str(row["pid"]),
                "persona": str(row["persona_summary"])
            }
            for _, row in df_personas.iterrows()
        ]
    }

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
    import tkinter as tk
    from tkinter import ttk, messagebox
    import webbrowser

    PORT = 8000
    OLLAMA_BASE = "http://localhost:11434"

    MODEL_SIZES = {
        "deepseek-r1:14b": "~9.0 GB", "gemma3:4b": "~3.3 GB", "gemma3:12b": "~8.1 GB",
        "llama3.1:8b": "~4.7 GB", "llama3.2:3b": "~2.0 GB", "mistral:7b-instruct": "~4.1 GB",
        "mistral-small3.2:24b": "~14.0 GB", "qwen2.5:7b": "~4.4 GB",
        "qwen2.5:7b-instruct": "~4.4 GB", "qwen2.5:14b": "~9.0 GB", "qwen3:14b": "~9.3 GB",
    }
    MODEL_LIST = list(MODEL_SIZES.keys()) + ["Other (enter below)"]

    def _ollama_ok():
        try:
            return requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3).status_code == 200
        except Exception:
            return False

    def _installed_models():
        try:
            r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
            return r.json().get("models", []) if r.ok else []
        except Exception:
            return []

    # ── FastAPI server (background) ──────────────────────────────────────
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error")
    server = uvicorn.Server(config)
    Thread(target=server.run, daemon=True).start()

    # ── Root window ──────────────────────────────────────────────────────
    root = tk.Tk()
    root.title("OLSEDG Helper")
    root.geometry("500x450")
    root.resizable(False, False)
    BG, BLUE, GREEN, RED, MUTED = "#f8fafc", "#1e40af", "#16a34a", "#dc2626", "#64748b"
    root.configure(bg=BG)

    hdr = tk.Frame(root, bg=BLUE, height=44)
    hdr.pack(fill="x"); hdr.pack_propagate(False)
    tk.Label(hdr, text="OLSEDG Helper", bg=BLUE, fg="white",
             font=("Helvetica", 13, "bold")).pack(side="left", padx=14)
    tk.Label(hdr, text=f"● http://127.0.0.1:{PORT}", bg=BLUE, fg="#93c5fd",
             font=("Helvetica", 10)).pack(side="right", padx=14)

    sty = ttk.Style(); sty.theme_use("clam")
    sty.configure("TNotebook", background=BG, borderwidth=0)
    sty.configure("TNotebook.Tab", padding=[14, 5], font=("Helvetica", 10))
    sty.map("TNotebook.Tab", background=[("selected", "#dbeafe")])
    sty.configure("TFrame", background=BG)

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=8, pady=6)

    # ── TAB 1 : Ollama Setup ─────────────────────────────────────────────
    tab1 = ttk.Frame(nb); nb.add(tab1, text="1. Ollama Setup")
    p1 = tk.Frame(tab1, bg=BG, padx=18, pady=16); p1.pack(fill="both", expand=True)

    r1h = tk.Frame(p1, bg=BG); r1h.pack(anchor="w")
    tk.Label(r1h, text="Validate Ollama Setup", font=("Helvetica", 13, "bold"), bg=BG).pack(side="left")
    tip = tk.Label(r1h, text="  ⓘ", font=("Helvetica", 11), fg=MUTED, bg=BG, cursor="hand2")
    tip.pack(side="left")

    _tw = [None]
    def _show_tip(e):
        if _tw[0]: return
        w = tk.Toplevel(root); w.overrideredirect(True); w.configure(bg="#1e293b")
        tk.Label(w, text=("Ollama runs open-source LLMs locally on your computer.\n"
                           "OLSEDG Helper sends prompts to it to generate synthetic\n"
                           "responses — no data ever leaves your machine."),
                 bg="#1e293b", fg="white", font=("Helvetica", 10),
                 padx=10, pady=8, justify="left").pack()
        w.geometry(f"+{e.x_root + 8}+{e.y_root + 8}"); _tw[0] = w
    def _hide_tip(e):
        if _tw[0]: _tw[0].destroy(); _tw[0] = None
    tip.bind("<Enter>", _show_tip); tip.bind("<Leave>", _hide_tip)

    t1_stat = tk.Label(p1, text="", font=("Helvetica", 11), bg=BG)
    t1_stat.pack(anchor="w", pady=(14, 0))
    t1_det = tk.Label(p1, text="", font=("Helvetica", 10), fg=MUTED, bg=BG, wraplength=430)
    t1_det.pack(anchor="w", pady=(4, 0))
    t1_lnk = tk.Label(p1, text="", font=("Helvetica", 10), fg="#2563eb", bg=BG, cursor="hand2")
    t1_lnk.pack(anchor="w", pady=(4, 0))
    t1_lnk.bind("<Button-1>", lambda e: webbrowser.open("https://ollama.ai"))
    t1_btn = tk.Button(p1, text="Check Again", bg=BLUE, fg="white", relief="flat",
                        padx=10, pady=4, font=("Helvetica", 10), cursor="hand2")
    t1_btn.pack(anchor="w", pady=(14, 0))

    def do_check():
        root.after(0, lambda: t1_stat.config(text="Checking…", fg=MUTED))
        ok = _ollama_ok()
        def u():
            if ok:
                t1_stat.config(text="✓  Ollama is installed and running.", fg=GREEN)
                t1_det.config(text="Proceed to Install Model to set up your LLM.")
                t1_lnk.config(text="")
            else:
                t1_stat.config(text="✗  Ollama not found or not running.", fg=RED)
                t1_det.config(text="Download and install Ollama, then click Check Again.")
                t1_lnk.config(text="→  Download Ollama at ollama.ai")
        root.after(0, u)
    t1_btn.config(command=lambda: Thread(target=do_check, daemon=True).start())

    # ── TAB 2 : Install Model ────────────────────────────────────────────
    tab2 = ttk.Frame(nb); nb.add(tab2, text="2. Install Model")
    p2 = tk.Frame(tab2, bg=BG, padx=18, pady=16); p2.pack(fill="both", expand=True)

    tk.Label(p2, text="Install a Model", font=("Helvetica", 13, "bold"), bg=BG).pack(anchor="w")
    tk.Label(p2, text="Choose a model for generating synthetic survey responses.",
             font=("Helvetica", 10), fg=MUTED, bg=BG).pack(anchor="w", pady=(2, 8))

    t2_var = tk.StringVar(value="Select a model")
    t2_cb = ttk.Combobox(p2, textvariable=t2_var, values=MODEL_LIST,
                          state="readonly", width=30, font=("Helvetica", 11))
    t2_cb.pack(anchor="w")

    # "Other" entry — packed/unpacked dynamically
    t2_of = tk.Frame(p2, bg=BG)
    t2_ov = tk.StringVar()
    tk.Entry(t2_of, textvariable=t2_ov, font=("Helvetica", 11), width=32).pack(anchor="w", pady=(4, 0))
    tk.Label(t2_of, text="Enter model name from ollama.com/library  (e.g. llama3.2:1b)",
             font=("Helvetica", 9), fg=MUTED, bg=BG).pack(anchor="w")

    t2_info = tk.Label(p2, text="", font=("Helvetica", 10), bg=BG, wraplength=430)
    t2_info.pack(anchor="w", pady=(8, 0))

    # Progress frame — packed/unpacked dynamically
    t2_pf = tk.Frame(p2, bg=BG)
    t2_pg = ttk.Progressbar(t2_pf, mode="determinate", length=430)
    t2_pg.pack(anchor="w")
    t2_pl = tk.Label(t2_pf, text="", font=("Helvetica", 9), fg=MUTED, bg=BG)
    t2_pl.pack(anchor="w", pady=(2, 0))

    t2_stat = tk.Label(p2, text="", font=("Helvetica", 10), bg=BG, wraplength=430)
    t2_stat.pack(anchor="w", pady=(4, 0))
    t2_lnk = tk.Label(p2, text="", font=("Helvetica", 10), fg="#2563eb", bg=BG, cursor="hand2")
    t2_lnk.pack(anchor="w")
    t2_lnk.bind("<Button-1>", lambda e: webbrowser.open("https://synthstudy.vercel.app"))

    t2_btn = tk.Button(p2, text="Install Model", bg=BLUE, fg="white", relief="flat",
                        padx=12, pady=5, font=("Helvetica", 10), cursor="hand2")
    t2_btn.pack(anchor="w", pady=(10, 0))

    def on_t2_select(e=None):
        val = t2_var.get()
        t2_pf.pack_forget()
        t2_stat.config(text=""); t2_lnk.config(text="")
        if val == "Other (enter below)":
            t2_of.pack(anchor="w", pady=(6, 0), before=t2_info)
            t2_info.config(text="")
        else:
            t2_of.pack_forget()
            if val not in ("Select a model", ""):
                sz = MODEL_SIZES.get(val, "")
                names = [m.get("name", "") for m in _installed_models()]
                already = val in names
                t2_info.config(
                    text=(f"Download size: {sz}  " if sz else "") + ("✓ Already installed" if already else ""),
                    fg=GREEN if already else MUTED)
            else:
                t2_info.config(text="")
    t2_cb.bind("<<ComboboxSelected>>", on_t2_select)

    def do_install():
        val = t2_var.get()
        model = (t2_ov.get().strip() if val == "Other (enter below)"
                 else (None if val in ("Select a model", "") else val))
        if not model:
            t2_stat.config(text="Please select or enter a model name.", fg=RED); return

        t2_btn.config(state="disabled", text="Installing…")
        t2_stat.config(text=f"Installing {model}…", fg=MUTED)
        t2_pg["value"] = 0; t2_pl.config(text=""); t2_lnk.config(text="")
        t2_pf.pack(anchor="w", pady=(8, 0), before=t2_stat)

        def on_p(status, total, completed):
            def u():
                if total and completed:
                    t2_pg["value"] = int(completed / total * 100)
                    t2_pl.config(text=f"{status}  {completed/1024**2:.0f} / {total/1024**2:.0f} MB")
                else:
                    t2_pl.config(text=status)
            root.after(0, u)

        def on_done():
            def u():
                t2_pg["value"] = 100; t2_pl.config(text="")
                t2_stat.config(text=f"✓  {model} is ready to use.", fg=GREEN)
                t2_lnk.config(text="→  Open SEDG at synthstudy.vercel.app")
                t2_btn.config(state="normal", text="Install Model")
            root.after(0, u)

        def on_err(msg):
            def u():
                t2_stat.config(text=f"Error: {msg}", fg=RED)
                t2_pf.pack_forget()
                t2_btn.config(state="normal", text="Install Model")
            root.after(0, u)

        def pull():
            try:
                resp = requests.post(f"{OLLAMA_BASE}/api/pull",
                                     json={"name": model}, stream=True, timeout=None)
                for line in resp.iter_lines():
                    if line:
                        d = json.loads(line)
                        on_p(d.get("status", ""), d.get("total", 0), d.get("completed", 0))
                        if d.get("status") == "success":
                            on_done(); return
                on_done()
            except Exception as ex:
                on_err(str(ex))
        Thread(target=pull, daemon=True).start()

    t2_btn.config(command=do_install)

    # ── TAB 3 : Manage Models ────────────────────────────────────────────
    tab3 = ttk.Frame(nb); nb.add(tab3, text="3. Manage Models")
    p3 = tk.Frame(tab3, bg=BG, padx=18, pady=16); p3.pack(fill="both", expand=True)

    t3h = tk.Frame(p3, bg=BG); t3h.pack(fill="x")
    tk.Label(t3h, text="Installed Models", font=("Helvetica", 13, "bold"), bg=BG).pack(side="left")
    t3_ref = tk.Button(t3h, text="↻ Refresh", font=("Helvetica", 9), relief="flat",
                        bg="#e2e8f0", padx=6, pady=3, cursor="hand2")
    t3_ref.pack(side="right")

    t3_list = tk.Frame(p3, bg=BG)
    t3_list.pack(fill="both", expand=True, pady=(10, 0))
    t3_stat = tk.Label(p3, text="", font=("Helvetica", 10), bg=BG)
    t3_stat.pack(anchor="w", pady=(4, 0))

    def refresh_t3():
        for w in t3_list.winfo_children(): w.destroy()
        t3_stat.config(text="")
        tk.Label(t3_list, text="Loading…", fg=MUTED, bg=BG, font=("Helvetica", 10)).pack(anchor="w")

        def do():
            models = _installed_models()
            def upd():
                for w in t3_list.winfo_children(): w.destroy()
                if not models:
                    tk.Label(t3_list, text="No models installed.", fg=MUTED, bg=BG,
                             font=("Helvetica", 10)).pack(anchor="w")
                    return
                for m in models:
                    name = m.get("name", "")
                    sz = m.get("size", 0)
                    row = tk.Frame(t3_list, bg="#f1f5f9", padx=8, pady=5)
                    row.pack(fill="x", pady=2)
                    tk.Label(row, text=name, font=("Helvetica", 10, "bold"),
                             bg="#f1f5f9").pack(side="left")
                    if sz:
                        tk.Label(row, text=f"  {sz/1024**3:.1f} GB", font=("Helvetica", 9),
                                 fg=MUTED, bg="#f1f5f9").pack(side="left")

                    def make_rm(n):
                        def rm():
                            if not messagebox.askyesno("Uninstall",
                                                        f"Remove {n}?\nThis will free up disk space."):
                                return
                            def do_rm():
                                try:
                                    r = requests.delete(f"{OLLAMA_BASE}/api/delete",
                                                        json={"name": n}, timeout=10)
                                    def af():
                                        t3_stat.config(
                                            text=f"✓  {n} removed." if r.ok else f"Failed to remove {n}.",
                                            fg=GREEN if r.ok else RED)
                                        refresh_t3()
                                    root.after(0, af)
                                except Exception as ex:
                                    root.after(0, lambda: t3_stat.config(text=str(ex), fg=RED))
                            Thread(target=do_rm, daemon=True).start()
                        return rm

                    tk.Button(row, text="Uninstall", font=("Helvetica", 9), relief="flat",
                              bg="#fee2e2", fg=RED, padx=6, pady=2, cursor="hand2",
                              command=make_rm(name)).pack(side="right")
            root.after(0, upd)
        Thread(target=do, daemon=True).start()

    t3_ref.config(command=refresh_t3)

    # ── Startup ──────────────────────────────────────────────────────────
    Thread(target=do_check, daemon=True).start()
    root.after(800, refresh_t3)

    def on_close():
        server.should_exit = True
        root.destroy()
        sys.exit(0)

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
