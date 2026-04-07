"""
Baseline inference script for Customer Support Env.
Strictly adheres to the OpenEnv competition stdout format rules.
"""

import os
import sys
import json
import requests
from typing import List, Optional
from openai import OpenAI

# ── Config ───────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000/env")

# The tasks from our tasks.py file
EVAL_TASK_IDS = ["easy_billing", "medium_tech", "hard_multi_issue"]

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


def env_reset(task_id: str) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(action_payload: dict) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/step", json=action_payload, timeout=30)
    r.raise_for_status()
    return r.json()


SYSTEM_PROMPT = """
You are an expert customer support AI. 
Read the user's email and output ONLY a JSON object with the following keys:
- "category": Must be one of: "Billing", "Tech Support", "Account Management"
- "urgency": Must be one of: "Low", "Medium", "High"
- "summary": A 3-5 word summary of the issue.

Output ONLY valid JSON. No markdown formatting. No explanation.
"""

def call_llm(email_text: str) -> dict:
    user_msg = f"Email to classify:\n{email_text}"
    
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0, 
        )
        raw = resp.choices[0].message.content.strip()
        
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
                
        return json.loads(raw.strip())
    except Exception as e:
        # if fails then  failssback  return a dummy fallbackk
        return {"category": "Error", "urgency": "Error", "summary": str(e)}

#episode runnner 

def run_episode(task_id: str):
    # 1. Reset Environment
    obs_data = env_reset(task_id)
    obs = obs_data["observation"]
    email = obs["email_text"]
    
    rewards: List[float] = []
    done = False
    step_num = 0
    error_msg = "null"


    print(f"[START] task={task_id} env=customer_support_env model={MODEL_NAME}", flush=True)

    step_num += 1
    action_payload = call_llm(email)

    try:
        result = env_step(action_payload)
        reward = result["reward"]
        done = result["done"]
        rewards.append(reward)
    except Exception as e:
        reward = 0.0
        done = True
        error_msg = f"'{str(e)}'"
        rewards.append(reward)


    action_str = json.dumps(action_payload).replace(" ", "") # Compact action string
    done_str = "true" if done else "false"
    
    print(
        f"[STEP] step={step_num} action={action_str} "
        f"reward={reward:.2f} done={done_str} error={error_msg}",
        flush=True,
    )


    final_score = max(rewards) if rewards else 0.0
    success_str = "true" if final_score >= 0.8 else "false" # Consider 0.8+ a success
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={success_str} steps={step_num} "
        f"score={final_score:.2f} rewards={rewards_str}",
        flush=True,
    )

if __name__ == "__main__":
    for task_id in EVAL_TASK_IDS:
        try:
            run_episode(task_id)
        except Exception as e:
            print(f"CRITICAL SCRIPT ERROR on task {task_id}: {e}", file=sys.stderr)