"""
Baseline inference script for Customer Support Env.
Strictly adheres to the OpenEnv competition stdout format rules.
"""

import os
import sys
import json
from typing import List
from openai import OpenAI

# 1. Import your native OpenEnv client and models
from client import CustomerSupportEnv
from models import CustomerSupportAction

# ── Config ───────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

# The tasks
EVAL_TASK_IDS = ["easy_billing", "medium_tech", "hard_multi_issue"]

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

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
        return {"category": "Error", "urgency": "Error", "summary": str(e)}

# episode runner 

def run_episode(task_id: str):
    print(f"[START] task={task_id} env=customer_support_env model={MODEL_NAME}", flush=True)

    # 2. Use the persistent WebSocket client from client.py
    # 2. Use the persistent WebSocket client from client.py
    with CustomerSupportEnv(base_url=ENV_BASE_URL).sync() as env:
        
        # Reset Environment securely
        obs_data = env.reset(task_id=task_id)
        email = obs_data.observation.email_text
        
        rewards: List[float] = []
        done = False
        step_num = 1
        error_msg = "null"

        # Call LLM
        action_payload = call_llm(email)

        try:
            # 3. Cast the JSON to your Pydantic Model and Step
            action = CustomerSupportAction(**action_payload)
            result = env.step(action)
            
            reward = result.reward
            done = result.done
            rewards.append(reward)
            
        except Exception as e:
            reward = 0.0
            done = True
            error_msg = f"'{str(e)}'"
            rewards.append(reward)

        action_str = json.dumps(action_payload).replace(" ", "") 
        done_str = "true" if done else "false"
        
        print(
            f"[STEP] step={step_num} action={action_str} "
            f"reward={reward:.2f} done={done_str} error={error_msg}",
            flush=True,
        )

        final_score = max(rewards) if rewards else 0.0
        success_str = "true" if final_score >= 0.8 else "false" 
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