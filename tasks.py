"""
Task bank for the Customer Support Environment.
Contains easy, medium, and hard emails for the AI to triage.
"""

from typing import Any, Dict, List

TASKS: List[Dict[str, Any]] = [
    #eazy task
    {
        "id": "easy_billing",
        "difficulty": "easy",
        "email_text": "Hello, I noticed I was charged $15 twice for my subscription this month. Please refund the extra charge immediately.",
        "expected_category": "Billing",
        "expected_urgency": "High",
    },

    ## mid task
    {
        "id": "medium_tech",
        "difficulty": "medium",
        "email_text": "Hey folks, hope you are having a good Tuesday. I was trying to log in today to check my stats but the system keeps throwing a 500 server error on the dashboard. I've cleared my cache but no luck. Let me know when u fix it!",
        "expected_category": "Tech Support",
        "expected_urgency": "Medium",
    },

    ####  difficult tasks
    {
        "id": "hard_multi_issue",
        "difficulty": "hard",
        "email_text": "I am so incredibly frustrated. My package arrived completely smashed, which is ridiculous. Because of this terrible experience, I want to completely delete my account and erase all my data from your servers right now.",
        "expected_category": "Account Management", 
        "expected_urgency": "High",
    }
]

TASK_BY_ID = {t["id"]: t for t in TASKS}