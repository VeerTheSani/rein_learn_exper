# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Customer Support Env Environment Implementation.
An agent reads a customer email and classifies it by category and urgency.
"""

from uuid import uuid4
from typing import Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import CustomerSupportAction, CustomerSupportObservation
    from ..tasks import TASKS
except ImportError:
    from models import CustomerSupportAction, CustomerSupportObservation
    from tasks import TASKS


class CustomerSupportEnvironment(Environment):
    """
    Real-world environment where an agent acts as a Support Ticket Triage system.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task = None
        self._task_index = 0
        self._cumulative_reward = 0.0

    def reset(self, task_id: Optional[str] = None) -> CustomerSupportObservation:
        """Starts a new task and hands the first email to the agent."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._cumulative_reward = 0.0

        # Load a specific task if requested, otherwise cycle through them
        if task_id:
            self._task = next((t for t in TASKS if t["id"] == task_id), TASKS[0])
        else:
            self._task = TASKS[self._task_index % len(TASKS)]
            self._task_index += 1

        return CustomerSupportObservation(
            task_id=self._task["id"],
            difficulty=self._task["difficulty"],
            email_text=self._task["email_text"],
            step_count=self._state.step_count,
            done=False,
            reward=0.0,
            reward_so_far=0.0
        )

    def step(self, action: CustomerSupportAction) -> CustomerSupportObservation: 
        """Takes the Agen answe and grades ii and returns the score  like this(0.0 to 1.0)."""
        if getattr(self, '_task', None) is None:
            raise RuntimeError("Episode is done or not started. Call reset() first.")

        self._state.step_count += 1
        reward = 0.0

        # grading system:::::
        
        # #### Did the agent output a valid summary string? (+0.2)
        if action.summary and len(action.summary.strip()) > 0:
            reward += 0.2

        #######  Did the agent guess the correct category?+0.4)
        if action.category.strip().lower() == self._task["expected_category"].strip().lower():
            reward += 0.4

        #Did the agent guess the correct urgency? (+0.4)
        if action.urgency.strip().lower() == self._task["expected_urgency"].strip().lower():
            reward += 0.4

        # ----------------------------------------------------------------------

        self._cumulative_reward = round(reward, 2)

        return CustomerSupportObservation(
            task_id=self._task["id"],
            difficulty=self._task["difficulty"],
            email_text=self._task["email_text"],
            step_count=self._state.step_count,
            done=True,  # Task ends after they submit the ticket
            reward=self._cumulative_reward, # Required by OpenEnv base spec
            reward_so_far=self._cumulative_reward
        )

    @property
    def state(self) -> State:
        return self._state