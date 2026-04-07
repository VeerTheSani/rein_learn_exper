# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Customer Support Env Environment.

The agent reads a customer email and outputs a structured JSON ticket 
containing the category, urgency, and a brief summary.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class CustomerSupportAction(Action):
    """AI Agentm output after reading the email."""
    
    category: str = Field(..., description="Must be one of: Billing, Tech Support, Account Management")
    urgency: str = Field(..., description="Must be one of: Low, Medium, High")
    summary: str = Field(..., description="A quick 3-to-5 word summary of the user's issue")


class CustomerSupportObservation(Observation):
    """environme  sends this to the AI Agent (the email and metadata)."""
    
    task_id: str = Field(default="", description="The ID of the current email task")
    difficulty: str = Field(default="", description="easy, medium, or hard")
    email_text: str = Field(default="", description="The actual customer email to be classified")
    step_count: int = Field(default=0, description="Number of steps taken so far")
    done: bool = Field(default=False, description="Whether the episode has ended")
    reward_so_far: float = Field(default=0.0, description="Cumulative reward this episode")