# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Customer Support Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import CustomerSupportAction, CustomerSupportObservation
except ImportError:
    from models import CustomerSupportAction, CustomerSupportObservation


class CustomerSupportEnv(
    EnvClient[CustomerSupportAction, CustomerSupportObservation, State]
):
    """
    Client for the Customer Support Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with CustomerSupportEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(CustomerSupportAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = CustomerSupportEnv.from_docker_image("customer_support_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(CustomerSupportAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: CustomerSupportAction) -> Dict:
        """
        Convert CustomerSupportAction to JSON payload for step message.

        Args:
            action: CustomerSupportAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "category": action.category,
            "urgency": action.urgency,
            "summary": action.summary,
        }

    def _parse_result(self, payload: Dict) -> StepResult[CustomerSupportObservation]:
        """
        Parse server response into StepResult[CustomerSupportObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with CustomerSupportObservation
        """
        obs_data = payload.get("observation", {})
        observation = CustomerSupportObservation(
            task_id=obs_data.get("task_id", ""),
            difficulty=obs_data.get("difficulty", ""),
            email_text=obs_data.get("email_text", ""),
            step_count=obs_data.get("step_count", 0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            reward_so_far=obs_data.get("reward_so_far", 0.0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
