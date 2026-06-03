from __future__ import annotations

import json
from typing import Any, Dict, Optional
from uuid import uuid4

from .session import SciResearchEnv


class SciResearchBackend:
    """In-memory backend for sampling and interacting with sci_research sessions."""

    def __init__(self) -> None:
        self._sessions: Dict[str, SciResearchEnv] = {}

    def sample_environment(self, **sampler_kwargs: Any) -> Dict[str, Any]:
        env = SciResearchEnv()
        task = env.sample_task(**sampler_kwargs)
        return self.create_session(task=task)

    def create_session(
        self,
        task: Optional[Dict[str, Any]] = None,
        **sampler_kwargs: Any,
    ) -> Dict[str, Any]:
        env = SciResearchEnv()
        if task is None:
            task = env.sample_task(**sampler_kwargs)
        env.set_task(task)
        observation = env.reset()
        session_id = uuid4().hex
        self._sessions[session_id] = env
        return {
            "session_id": session_id,
            "task_type": "SCI_RESEARCH",
            "task_description": env.get_task_goal(),
            "observation": observation,
            "tool_prompt": env.get_function_tools_prompt(),
        }

    def close_session(self, session_id: str) -> Dict[str, Any]:
        existed = self._sessions.pop(session_id, None) is not None
        return {"success": existed, "session_id": session_id}

    def get_session(self, session_id: str) -> SciResearchEnv:
        if session_id not in self._sessions:
            raise KeyError(f"Unknown sci_research session: {session_id}")
        return self._sessions[session_id]

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        env = self.get_session(session_id)
        return {
            "session_id": session_id,
            "task_type": "SCI_RESEARCH",
            "task_description": env.get_task_goal(),
            "summary": env.public_state(),
            "function_tools": env.get_function_tools(),
        }

    def export_internal_task(self, session_id: str) -> Dict[str, Any]:
        env = self.get_session(session_id)
        return env.get_task()

    def dispatch_function_call(self, session_id: str, function_call: Dict[str, Any]) -> Dict[str, Any]:
        env = self.get_session(session_id)
        return env.dispatch_function_call(function_call)

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Service-style entrypoint for backend-driven integrations.

        Supported actions:
        - ``sample_environment``: sample a new task and create a session
        - ``create_session``: create a session from a provided task or sampler kwargs
        - ``get_session_summary``: return task summary and function tools
        - ``dispatch_function_call``: execute a tool call against an existing session
        - ``close_session``: dispose an existing session
        """
        if not isinstance(request, dict):
            return {"success": False, "message": "Request must be a dict."}

        action = request.get("action")
        if not action:
            return {"success": False, "message": "Request is missing required field 'action'."}

        try:
            if action == "sample_environment":
                sampler_kwargs = request.get("sampler_kwargs", {})
                response = self.sample_environment(**sampler_kwargs)
                return {"success": True, **response}

            if action == "create_session":
                sampler_kwargs = request.get("sampler_kwargs", {})
                task = request.get("task")
                response = self.create_session(task=task, **sampler_kwargs)
                return {"success": True, **response}

            if action == "get_session_summary":
                session_id = request.get("session_id")
                if not session_id:
                    return {"success": False, "message": "Missing session_id for get_session_summary."}
                response = self.get_session_summary(session_id)
                return {"success": True, **response}

            if action == "export_internal_task":
                session_id = request.get("session_id")
                if not session_id:
                    return {"success": False, "message": "Missing session_id for export_internal_task."}
                response = self.export_internal_task(session_id)
                return {"success": True, "task": response}

            if action == "dispatch_function_call":
                session_id = request.get("session_id")
                if not session_id:
                    return {"success": False, "message": "Missing session_id for dispatch_function_call."}
                function_call = request.get("function_call")
                if function_call is None:
                    return {"success": False, "message": "Missing function_call for dispatch_function_call."}
                response = self.dispatch_function_call(session_id, function_call)
                if isinstance(response, dict) and "success" not in response:
                    return {"success": True, "result": response}
                return response

            if action == "close_session":
                session_id = request.get("session_id")
                if not session_id:
                    return {"success": False, "message": "Missing session_id for close_session."}
                return self.close_session(session_id)

            return {"success": False, "message": f"Unknown backend action: {action}"}
        except KeyError as exc:
            return {"success": False, "message": str(exc)}
        except Exception as exc:
            return {"success": False, "message": f"Backend error during {action}: {exc}"}

    def handle_json_request(self, request_json: str) -> str:
        """JSON wrapper around ``handle_request`` for process or HTTP adapters."""
        try:
            request = json.loads(request_json)
        except json.JSONDecodeError as exc:
            return json.dumps({"success": False, "message": f"Invalid JSON request: {exc}"})
        return json.dumps(self.handle_request(request), ensure_ascii=False)

    # === Evaluation API (god-view, not exposed to agents) ===

    def eval_find_synthesis_routes(self, session_id: str, **kwargs) -> Dict[str, Any]:
        """Find synthesis routes using full reaction graph. For evaluation only."""
        env = self.get_session(session_id)
        return env.find_synthesis_routes(**kwargs)

    def eval_find_cheapest_medicinal_pathway(self, session_id: str, **kwargs) -> Dict[str, Any]:
        """Find optimal medicinal pathway. For evaluation only."""
        env = self.get_session(session_id)
        return env.find_cheapest_medicinal_pathway(**kwargs)

    def eval_score_synthesis_route(self, session_id: str, **kwargs) -> Dict[str, Any]:
        """Score a route with full GT details (components, target_profile, generated_plan). For evaluation only."""
        env = self.get_session(session_id)
        return env._score_synthesis_route_full(**kwargs)

    def eval_score_synthesis_plan(self, session_id: str, **kwargs) -> Dict[str, Any]:
        """Score a plan with full GT details (components, target_profile, pathway_evaluation). For evaluation only."""
        env = self.get_session(session_id)
        return env._score_synthesis_plan_full(**kwargs)

    def eval_get_best_submission(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return the best submission (full unsanitized scorecard) or None. For evaluation only."""
        env = self.get_session(session_id)
        return env.get_best_submission()

    def eval_export_world(self, session_id: str) -> Dict[str, Any]:
        """Export the full world data for offline analysis. For evaluation only."""
        env = self.get_session(session_id)
        return env.get_task()
