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

    def eval_compute_optimal_cost(self, session_id: str, **kwargs) -> Dict[str, Any]:
        """Compute ground-truth minimum cost for the task. For evaluation only."""
        env = self.get_session(session_id)
        constraints = env._get_constraints()
        return env.compute_optimal_cost(
            min_medicinal_value=kwargs.get("min_medicinal_value", constraints["min_medicinal"]),
            max_toxicity=kwargs.get("max_toxicity", constraints["max_toxicity"]),
            min_yield_g=kwargs.get("min_yield_g", constraints["min_yield_g"]),
            max_time_seconds=kwargs.get("max_time_seconds", constraints["max_time_seconds"]),
            required_phase=kwargs.get("required_phase", constraints.get("required_phase")),
            phase_temp_C=kwargs.get("phase_temp_C", constraints.get("phase_temp_C", 25.0)),
            max_routes_per_target=kwargs.get("max_routes_per_target", 5),
            max_steps=kwargs.get("max_steps", 6),
        )

    def eval_get_best_submission(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return the best submission or None. For evaluation only."""
        env = self.get_session(session_id)
        return env.get_best_submission()

    def eval_get_session_metrics(self, session_id: str) -> Dict[str, Any]:
        """Return session metrics (time, cost, production). For evaluation only."""
        env = self.get_session(session_id)
        return {
            "elapsed_time": env._elapsed_time,
            "time_budget": env._time_budget(),
            "total_cost": env._total_cost,
            "total_produced": dict(env._total_produced),
            "finished": env._finished,
        }

    def eval_compute_score(self, session_id: str, **kwargs) -> Dict[str, Any]:
        """Compute the final episode score. For evaluation only.

        Score = min(yield_ratio, 1.0) * min(cost_efficiency, 1.0)
        where:
          yield_ratio = total_yield / required_yield
          cost_efficiency = optimal_cost / actual_cost

        Returns 0.0 if no qualifying compound was produced.
        """
        env = self.get_session(session_id)
        constraints = env._get_constraints()

        best = env.get_best_submission()
        if best is None:
            return {
                "score": 0.0,
                "reason": "no_qualifying_submission",
                "details": {
                    "total_cost": round(env._total_cost, 2),
                    "elapsed_time": round(env._elapsed_time, 1),
                },
            }

        actual_cost = best["total_experiment_cost"]
        actual_yield = best["total_yield"]
        required_yield = constraints["min_yield_g"]

        optimal_result = env.compute_optimal_cost(
            min_medicinal_value=constraints["min_medicinal"],
            max_toxicity=constraints["max_toxicity"],
            min_yield_g=required_yield,
            max_time_seconds=constraints["max_time_seconds"],
            required_phase=constraints.get("required_phase"),
            phase_temp_C=constraints.get("phase_temp_C", 25.0),
        )

        if not optimal_result.get("found") or optimal_result.get("optimal_cost") is None:
            cost_efficiency = 1.0
        else:
            optimal_cost = optimal_result["optimal_cost"]
            cost_efficiency = min(optimal_cost / max(actual_cost, 1e-9), 1.0)

        yield_ratio = min(actual_yield / max(required_yield, 1e-9), 1.0)
        score = round(yield_ratio * cost_efficiency, 4)

        return {
            "score": score,
            "reason": "scored",
            "yield_ratio": round(yield_ratio, 4),
            "cost_efficiency": round(cost_efficiency, 4),
            "details": {
                "actual_cost": actual_cost,
                "optimal_cost": optimal_result.get("optimal_cost"),
                "actual_yield": actual_yield,
                "required_yield": required_yield,
                "target_compound": best["target_compound"],
                "elapsed_time": round(env._elapsed_time, 1),
            },
        }

    def eval_export_world(self, session_id: str) -> Dict[str, Any]:
        """Export the full world data for offline analysis. For evaluation only."""
        env = self.get_session(session_id)
        return env.get_task()
