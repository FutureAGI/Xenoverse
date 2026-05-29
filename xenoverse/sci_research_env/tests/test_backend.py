from __future__ import annotations

import json
import tempfile
import unittest

from xenoverse.sci_research import LegacyChemistryEnvironment, SciResearchBackend


class SciResearchBackendTests(unittest.TestCase):
    def _create_session(self, backend: SciResearchBackend, seed: int = 7) -> dict:
        response = backend.handle_request(
            {
                "action": "sample_environment",
                "sampler_kwargs": {"seed": seed, "max_attempts": 20},
            }
        )
        self.assertTrue(response["success"])
        self.assertIn("session_id", response)
        self.assertEqual(response["observation"]["task_type"], "SCI_RESEARCH")
        self.assertIn("task_description", response)
        return response

    def test_sample_environment_returns_session_and_tools(self) -> None:
        backend = SciResearchBackend()

        response = self._create_session(backend, seed=11)
        public_state = response["observation"]["public_state"]
        tools = response["observation"]["function_tools"]

        self.assertEqual(response["task_type"], "SCI_RESEARCH")
        self.assertIn("objective", response["task_description"])
        self.assertIn("world_id", public_state)
        self.assertIn("inventory_size", public_state)
        self.assertTrue(any(tool["function"]["name"] == "list_purchasable" for tool in tools))
        self.assertTrue(any(tool["function"]["name"] == "restate_task_goal" for tool in tools))

    def test_backend_dispatch_function_call_round_trip(self) -> None:
        backend = SciResearchBackend()
        session = self._create_session(backend, seed=13)
        session_id = session["session_id"]

        purchasable_response = backend.handle_request(
            {
                "action": "dispatch_function_call",
                "session_id": session_id,
                "function_call": {"name": "list_purchasable", "arguments": {}},
            }
        )
        self.assertTrue(purchasable_response["success"])
        purchasable = purchasable_response["result"]
        self.assertIsInstance(purchasable, dict)
        self.assertTrue(purchasable)

        first_name = next(iter(purchasable))
        purchase_response = backend.handle_request(
            {
                "action": "dispatch_function_call",
                "session_id": session_id,
                "function_call": {
                    "name": "purchase",
                    "arguments": {"chemical_name": first_name, "amount_grams": 5.0},
                },
            }
        )
        self.assertTrue(purchase_response["success"])

        inventory_response = backend.handle_request(
            {
                "action": "dispatch_function_call",
                "session_id": session_id,
                "function_call": {"name": "get_inventory", "arguments": {}},
            }
        )
        self.assertTrue(inventory_response["success"])
        inventory = inventory_response["result"]
        self.assertIn(first_name, inventory)
        self.assertGreater(inventory[first_name]["amount_g"], 0)

        recap_response = backend.handle_request(
            {
                "action": "dispatch_function_call",
                "session_id": session_id,
                "function_call": {"name": "recap_recent_activity", "arguments": {"last_n": 2}},
            }
        )
        self.assertTrue(recap_response["success"])
        self.assertGreaterEqual(recap_response["num_returned"], 1)

    def test_json_request_and_close_session_flow(self) -> None:
        backend = SciResearchBackend()

        raw_response = backend.handle_json_request(
            json.dumps(
                {
                    "action": "sample_environment",
                    "sampler_kwargs": {"seed": 17, "max_attempts": 20},
                }
            )
        )
        session = json.loads(raw_response)
        self.assertTrue(session["success"])
        session_id = session["session_id"]

        summary_response = backend.handle_request(
            {"action": "get_session_summary", "session_id": session_id}
        )
        self.assertTrue(summary_response["success"])
        self.assertEqual(
            summary_response["summary"]["world_id"],
            session["observation"]["public_state"]["world_id"],
        )

        close_response = backend.handle_request(
            {"action": "close_session", "session_id": session_id}
        )
        self.assertTrue(close_response["success"])

        missing_response = backend.handle_request(
            {"action": "get_session_summary", "session_id": session_id}
        )
        self.assertFalse(missing_response["success"])

    def test_score_synthesis_plan_returns_scorecard(self) -> None:
        backend = SciResearchBackend()
        session = self._create_session(backend, seed=19)
        session_id = session["session_id"]

        internal_task_response = backend.handle_request(
            {"action": "export_internal_task", "session_id": session_id}
        )
        self.assertTrue(internal_task_response["success"])
        internal_task = internal_task_response["task"]

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as handle:
            json.dump(internal_task["world"], handle)
            world_path = handle.name

        try:
            legacy_env = LegacyChemistryEnvironment(world_path)
            search = legacy_env.find_cheapest_medicinal_pathway(
                min_medicinal_value=3.0,
                max_toxicity=4.0,
                per_m1_g=10.0,
            )
            self.assertTrue(search["success"])
            self.assertTrue(search["found"])

            target = search["best_pathway"]["target"]
            target_id = legacy_env._name_to_id(target)
            chains = legacy_env._find_reaction_chains(target_id, max_routes=1, max_steps=8)
            self.assertTrue(chains)
            steps = legacy_env._build_pathway_steps(chains[0], per_m1_g=10.0)
        finally:
            import os
            os.unlink(world_path)

        score_response = backend.handle_request(
            {
                "action": "dispatch_function_call",
                "session_id": session_id,
                "function_call": {
                    "name": "score_synthesis_plan",
                    "arguments": {"target_compound": target, "steps": steps},
                },
            }
        )
        self.assertTrue(score_response["success"])
        self.assertIn("aggregate_score", score_response)
        self.assertIn("verdict", score_response)
        self.assertIn("components", score_response)
        self.assertIn("reasoning", score_response)

    def test_score_synthesis_route_accepts_high_level_route(self) -> None:
        backend = SciResearchBackend()
        session = self._create_session(backend, seed=23)
        session_id = session["session_id"]

        internal_task_response = backend.handle_request(
            {"action": "export_internal_task", "session_id": session_id}
        )
        internal_task = internal_task_response["task"]

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as handle:
            json.dump(internal_task["world"], handle)
            world_path = handle.name

        try:
            legacy_env = LegacyChemistryEnvironment(world_path)
            search = legacy_env.find_cheapest_medicinal_pathway(
                min_medicinal_value=3.0,
                max_toxicity=4.0,
                per_m1_g=10.0,
            )
            self.assertTrue(search["success"])
            self.assertTrue(search["found"])
            best = search["best_pathway"]
            route_steps = [
                {
                    "reactants": step["reactants"],
                    "catalysts": step["catalysts"],
                }
                for step in best["route"]["steps"]
            ]
        finally:
            import os
            os.unlink(world_path)

        route_score = backend.handle_request(
            {
                "action": "dispatch_function_call",
                "session_id": session_id,
                "function_call": {
                    "name": "score_synthesis_route",
                    "arguments": {
                        "target_compound": best["target"],
                        "steps": route_steps,
                    },
                },
            }
        )
        self.assertTrue(route_score["success"])
        self.assertIn("aggregate_score", route_score)
        self.assertIn("generated_plan", route_score)
        self.assertEqual(route_score["submitted_route"]["target_compound"], best["target"])


if __name__ == "__main__":
    unittest.main()
