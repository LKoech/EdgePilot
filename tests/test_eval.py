"""Tests for the eval harness — verifies scenarios and report metrics."""


from eval.harness import (
    build_eval_registry,
    run_scenario_with_recovery,
    run_scenario_without_recovery,
)
from eval.scenarios import SCENARIOS


class TestEvalScenarios:
    def test_all_scenarios_have_unique_ids(self) -> None:
        ids = [s.id for s in SCENARIOS]
        assert len(ids) == len(set(ids)), "Duplicate scenario IDs found"

    def test_scenario_count(self) -> None:
        assert len(SCENARIOS) >= 20, f"Need >= 20 scenarios, have {len(SCENARIOS)}"

    def test_happy_path_scenario(self) -> None:
        registry = build_eval_registry()
        scenario = next(s for s in SCENARIOS if s.id == "time_basic")
        _, eval_result = run_scenario_with_recovery(scenario, registry)
        assert eval_result.success is True
        assert eval_result.recovery_events == 0
        assert eval_result.passed is True

    def test_validation_failure_scenario(self) -> None:
        registry = build_eval_registry()
        scenario = next(s for s in SCENARIOS if s.id == "val_unknown_tool")
        _, eval_result = run_scenario_with_recovery(scenario, registry)
        assert eval_result.success is True
        assert eval_result.recovery_events > 0
        assert eval_result.validation_failures > 0
        assert eval_result.passed is True

    def test_execution_failure_scenario(self) -> None:
        registry = build_eval_registry()
        scenario = next(
            s for s in SCENARIOS if s.id == "exec_flaky_fails_then_recovers"
        )
        _, eval_result = run_scenario_with_recovery(scenario, registry)
        assert eval_result.success is True
        assert eval_result.recovery_events > 0
        assert eval_result.execution_failures > 0

    def test_budget_exhaustion_scenario(self) -> None:
        registry = build_eval_registry()
        scenario = next(
            s for s in SCENARIOS if s.id == "exec_flaky_exhausts_budget"
        )
        _, eval_result = run_scenario_with_recovery(scenario, registry)
        assert eval_result.success is False
        assert eval_result.passed is True  # expected to fail

    def test_recovery_improves_completion(self) -> None:
        """Scenarios that need recovery should fail without it."""
        recovery_scenarios = [
            s for s in SCENARIOS
            if s.expect_recovery and s.expect_success
        ]
        assert len(recovery_scenarios) > 0

        for scenario in recovery_scenarios:
            # With recovery: should succeed
            _, with_result = run_scenario_with_recovery(
                scenario, build_eval_registry()
            )
            assert with_result.success is True, (
                f"{scenario.id} should succeed with recovery"
            )

            # Without recovery: should fail (only first call matters)
            without_success = run_scenario_without_recovery(
                scenario, build_eval_registry()
            )
            # If first call fails, without recovery it should fail
            if with_result.recovery_events > 0:
                assert without_success is False, (
                    f"{scenario.id} should fail without recovery"
                )
