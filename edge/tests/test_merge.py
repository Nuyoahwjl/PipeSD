import importlib
import sys
import unittest


class MergeModuleTests(unittest.TestCase):
    def test_merge_module_imports_without_pandas_installed(self):
        previous_pandas = sys.modules.pop("pandas", None)
        previous_merge = sys.modules.pop("src.merge", None)
        original_import = __import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "pandas":
                raise ModuleNotFoundError("No module named 'pandas'")
            return original_import(name, globals, locals, fromlist, level)

        try:
            import builtins

            builtins.__import__, saved_import = guarded_import, builtins.__import__
            merge_module = importlib.import_module("src.merge")
            self.assertTrue(hasattr(merge_module, "dynamic_token_scheduling_dp"))
        finally:
            import builtins

            builtins.__import__ = saved_import
            sys.modules.pop("src.merge", None)
            if previous_merge is not None:
                sys.modules["src.merge"] = previous_merge
            if previous_pandas is not None:
                sys.modules["pandas"] = previous_pandas

    def test_paper_scheduler_starts_with_twenty_token_window(self):
        from src.merge import PaperDPScheduler

        scheduler = PaperDPScheduler(alpha=0.02, beta=0.01, gamma=0.03)

        self.assertEqual(sum(scheduler.plan()), 20)

    def test_paper_scheduler_updates_window_from_recent_draft_lengths(self):
        from src.merge import PaperDPScheduler

        scheduler = PaperDPScheduler(alpha=0.02, beta=0.01, gamma=0.03)
        scheduler.observe_draft_length(4)
        scheduler.observe_draft_length(6)

        self.assertEqual(scheduler.window, 5)
        self.assertEqual(sum(scheduler.plan()), 5)

    def test_paper_scheduler_uses_twenty_percent_parameter_gate(self):
        from src.merge import PaperDPScheduler

        scheduler = PaperDPScheduler(alpha=1.0, beta=1.0, gamma=1.0)

        self.assertFalse(scheduler.update_parameters(alpha=1.2))
        self.assertTrue(scheduler.update_parameters(alpha=1.21))

    def test_plan_index_cycles_across_windows(self):
        from src.merge import next_plan_index

        plan = [1, 3, 8]
        index = 0
        observed = []
        for _ in range(7):
            observed.append(plan[index])
            index = next_plan_index(index, plan)

        self.assertEqual(observed, [1, 3, 8, 1, 3, 8, 1])

    def test_online_environment_estimator_regresses_comm_and_generation(self):
        from src.merge import OnlineEnvironmentEstimator

        estimator = OnlineEnvironmentEstimator(history_size=100, min_comm_samples=2)
        estimator.observe_communication(1, 0.15)
        estimator.observe_communication(3, 0.35)
        estimator.observe_generation(2, 0.08)

        estimates = estimator.estimate()

        self.assertAlmostEqual(estimates["alpha"], 0.05, places=6)
        self.assertAlmostEqual(estimates["beta"], 0.1, places=6)
        self.assertAlmostEqual(estimates["gamma"], 0.04, places=6)

    def test_online_environment_estimator_reports_missing_bootstrap_sizes(self):
        from src.merge import OnlineEnvironmentEstimator

        estimator = OnlineEnvironmentEstimator(history_size=100, min_comm_samples=8)
        estimator.observe_communication(1, 0.1)
        estimator.observe_communication(3, 0.3)

        self.assertEqual(estimator.missing_batch_sizes(range(1, 9)), [2, 4, 5, 6, 7, 8])


if __name__ == "__main__":
    unittest.main()
