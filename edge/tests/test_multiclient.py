import unittest


from src.multiclient import (
    build_client_command,
    build_client_result_tag,
    partition_sample_indices,
    summarize_multiclient_metrics,
)


class MultiClientHelpersTests(unittest.TestCase):
    def test_partition_distinct_spreads_samples_without_overlap(self):
        assignments = partition_sample_indices(
            total_samples=8,
            num_clients=4,
            workload_mode="distinct",
            pilot_samples=8,
        )

        self.assertEqual(assignments, [[0, 1], [2, 3], [4, 5], [6, 7]])

    def test_partition_same_reuses_the_same_subset_for_each_client(self):
        assignments = partition_sample_indices(
            total_samples=8,
            num_clients=4,
            workload_mode="same",
            pilot_samples=8,
        )

        self.assertEqual(assignments, [[0, 1], [0, 1], [0, 1], [0, 1]])

    def test_summarize_multiclient_metrics_reports_throughput_and_energy(self):
        metrics = summarize_multiclient_metrics(
            entries=[
                {"output_length": 12, "gpu_power_integral_joules": 8.0},
                {"output_length": 18, "gpu_power_integral_joules": 12.0},
            ],
            makespan=5.0,
        )

        self.assertEqual(metrics["total_output_tokens"], 30)
        self.assertEqual(metrics["num_completed_samples"], 2)
        self.assertEqual(metrics["total_cloud_energy_joules"], 20.0)
        self.assertEqual(metrics["token_throughput_tps"], 6.0)
        self.assertEqual(metrics["sample_throughput_sps"], 0.4)
        self.assertAlmostEqual(metrics["energy_per_token_joules"], 20.0 / 30.0)
        self.assertEqual(metrics["energy_per_sample_joules"], 10.0)
        self.assertIsNone(metrics["sample_window_makespan_seconds"])
        self.assertIsNone(metrics["sample_window_token_throughput_tps"])
        self.assertIsNone(metrics["sample_window_sample_throughput_sps"])

    def test_summarize_multiclient_metrics_reports_sample_window_when_timestamps_exist(self):
        metrics = summarize_multiclient_metrics(
            entries=[
                {
                    "output_length": 12,
                    "gpu_power_integral_joules": 8.0,
                    "sample_started_at": 10.0,
                    "sample_finished_at": 16.0,
                },
                {
                    "output_length": 18,
                    "gpu_power_integral_joules": 12.0,
                    "sample_started_at": 12.0,
                    "sample_finished_at": 18.0,
                },
            ],
            makespan=20.0,
        )

        self.assertEqual(metrics["sample_window_makespan_seconds"], 8.0)
        self.assertEqual(metrics["sample_window_token_throughput_tps"], 30.0 / 8.0)
        self.assertEqual(metrics["sample_window_sample_throughput_sps"], 2.0 / 8.0)

    def test_build_client_result_tag_is_stable_and_distinct(self):
        self.assertEqual(build_client_result_tag("pilot", 0), "pilot_client0")
        self.assertEqual(build_client_result_tag("pilot", 3), "pilot_client3")

    def test_build_client_command_includes_range_offset_and_tag(self):
        command = build_client_command(
            python_bin="python3",
            dataset="humaneval",
            algorithm="pipesd",
            start_index=4,
            end_index=7,
            task_id_offset=2000,
            result_tag="pilot_client1",
            extra_args=["--bandwidth_MBps", "2.5"],
        )

        self.assertEqual(command[0], "python3")
        self.assertIn("app/run_edge.py", command[1])
        self.assertIn("--dataset", command)
        self.assertIn("humaneval", command)
        self.assertIn("--algorithm", command)
        self.assertIn("pipesd", command)
        self.assertIn("--start_index_of_sample", command)
        self.assertIn("4", command)
        self.assertIn("--end_index_of_sample", command)
        self.assertIn("7", command)
        self.assertIn("--task_id_offset", command)
        self.assertIn("2000", command)
        self.assertIn("--result_tag", command)
        self.assertIn("pilot_client1", command)
        self.assertEqual(command[-2:], ["--bandwidth_MBps", "2.5"])


if __name__ == "__main__":
    unittest.main()
