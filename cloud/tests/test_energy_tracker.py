import importlib.util
import sys
import time
import types
import unittest
from pathlib import Path


def load_energy_tracker():
    torch = types.ModuleType("torch")
    torch.nn = types.ModuleType("torch.nn")
    torch.nn.functional = types.ModuleType("torch.nn.functional")
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = torch.nn
    sys.modules["torch.nn.functional"] = torch.nn.functional

    module_path = Path(__file__).parents[1] / "src" / "util.py"
    spec = importlib.util.spec_from_file_location("cloud_util_energy_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.EnergyTracker


EnergyTracker = load_energy_tracker()


class ConstantPowerMonitor:
    supports_power = True

    def read_power_watts(self):
        return 100.0


class RecordingTask:
    def __init__(self):
        self.task_id = 1
        self.total_gpu_power_integral_joules = 0.0
        self.last_verify_power_integral = 0.0
        self.measurements = []

    def record_energy_measurement(self, **measurement):
        self.measurements.append(measurement)
        self.total_gpu_power_integral_joules += measurement["energy_joules"]


class EnergyTrackerTests(unittest.TestCase):
    def test_records_energy_duration_and_sample_count_for_one_stage(self):
        task = RecordingTask()

        with EnergyTracker(
            ConstantPowerMonitor(), task, "verify_total", sample_interval=0.001
        ):
            time.sleep(0.006)

        self.assertEqual(len(task.measurements), 1)
        measurement = task.measurements[0]
        self.assertEqual(measurement["stage"], "verify_total")
        self.assertGreaterEqual(measurement["sample_count"], 2)
        self.assertGreater(measurement["duration_seconds"], 0.0)
        self.assertGreater(measurement["energy_joules"], 0.0)
        self.assertAlmostEqual(
            measurement["energy_joules"],
            100.0 * measurement["duration_seconds"],
            delta=0.2,
        )


if __name__ == "__main__":
    unittest.main()
