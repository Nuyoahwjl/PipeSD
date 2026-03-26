import importlib.util
import builtins
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "app" / "run_edge.py"


def install_stub_modules():
    torch = types.ModuleType("torch")

    def no_grad():
        def decorator(fn):
            return fn
        return decorator

    torch.no_grad = no_grad
    torch.multiprocessing = types.ModuleType("torch.multiprocessing")
    torch.nn = types.ModuleType("torch.nn")
    torch.nn.functional = types.ModuleType("torch.nn.functional")
    sys.modules["torch"] = torch
    sys.modules["torch.multiprocessing"] = torch.multiprocessing
    sys.modules["torch.nn"] = torch.nn
    sys.modules["torch.nn.functional"] = torch.nn.functional

    transformers = types.ModuleType("transformers")
    transformers.utils = types.SimpleNamespace(logging=types.SimpleNamespace(set_verbosity=lambda *_: None))
    sys.modules["transformers"] = transformers

    msgpack = types.ModuleType("msgpack")
    msgpack.packb = lambda payload: payload
    sys.modules["msgpack"] = msgpack

    pandas = types.ModuleType("pandas")
    pandas.DataFrame = object
    sys.modules["pandas"] = pandas

    llama_cpp = types.ModuleType("llama_cpp")

    class StubLlama:
        def __init__(self, *args, **kwargs):
            self.n_tokens = 0

    llama_cpp.Llama = StubLlama
    sys.modules["llama_cpp"] = llama_cpp

    util = types.ModuleType("src.util")
    util.seed_everything = lambda *_: None
    util.parse_arguments = lambda: None
    sys.modules["src.util"] = util

    engine = types.ModuleType("src.engine")

    class Decoding:
        def __init__(self, args):
            self.args = args

    engine.Decoding = Decoding
    sys.modules["src.engine"] = engine

    src_pkg = types.ModuleType("src")
    src_pkg.util = util
    src_pkg.engine = engine
    sys.modules["src"] = src_pkg

    sys.modules.pop("skopt", None)
    sys.modules.pop("skopt.space", None)


class RunEdgeOptionalSkoptTests(unittest.TestCase):
    def test_import_succeeds_without_skopt_installed(self):
        module_name = "run_edge_without_skopt"
        sys.modules.pop(module_name, None)
        saved_modules = dict(sys.modules)

        try:
            install_stub_modules()
            spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            original_import = builtins.__import__

            def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "skopt" or name.startswith("skopt."):
                    raise ModuleNotFoundError("No module named 'skopt'")
                return original_import(name, globals, locals, fromlist, level)

            with mock.patch("builtins.__import__", side_effect=guarded_import):
                spec.loader.exec_module(module)
        finally:
            sys.modules.clear()
            sys.modules.update(saved_modules)

        self.assertTrue(hasattr(module, "CloudEdgeSpeculativeEval"))

    def test_import_preserves_existing_cuda_visible_devices(self):
        module_name = "run_edge_preserve_cuda_visible_devices"
        sys.modules.pop(module_name, None)
        saved_modules = dict(sys.modules)

        try:
            install_stub_modules()
            spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None

            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False):
                spec.loader.exec_module(module)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "1")
        finally:
            sys.modules.clear()
            sys.modules.update(saved_modules)


if __name__ == "__main__":
    unittest.main()
