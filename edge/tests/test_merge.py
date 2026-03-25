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


if __name__ == "__main__":
    unittest.main()
