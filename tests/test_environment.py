from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from web.environment import load_project_environment


class EnvironmentLoadingTests(unittest.TestCase):
    def test_dotenv_loads_missing_values_without_overriding_process_environment(self) -> None:
        existing_key = "RLAE_TEST_EXISTING_ENVIRONMENT_VALUE"
        missing_key = "RLAE_TEST_MISSING_ENVIRONMENT_VALUE"
        original_existing = os.environ.get(existing_key)
        original_missing = os.environ.get(missing_key)
        os.environ[existing_key] = "from-process"
        os.environ.pop(missing_key, None)
        try:
            with tempfile.TemporaryDirectory() as directory:
                dotenv_path = Path(directory) / ".env"
                dotenv_path.write_text(
                    f"{existing_key}=from-file\n{missing_key}=loaded-from-file\n",
                    encoding="utf-8",
                )
                self.assertTrue(load_project_environment(dotenv_path))
            self.assertEqual(os.environ[existing_key], "from-process")
            self.assertEqual(os.environ[missing_key], "loaded-from-file")
        finally:
            if original_existing is None:
                os.environ.pop(existing_key, None)
            else:
                os.environ[existing_key] = original_existing
            if original_missing is None:
                os.environ.pop(missing_key, None)
            else:
                os.environ[missing_key] = original_missing


if __name__ == "__main__":
    unittest.main()
