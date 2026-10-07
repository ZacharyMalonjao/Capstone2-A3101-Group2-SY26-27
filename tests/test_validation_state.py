import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from validation import clear_validation_state, set_validation_state


class ValidationStateTests(unittest.TestCase):
    def test_set_validation_state_replaces_stale_messages(self):
        state = {}

        state = set_validation_state(state, ["Invalid region"], ["Out of range"])
        self.assertEqual(state["validation_errors"], ["Invalid region"])
        self.assertEqual(state["validation_warnings"], ["Out of range"])

        state = set_validation_state(state, ["New issue"], [])
        self.assertEqual(state["validation_errors"], ["New issue"])
        self.assertEqual(state["validation_warnings"], [])

    def test_clear_validation_state_removes_messages(self):
        state = {
            "validation_errors": ["Invalid region"],
            "validation_warnings": ["Out of range"],
        }

        cleared = clear_validation_state(state)

        self.assertEqual(cleared["validation_errors"], [])
        self.assertEqual(cleared["validation_warnings"], [])


if __name__ == "__main__":
    unittest.main()
