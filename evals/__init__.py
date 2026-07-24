"""Development-only Braintrust evaluations for Magpie.

Nothing in :mod:`magpie` imports this package.  The optional dependencies in
``requirements-eval.txt`` therefore stay outside the production runtime.
"""

PROJECT_NAME = "magpie-claim-engine"
TASK_NAMES = ("visible", "memory", "relations", "collision")
VARIANT_NAMES = ("V0", "V1", "V2", "V3", "V4")

