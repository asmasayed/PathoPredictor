"""Cross-module glue without modifying package internals (e.g. module3_lstm)."""

from src.integration.seir_projection import (
    project_seir_using_module2_parameters,
    run_hybrid_seir_with_module2_rates,
)

__all__ = ["project_seir_using_module2_parameters", "run_hybrid_seir_with_module2_rates"]
