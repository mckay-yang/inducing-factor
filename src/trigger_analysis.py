"""Legacy compatibility module.

The inducing-factor workflow now lives in the `inducing_factor` package.
"""

from inducing_factor.core.attribution_system import EcoDisturbanceAttributionSystem

__all__ = ["EcoDisturbanceAttributionSystem"]
