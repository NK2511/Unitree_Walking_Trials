"""Swappable Policy Interface for Unitree G1 PhD Research.

To use a different brain architecture, change the import in your experiment:
    from policies.mlp_policy import MlpPolicy          # baseline
    from policies.modular import ModularPolicy          # Approach 1
    from policies.rnn import RnnPolicy                  # future
"""

from policies.mlp_policy import MlpPolicy  # noqa: F401
