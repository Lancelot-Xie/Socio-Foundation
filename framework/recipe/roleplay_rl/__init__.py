"""Role-playing RL extensions for Simulation.

The package intentionally keeps task reward shaping separate from the policy
objective:

* :mod:`reward_pipeline` implements AnthroDial-style adaptive multi-dimensional
  rewards and judge-confidence calibration.
* :mod:`algorithms` implements the CRPO dual-stream advantage estimator with a
  FoldGRPO-compatible multi-agent de-duplication path.
* :mod:`anchors` selects and marks profile-ablated generic anchor rollouts.
"""

from recipe.roleplay_rl.reward_pipeline import RoleplayRewardProcessor

__all__ = ["RoleplayRewardProcessor"]
