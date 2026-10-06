"""Register role-playing algorithms with verl's policy registry."""

from recipe.roleplay_rl.algorithms import compute_crpo_foldgrpo_advantage
from verl.trainer.ppo.core_algos import register_adv_est


register_adv_est("crpo_foldgrpo")(compute_crpo_foldgrpo_advantage)
