"""Supplemental compatibility suite identifiers, with no global registration."""

TASKS = ("hitom", "paratomi", "mistakes", "twinvoice", "socsci210", "sim_doc", "sim_math")
BENCHMARK_IDS = frozenset(f"supplemental_{task}" for task in TASKS)
EVALUATED_ROLES = {
    f"supplemental_{task}": "evaluated_user" if task in {"sim_doc", "sim_math"} else "evaluated_model"
    for task in TASKS
}
REVISION = "supplemental-local-eval-snapshot-v1"


def build_adapter(benchmark_id, config):
    from .adapter import SupplementalAdapter
    adapter = SupplementalAdapter(benchmark_id, config=config)
    roles = [EVALUATED_ROLES[benchmark_id]]
    if adapter.interactive:
        roles.extend(("fixed_assistant", "judge"))
    return adapter, tuple(roles), {}, {}
