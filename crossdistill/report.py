"""Paired, group-bootstrap comparisons; invalid evaluations stay missing."""
import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from opd.checkpoints import atomic_json
from opd.data import read_rows


def compare(before, after, repetitions=2000):
    old = {r['id']: r for r in before}
    new = {r['id']: r for r in after}
    if len(old) != len(before) or len(new) != len(after) or old.keys() != new.keys():
        raise ValueError('Paired evaluation requires exactly the same unique sample IDs')
    grouped = defaultdict(list)
    coverage = defaultdict(lambda: [0, 0])
    for key in old:
        a, b = old[key], new[key]
        if a['task_id'] != b['task_id'] or a.get('group_id', key) != b.get('group_id', key):
            raise ValueError(f'Pair metadata mismatch: {key}')
        task = a['task_id']
        ea, eb = a['evaluation'], b['evaluation']
        dims = set(ea.get('scores', {})) | set(eb.get('scores', {}))
        coverage[task][0] += 1
        valid = ea['valid'] and eb['valid']
        coverage[task][1] += int(valid)
        if not valid:
            continue
        for dim in dims:
            if dim in ea['scores'] and dim in eb['scores']:
                va = ea['scores'][dim] if ea['constraint_pass'] else 0.
                vb = eb['scores'][dim] if eb['constraint_pass'] else 0.
                grouped[(task, dim)].append((a.get('group_id', key), va, vb))
    rows = []
    for (task, dim), records in sorted(grouped.items()):
        clusters = defaultdict(list)
        for group, a, b in records:
            clusters[group].append((a, b))
        # Equal weight per independent group, not per correlated turn.
        groups = [(statistics.mean(a for a, _ in items), statistics.mean(b for _, b in items))
                  for items in clusters.values()]
        differences = [b-a for a, b in groups]
        rng = random.Random(42)
        samples = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(repetitions)) if len(groups) > 1 else []
        rows.append({'task': task, 'dimension': dim, 'paired_samples': len(records), 'groups': len(groups),
                     'baseline': statistics.mean(a for a, _ in groups), 'distilled': statistics.mean(b for _, b in groups),
                     'delta': statistics.mean(differences),
                     'ci95_low': samples[int(.025 * repetitions)] if samples else None,
                     'ci95_high': samples[min(repetitions-1, int(.975 * repetitions))] if samples else None,
                     'total_samples': coverage[task][0], 'both_valid_samples': coverage[task][1]})
    return rows, {t: {'total': n, 'both_valid': v} for t, (n, v) in coverage.items()}


def summarize(root):
    root = Path(root)
    result = json.loads((root / 'results.json').read_text())
    if result['dry_run']:
        raise ValueError('A dry-run has no measured results')
    rows, summaries = [], []
    for run in result['runs']:
        metrics, coverage = compare(list(read_rows(run['baseline'])), list(read_rows(run['evaluation'])))
        prefix = {k: run[k] for k in ('student', 'method', 'seed')}
        rows.extend({**prefix, **r} for r in metrics)
        tasks = defaultdict(list)
        for row in metrics:
            tasks[row['task']].append(row['delta'])
        complete = bool(coverage) and all(c['total'] == c['both_valid'] for c in coverage.values()) and set(tasks) == set(coverage)
        summaries.append({**prefix, 'task_macro_delta_on_valid_pairs': statistics.mean(statistics.mean(v) for v in tasks.values()) if tasks else None,
                          'complete_pair_coverage': complete, 'coverage': coverage,
                          'note': 'Macro is diagnostic when coverage is incomplete; inspect task/dimension CSV.'})
    output = {'runs': summaries, 'rows': rows,
              'scope': 'Static held-out prompt proxy; no claim of full interactive task success.',
              'ci': '95% percentile bootstrap of paired group means within task/dimension; 2000 resamples; no multiplicity correction.',
              'conclusion': 'Inspect measured gains and intervals; the code does not predetermine that transfer succeeds.'}
    atomic_json(root / 'comparison.json', output)
    with (root / 'comparison.csv').open('w', newline='') as f:
        fields = list(rows[0]) if rows else ['student', 'method', 'seed', 'task', 'dimension', 'delta']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_root', type=Path)
    summarize(parser.parse_args().output_root)
