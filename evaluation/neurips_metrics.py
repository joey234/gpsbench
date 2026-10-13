"""NeurIPS headline metrics applied to the current validated answer extraction.

Distance, Bearing and Area use clipped 100 * (1 - mean normalized error),
conditional on finite parsed errors below 1000, as in the historical scorer.
Interpolation uses the NeurIPS main-text +/-0.5-degree threshold. Other task
matching rules retain the current fixes. No historical model answers are changed.
"""
from __future__ import annotations
import math
from copy import deepcopy

VERSION = 'gpsbench-neurips-metrics-restored-v1'
ERROR_TASKS = frozenset({'Distance', 'Bearing', 'Area'})
INTERPOLATION_TOLERANCE = .5
MAX_RELATIVE_ERROR = 1000.


def included_error(row):
    e = row.get('error')
    return bool(row.get('parsed')) and e is not None and math.isfinite(e) and 0 <= e < MAX_RELATIVE_ERROR


def task_score(rows, task):
    if task in ERROR_TASKS:
        errors = [r['error'] for r in rows if included_error(r)]
        return max(0., 100. * (1. - sum(errors) / len(errors))) if errors else 0.
    return 100. * sum(bool(r['correct']) for r in rows) / len(rows) if rows else 0.


def restored_rows(rows, task, sources):
    assert len(rows) == len(sources)
    result = deepcopy(rows)
    if task == 'Interpolation':
        for r, s in zip(result, sources):
            p, g = r.get('predicted'), s['ground_truth']['interpolated_point']
            r['correct'] = p is not None and abs(p[0] - g['lat']) <= .5 and abs(p[1] - g['lon']) <= .5
    return result
