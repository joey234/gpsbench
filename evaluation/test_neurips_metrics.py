import unittest
from evaluation.neurips_metrics import task_score, included_error, restored_rows

class NeurIPSMetricTests(unittest.TestCase):
    def test_error_score_is_not_tolerance_accuracy(self):
        rows = [{'parsed': True, 'error': .02, 'correct': True}, {'parsed': True, 'error': .20, 'correct': False}]
        self.assertAlmostEqual(task_score(rows, 'Distance'), 89.)
    def test_invalid_and_extreme_errors_have_original_exclusion_rule(self):
        rows = [{'parsed': True, 'error': .1}, {'parsed': False, 'error': None}, {'parsed': True, 'error': 1000.}]
        self.assertAlmostEqual(task_score(rows, 'Area'), 90.)
        self.assertFalse(included_error(rows[-1]))
    def test_clipping_occurs_after_averaging(self):
        rows = [{'parsed': True, 'error': 0.}, {'parsed': True, 'error': 3.}]
        self.assertEqual(task_score(rows, 'Area'), 0.)
    def test_missing_numerical_answers_cannot_produce_perfect_score(self):
        self.assertEqual(task_score([{'parsed': False, 'error': None}], 'Distance'), 0.)
    def test_interpolation_uses_neurips_half_degree_rule(self):
        rows = [{'correct': False, 'parsed': True, 'predicted': [10.5, 20.5]}, {'correct': False, 'parsed': True, 'predicted': [10.5, 20.501]}, {'correct': False, 'parsed': False, 'predicted': None}]
        sources = [{'ground_truth': {'interpolated_point': {'lat': 10., 'lon': 20.}}}] * 3
        changed = restored_rows(rows, 'Interpolation', sources)
        self.assertEqual([r['correct'] for r in changed], [True, False, False])
        self.assertFalse(rows[0]['correct'])
    def test_other_tasks_remain_binary(self):
        self.assertEqual(task_score([{'correct': True}, {'correct': False}], 'Place'), 50.)
if __name__ == '__main__':
    unittest.main()
