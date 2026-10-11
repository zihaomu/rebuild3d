#!/usr/bin/env python3
"""Guard against false performance passes caused by missing or uncertain evidence."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('summary', Path(__file__).with_name('summarize-ui-performance.py'))
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)


def at(seconds):
    return f'2026-10-10T00:{seconds // 60:02d}:{seconds % 60:02d}+00:00'


def call(lo, hi, **extra):
    return {'startedAtUTC': at(lo), 'finishedAtUTC': at(hi), **extra}


def sample(label, index, lower, upper):
    return {'dataset': 'A', 'label': label, 'round': index,
            'inferenceSeconds': 20 if label == 'candidate' else 30,
            'inferencePeakPhysicalFootprintBytes': 60 if label == 'candidate' else 100,
            'swapUsedDeltaBytes': 0, 'swapOutDeltaBytes': 0,
            'waitingTimeBounds': {'lowerSeconds': lower, 'upperSeconds': upper}}


class EvidenceTests(unittest.TestCase):
    def test_click_and_observation_uncertainty(self):
        ui = {'clickCall': call(0, 2), 'observations': [
            call(100, 104, state='not-visible', evidence='actual busy window'),
            call(105, 110, state='visible', texturedModelVisible=True, evidence='actual new model screenshot')]}
        bounds = S.waiting_bounds(ui, at(99))
        self.assertEqual(bounds['lowerSeconds'], 98)
        self.assertEqual(bounds['upperSeconds'], 110)

    def test_backend_completion_is_only_a_lower_bound(self):
        ui = {'clickCall': call(0, 2), 'observations': [
            call(110, 112, state='visible', texturedModelVisible=True, evidence='actual screenshot')]}
        bounds = S.waiting_bounds(ui, at(100))
        self.assertEqual(bounds['lowerSeconds'], 98)
        self.assertEqual(bounds['upperSeconds'], 112)

    def test_backend_success_without_visible_model_rejected(self):
        with self.assertRaises(AssertionError):
            S.waiting_bounds({'clickCall': call(0, 2), 'observations': [
                call(110, 112, state='error', evidence='window timeout')]}, at(100))

    def test_missing_footprint_is_not_zero(self):
        with self.assertRaises(AssertionError):
            S.peak([{'memory': 100}, {'memory': None}], 'memory')
        with self.assertRaises(AssertionError):
            S.peak([], 'memory')

    def test_uncertain_waiting_ratio_does_not_pass(self):
        rows = [sample(label, i, lo, hi) for i in (1, 2, 3)
                for label, lo, hi in [('baseline', 90, 110), ('candidate', 70, 85)]]
        result = S.compare(rows)
        self.assertEqual(result['A']['gates']['actualUIWaiting'], 'inconclusive-observation-uncertainty')
        self.assertIsNone(result['B']['gates'])

    def test_three_distinct_rounds_required(self):
        rows = [sample(label, 1, 50, 55) for _ in range(3) for label in ('baseline', 'candidate')]
        self.assertIsNone(S.compare(rows)['A']['gates'])

    def test_conservative_waiting_gain_passes(self):
        rows = [sample(label, i, lo, hi) for i in (1, 2, 3)
                for label, lo, hi in [('baseline', 100, 105), ('candidate', 70, 80)]]
        self.assertEqual(S.compare(rows)['A']['gates']['actualUIWaiting'], 'passed')


if __name__ == '__main__':
    unittest.main()
