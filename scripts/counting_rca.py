"""Counting metrics with explicit zero-count and invalid-answer handling."""
import re

WORDS = dict(zip('zero one two three four five six seven eight nine ten'.split(), range(11)))


def parse_count(response):
    # Exactly the official first-sentence normalization, with strict parsing for RCA.
    answer = response.split('.')[0].strip().lower()
    if answer in WORDS:
        return WORDS[answer]
    if re.fullmatch(r'\d+', answer, flags=re.ASCII):
        return int(answer)
    return None


def score_count(prediction, truth):
    if not isinstance(truth, int) or truth < 0:
        raise ValueError('Ground truth must be a nonnegative integer')
    if prediction is not None and (not isinstance(prediction, int) or prediction < 0):
        raise ValueError('Prediction must be a nonnegative integer or None')
    em = float(prediction is not None and prediction == truth)
    raw = None if truth == 0 else (0.0 if prediction is None else 1.0-abs(prediction-truth)/truth)
    return dict(numeric_em=em, rca_positive=raw,
                rca_positive_clipped=None if raw is None else max(0.0,raw),
                rca_zero_extended=em if truth == 0 else raw)
