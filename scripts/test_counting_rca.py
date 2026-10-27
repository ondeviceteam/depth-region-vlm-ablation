"""Check paper-formula edge cases and preserve a fixed evaluation denominator."""
import pytest
from counting_rca import parse_count, score_count


@pytest.mark.parametrize('text,expected',[('two',2),('  THREE. ',3),('12',12),('zero',0),
    ('There are two people.',None),('',None),('-1',None),('2 people',None)])
def test_strict_official_response_parsing(text,expected):
    assert parse_count(text)==expected


def test_counting_rca_is_relative_and_not_exact_match():
    assert score_count(3,4)['rca_positive']==.75
    assert score_count(5,4)['rca_positive']==.75
    assert score_count(0,4)['rca_positive']==0
    assert score_count(10,2)['rca_positive']==-3
    assert score_count(10,2)['rca_positive_clipped']==0


def test_zero_and_invalid_are_explicit():
    assert score_count(0,0)['rca_positive'] is None
    assert score_count(1,0)['rca_zero_extended']==0
    assert score_count(0,0)['rca_zero_extended']==1
    assert score_count(None,4)['rca_positive']==0
    assert score_count(None,0)['numeric_em']==0


def test_negative_ground_truth_rejected():
    with pytest.raises(ValueError):score_count(1,-1)
