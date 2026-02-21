from omega_model import OMEGAModel, PredictionTracker


def _feature_row(regime='bull_low'):
    f = {
        'regime_bull_low': 0.0,
        'regime_bear_high': 0.0,
        'regime_sideways': 0.0,
        'regime_bull_high': 0.0,
    }
    if regime == 'bull_low':
        f['regime_bull_low'] = 1.0
    elif regime == 'bear_high':
        f['regime_bear_high'] = 1.0
    elif regime == 'sideways':
        f['regime_sideways'] = 1.0
    elif regime == 'bull_high':
        f['regime_bull_high'] = 1.0
    return f


def _pred(signal, actual_return, regime='bull_low'):
    return {
        'pred': {'signal': signal},
        'outcome': {'actual_return': float(actual_return)},
        'features': _feature_row(regime=regime),
    }


def test_reliability_gate_blocks_when_history_insufficient():
    tracker = PredictionTracker.__new__(PredictionTracker)
    tracker.predictions = [_pred('BUY', 0.01, 'bull_low') for _ in range(12)]

    gate = tracker.get_reliability_gate_decision(
        signal='BUY',
        features=_feature_row('bull_low'),
        min_samples=40,
    )

    assert gate['required'] is True
    assert gate['passed'] is False
    assert gate['reason'].startswith('insufficient_history')


def test_reliability_gate_passes_with_strong_oos_stats():
    tracker = PredictionTracker.__new__(PredictionTracker)

    preds = []
    preds.extend(_pred('BUY', 0.012, 'bull_low') for _ in range(42))
    preds.extend(_pred('BUY', -0.004, 'bull_low') for _ in range(18))
    preds.extend(_pred('SELL', -0.010, 'bear_high') for _ in range(20))
    tracker.predictions = preds

    gate = tracker.get_reliability_gate_decision(
        signal='BUY',
        features=_feature_row('bull_low'),
        min_samples=40,
        min_accuracy_pct=58.0,
        min_avg_edge_pct=0.10,
    )

    assert gate['required'] is True
    assert gate['passed'] is True
    assert gate['reason'] == 'pass'
    assert gate['total_samples'] >= 40


def test_reliability_gate_fails_on_low_accuracy():
    tracker = PredictionTracker.__new__(PredictionTracker)

    preds = []
    preds.extend(_pred('BUY', 0.010, 'bull_low') for _ in range(20))
    preds.extend(_pred('BUY', -0.010, 'bull_low') for _ in range(30))
    preds.extend(_pred('SELL', -0.008, 'bear_high') for _ in range(10))
    tracker.predictions = preds

    gate = tracker.get_reliability_gate_decision(
        signal='BUY',
        features=_feature_row('bull_low'),
        min_samples=40,
        min_accuracy_pct=58.0,
    )

    assert gate['required'] is True
    assert gate['passed'] is False
    assert gate['reason'].startswith('low_global_accuracy')


def test_omega_model_applies_oos_gate_to_directional_signal():
    tracker = PredictionTracker.__new__(PredictionTracker)
    tracker.predictions = [_pred('BUY', 0.01, 'bull_low') for _ in range(10)]

    model = OMEGAModel.__new__(OMEGAModel)
    model.tracker = tracker

    gated_signal, gate = model._apply_oos_reliability_gate(
        signal='BUY',
        features=_feature_row('bull_low'),
        oos_min_samples=40,
    )

    assert gated_signal == 'HOLD'
    assert gate['required'] is True
    assert gate['passed'] is False


def test_omega_model_oos_gate_keeps_hold_unchanged():
    tracker = PredictionTracker.__new__(PredictionTracker)
    tracker.predictions = [_pred('BUY', 0.01, 'bull_low') for _ in range(10)]

    model = OMEGAModel.__new__(OMEGAModel)
    model.tracker = tracker

    gated_signal, gate = model._apply_oos_reliability_gate(
        signal='HOLD',
        features=_feature_row('bull_low'),
        oos_min_samples=40,
    )

    assert gated_signal == 'HOLD'
    assert gate['required'] is False
    assert gate['passed'] is True
