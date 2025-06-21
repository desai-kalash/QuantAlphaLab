from src.labeling import classify_signal

def test_classify_signal_buy():
    assert classify_signal(0.03) == 1  # more than +2% return

def test_classify_signal_sell():
    assert classify_signal(-0.03) == -1  # less than -2% return

def test_classify_signal_hold():
    assert classify_signal(0.0) == 0  # flat return
