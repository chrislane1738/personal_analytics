from topstep.state_manager import StateManager, EvalState


def test_normal_state_at_zero_pnl():
    sm = StateManager()
    assert sm.get_state(0.0) == EvalState.NORMAL
    assert sm.get_position_multiplier(0.0) == 1.0
    assert sm.get_stop_multiplier(0.0) == 1.0


def test_careful_state():
    sm = StateManager()
    assert sm.get_state(800.0) == EvalState.CAREFUL
    assert sm.get_position_multiplier(800.0) == 0.7
    assert sm.get_stop_multiplier(800.0) == 0.7


def test_repeat_state():
    sm = StateManager()
    assert sm.get_state(2000.0) == EvalState.REPEAT
    assert sm.get_position_multiplier(2000.0) == 0.8
    assert sm.get_stop_multiplier(2000.0) == 0.9


def test_aggressive_state():
    sm = StateManager()
    assert sm.get_state(-700.0) == EvalState.AGGRESSIVE
    assert sm.get_position_multiplier(-700.0) == 1.3
    assert sm.get_stop_multiplier(-700.0) == 1.2


def test_yolo_state():
    sm = StateManager()
    assert sm.get_state(-1200.0) == EvalState.YOLO
    assert sm.get_position_multiplier(-1200.0) == 1.8
    assert sm.get_stop_multiplier(-1200.0) == 1.5


def test_hail_mary_state():
    sm = StateManager()
    assert sm.get_state(-1700.0) == EvalState.HAIL_MARY
    assert sm.get_position_multiplier(-1700.0) == 2.5
    assert sm.get_stop_multiplier(-1700.0) == 2.0


def test_boundary_values():
    sm = StateManager()
    assert sm.get_state(500.0) == EvalState.CAREFUL
    assert sm.get_state(499.99) == EvalState.NORMAL
    assert sm.get_state(1500.0) == EvalState.REPEAT
    assert sm.get_state(-500.0) == EvalState.AGGRESSIVE
    assert sm.get_state(-1000.0) == EvalState.YOLO
    assert sm.get_state(-1500.0) == EvalState.HAIL_MARY


def test_disabled_state_manager():
    sm = StateManager(enabled=False)
    assert sm.get_position_multiplier(-1700.0) == 1.0
    assert sm.get_stop_multiplier(-1700.0) == 1.0
    assert sm.get_state(-1700.0) == EvalState.HAIL_MARY  # state still computed
