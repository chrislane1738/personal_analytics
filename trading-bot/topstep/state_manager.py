from enum import Enum, auto


class EvalState(Enum):
    NORMAL = auto()
    CAREFUL = auto()
    REPEAT = auto()
    AGGRESSIVE = auto()
    YOLO = auto()
    HAIL_MARY = auto()


_STATE_CONFIG = {
    EvalState.NORMAL: (1.0, 1.0),  # (position_mult, stop_mult)
    EvalState.CAREFUL: (0.7, 0.7),
    EvalState.REPEAT: (0.8, 0.9),
    EvalState.AGGRESSIVE: (1.3, 1.2),
    EvalState.YOLO: (1.8, 1.5),
    EvalState.HAIL_MARY: (2.5, 2.0),
}


class StateManager:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def get_state(self, cumulative_pnl: float) -> EvalState:
        if cumulative_pnl >= 1500.0:
            return EvalState.REPEAT
        elif cumulative_pnl >= 500.0:
            return EvalState.CAREFUL
        elif cumulative_pnl > -500.0:
            return EvalState.NORMAL
        elif cumulative_pnl > -1000.0:
            return EvalState.AGGRESSIVE
        elif cumulative_pnl > -1500.0:
            return EvalState.YOLO
        else:
            return EvalState.HAIL_MARY

    def get_position_multiplier(self, cumulative_pnl: float) -> float:
        if not self.enabled:
            return 1.0
        return _STATE_CONFIG[self.get_state(cumulative_pnl)][0]

    def get_stop_multiplier(self, cumulative_pnl: float) -> float:
        if not self.enabled:
            return 1.0
        return _STATE_CONFIG[self.get_state(cumulative_pnl)][1]
