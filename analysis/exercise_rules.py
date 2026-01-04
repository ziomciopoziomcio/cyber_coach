from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class ErrorSeverity(Enum):
    """Poziom powagi błędu."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ExerciseError:
    """Reprezentuje wykryty błęd w ćwiczeniu."""
    frame: int
    timestamp: float
    joint: str
    error_type: str
    severity: ErrorSeverity
    message: str
    actual_value: Optional[float] = None
    expected_range: Optional[Tuple[float, float]] = None

class JointStatus(Enum):
    """Status stawu podczas analizy."""
    OK = "ok"              # kąt poprawny
    ERROR = "error"        # kąt niepoprawny
    MISSING = "missing"    # punkt niewidoczny/brak danych

class ShoulderPressRules:
    """Reguły walidacji dla ćwiczenia Shoulder Press."""

    # Zakresy kątów dla widoku z przodu
    FRONT_VIEW_RANGES = {
        'left_shoulder': (40, 180),
        'right_shoulder': (40, 180),
        'left_elbow': (35, 180),
        'right_elbow': (35, 180),
    }

    # Zakresy kątów dla widoku z boku
    SIDE_VIEW_RANGES = {
        'left_shoulder': (0, 160),
        'left_elbow': (8, 180),
        'left_hip': (100, 133),
    }

    def __init__(self, view_type: str = 'front'):
        self.view_type = view_type
        self.ranges = self.FRONT_VIEW_RANGES if view_type == 'front' else self.SIDE_VIEW_RANGES

    def check_angles(self, angles: Dict[str, Optional[float]]) -> Dict[str, bool]:
        """
        Sprawdza kąty i zwraca dict z informacją czy są OK.

        Returns:
            Dict gdzie klucz to nazwa stawu, wartość to True (OK) lub False (błąd)
        """
        results = {}

        for joint, (min_angle, max_angle) in self.ranges.items():
            angle = angles.get(joint)

            if angle is None:
                results[joint] = False  # brak danych = błąd
            else:
                results[joint] = min_angle <= angle <= max_angle

        return results

    def is_pose_correct(self, angles: Dict[str, Optional[float]]) -> bool:
        """Zwraca True jeśli cała poza jest poprawna."""
        results = self.check_angles(angles)
        return all(results.values())