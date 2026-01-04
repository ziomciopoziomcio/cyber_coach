from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class JointStatus(Enum):
    """Status stawu podczas analizy."""
    OK = "ok"
    ERROR = "error"
    MISSING = "missing"


@dataclass
class Repetition:
    """Reprezentuje jedno powtórzenie."""
    start_frame: int
    end_frame: int
    min_angle: float
    max_angle: float
    rom: float
    is_complete: bool
    errors: List[str]


class ShoulderPressRules:
    """Reguły walidacji dla ćwiczenia Shoulder Press z konfigurowalnymi progami ROM."""

    # Zakresy kątów dla POPRAWNEJ TECHNIKI (błędy)
    FRONT_VIEW_RANGES = {
        'left_shoulder': (35, 180),
        'right_shoulder': (35, 180),
        'left_elbow': (35, 180),
        'right_elbow': (35, 180),
    }

    SIDE_VIEW_RANGES = {
        'left_shoulder': (0, 160),
        'left_elbow': (8, 180),
        'left_hip': (100, 133),
    }

    # NOWE: Progi dla POWTÓRZEŃ (min/max kąt żeby zliczyć rep)
    # Format: {staw: (min_angle_dla_dolu, max_angle_dla_gory)}
    FRONT_VIEW_ROM_THRESHOLDS = {
        'left_shoulder': (50, 150),  # ramiona: 50° -> 150°
        'right_shoulder': (50, 150),
        'left_elbow': (50, 160),  # łokcie: 50° -> 160°
        'right_elbow': (50, 160),
    }

    SIDE_VIEW_ROM_THRESHOLDS = {
        'left_shoulder': (10, 140),  # ramiona: 10° -> 140°
        'left_elbow': (20, 160),  # łokieć: 20° -> 160°
    }

    MIN_ROM = 100.0  # minimalny ROM (różnica max-min) dla "pełnego" powtórzenia
    PEAK_DETECTION_WINDOW = 10  # ile klatek do wykrywania piku
    MIN_PEAK_PROMINENCE = 15.0  # minimalna różnica między pikiem a doliną

    def __init__(self, view_type: str = 'front'):
        self.view_type = view_type.lower()
        if self.view_type == 'front':
            self.angle_ranges = self.FRONT_VIEW_RANGES
            self.rom_thresholds = self.FRONT_VIEW_ROM_THRESHOLDS
            # Używamy średniej z ramion i łokci
            self.primary_joints = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow']
        elif self.view_type == 'side':
            self.angle_ranges = self.SIDE_VIEW_RANGES
            self.rom_thresholds = self.SIDE_VIEW_ROM_THRESHOLDS
            self.primary_joints = ['left_shoulder', 'left_elbow']
        else:
            raise ValueError(f"Nieznany typ widoku: {view_type}")

        # Historia kątów do detekcji pików
        self.angle_history: List[Tuple[int, float]] = []  # (frame_idx, avg_angle)
        self.last_peak_frame = -1
        self.last_valley_frame = -1
        self.last_peak_angle = None
        self.last_valley_angle = None

        self.repetitions: List[Repetition] = []

    def check_angles(self, angles: Dict[str, Optional[float]]) -> Dict[str, JointStatus]:
        """Sprawdza czy kąty są w dozwolonych zakresach (dla błędów techniki)."""
        results = {}
        for joint, angle in angles.items():
            if joint not in self.angle_ranges:
                continue

            if angle is None:
                results[joint] = JointStatus.MISSING
                continue

            min_angle, max_angle = self.angle_ranges[joint]
            if min_angle <= angle <= max_angle:
                results[joint] = JointStatus.OK
            else:
                results[joint] = JointStatus.ERROR

        return results

    def has_angle_errors(self, angles: Dict[str, Optional[float]]) -> bool:
        """Sprawdza czy są błędy w kątach (TYLKO widoczne kąty poza zakresem)."""
        angle_status = self.check_angles(angles)
        return any(status == JointStatus.ERROR for status in angle_status.values())

    def _get_average_angle(self, angles: Dict[str, Optional[float]]) -> Optional[float]:
        """Zwraca średni kąt z głównych stawów (ignoruje None)."""
        valid_angles = [angles[j] for j in self.primary_joints if angles.get(j) is not None]
        if not valid_angles:
            return None
        return float(np.mean(valid_angles))

    # analysis/exercise_rules.py

    def _check_rom_thresholds(self, min_angle: float, max_angle: float) -> bool:
        """
        Sprawdza czy ruch osiągnął wymagane progi.

        NOWA LOGIKA:
        - ROM musi być >= MIN_ROM (np. 40°)
        - ZAKRES [min_angle, max_angle] musi "pokrywać" wymagany zakres [low, high]
          dla KTÓREGOKOLWIEK stawu
        """
        rom = max_angle - min_angle

        # Warunek 1: minimalny ROM
        if rom < self.MIN_ROM:
            return False

        # Warunek 2: zakres musi "przecinać" wymagany zakres któregoś stawu
        for joint_name, (low_thresh, high_thresh) in self.rom_thresholds.items():
            # Sprawdź czy zakres [min_angle, max_angle] "pokrywa" zakres [low_thresh, high_thresh]
            # Wystarczy że:
            # - min_angle jest w rozsądnej odległości od low_thresh (np. ±20°)
            # - max_angle jest w rozsądnej odległości od high_thresh (np. ±20°)

            TOLERANCE = 20.0  # tolerancja w stopniach

            covers_bottom = min_angle <= (low_thresh + TOLERANCE)
            covers_top = max_angle >= (high_thresh - TOLERANCE)

            if covers_bottom and covers_top:
                return True  # ten staw osiągnął wymagany zakres

        return False  # żaden staw nie osiągnął zakresu

    def _is_local_maximum(self, idx: int) -> bool:
        """Sprawdza czy punkt w historii jest lokalnym maksimum."""
        if idx < self.PEAK_DETECTION_WINDOW or idx >= len(self.angle_history) - self.PEAK_DETECTION_WINDOW:
            return False

        center_angle = self.angle_history[idx][1]

        for i in range(idx - self.PEAK_DETECTION_WINDOW, idx + self.PEAK_DETECTION_WINDOW + 1):
            if i != idx and self.angle_history[i][1] >= center_angle:
                return False

        return True

    def _is_local_minimum(self, idx: int) -> bool:
        """Sprawdza czy punkt w historii jest lokalnym minimum."""
        if idx < self.PEAK_DETECTION_WINDOW or idx >= len(self.angle_history) - self.PEAK_DETECTION_WINDOW:
            return False

        center_angle = self.angle_history[idx][1]

        for i in range(idx - self.PEAK_DETECTION_WINDOW, idx + self.PEAK_DETECTION_WINDOW + 1):
            if i != idx and self.angle_history[i][1] <= center_angle:
                return False

        return True

    def update_repetition_tracking(
            self,
            angles: Dict[str, Optional[float]],
            frame_idx: int
    ) -> Optional[Repetition]:
        """Wykrywa powtórzenia przez lokalne maksima/minima."""

        avg_angle = self._get_average_angle(angles)
        if avg_angle is None:
            return None

        self.angle_history.append((frame_idx, avg_angle))

        if len(self.angle_history) > 200:
            self.angle_history.pop(0)

        if len(self.angle_history) < 2 * self.PEAK_DETECTION_WINDOW + 1:
            return None

        check_idx = len(self.angle_history) - self.PEAK_DETECTION_WINDOW - 1
        check_frame, check_angle = self.angle_history[check_idx]

        # Wykryj pik (maksimum)
        if self._is_local_maximum(check_idx):
            # Jeśli mamy poprzednią dolinę, sprawdź czy jest pełne powtórzenie
            if self.last_valley_frame >= 0 and self.last_valley_angle is not None:
                # POPRAWKA: oblicz min/max z CAŁEGO zakresu między doliną a pikiem
                frames_between = [
                    angle for frame, angle in self.angle_history
                    if self.last_valley_frame <= frame <= check_frame
                ]

                if frames_between:
                    min_angle = min(frames_between)
                    max_angle = max(frames_between)
                    rom = max_angle - min_angle

                    # Sprawdź progi
                    is_complete = self._check_rom_thresholds(min_angle, max_angle)

                    rep = Repetition(
                        start_frame=self.last_valley_frame,
                        end_frame=check_frame,
                        min_angle=min_angle,
                        max_angle=max_angle,
                        rom=rom,
                        is_complete=is_complete,
                        errors=[] if is_complete else ["ROM za mały lub nie osiągnięto progów"]
                    )

                    self.repetitions.append(rep)

                    # Reset dla następnego powtórzenia
                    self.last_valley_frame = -1
                    self.last_valley_angle = None

                    return rep

            # Zapisz pik
            self.last_peak_frame = check_frame
            self.last_peak_angle = check_angle

        # Wykryj dolinę (minimum)
        elif self._is_local_minimum(check_idx):
            # Zapisz dolinę (początek potencjalnego powtórzenia)
            self.last_valley_frame = check_frame
            self.last_valley_angle = check_angle

        return None

    def get_repetition_summary(self) -> Dict:
        """Zwraca podsumowanie powtórzeń."""
        if not self.repetitions:
            return {
                'total_reps': 0,
                'complete_reps': 0,
                'incomplete_reps': 0,
                'avg_rom': 0.0
            }

        complete = [r for r in self.repetitions if r.is_complete]

        return {
            'total_reps': len(self.repetitions),
            'complete_reps': len(complete),
            'incomplete_reps': len(self.repetitions) - len(complete),
            'avg_rom': np.mean([r.rom for r in self.repetitions]) if self.repetitions else 0.0
        }
