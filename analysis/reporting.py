# kod do zbierania metryk runtime i generowania raportów
import os
import time
import json
import csv
from datetime import datetime
from typing import Any, Dict, List, Optional
import threading

import numpy as np

try:
    import psutil
except Exception:
    psutil = None

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import cv2
except Exception:
    cv2 = None


class Reporter:
    """
    Zbiera metryki runtime i repy, generuje CSV/JSON oraz wykresy.
    Integracja: w `cyber_trainer/camera.py` utwórz instancję i wywołuj:
      reporter.record_frame(frame_idx, fps, angles, enabled, view_name, frame=frame)
      reporter.record_rep(rep, view_name)
      reporter.generate_report() przy zakończeniu sesji
    """

    def __init__(self, save_root: str = "reports"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.outdir = os.path.join(save_root, f"session_{ts}")
        os.makedirs(self.outdir, exist_ok=True)
        self.start_time = time.time()
        self.frames: List[Dict[str, Any]] = []
        self.reps: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self.snapshots_dir = os.path.join(self.outdir, "snapshots")
        os.makedirs(self.snapshots_dir, exist_ok=True)

    def _sample_system(self) -> Dict[str, Any]:
        info = {"ts": time.time(), "proc_rss_bytes": None, "sys_used_bytes": None, "gpu": None}
        if psutil:
            try:
                p = psutil.Process()
                info["proc_rss_bytes"] = getattr(p.memory_info(), "rss", None)
                info["sys_used_bytes"] = getattr(psutil.virtual_memory(), "used", None)
            except Exception:
                pass
        if _NVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                info["gpu"] = {
                    "mem_used": int(mem.used),
                    "mem_total": int(mem.total),
                    "gpu_util": int(util.gpu),
                    "mem_util": int(util.memory)
                }
            except Exception:
                info["gpu"] = None
        return info

    def record_frame(
        self,
        frame_idx: int,
        fps: float,
        angles: Optional[Dict[str, Optional[float]]],
        detection_enabled: bool,
        view_name: str,
        frame: Optional[Any] = None,
    ):
        """Wywoływane raz na klatkę. Opcjonalnie zapisuje zrzut co N-te klatki."""
        sample = self._sample_system()
        entry = {
            "frame_idx": frame_idx,
            "time": time.time(),
            "fps": float(fps),
            "view": view_name,
            "detection_enabled": bool(detection_enabled),
            "angles": angles,
            "sys_sample": sample,
        }
        with self._lock:
            self.frames.append(entry)

        # przykładowo: zapis co 300 klatek miniaturki
        if frame is not None and frame_idx % 300 == 0:
            try:
                path = os.path.join(self.snapshots_dir, f"frame_{view_name}_{frame_idx}.jpg")
                if cv2 is not None:
                    cv2.imwrite(path, frame)
            except Exception:
                pass

    def record_rep(self, rep: Any, view_name: str, frame_image: Optional[Any] = None):
        """Wywołać gdy powstanie Repetition. `rep` może być obiektem dataclass z atrybutami."""
        with self._lock:
            rep_dict = {
                "start_frame": getattr(rep, "start_frame", None),
                "end_frame": getattr(rep, "end_frame", None),
                "min_angle": float(getattr(rep, "min_angle", np.nan)),
                "max_angle": float(getattr(rep, "max_angle", np.nan)),
                "rom": float(getattr(rep, "rom", np.nan)),
                "is_complete": bool(getattr(rep, "is_complete", False)),
                "errors": getattr(rep, "errors", []),
                "view": view_name,
                "ts": time.time(),
            }
            self.reps.append(rep_dict)

        if frame_image is not None and cv2 is not None:
            try:
                fname = f"rep_{view_name}_{rep_dict['end_frame']}.jpg"
                cv2.imwrite(os.path.join(self.snapshots_dir, fname), frame_image)
            except Exception:
                pass

    def _save_json_csv(self):
        # JSON summary
        summary = {
            "start_time": self.start_time,
            "end_time": time.time(),
            "n_frames": len(self.frames),
            "n_reps": len(self.reps),
            "overall_efficiency_pct": self._compute_efficiency() * 100.0 if self._compute_efficiency() is not None else None,
        }
        with open(os.path.join(self.outdir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "frames": self.frames, "reps": self.reps}, f, default=str, indent=2)

        # frames CSV (flatten basic fields)
        frames_csv = os.path.join(self.outdir, "frames.csv")
        with open(frames_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_idx", "time", "fps", "view", "detection_enabled"])
            for fr in self.frames:
                writer.writerow([fr["frame_idx"], fr["time"], fr["fps"], fr["view"], fr["detection_enabled"]])

        # reps CSV
        reps_csv = os.path.join(self.outdir, "reps.csv")
        with open(reps_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["start_frame", "end_frame", "min_angle", "max_angle", "rom", "is_complete", "view", "ts", "errors"])
            for r in self.reps:
                writer.writerow([r["start_frame"], r["end_frame"], r["min_angle"], r["max_angle"], r["rom"], r["is_complete"], r["view"], r["ts"], "|".join(r.get("errors", []))])

    def _compute_efficiency(self) -> Optional[float]:
        if not self.reps:
            return None
        total = len(self.reps)
        complete = len([r for r in self.reps if r.get("is_complete")])
        return complete / total if total else None

    def _make_plots(self):
        if plt is None:
            return
        # FPS over time
        if self.frames:
            times = np.array([f["time"] - self.start_time for f in self.frames])
            fps = np.array([f["fps"] for f in self.frames])
            plt.figure(figsize=(8, 3))
            plt.plot(times, fps, label="FPS")
            plt.xlabel("s od startu")
            plt.ylabel("FPS")
            plt.title("FPS over time")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.outdir, "fps_over_time.png"))
            plt.close()

        # ROM histogram and eff over time
        if self.reps:
            roms = np.array([r["rom"] for r in self.reps if r["rom"] is not None and not np.isnan(r["rom"])])
            plt.figure(figsize=(6, 4))
            plt.hist(roms, bins=20)
            plt.xlabel("ROM (deg)")
            plt.ylabel("Liczba powtórzeń")
            plt.title("Histogram ROM")
            plt.tight_layout()
            plt.savefig(os.path.join(self.outdir, "rom_histogram.png"))
            plt.close()

            # efficiency cumulative
            cum_total = []
            cum_complete = []
            t = []
            total = 0
            complete = 0
            for r in self.reps:
                total += 1
                if r.get("is_complete"):
                    complete += 1
                cum_total.append(total)
                cum_complete.append(complete)
                t.append(r["ts"] - self.start_time)
            eff = np.array(cum_complete) / np.array(cum_total)
            plt.figure(figsize=(8, 3))
            plt.plot(t, eff * 100)
            plt.xlabel("s od startu")
            plt.ylabel("Efektywność \% (skumulowana)")
            plt.title("Efektywność w czasie")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.outdir, "efficiency_over_time.png"))
            plt.close()

    def generate_report(self):
        """Zapis plików i wykresów - wywołać przy zamykaniu aplikacji."""
        try:
            self._save_json_csv()
            self._make_plots()
        finally:
            if _NVML_AVAILABLE:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
        return self.outdir
