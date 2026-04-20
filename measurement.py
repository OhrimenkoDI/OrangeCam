"""Модуль измерения углов по кликам мышью на fisheye-изображении."""

from __future__ import annotations

import csv
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .utils import (
    TARGET_WIDTH,
    draw_crosshair,
    format_marker_table,
    load_calib,
    open_camera,
    pixel_to_angle,
)


@dataclass(slots=True)
class Marker:
    """Описывает один поставленный маркер."""

    label: str
    u: int
    v: int
    theta_deg: float
    phi_deg: float
    color: tuple[int, int, int]


class MarkerLabelWorker:
    """Фоновый ввод подписей маркеров через терминал."""

    def __init__(self) -> None:
        """Создаёт очереди и поток чтения `input()`."""

        self.request_queue: queue.SimpleQueue[tuple[int, int, int, float, float]] = queue.SimpleQueue()
        self.result_queue: queue.SimpleQueue[Marker] = queue.SimpleQueue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, name="marker-label-worker", daemon=True)
        self.thread.start()

    def request_label(self, index: int, u: int, v: int, theta_deg: float, phi_deg: float) -> None:
        """Ставит запрос на ввод подписи в очередь."""

        self.request_queue.put((index, u, v, theta_deg, phi_deg))

    def poll(self) -> list[Marker]:
        """Возвращает все готовые маркеры, введённые пользователем."""

        results: list[Marker] = []
        while True:
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def stop(self) -> None:
        """Останавливает поток ввода."""

        self.stop_event.set()

    def _worker(self) -> None:
        """Последовательно спрашивает подписи для новых точек."""

        palette = self._palette()
        while not self.stop_event.is_set():
            request = self.request_queue.get()
            index, u, v, theta_deg, phi_deg = request
            default_label = f"P{index}"
            print(
                f"[{default_label}] label: ",
                end="",
                flush=True,
            )
            try:
                label = input().strip()
            except EOFError:
                label = ""
            if not label:
                label = default_label
            color = palette[(index - 1) % len(palette)]
            marker = Marker(
                label=label,
                u=u,
                v=v,
                theta_deg=theta_deg,
                phi_deg=phi_deg,
                color=color,
            )
            print(
                f"[{default_label}] label: {marker.label}   "
                f"u={u}   v={v}   theta={theta_deg:.2f}°   phi={phi_deg:.1f}°",
                flush=True,
            )
            self.result_queue.put(marker)

    @staticmethod
    def _palette() -> list[tuple[int, int, int]]:
        """Возвращает фиксированную палитру для маркеров."""

        return [
            (56, 56, 255),
            (60, 180, 75),
            (255, 140, 0),
            (255, 0, 255),
            (0, 215, 255),
            (220, 20, 60),
            (180, 105, 255),
            (255, 255, 0),
        ]


class FisheyeAngleMeasurementApp:
    """Интерактивный инструмент измерения углов по fisheye-кадру."""

    def __init__(self, camera_index: int, calib_path: str | Path) -> None:
        """Подготавливает приложение измерения."""

        self.camera_index = camera_index
        self.calib_path = Path(calib_path)
        self.window_name = "Fisheye angle tool"
        self.calibration = load_calib(self.calib_path)
        self.markers: list[Marker] = []
        self.label_worker = MarkerLabelWorker()
        self.center = self._center_from_k(self.calibration.k)
        self.next_marker_index = 1

    def run(self) -> None:
        """Запускает основной цикл отображения и обработки клавиш."""

        self._print_loaded_calibration()
        print("Click on frame to place markers. Press S to save, Q to quit.", flush=True)

        cap = open_camera(self.camera_index)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, TARGET_WIDTH * 2, TARGET_HEIGHT * 2)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Не удалось получить кадр с камеры.")

                self._flush_pending_labels()
                display_frame = self._render_frame(frame)
                cv2.imshow(self.window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in (ord("s"), ord("S")):
                    self._save_markers()
                if key in (ord("d"), ord("D")):
                    self._delete_last_marker()
                if key in (ord("r"), ord("R")):
                    self._reload_calibration()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.label_worker.stop()

    def _render_frame(self, frame: np.ndarray) -> np.ndarray:
        """Рисует перекрестие, маркеры и лучи до них."""

        output = frame.copy()
        center_layer = output.copy()
        draw_crosshair(center_layer, self.center, color=(255, 255, 255), size=14, thickness=1)
        output = cv2.addWeighted(center_layer, 0.95, output, 0.05, 0.0)

        line_layer = output.copy()
        for marker in self.markers:
            cv2.line(line_layer, self.center, (marker.u, marker.v), marker.color, 1, cv2.LINE_AA)
        output = cv2.addWeighted(line_layer, 0.32, output, 0.68, 0.0)

        for marker in self.markers:
            cv2.circle(output, (marker.u, marker.v), 5, marker.color, -1, cv2.LINE_AA)
            cv2.putText(
                output,
                marker.label,
                (marker.u + 8, marker.v - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                marker.color,
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            output,
            "LMB: marker  S: save  D: undo  R: reload calib  Q: quit",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return output

    def _flush_pending_labels(self) -> None:
        """Забирает готовые подписи из фонового потока."""

        for marker in self.label_worker.poll():
            self.markers.append(marker)

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        """Обрабатывает постановку нового маркера по клику."""

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        theta_deg, phi_deg = pixel_to_angle(x, y, self.calibration.k, self.calibration.d)
        index = self.next_marker_index
        self.next_marker_index += 1
        self.label_worker.request_label(index=index, u=x, v=y, theta_deg=theta_deg, phi_deg=phi_deg)

    def _save_markers(self) -> None:
        """Печатает таблицу и сохраняет CSV."""

        if not self.markers:
            print("Нет маркеров для сохранения.", flush=True)
            return

        rows = [(marker.label, marker.u, marker.v, marker.theta_deg, marker.phi_deg) for marker in self.markers]
        table = format_marker_table(rows)
        print(table, flush=True)

        csv_path = Path("markers.csv")
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["label", "u", "v", "theta_deg", "phi_deg"])
            for row in rows:
                writer.writerow(
                    [
                        row[0],
                        row[1],
                        row[2],
                        f"{row[3]:.6f}",
                        f"{row[4]:.6f}",
                    ]
                )
        print(f"Маркеры сохранены в {csv_path}", flush=True)

    def _delete_last_marker(self) -> None:
        """Удаляет последний установленный маркер."""

        if not self.markers:
            print("Список маркеров пуст.", flush=True)
            return
        removed = self.markers.pop()
        print(f"Удалён маркер {removed.label}", flush=True)

    def _reload_calibration(self) -> None:
        """Перезагружает файл калибровки без перезапуска."""

        self.calibration = load_calib(self.calib_path)
        self.center = self._center_from_k(self.calibration.k)
        self._print_loaded_calibration()

    def _print_loaded_calibration(self) -> None:
        """Печатает краткую сводку по параметрам камеры."""

        fx = float(self.calibration.k[0, 0])
        fy = float(self.calibration.k[1, 1])
        cx = float(self.calibration.k[0, 2])
        cy = float(self.calibration.k[1, 2])
        rms_text = f"{self.calibration.rms:.2f}" if self.calibration.rms is not None else "N/A"
        print(
            f"Loaded calibration: f={0.5 * (fx + fy):.1f}px  "
            f"cx={cx:.1f}  cy={cy:.1f}  RMS={rms_text}",
            flush=True,
        )

    @staticmethod
    def _center_from_k(k: np.ndarray) -> tuple[int, int]:
        """Берёт центр из матрицы камеры."""

        return int(round(float(k[0, 2]))), int(round(float(k[1, 2])))
