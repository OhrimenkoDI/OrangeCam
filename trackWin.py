import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

TARGET_FPS = 60
TARGET_FOURCC = "MJPG"
RESOLUTION_PRESETS = {
    "1": (1920, 1080),
    "2": (1280, 720),
    "3": (640, 480),
}


def _safe_crop(frame: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray | None:
    """Returns a clipped crop of an object box if it is valid."""
    h, w = frame.shape[:2]
    xi1 = max(0, min(w - 1, int(x1)))
    yi1 = max(0, min(h - 1, int(y1)))
    xi2 = max(0, min(w, int(x2)))
    yi2 = max(0, min(h, int(y2)))

    if xi2 <= xi1 or yi2 <= yi1:
        return None

    return frame[yi1:yi2, xi1:xi2].copy()


def _placeholder(width: int = 360, height: int = 240, text: str = "Object not found") -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        img,
        text,
        (15, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    return img


def _parse_source(raw_source: str) -> int | str:
    source_candidate = raw_source.strip()
    if source_candidate.isdigit():
        return int(source_candidate)
    return source_candidate


def _choose_resolution() -> tuple[int, int]:
    print("Select resolution preset:")
    print("  1 - 1920x1080")
    print("  2 - 1280x720")
    print("  3 - 640x480")
    choice = input("Enter 1/2/3 [2]: ").strip() or "2"
    if choice not in RESOLUTION_PRESETS:
        print("Invalid choice. Using default 1280x720.")
        choice = "2"
    return RESOLUTION_PRESETS[choice]


def _fourcc_to_str(value: float) -> str:
    code = int(value)
    if code <= 0:
        return "N/A"
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))


def _open_usb_camera_dshow(camera_index: int, width: int, height: int) -> cv2.VideoCapture | None:
    params = [
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*TARGET_FOURCC),
        cv2.CAP_PROP_FRAME_WIDTH,
        width,
        cv2.CAP_PROP_FRAME_HEIGHT,
        height,
        cv2.CAP_PROP_FPS,
        TARGET_FPS,
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    ]
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW, params)
    if not cap.isOpened():
        return None

    for _ in range(10):
        cap.read()
    return cap


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="YOLOv8 tracking from camera or video file with real-time playback pacing."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: camera index (for example 0) or file path (for example video.mp4).",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    model_path = Path(__file__).resolve().parent / "best.pt"#license-plate-finetune-v1n.pt"
    if not model_path.exists():
        print(f"[ERROR] Weights file not found: {model_path}")
        return

    model = YOLO(str(model_path))

    source = _parse_source(args.source)
    is_camera_source = isinstance(source, int)

    if is_camera_source:
        target_width, target_height = _choose_resolution()
        cap = _open_usb_camera_dshow(source, target_width, target_height)
        if cap is None:
            print(
                f"[ERROR] Could not open camera {source} with "
                f"DSHOW {target_width}x{target_height}@{TARGET_FPS} {TARGET_FOURCC}"
            )
            return

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_fourcc = _fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))
        print(
            "[INFO] Camera mode: "
            f"request={target_width}x{target_height}@{TARGET_FPS} {TARGET_FOURCC}, "
            f"actual={actual_w}x{actual_h}, fps={actual_fps:.2f}, fourcc={actual_fourcc}"
        )
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Could not open source: {args.source}")
            return

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps is None or source_fps <= 0 or np.isnan(source_fps):
        source_fps = 30.0
    frame_interval_sec = 1.0 / source_fps

    frame_num = 0
    window_name = "YOLOv8 Tracking"
    captured_track_id: int | None = None
    active_object_windows: set[int] = set()
    current_fps = 0.0
    fps_frames = 0
    fps_started_at = time.perf_counter()

    print("[INFO] Controls: q - quit, c - recapture target ID.")

    try:
        while True:
            loop_start = time.perf_counter()

            ok, frame = cap.read()
            if not ok:
                if is_camera_source:
                    print("[ERROR] Could not read frame from camera.")
                else:
                    print("[INFO] End of video file reached.")
                break

            frame_num += 1
            frame_for_crop = frame.copy()

            frame_h, frame_w = frame.shape[:2]
            frame_center_x = frame_w / 2.0
            frame_center_y = frame_h / 2.0

            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )

            result = results[0]
            boxes = result.boxes

            cv2.circle(
                frame,
                (int(frame_center_x), int(frame_center_y)),
                radius=4,
                color=(0, 255, 255),
                thickness=-1,
            )

            id_to_bbox: dict[int, tuple[float, float, float, float]] = {}

            if boxes is not None and len(boxes) > 0:
                xyxy_list = boxes.xyxy.cpu().tolist()
                if boxes.id is not None:
                    ids = boxes.id.int().cpu().tolist()
                else:
                    ids = [None] * len(xyxy_list)

                for i, xyxy in enumerate(xyxy_list):
                    track_id = ids[i]
                    if track_id is not None:
                        id_to_bbox[track_id] = tuple(xyxy)

                if captured_track_id not in id_to_bbox:
                    if id_to_bbox:
                        captured_track_id = min(
                            id_to_bbox,
                            key=lambda tid: (
                                ((id_to_bbox[tid][0] + id_to_bbox[tid][2]) / 2.0 - frame_center_x) ** 2
                                + ((id_to_bbox[tid][1] + id_to_bbox[tid][3]) / 2.0 - frame_center_y) ** 2
                            ),
                        )
                    else:
                        captured_track_id = None

                for i, xyxy in enumerate(xyxy_list):
                    x1, y1, x2, y2 = xyxy
                    track_id = ids[i]

                    abs_cx = (x1 + x2) / 2.0
                    abs_cy = (y1 + y2) / 2.0
                    rel_x = abs_cx - frame_center_x
                    rel_y = abs_cy - frame_center_y

                    abs_cx_i = int(round(abs_cx))
                    abs_cy_i = int(round(abs_cy))
                    rel_x_i = int(round(rel_x))
                    rel_y_i = int(round(rel_y))

                    if track_id is None:
                        color = (0, 255, 0)
                        id_text = "N/A"
                    else:
                        color = (
                            int((37 * track_id) % 255),
                            int((17 * track_id) % 255),
                            int((29 * track_id) % 255),
                        )
                        id_text = str(track_id)

                    is_selected = track_id is not None and track_id == captured_track_id
                    thickness = 3 if is_selected else 2

                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        color,
                        thickness,
                    )

                    overlay_text = f"ID: {id_text}, X: {rel_x_i}, Y: {rel_y_i}"
                    if is_selected:
                        overlay_text += " [CAPTURED]"

                    text_x = int(x1)
                    text_y = int(y1) - 10 if int(y1) - 10 > 10 else int(y1) + 20

                    cv2.putText(
                        frame,
                        overlay_text,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                    print(
                        f"Frame: {frame_num}, ID: {id_text}, "
                        f"Abs_Center: ({abs_cx_i}, {abs_cy_i}), "
                        f"Rel_to_Center: ({rel_x_i}, {rel_y_i})",
                        flush=True,
                    )
            else:
                captured_track_id = None

            current_ids = set(id_to_bbox.keys())
            for track_id in sorted(current_ids):
                window_name_id = f"Object ID {track_id}"
                cropped = _safe_crop(frame_for_crop, *id_to_bbox[track_id])
                if cropped is not None and cropped.size > 0:
                    cv2.imshow(window_name_id, cropped)
                else:
                    cv2.imshow(window_name_id, _placeholder(text=f"ID {track_id}: invalid crop"))

            stale_ids = active_object_windows - current_ids
            for stale_id in stale_ids:
                try:
                    cv2.destroyWindow(f"Object ID {stale_id}")
                except cv2.error:
                    pass

            active_object_windows = current_ids

            fps_frames += 1
            now = time.perf_counter()
            elapsed = now - fps_started_at
            if elapsed >= 0.5:
                current_fps = fps_frames / elapsed
                fps_frames = 0
                fps_started_at = now

            cv2.putText(
                frame,
                f"FPS: {current_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame)

            processing_sec = time.perf_counter() - loop_start
            if is_camera_source:
                wait_ms = 1
            else:
                remaining_sec = frame_interval_sec - processing_sec
                wait_ms = 1 if remaining_sec <= 0 else max(1, int(remaining_sec * 1000))

            # q - exit, c - release captured object id and pick closest one again.
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                captured_track_id = None

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
