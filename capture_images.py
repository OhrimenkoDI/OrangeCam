import os
import platform
import time

import cv2

OUTPUT_DIR = "dataset/images"
CAMERA_NAME = "RYS HFR USB2.0 Camera"
CAMERA_SOURCE = "/dev/v4l/by-id/usb-RYS_HFR_USB2.0_Camera_RYS_HFR_USB2.0_Camera-video-index0"
CAMERA_METADATA_SOURCE = "/dev/v4l/by-id/usb-RYS_HFR_USB2.0_Camera_RYS_HFR_USB2.0_Camera-video-index1"
MAX_IMAGES = 100
CAMERA_BACKEND = "v4l2"
HEADLESS_MODE = False
SNAPSHOT_INTERVAL_SEC = 1.0

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_FPS = 120
TARGET_FOURCC = "MJPG"
RETICLE_DIAMETER_PX = 15
BACKEND_LABELS = {
    "auto": "AUTO",
    "dshow": "DSHOW",
    "v4l2": "V4L2",
}


def fourcc_to_str(value):
    code = int(value)
    if code <= 0:
        return "N/A"
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))


def get_backend_candidates(requested_backend):
    if requested_backend != "auto":
        return [requested_backend]

    system_name = platform.system().lower()
    if system_name == "windows":
        return ["dshow", "auto"]
    if system_name == "linux":
        return ["v4l2", "auto"]
    return ["auto"]


def open_camera_fixed_mode(camera_source, target_width, target_height, requested_backend):
    backend_candidates = get_backend_candidates(requested_backend)
    params = [
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*TARGET_FOURCC),
        cv2.CAP_PROP_FRAME_WIDTH,
        target_width,
        cv2.CAP_PROP_FRAME_HEIGHT,
        target_height,
        cv2.CAP_PROP_FPS,
        TARGET_FPS,
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    ]
    backend_map = {
        "auto": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,
        "v4l2": cv2.CAP_V4L2,
    }

    for backend_name in backend_candidates:
        cap = cv2.VideoCapture(camera_source, backend_map[backend_name], params)
        if not cap.isOpened():
            cap.release()
            continue

        for _ in range(10):
            cap.read()
        return cap, backend_name

    return None, backend_candidates[0]


def print_runtime_info(camera_source, backend_name):
    print("Capture settings:")
    print(f"  camera name : {CAMERA_NAME}")
    print(f"  source      : {camera_source}")
    print(f"  metadata    : {CAMERA_METADATA_SOURCE}")
    print(f"  backend     : {BACKEND_LABELS.get(backend_name, backend_name.upper())}")


def print_mode_info(cap, backend_name, first_frame, target_width, target_height):
    backend_label = BACKEND_LABELS.get(backend_name, backend_name.upper())
    if not cap.isOpened():
        return

    reported_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    reported_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    reported_fourcc = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))

    frame_height, frame_width = first_frame.shape[:2]
    actual_width = reported_width if reported_width > 0 else frame_width
    actual_height = reported_height if reported_height > 0 else frame_height

    print("Selected mode:")
    print(
        f"  request: {target_width}x{target_height}@{TARGET_FPS} {TARGET_FOURCC}"
    )
    print(
        f"  actual : {actual_width}x{actual_height}, "
        f"fps={reported_fps:.2f}, fourcc={reported_fourcc}, backend={backend_label}"
    )


def draw_reticle(frame):
    height, width = frame.shape[:2]
    center = (width // 2, height // 2)
    radius = RETICLE_DIAMETER_PX // 2
    cv2.circle(frame, center, radius, (0, 0, 255), 1, cv2.LINE_AA)


def can_use_preview():
    return hasattr(cv2, "imshow") and bool(os.environ.get("DISPLAY"))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    target_width, target_height = TARGET_WIDTH, TARGET_HEIGHT
    print_runtime_info(CAMERA_SOURCE, CAMERA_BACKEND)
    cap, opened_backend = open_camera_fixed_mode(
        CAMERA_SOURCE,
        target_width,
        target_height,
        CAMERA_BACKEND,
    )
    if cap is None:
        print(
            f"Error: cannot open camera source {CAMERA_SOURCE} "
            f"with {BACKEND_LABELS.get(CAMERA_BACKEND, CAMERA_BACKEND.upper())} "
            f"{target_width}x{target_height}@{TARGET_FPS} {TARGET_FOURCC}"
        )
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: first frame read failed")
        cap.release()
        return

    print_mode_info(cap, opened_backend, frame, target_width, target_height)
    print("\nDataset capture started")
    preview_enabled = not HEADLESS_MODE and can_use_preview()
    if preview_enabled:
        print("Press:")
        print("  SPACE - take snapshot")
        print("  ESC   - finish capture")
    else:
        print("Headless mode active")
        print(f"  snapshot interval: {SNAPSHOT_INTERVAL_SEC:.2f} sec")

    count = 0
    current_fps = 0.0
    fps_frames = 0
    fps_started_at = time.perf_counter()
    last_saved_at = 0.0
    while count < MAX_IMAGES:
        ret, frame = cap.read()
        if not ret:
            print("Frame read error")
            break

        fps_frames += 1
        now = time.perf_counter()
        elapsed = now - fps_started_at
        if elapsed >= 0.5:
            current_fps = fps_frames / elapsed
            fps_frames = 0
            fps_started_at = now

        preview_frame = frame.copy()
        frame_height = preview_frame.shape[0]
        cv2.putText(
            preview_frame,
            f"FPS: {current_fps:.1f}",
            (10, frame_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        draw_reticle(preview_frame)

        if preview_enabled:
            cv2.imshow("Data capture - SPACE: snapshot, ESC: exit", preview_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                print(f"\nCapture finished. Saved {count} images.")
                break
            if key == 32:
                filename = os.path.join(OUTPUT_DIR, f"object_{count:03d}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")
                count += 1
                time.sleep(0.3)
        else:
            if now - last_saved_at < max(SNAPSHOT_INTERVAL_SEC, 0.0):
                continue

            filename = os.path.join(OUTPUT_DIR, f"object_{count:03d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
            count += 1
            last_saved_at = now

    cap.release()
    if preview_enabled:
        cv2.destroyAllWindows()
    print(f"\nTotal saved: {count} images in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
