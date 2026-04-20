import os
import time

import cv2

OUTPUT_DIR = "dataset/images"
CAMERA_INDEX = 0
MAX_IMAGES = 100

TARGET_FPS = 60
TARGET_FOURCC = "MJPG"
RESOLUTION_PRESETS = {
    "1": (1920, 1080),
    "2": (1280, 720),
    "3": (640, 480),
}
RETICLE_DIAMETER_PX = 15


def fourcc_to_str(value):
    code = int(value)
    if code <= 0:
        return "N/A"
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))


def choose_resolution():
    print("Select resolution preset:")
    print("  1 - 1920x1080")
    print("  2 - 1280x720")
    print("  3 - 640x480")
    choice = input("Enter 1/2/3 [2]: ").strip() or "2"
    if choice not in RESOLUTION_PRESETS:
        print("Invalid choice. Using default 1280x720.")
        choice = "2"
    return RESOLUTION_PRESETS[choice]


def open_camera_fixed_mode(target_width, target_height):
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
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW, params)
    if not cap.isOpened():
        return None

    for _ in range(10):
        cap.read()
    return cap


def print_mode_info(cap, first_frame, target_width, target_height):
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
        f"fps={reported_fps:.2f}, fourcc={reported_fourcc}"
    )


def draw_reticle(frame):
    height, width = frame.shape[:2]
    center = (width // 2, height // 2)
    radius = RETICLE_DIAMETER_PX // 2
    cv2.circle(frame, center, radius, (0, 0, 255), 1, cv2.LINE_AA)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    target_width, target_height = choose_resolution()
    cap = open_camera_fixed_mode(target_width, target_height)
    if cap is None:
        print(
            f"Error: cannot open camera index {CAMERA_INDEX} "
            f"with DSHOW {target_width}x{target_height}@{TARGET_FPS} {TARGET_FOURCC}"
        )
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: first frame read failed")
        cap.release()
        return

    print_mode_info(cap, frame, target_width, target_height)
    print("\nDataset capture started")
    print("Press:")
    print("  SPACE - take snapshot")
    print("  ESC   - finish capture")

    count = 0
    current_fps = 0.0
    fps_frames = 0
    fps_started_at = time.perf_counter()
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

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTotal saved: {count} images in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
