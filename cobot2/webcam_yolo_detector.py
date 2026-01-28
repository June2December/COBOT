# webcam_yolo_detector v0.000 2026-01-28
# [이번 버전에서 수정된 사항]
# - (기능구현) 노트북 웹캠(OpenCV) 입력으로 Ultralytics YOLO 추론 파이프라인 기본틀 구현
# - (기능구현) 특정 클래스(target_class_name)만 필터링 후, 가장 큰 bbox 1개를 타겟으로 선택
# - (기능구현) 타겟 bbox 중심 기반 error_x_norm/error_y_norm 계산 + 화면 오버레이(FPS/클래스/신뢰도/에러)

"""
Webcam YOLO detector (MVP skeleton)

- Input: OpenCV VideoCapture webcam frames
- Output: On-screen overlay + stdout logs (optional)
- Core:
  1) YOLO inference
  2) Filter detections by a specific class name
  3) Select a single target (largest bbox area)
  4) Compute normalized center error (error_x_norm, error_y_norm)

Notes:
- This file is intentionally hardware-agnostic: later, replace webcam frame source
  with RealSense ROS topic (cv_bridge) and keep the same post-processing logic.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection2D:
    class_id: int
    class_name: str
    confidence: float
    x1_px: float
    y1_px: float
    x2_px: float
    y2_px: float

    @property
    def width_px(self) -> float:
        return max(0.0, self.x2_px - self.x1_px)

    @property
    def height_px(self) -> float:
        return max(0.0, self.y2_px - self.y1_px)

    @property
    def area_px2(self) -> float:
        return self.width_px * self.height_px

    @property
    def center_px(self) -> Tuple[float, float]:
        return (self.x1_px + self.width_px / 2.0, self.y1_px + self.height_px / 2.0)


def build_class_name_to_id(model: YOLO) -> Dict[str, int]:
    # Ultralytics model has model.names: {id: name} or list
    names = model.names
    if isinstance(names, dict):
        return {str(name): int(class_id) for class_id, name in names.items()}
    return {str(name): int(i) for i, name in enumerate(names)}


def parse_detections(result, class_id_to_name: Dict[int, str]) -> List[Detection2D]:
    detections: List[Detection2D] = []
    if result.boxes is None:
        return detections

    boxes_xyxy = result.boxes.xyxy
    confidences = result.boxes.conf
    class_ids = result.boxes.cls

    if boxes_xyxy is None or confidences is None or class_ids is None:
        return detections

    boxes_xyxy = boxes_xyxy.cpu().numpy()
    confidences = confidences.cpu().numpy()
    class_ids = class_ids.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), conf, class_id in zip(boxes_xyxy, confidences, class_ids):
        class_name = class_id_to_name.get(int(class_id), f"class_{class_id}")
        detections.append(
            Detection2D(
                class_id=int(class_id),
                class_name=class_name,
                confidence=float(conf),
                x1_px=float(x1),
                y1_px=float(y1),
                x2_px=float(x2),
                y2_px=float(y2),
            )
        )
    return detections


def filter_by_class_name(
    detections: List[Detection2D],
    target_class_name: str,
    min_confidence: float,
) -> List[Detection2D]:
    target_class_name_lower = target_class_name.strip().lower()
    return [
        det
        for det in detections
        if det.class_name.lower() == target_class_name_lower and det.confidence >= min_confidence
    ]


def select_largest_bbox(detections: List[Detection2D]) -> Optional[Detection2D]:
    if not detections:
        return None
    return max(detections, key=lambda d: d.area_px2)


def compute_normalized_center_error(
    target_center_px: Tuple[float, float],
    frame_width_px: int,
    frame_height_px: int,
) -> Tuple[float, float]:
    image_center_x_px = frame_width_px / 2.0
    image_center_y_px = frame_height_px / 2.0
    target_center_x_px, target_center_y_px = target_center_px

    # Normalize to [-1, 1] approximately (depending on bounds)
    error_x_norm = (target_center_x_px - image_center_x_px) / max(image_center_x_px, 1.0)
    error_y_norm = (target_center_y_px - image_center_y_px) / max(image_center_y_px, 1.0)
    return float(error_x_norm), float(error_y_norm)


def draw_overlay(
    frame_bgr: np.ndarray,
    fps: float,
    target_detection: Optional[Detection2D],
    error_x_norm: Optional[float],
    error_y_norm: Optional[float],
) -> np.ndarray:
    overlay = frame_bgr.copy()
    frame_height_px, frame_width_px = overlay.shape[:2]

    # Draw image center crosshair
    center_x = int(frame_width_px / 2)
    center_y = int(frame_height_px / 2)
    cv2.drawMarker(overlay, (center_x, center_y), (255, 255, 255), markerType=cv2.MARKER_CROSS, thickness=2)

    # FPS text
    cv2.putText(
        overlay,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if target_detection is not None:
        x1, y1, x2, y2 = map(int, [target_detection.x1_px, target_detection.y1_px, target_detection.x2_px, target_detection.y2_px])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cx, cy = target_detection.center_px
        cv2.circle(overlay, (int(cx), int(cy)), 6, (0, 255, 0), -1)

        label = f"{target_detection.class_name} conf={target_detection.confidence:.2f}"
        cv2.putText(
            overlay,
            label,
            (x1, max(10, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if error_x_norm is not None and error_y_norm is not None:
            cv2.putText(
                overlay,
                f"error_norm: x={error_x_norm:+.3f}, y={error_y_norm:+.3f}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    else:
        cv2.putText(
            overlay,
            "target: NONE",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )

    return overlay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--frame-width", type=int, default=640)
    parser.add_argument("--frame-height", type=int, default=480)

    parser.add_argument("--model", type=str, default="yolo11n.pt")  # will auto-download if available
    parser.add_argument("--target-class-name", type=str, default="person")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)

    parser.add_argument("--show", action="store_true", default=True)
    parser.add_argument("--print-error", action="store_true", default=False)
    args = parser.parse_args()

    # Load YOLO
    yolo_model = YOLO(args.model)
    class_name_to_id = build_class_name_to_id(yolo_model)
    class_id_to_name = {v: k for k, v in class_name_to_id.items()}

    if args.target_class_name.strip().lower() not in [name.lower() for name in class_name_to_id.keys()]:
        print(f"[WARN] target_class_name='{args.target_class_name}' not found in model class names.")
        print(f"[INFO] available classes example (show first 20): {list(class_name_to_id.keys())[:20]}")

    # Open webcam
    video_capture = cv2.VideoCapture(args.camera_index)
    if not video_capture.isOpened():
        raise RuntimeError(f"Failed to open webcam camera_index={args.camera_index}")

    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.frame_width)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frame_height)

    last_frame_time_sec = time.time()
    fps_ema = 0.0
    fps_alpha = 0.1  # smoothing

    print(
        "[INFO] Start webcam YOLO. "
        f"model={args.model}, target_class={args.target_class_name}, min_conf={args.min_confidence}, imgsz={args.imgsz}"
    )
    print("[INFO] Press 'q' to quit.")

    while True:
        ok, frame_bgr = video_capture.read()
        if not ok or frame_bgr is None:
            print("[WARN] Failed to read frame from webcam.")
            continue

        frame_height_px, frame_width_px = frame_bgr.shape[:2]

        # YOLO expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Inference
        results = yolo_model.predict(
            source=frame_rgb,
            imgsz=args.imgsz,
            conf=args.min_confidence,
            verbose=False,
        )
        result = results[0]

        detections_all = parse_detections(result, class_id_to_name)
        detections_target_class = filter_by_class_name(
            detections_all,
            target_class_name=args.target_class_name,
            min_confidence=args.min_confidence,
        )
        target_detection = select_largest_bbox(detections_target_class)

        error_x_norm: Optional[float] = None
        error_y_norm: Optional[float] = None
        if target_detection is not None:
            error_x_norm, error_y_norm = compute_normalized_center_error(
                target_center_px=target_detection.center_px,
                frame_width_px=frame_width_px,
                frame_height_px=frame_height_px,
            )
            if args.print_error:
                print(f"error_norm x={error_x_norm:+.3f} y={error_y_norm:+.3f}")

        # FPS
        now_sec = time.time()
        dt = max(now_sec - last_frame_time_sec, 1e-6)
        inst_fps = 1.0 / dt
        fps_ema = inst_fps if fps_ema <= 0.0 else (fps_alpha * inst_fps + (1.0 - fps_alpha) * fps_ema)
        last_frame_time_sec = now_sec

        if args.show:
            overlay = draw_overlay(frame_bgr, fps=fps_ema, target_detection=target_detection, error_x_norm=error_x_norm, error_y_norm=error_y_norm)
            cv2.imshow("webcam_yolo_detector", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    video_capture.release()
    cv2.destroyAllWindows()
    print("[INFO] Exit.")


if __name__ == "__main__":
    main()
