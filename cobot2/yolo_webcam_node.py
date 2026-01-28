# yolo_webcam_node v0.101 2026-01-28
# [이번 버전에서 수정된 사항]
# - (버그수정) webcam opened 로그가 _camera_index 초기화 이전에 실행되어 AttributeError가 발생하던 문제 수정(로그 위치 이동)
# - (기능구현) 웹캠 오픈 성공 및 첫 프레임 수신 시점을 터미널에 1회씩 명확히 출력하는 로그 추가

"""
YOLO Webcam Node (ROS2)

Publishes:
- /follow/error_norm (std_msgs/Float32MultiArray): [error_x_norm, error_y_norm]
  * error_x_norm: + means target is to the right of image center
  * error_y_norm: + means target is below image center

Targeting:
- Filter by target_class_name
- Select the largest bbox (area) among filtered detections
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
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


def _build_class_maps(model: YOLO) -> Tuple[Dict[str, int], Dict[int, str]]:
    names = model.names
    if isinstance(names, dict):
        name_to_id = {str(name): int(class_id) for class_id, name in names.items()}
        id_to_name = {int(class_id): str(name) for class_id, name in names.items()}
        return name_to_id, id_to_name
    name_to_id = {str(name): int(i) for i, name in enumerate(names)}
    id_to_name = {int(i): str(name) for i, name in enumerate(names)}
    return name_to_id, id_to_name


def _parse_detections(result, id_to_name: Dict[int, str]) -> List[Detection2D]:
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
        class_name = id_to_name.get(int(class_id), f"class_{class_id}")
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


def _filter_by_class_name(
    detections: List[Detection2D],
    target_class_name: str,
    min_confidence: float,
) -> List[Detection2D]:
    target_name = target_class_name.strip().lower()
    return [
        d
        for d in detections
        if d.class_name.lower() == target_name and d.confidence >= min_confidence
    ]


def _select_largest(detections: List[Detection2D]) -> Optional[Detection2D]:
    if not detections:
        return None
    return max(detections, key=lambda d: d.area_px2)


def _compute_error_norm(
    target_center_px: Tuple[float, float],
    frame_width_px: int,
    frame_height_px: int,
) -> Tuple[float, float]:
    image_center_x_px = frame_width_px / 2.0
    image_center_y_px = frame_height_px / 2.0
    target_center_x_px, target_center_y_px = target_center_px

    error_x_norm = (target_center_x_px - image_center_x_px) / max(image_center_x_px, 1.0)
    error_y_norm = (target_center_y_px - image_center_y_px) / max(image_center_y_px, 1.0)
    return float(error_x_norm), float(error_y_norm)


class YoloWebcamNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_webcam_node")

        # Params
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("frame_width", 640)
        self.declare_parameter("frame_height", 480)

        self.declare_parameter("model", "yolo11n.pt")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("target_class_name", "person")
        self.declare_parameter("min_confidence", 0.5)

        self.declare_parameter("publish_topic", "/follow/error_norm")
        self.declare_parameter("control_hz", 15.0)

        self.declare_parameter("show_debug", True)
        self.declare_parameter("print_error", False)

        self._camera_index: int = int(self.get_parameter("camera_index").value)
        self._frame_width: int = int(self.get_parameter("frame_width").value)
        self._frame_height: int = int(self.get_parameter("frame_height").value)

        self._model_path: str = str(self.get_parameter("model").value)
        self._imgsz: int = int(self.get_parameter("imgsz").value)
        self._target_class_name: str = str(self.get_parameter("target_class_name").value)
        self._min_confidence: float = float(self.get_parameter("min_confidence").value)

        self._publish_topic: str = str(self.get_parameter("publish_topic").value)
        self._control_hz: float = float(self.get_parameter("control_hz").value)

        self._show_debug: bool = bool(self.get_parameter("show_debug").value)
        self._print_error: bool = bool(self.get_parameter("print_error").value)

        # Publisher
        self._pub_error = self.create_publisher(Float32MultiArray, self._publish_topic, 10)

        # YOLO
        self._yolo_model = YOLO(self._model_path)
        _, self._id_to_name = _build_class_maps(self._yolo_model)

        # Webcam
        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open webcam index={self._camera_index}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_height)

        self.get_logger().info(f"[YOLO_WEBCAM] webcam opened (index={self._camera_index})")

        # First-frame log flag
        self._first_frame_logged: bool = False

        # FPS
        self._last_time_sec = time.time()
        self._fps_ema = 0.0
        self._fps_alpha = 0.1

        # Timer
        period_sec = 1.0 / max(self._control_hz, 1.0)
        self.create_timer(period_sec, self._on_timer)

        self.get_logger().info(
            "[YOLO_WEBCAM] ready "
            f"(cam={self._camera_index}, size={self._frame_width}x{self._frame_height}, "
            f"model={self._model_path}, target_class={self._target_class_name}, conf>={self._min_confidence}, "
            f"imgsz={self._imgsz}, hz={self._control_hz:.1f}, pub={self._publish_topic})"
        )

    def destroy_node(self) -> bool:
        try:
            if self._cap is not None:
                self._cap.release()
            if self._show_debug:
                cv2.destroyAllWindows()
        except Exception:
            pass
        return super().destroy_node()

    def _draw_debug(
        self,
        frame_bgr: np.ndarray,
        fps: float,
        target: Optional[Detection2D],
        error_x_norm: Optional[float],
        error_y_norm: Optional[float],
    ) -> None:
        frame_height_px, frame_width_px = frame_bgr.shape[:2]
        center_x = int(frame_width_px / 2)
        center_y = int(frame_height_px / 2)
        cv2.drawMarker(
            frame_bgr,
            (center_x, center_y),
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            thickness=2,
        )

        cv2.putText(
            frame_bgr,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

        if target is not None:
            x1, y1, x2, y2 = map(int, [target.x1_px, target.y1_px, target.x2_px, target.y2_px])
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cx, cy = target.center_px
            cv2.circle(frame_bgr, (int(cx), int(cy)), 6, (0, 255, 0), -1)

            cv2.putText(
                frame_bgr,
                f"{target.class_name} conf={target.confidence:.2f}",
                (x1, max(10, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            if error_x_norm is not None and error_y_norm is not None:
                cv2.putText(
                    frame_bgr,
                    f"error_norm: x={error_x_norm:+.3f}, y={error_y_norm:+.3f}",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
        else:
            cv2.putText(
                frame_bgr,
                "target: NONE",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 255),
                2,
            )

        cv2.imshow("yolo_webcam_node", frame_bgr)
        cv2.waitKey(1)

    def _on_timer(self) -> None:
        ok, frame_bgr = self._cap.read()
        if not ok or frame_bgr is None:
            self.get_logger().warn("[YOLO_WEBCAM] failed to read frame")
            return

        frame_height_px, frame_width_px = frame_bgr.shape[:2]

        if not self._first_frame_logged:
            self.get_logger().info(f"[YOLO_WEBCAM] first frame received ({frame_width_px}x{frame_height_px})")
            self._first_frame_logged = True

        # YOLO expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        results = self._yolo_model.predict(
            source=frame_rgb,
            imgsz=self._imgsz,
            conf=self._min_confidence,
            verbose=False,
        )
        result = results[0]

        detections_all = _parse_detections(result, self._id_to_name)
        detections_target = _filter_by_class_name(
            detections_all,
            self._target_class_name,
            self._min_confidence,
        )
        target = _select_largest(detections_target)

        error_x_norm: Optional[float] = None
        error_y_norm: Optional[float] = None

        if target is not None:
            error_x_norm, error_y_norm = _compute_error_norm(
                target.center_px,
                frame_width_px,
                frame_height_px,
            )

            msg = Float32MultiArray()
            msg.data = [error_x_norm, error_y_norm]
            self._pub_error.publish(msg)

            if self._print_error:
                self.get_logger().info(f"[YOLO_WEBCAM] error_norm x={error_x_norm:+.3f} y={error_y_norm:+.3f}")

        # FPS
        now_sec = time.time()
        dt = max(now_sec - self._last_time_sec, 1e-6)
        inst_fps = 1.0 / dt
        self._fps_ema = (
            inst_fps
            if self._fps_ema <= 0.0
            else (self._fps_alpha * inst_fps + (1.0 - self._fps_alpha) * self._fps_ema)
        )
        self._last_time_sec = now_sec

        if self._show_debug:
            self._draw_debug(frame_bgr, self._fps_ema, target, error_x_norm, error_y_norm)


def main() -> None:
    rclpy.init()
    node = YoloWebcamNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
