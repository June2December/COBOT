# yolo_camera_node v0.300 2026-01-29
# [이번 버전에서 수정된 사항]
# - (기능구현) ROS2 Image 토픽(sensor_msgs/Image) 구독 기반 YOLO 추론 노드 구현(RealSense 포함)
# - (기능구현) 특정 클래스(target_class_name)만 필터링 후 가장 큰 bbox 1개를 타겟으로 선택
# - (기능구현) 타겟 중심 기반 error_norm(Float32MultiArray) 퍼블리시 (/follow/error_norm)
# - (기능구현) show_debug=True 시 디버그 윈도우(bbox/center/error/FPS) 출력
# - (기능구현) 카메라 upside-down 장착 대응: 입력 프레임 상하/좌우 반전 옵션(input_flip_v/input_flip_h) 추가
# - (버그수정) entry_point에서 호출 가능한 main() 함수 제공

"""
YOLO Camera Node (ROS2)

Subscribes:
- image_topic (sensor_msgs/Image)  e.g. /camera/color/image_raw

Publishes:
- /follow/error_norm (std_msgs/Float32MultiArray): [error_x_norm, error_y_norm]
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from ultralytics import YOLO

import os
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import qos_profile_sensor_data
# from datetime import datetime

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1_px: float
    y1_px: float
    x2_px: float
    y2_px: float

    @property
    def center_px(self) -> Tuple[float, float]:
        return ((self.x1_px + self.x2_px) * 0.5, (self.y1_px + self.y2_px) * 0.5)

    @property
    def area(self) -> float:
        return max(0.0, self.x2_px - self.x1_px) * max(0.0, self.y2_px - self.y1_px)


def _parse_detections(result, id_to_name: Dict[int, str]) -> List[Detection]:
    """Ultralytics result -> list[Detection]."""
    dets: List[Detection] = []
    if result.boxes is None or len(result.boxes) == 0:
        return dets

    xyxy = result.boxes.xyxy
    confs = result.boxes.conf
    clss = result.boxes.cls

    xyxy_np = xyxy.cpu().numpy()
    confs_np = confs.cpu().numpy()
    clss_np = clss.cpu().numpy()

    for (x1, y1, x2, y2), conf, cls_id in zip(xyxy_np, confs_np, clss_np):
        cid = int(cls_id)
        dets.append(
            Detection(
                class_id=cid,
                class_name=id_to_name.get(cid, str(cid)),
                confidence=float(conf),
                x1_px=float(x1),
                y1_px=float(y1),
                x2_px=float(x2),
                y2_px=float(y2),
            )
        )
    return dets


def _filter_class(dets: List[Detection], class_name: str, min_conf: float) -> List[Detection]:
    return [d for d in dets if d.class_name == class_name and d.confidence >= min_conf]


def _select_largest(dets: List[Detection]) -> Optional[Detection]:
    if not dets:
        return None
    return max(dets, key=lambda d: d.area)


def _compute_error_norm(center_px: Tuple[float, float], w: int, h: int) -> Tuple[float, float]:
    """Normalize center error to [-1, +1] based on image center."""
    cx, cy = center_px
    ex = (cx - (w * 0.5)) / (w * 0.5)
    ey = (cy - (h * 0.5)) / (h * 0.5)
    ex = float(np.clip(ex, -1.0, 1.0))
    ey = float(np.clip(ey, -1.0, 1.0))
    return ex, ey


class YoloCameraNode(Node):
    def __init__(self):
        super().__init__("yolo_camera_node")

        self._bridge = CvBridge()

        # -------------------------------
        # Parameters
        # -------------------------------
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("model", "Day.pt")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("target_class_name", "person")
        self.declare_parameter("min_confidence", 0.6)
        self.declare_parameter("publish_topic", "/follow/error_norm")
        self.declare_parameter("show_debug", True)
        self.declare_parameter("input_flip_v", True)   # 카메라 상하 반전(기본 ON)
        self.declare_parameter("input_flip_h", False)  # 필요 시 좌우 반전

        self._image_topic = str(self.get_parameter("image_topic").value)
        self._model_filename = str(self.get_parameter("model").value)
        self._imgsz = int(self.get_parameter("imgsz").value)
        self._target_class = str(self.get_parameter("target_class_name").value)
        self._min_conf = float(self.get_parameter("min_confidence").value)
        self._publish_topic = str(self.get_parameter("publish_topic").value)
        self._show_debug = bool(self.get_parameter("show_debug").value)
        self._input_flip_v = bool(self.get_parameter("input_flip_v").value)
        self._input_flip_h = bool(self.get_parameter("input_flip_h").value)

        # -------------------------------
        # YOLO model
        # -------------------------------
        pkg_share = get_package_share_directory("cobot2")     # 패키지명 정확히
        resource_dir = os.path.join(pkg_share, "resource")
        self._model_path = os.path.join(resource_dir, self._model_filename)

        if not os.path.exists(self._model_path):
            raise FileNotFoundError(f"YOLO model not found: {self._model_path}")
        self.get_logger().info(f"[YOLO_CAMERA] loading model: {self._model_path}")
        self._yolo = YOLO(self._model_path).to("cuda")

        # id -> name mapping (from loaded model)
        try:
            self._id_to_name: Dict[int, str] = dict(self._yolo.names)  # type: ignore[arg-type]
        except Exception:
            self._id_to_name = {}

        # -------------------------------
        # ROS pubs/subs
        # -------------------------------
        self._pub = self.create_publisher(Float32MultiArray, self._publish_topic, 10)
        self.create_subscription(Image, self._image_topic, self._on_image, 10)

        # Debug / FPS
        self._t_prev = time.time()
        self._fps_ema = 0.0

        self.get_logger().info(
            f"[YOLO_CAMERA] ready (sub={self._image_topic}, model={self._model_path}, "
            f"imgsz={self._imgsz}, target={self._target_class}, conf>={self._min_conf}, pub={self._publish_topic}, "
            f"flip_v={self._input_flip_v}, flip_h={self._input_flip_h})"
        )

    def _publish_error(self, ex: float, ey: float) -> None:
        msg = Float32MultiArray()
        msg.data = [float(ex), float(ey)]
        self._pub.publish(msg)

    def _on_image(self, msg: Image) -> None:
        frame_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # 카메라가 upside-down(상하 반전) 장착된 경우 입력 프레임을 먼저 뒤집어
        # 추론/에러계산/디버그표시를 모두 같은 좌표계로 통일한다.
        if self._input_flip_v and self._input_flip_h:
            frame_bgr = cv2.flip(frame_bgr, -1)  # both
        elif self._input_flip_v:
            frame_bgr = cv2.flip(frame_bgr, 0)   # vertical
        elif self._input_flip_h:
            frame_bgr = cv2.flip(frame_bgr, 1)   # horizontal

        h, w = frame_bgr.shape[:2]

        # inference
        results = self._yolo.predict(
            source=frame_bgr,
            imgsz=self._imgsz,
            conf=self._min_conf,
            device=0,
            verbose=False,
        )
        result = results[0]

        dets_all = _parse_detections(result, self._id_to_name)
        dets_target = _filter_class(dets_all, self._target_class, self._min_conf)
        target = _select_largest(dets_target)

        if target is not None:
            ex, ey = _compute_error_norm(target.center_px, w, h)
            self._publish_error(ex, ey)
        else:
            self._publish_error(0.0, 0.0)

        if self._show_debug:
            self._draw_debug(frame_bgr, target, w, h)

    def _draw_debug(self, frame: np.ndarray, target: Optional[Detection], w: int, h: int) -> None:
        # FPS (EMA)
        t = time.time()
        dt = max(1e-6, t - self._t_prev)
        self._t_prev = t
        fps = 1.0 / dt
        self._fps_ema = 0.9 * self._fps_ema + 0.1 * fps if self._fps_ema > 0.0 else fps

        # center crosshair
        cv2.circle(frame, (w // 2, h // 2), 6, (0, 255, 255), -1)
        cv2.putText(frame, f"FPS: {self._fps_ema:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if target is not None:
            x1, y1, x2, y2 = map(int, [target.x1_px, target.y1_px, target.x2_px, target.y2_px])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cx, cy = target.center_px
            cv2.circle(frame, (int(cx), int(cy)), 6, (0, 255, 0), -1)

            ex, ey = _compute_error_norm(target.center_px, w, h)
            cv2.putText(
                frame,
                f"{target.class_name} conf={target.confidence:.2f}  err=({ex:+.3f},{ey:+.3f})",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("yolo_camera_node", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = YoloCameraNode()
    try:
        rclpy.spin(node)
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
