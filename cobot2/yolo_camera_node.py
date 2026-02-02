# yolo_camera_node v0.412 2026-02-02
# [이번 버전에서 수정된 사항]
# - (버그수정) model 가중치 파일 상대경로(FileNotFoundError: 'day.pt') 해결:
#   - model 값이 절대경로가 아니면 여러 후보 경로에서 자동 탐색 후 로드
#   - 탐색 후보: (1) 입력 그대로(절대/상대) (2) 현재 파일 폴더 (3) package share (4) workspace src 추정
# - (유지) day_image_topic 기본값=/camera/camera/color/image_raw, 락온/유지(0.93/0.4) + tracker, error_norm, lock_done 1회 publish 유지

from __future__ import annotations

import datetime
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray, String
from ultralytics import YOLO

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:
    get_package_share_directory = None  # type: ignore[assignment]


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1_px: float
    y1_px: float
    x2_px: float
    y2_px: float
    track_id: Optional[int] = None

    @property
    def center_px(self) -> Tuple[float, float]:
        return ((self.x1_px + self.x2_px) * 0.5, (self.y1_px + self.y2_px) * 0.5)

    @property
    def area(self) -> float:
        return max(0.0, self.x2_px - self.x1_px) * max(0.0, self.y2_px - self.y1_px)


def _compute_error_norm(center_px: Tuple[float, float], w: int, h: int) -> Tuple[float, float]:
    cx, cy = center_px
    ex = (cx - (w * 0.5)) / (w * 0.5)
    ey = (cy - (h * 0.5)) / (h * 0.5)
    ex = float(np.clip(ex, -1.0, 1.0))
    ey = float(np.clip(ey, -1.0, 1.0))
    return ex, ey


def _resolve_model_path(model: str, package_name: str = "cobot2") -> Tuple[str, List[str]]:
    """
    Ultralytics YOLO(weights) 경로를 최대한 자동으로 찾아준다.
    - model이 절대경로면 그대로 사용
    - 상대경로면 여러 후보를 만들어 존재하는 첫 번째를 사용
    반환: (resolved_path, tried_paths)
    """
    tried: List[str] = []
    model = str(model).strip()

    if not model:
        return model, tried

    # 1) 사용자가 준 경로 그대로 (상대/절대)
    tried.append(model)
    if os.path.isabs(model) and os.path.isfile(model):
        return model, tried
    if os.path.isfile(model):
        return os.path.abspath(model), tried

    # 2) 현재 파일(yolo_camera_node.py)이 있는 폴더 기준
    here_dir = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here_dir, model)
    tried.append(cand)
    if os.path.isfile(cand):
        return cand, tried

    # 3) install/share/<pkg> 기준 (data_files로 weights를 설치했다면 여기서 잡힘)
    if get_package_share_directory is not None:
        try:
            share_dir = get_package_share_directory(package_name)
            cand2 = os.path.join(share_dir, model)
            tried.append(cand2)
            if os.path.isfile(cand2):
                return cand2, tried

            # 흔히 share 아래에 weights/로 넣는 경우
            cand3 = os.path.join(share_dir, "weights", model)
            tried.append(cand3)
            if os.path.isfile(cand3):
                return cand3, tried
        except Exception:
            pass

    # 4) workspace src 경로 추정 (/home/<user>/<ws>/src/<pkg>/<pkg>/<model>)
    # install/cobot2/lib/python3.10/site-packages/cobot2/.. 형태에서 ws 루트를 역으로 못 박긴 어렵지만,
    # 흔히 /home/gom/cobot_ws/src/cobot2/cobot2 에 파일이 있으니 그 패턴을 후보로 추가
    # (실제 경로가 다르면 launch에서 model 절대경로 권장)
    for guess in [
        os.path.expanduser(f"~/cobot_ws/src/{package_name}/{package_name}/{model}"),
        os.path.expanduser(f"~/cobot_ws/src/{package_name}/{model}"),
    ]:
        tried.append(guess)
        if os.path.isfile(guess):
            return guess, tried

    return model, tried


class YoloCameraNode(Node):
    def __init__(self):
        super().__init__("yolo_camera_node")
        self._bridge = CvBridge()

        # -------------------------------
        # Parameters (기본)
        # -------------------------------
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("model", "day.pt")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("target_class_name", "person")

        # 락온/유지 threshold
        self.declare_parameter("lock_conf_high", 0.93)
        self.declare_parameter("maintain_conf_low", 0.4)
        self.declare_parameter("lost_timeout_sec", 0.6)

        # tracker
        self.declare_parameter("use_tracker", True)
        self.declare_parameter("tracker_yaml", "bytetrack.yaml")

        # 출력
        self.declare_parameter("publish_topic", "/follow/error_norm")
        self.declare_parameter("show_debug", True)

        # 입력 뒤집기
        self.declare_parameter("input_flip_v", True)
        self.declare_parameter("input_flip_h", False)

        # 락온 완료 더미 토픽 (1회)
        self.declare_parameter("lock_done_topic", "/follow/lock_done")
        self.declare_parameter("lock_done_delay_sec", 1.0)

        # 시간 기반 토픽 전환
        self.declare_parameter("enable_time_based_switch", False)
        self.declare_parameter("day_image_topic", "/camera/camera/color/image_raw")  # ✅ 원래 쓰던 토픽
        self.declare_parameter("night_image_topic", "/camera/camera/infra1/image_rect_raw")
        self.declare_parameter("day_start_hms", [7, 30, 0])
        self.declare_parameter("night_start_hms", [17, 44, 0])
        self.declare_parameter("time_check_period_sec", 1.0)

        # 외부 override 토픽 (String.data = 새 image_topic)
        self.declare_parameter("image_topic_override_topic", "/follow/image_topic_override")

        # -------------------------------
        # Read params
        # -------------------------------
        self._image_topic: str = str(self.get_parameter("image_topic").value)
        self._model_path_raw: str = str(self.get_parameter("model").value)
        self._imgsz: int = int(self.get_parameter("imgsz").value)
        self._target_class: str = str(self.get_parameter("target_class_name").value)

        self._lock_conf_high: float = float(self.get_parameter("lock_conf_high").value)
        self._maintain_conf_low: float = float(self.get_parameter("maintain_conf_low").value)
        self._lost_timeout_sec: float = float(self.get_parameter("lost_timeout_sec").value)

        self._use_tracker: bool = bool(self.get_parameter("use_tracker").value)
        self._tracker_yaml: str = str(self.get_parameter("tracker_yaml").value)

        self._publish_topic: str = str(self.get_parameter("publish_topic").value)
        self._show_debug: bool = bool(self.get_parameter("show_debug").value)

        self._input_flip_v: bool = bool(self.get_parameter("input_flip_v").value)
        self._input_flip_h: bool = bool(self.get_parameter("input_flip_h").value)

        self._lock_done_topic: str = str(self.get_parameter("lock_done_topic").value)
        self._lock_done_delay_sec: float = float(self.get_parameter("lock_done_delay_sec").value)

        self._enable_time_switch: bool = bool(self.get_parameter("enable_time_based_switch").value)
        self._day_topic: str = str(self.get_parameter("day_image_topic").value)
        self._night_topic: str = str(self.get_parameter("night_image_topic").value)
        self._day_hms: List[int] = list(self.get_parameter("day_start_hms").value)
        self._night_hms: List[int] = list(self.get_parameter("night_start_hms").value)
        self._time_check_period: float = float(self.get_parameter("time_check_period_sec").value)

        self._override_topic: str = str(self.get_parameter("image_topic_override_topic").value)

        # -------------------------------
        # YOLO model (path resolve)
        # -------------------------------
        resolved, tried = _resolve_model_path(self._model_path_raw, package_name="cobot2")
        self._model_path = resolved

        if not os.path.isfile(self._model_path):
            self.get_logger().error(f"[YOLO_CAMERA] model not found: '{self._model_path_raw}'")
            self.get_logger().error("[YOLO_CAMERA] tried paths:")
            for p in tried:
                self.get_logger().error(f"  - {p}")
            raise FileNotFoundError(f"model file not found: {self._model_path_raw}")

        self.get_logger().info(f"[YOLO_CAMERA] loading model: {self._model_path}")
        self._yolo = YOLO(self._model_path)

        try:
            self._id_to_name: Dict[int, str] = dict(self._yolo.names)  # type: ignore[arg-type]
        except Exception:
            self._id_to_name = {}

        # -------------------------------
        # ROS pubs/subs
        # -------------------------------
        self._pub_err = self.create_publisher(Float32MultiArray, self._publish_topic, 10)
        self._pub_lock_done = self.create_publisher(Bool, self._lock_done_topic, 10)

        self._sub_image = None
        self._switch_image_topic(self._image_topic, reason="init")
        self.create_subscription(String, self._override_topic, self._on_override_topic, 10)

        self._is_day: Optional[bool] = None
        if self._enable_time_switch:
            self._timer = self.create_timer(self._time_check_period, self._on_time_check)

        # Lock-on state
        self._locked_id: Optional[int] = None
        self._locked_last_seen_t: float = 0.0
        self._lock_acquired_t: float = 0.0
        self._lock_done_published: bool = False

        # Debug / FPS
        self._t_prev = time.time()
        self._fps_ema = 0.0

        self.get_logger().info(
            f"[YOLO_CAMERA] ready (sub={self._image_topic}, model={self._model_path_raw} -> {self._model_path}, "
            f"target={self._target_class}, lock>={self._lock_conf_high}, keep>={self._maintain_conf_low}, "
            f"time_switch={self._enable_time_switch}, lock_done_delay={self._lock_done_delay_sec:.2f}s)"
        )

    # -------------------------------
    # Topic switching
    # -------------------------------
    def _reset_lock_state(self) -> None:
        self._locked_id = None
        self._locked_last_seen_t = 0.0
        self._lock_acquired_t = 0.0
        self._lock_done_published = False

    def _switch_image_topic(self, new_topic: str, *, reason: str) -> None:
        new_topic = str(new_topic).strip()
        if not new_topic:
            return
        if new_topic == self._image_topic and self._sub_image is not None:
            return

        if self._sub_image is not None:
            try:
                self.destroy_subscription(self._sub_image)
            except Exception:
                pass
            self._sub_image = None

        self._image_topic = new_topic
        self._sub_image = self.create_subscription(Image, self._image_topic, self._on_image, 10)

        self._reset_lock_state()
        self.get_logger().warn(f"[YOLO_CAMERA] image_topic switched -> {self._image_topic} ({reason})")

    def _on_override_topic(self, msg: String) -> None:
        topic = str(msg.data).strip()
        if not topic:
            return
        self._switch_image_topic(topic, reason="override")

    def _is_daytime(self) -> bool:
        now = datetime.datetime.now().time()
        day_start = datetime.time(*[int(x) for x in self._day_hms[:3]])
        night_start = datetime.time(*[int(x) for x in self._night_hms[:3]])

        if day_start < night_start:
            return day_start <= now < night_start
        return now >= day_start or now < night_start

    def _on_time_check(self) -> None:
        is_day = self._is_daytime()
        if self._is_day is None:
            self._is_day = is_day
            target_topic = self._day_topic if is_day else self._night_topic
            self._switch_image_topic(target_topic, reason="time_init")
            return

        if is_day != self._is_day:
            self._is_day = is_day
            target_topic = self._day_topic if is_day else self._night_topic
            self._switch_image_topic(target_topic, reason="time_change")

    # -------------------------------
    # Publish
    # -------------------------------
    def _publish_error(self, ex: float, ey: float) -> None:
        msg = Float32MultiArray()
        msg.data = [float(ex), float(ey)]
        self._pub_err.publish(msg)

    def _maybe_publish_lock_done(self) -> None:
        if self._lock_done_published:
            return
        if self._locked_id is None or self._lock_acquired_t <= 0.0:
            return
        if (time.time() - self._lock_acquired_t) >= max(0.0, self._lock_done_delay_sec):
            self._pub_lock_done.publish(Bool(data=True))
            self._lock_done_published = True
            self.get_logger().warn("[YOLO_CAMERA] lock_done published (dummy)")

    # -------------------------------
    # YOLO
    # -------------------------------
    def _run_inference(self, frame_bgr: np.ndarray):
        if self._use_tracker:
            return self._yolo.track(
                source=frame_bgr,
                conf=self._maintain_conf_low,
                persist=True,
                tracker=self._tracker_yaml,
                verbose=False,
            )
        return self._yolo.predict(
            source=frame_bgr,
            imgsz=self._imgsz,
            conf=self._maintain_conf_low,
            verbose=False,
        )

    def _extract_detections(self, result) -> List[Detection]:
        dets: List[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return dets

        xyxy = boxes.xyxy
        confs = boxes.conf
        clss = boxes.cls
        ids = getattr(boxes, "id", None)

        xyxy_np = xyxy.cpu().numpy()
        confs_np = confs.cpu().numpy()
        clss_np = clss.cpu().numpy()
        ids_np = ids.cpu().numpy().astype(int) if ids is not None else None

        for i, ((x1, y1, x2, y2), conf, cls_id) in enumerate(zip(xyxy_np, confs_np, clss_np)):
            cid = int(cls_id)
            cname = self._id_to_name.get(cid, str(cid))
            tid = int(ids_np[i]) if ids_np is not None else None
            dets.append(
                Detection(
                    class_id=cid,
                    class_name=cname,
                    confidence=float(conf),
                    x1_px=float(x1),
                    y1_px=float(y1),
                    x2_px=float(x2),
                    y2_px=float(y2),
                    track_id=tid,
                )
            )
        return dets

    def _pick_target_with_lock(self, dets: List[Detection]) -> Optional[Detection]:
        now = time.time()
        dets = [d for d in dets if d.class_name == self._target_class]

        if self._locked_id is not None:
            for d in dets:
                if d.track_id == self._locked_id:
                    self._locked_last_seen_t = now
                    return d
            if (now - self._locked_last_seen_t) > self._lost_timeout_sec:
                self._reset_lock_state()
            return None

        candidates = [d for d in dets if d.confidence >= self._lock_conf_high]
        if not candidates:
            return None

        target = max(candidates, key=lambda d: d.area)
        if target.track_id is not None:
            self._locked_id = target.track_id
            self._locked_last_seen_t = now
            self._lock_acquired_t = now
            self._lock_done_published = False
        return target

    # -------------------------------
    # Image callback
    # -------------------------------
    def _on_image(self, msg: Image) -> None:
        frame_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        if self._input_flip_v and self._input_flip_h:
            frame_bgr = cv2.flip(frame_bgr, -1)
        elif self._input_flip_v:
            frame_bgr = cv2.flip(frame_bgr, 0)
        elif self._input_flip_h:
            frame_bgr = cv2.flip(frame_bgr, 1)

        h, w = frame_bgr.shape[:2]

        results = self._run_inference(frame_bgr)
        result0 = results[0]

        dets_all = self._extract_detections(result0)
        target = self._pick_target_with_lock(dets_all)

        if target is not None:
            ex, ey = _compute_error_norm(target.center_px, w, h)
            self._publish_error(ex, ey)
        else:
            self._publish_error(0.0, 0.0)

        self._maybe_publish_lock_done()

        if self._show_debug:
            self._draw_debug(frame_bgr, target, w, h)

    def _draw_debug(self, frame: np.ndarray, target: Optional[Detection], w: int, h: int) -> None:
        t = time.time()
        dt = max(1e-6, t - self._t_prev)
        self._t_prev = t
        fps = 1.0 / dt
        self._fps_ema = 0.9 * self._fps_ema + 0.1 * fps if self._fps_ema > 0.0 else fps

        cv2.circle(frame, (w // 2, h // 2), 6, (0, 255, 255), -1)
        cv2.putText(frame, f"FPS: {self._fps_ema:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        lock_txt = f"LOCK: {self._locked_id}" if self._locked_id is not None else "LOCK: -"
        cv2.putText(frame, lock_txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        done_txt = "DONE:1" if self._lock_done_published else "DONE:0"
        cv2.putText(frame, done_txt, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if target is not None:
            x1, y1, x2, y2 = map(int, [target.x1_px, target.y1_px, target.x2_px, target.y2_px])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cx, cy = target.center_px
            cv2.circle(frame, (int(cx), int(cy)), 6, (0, 255, 0), -1)

            ex, ey = _compute_error_norm(target.center_px, w, h)
            tid = target.track_id if target.track_id is not None else -1
            cv2.putText(
                frame,
                f"{target.class_name} id={tid} conf={target.confidence:.2f} err=({ex:+.3f},{ey:+.3f})",
                (10, 125),
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
