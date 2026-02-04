# follow_ui_node v0.000 2026-02-04
# [이번 버전에서 수정된 사항]
# - (기능구현) PyQt 기반 2패널 UI: 왼쪽 영상(/follow/annotated_image 우선), 오른쪽 로그 리스트(/follow/ui_event)
# - (기능구현) 이벤트 수신 시 UI에서 타임스탬프를 붙여 로그에 누적(최대 N줄 유지)
# - (기능구현) annotated 토픽이 끊기면 원본 이미지 토픽(raw_image_topic)으로 자동 fallback (옵션)

from __future__ import annotations

import datetime
import threading
import time
from typing import Optional

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from PyQt5 import QtCore, QtGui, QtWidgets


def _now_stamp() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]


class FollowUiRosNode(Node):
    def __init__(self, ui_bridge: "UiBridge") -> None:
        super().__init__("follow_ui_node")
        self._ui_bridge = ui_bridge
        self._bridge = CvBridge()

        self.declare_parameter("annotated_image_topic", "/follow/annotated_image")
        self.declare_parameter("raw_image_topic", "/camera/color/image_raw")
        self.declare_parameter("ui_event_topic", "/follow/ui_event")
        self.declare_parameter("annotated_stale_sec", 0.5)
        self.declare_parameter("image_fit_mode", "contain")  # contain|stretch

        self._annotated_topic = str(self.get_parameter("annotated_image_topic").value)
        self._raw_topic = str(self.get_parameter("raw_image_topic").value)
        self._event_topic = str(self.get_parameter("ui_event_topic").value)
        self._annotated_stale_sec = float(self.get_parameter("annotated_stale_sec").value)
        self._fit_mode = str(self.get_parameter("image_fit_mode").value)

        self._last_annotated_t = 0.0
        self._last_raw_t = 0.0

        self.create_subscription(Image, self._annotated_topic, self._on_annotated, 10)
        self.create_subscription(Image, self._raw_topic, self._on_raw, 10)
        self.create_subscription(String, self._event_topic, self._on_event, 50)

        self.get_logger().info(
            f"UI topics: annotated={self._annotated_topic}, raw={self._raw_topic}, event={self._event_topic}"
        )

    def _imgmsg_to_bgr(self, msg: Image):
        try:
            enc = (msg.encoding or "").lower()
            if enc in ("mono8", "8uc1"):
                gray = self._bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return None

    def _on_annotated(self, msg: Image) -> None:
        bgr = self._imgmsg_to_bgr(msg)
        if bgr is None:
            return
        self._last_annotated_t = time.time()
        self._ui_bridge.push_frame(bgr, source="annotated", fit_mode=self._fit_mode)

    def _on_raw(self, msg: Image) -> None:
        bgr = self._imgmsg_to_bgr(msg)
        if bgr is None:
            return
        self._last_raw_t = time.time()
        if (time.time() - self._last_annotated_t) > self._annotated_stale_sec:
            self._ui_bridge.push_frame(bgr, source="raw", fit_mode=self._fit_mode)

    def _on_event(self, msg: String) -> None:
        text = (msg.data or "").strip()
        if not text:
            return
        self._ui_bridge.push_log(f"[{_now_stamp()}] {text}")


class UiBridge(QtCore.QObject):
    frame_signal = QtCore.pyqtSignal(object, str, str)  # bgr, source, fit_mode
    log_signal = QtCore.pyqtSignal(str)

    def push_frame(self, bgr, source: str, fit_mode: str) -> None:
        self.frame_signal.emit(bgr, source, fit_mode)

    def push_log(self, line: str) -> None:
        self.log_signal.emit(line)


class FollowUiWindow(QtWidgets.QMainWindow):
    def __init__(self, bridge: UiBridge) -> None:
        super().__init__()
        self.setWindowTitle("Follow UI (Video + Log)")
        self.resize(1280, 720)

        self._max_logs = 300

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        self._video_label = QtWidgets.QLabel("Waiting for image...")
        self._video_label.setAlignment(QtCore.Qt.AlignCenter)
        self._video_label.setMinimumSize(640, 480)
        self._video_label.setStyleSheet("background-color: #111; color: #ddd; font-size: 18px;")

        self._source_label = QtWidgets.QLabel("source: -")
        self._source_label.setStyleSheet("color: #888; padding: 4px;")

        self._log_list = QtWidgets.QListWidget()
        self._log_list.setStyleSheet("font-size: 14px;")
        self._log_list.setUniformItemSizes(True)

        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.clicked.connect(self._log_list.clear)

        left_box = QtWidgets.QVBoxLayout()
        left_box.addWidget(self._video_label, stretch=1)
        left_box.addWidget(self._source_label, stretch=0)

        right_box = QtWidgets.QVBoxLayout()
        right_box.addWidget(self._log_list, stretch=1)
        right_box.addWidget(clear_btn, stretch=0)

        root = QtWidgets.QHBoxLayout()
        root.addLayout(left_box, stretch=3)
        root.addLayout(right_box, stretch=2)
        central.setLayout(root)

        bridge.frame_signal.connect(self._on_frame)
        bridge.log_signal.connect(self._on_log)

    def _on_log(self, line: str) -> None:
        self._log_list.addItem(line)
        while self._log_list.count() > self._max_logs:
            self._log_list.takeItem(0)
        self._log_list.scrollToBottom()

    def _on_frame(self, bgr, source: str, fit_mode: str) -> None:
        self._source_label.setText(f"source: {source}")

        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)

        if fit_mode == "stretch":
            self._video_label.setPixmap(
                pix.scaled(self._video_label.size(), QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
            )
        else:
            self._video_label.setPixmap(
                pix.scaled(self._video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            )


def _spin_ros(node: Node, executor: SingleThreadedExecutor, stop_event: threading.Event) -> None:
    while rclpy.ok() and not stop_event.is_set():
        executor.spin_once(timeout_sec=0.1)


def main(args=None) -> None:
    rclpy.init(args=args)

    app = QtWidgets.QApplication([])
    bridge = UiBridge()
    window = FollowUiWindow(bridge)
    window.show()

    ros_node = FollowUiRosNode(bridge)
    executor = SingleThreadedExecutor()
    executor.add_node(ros_node)

    stop_event = threading.Event()
    th = threading.Thread(target=_spin_ros, args=(ros_node, executor, stop_event), daemon=True)
    th.start()

    try:
        app.exec_()
    finally:
        stop_event.set()
        try:
            executor.shutdown()
        except Exception:
            pass
        try:
            ros_node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()
