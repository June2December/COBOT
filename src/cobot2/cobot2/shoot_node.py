#!/usr/bin/env python3
import time
import threading
import traceback

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from dsr_msgs2.srv import MoveJoint

from cobot2.speak import speak


class ShootRunner(Node):
    def __init__(self):
        super().__init__("shoot_runner")

        # ---- params ----
        self.declare_parameter("trigger_topic", "/shoot_trigger")
        self.declare_parameter("done_topic", "/shoot_done")
        self.declare_parameter("vel", 30.0)
        self.declare_parameter("acc", 50.0)
        self.declare_parameter("cooldown_sec", 2.0)
        self.declare_parameter("robot_ns", "/dsr01")
        self.declare_parameter("service_timeout_sec", 20.0)
        
        # TTS params
        self.declare_parameter("tts_enabled", True)
        self.declare_parameter("tts_text", "사격 개시")

        self.tts_enabled = bool(self.get_parameter("tts_enabled").value)
        self.tts_text = str(self.get_parameter("tts_text").value)

        self.trigger_topic = self.get_parameter("trigger_topic").value
        self.done_topic = self.get_parameter("done_topic").value
        self.vel = float(self.get_parameter("vel").value)
        self.acc = float(self.get_parameter("acc").value)
        self.cooldown_sec = float(self.get_parameter("cooldown_sec").value)

        self.robot_ns = self.get_parameter("robot_ns").value.rstrip("/")
        if self.robot_ns == "":
            self.robot_ns = "/dsr01"
        self.service_timeout_sec = float(self.get_parameter("service_timeout_sec").value)

        # ---- pub/sub ----
        self.sub = self.create_subscription(Bool, self.trigger_topic, self._on_trigger, 10)
        self.pub_done = self.create_publisher(Bool, self.done_topic, 10)

        # ---- state ----
        self._busy = False
        self._lock = threading.Lock()
        self._last_time = 0.0

        # ---- service client ----
        self._srv_move_joint_name = f"{self.robot_ns}/motion/move_joint"
        self._cli_move_joint = self.create_client(MoveJoint, self._srv_move_joint_name)

        self.get_logger().info(
            f"Subscribed: {self.trigger_topic} (Bool). True -> shoot\n"
            f"Publish done: {self.done_topic} (Bool)\n"
            f"Using service: {self._srv_move_joint_name}\n"
            f"TTS: {'ON' if self.tts_enabled else 'OFF'} / \"{self.tts_text}\""
        )

    def _on_trigger(self, msg: Bool):
        if not msg.data:
            return

        now = time.time()
        with self._lock:
            if self._busy:
                self.get_logger().warn("Ignored: already running.")
                return
            if now - self._last_time < self.cooldown_sec:
                self.get_logger().warn("Ignored: cooldown.")
                return
            self._busy = True
            self._last_time = now

        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        if not self._cli_move_joint.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(f"Service not available: {self._srv_move_joint_name}")
            return False
        ok = False
        try:
            # ✅ 먼저 TTS
            if self.tts_enabled and self.tts_text.strip():
                speak(self.tts_text)
            ok = self._shoot_motion()

        except Exception:
            self.get_logger().error("Shoot failed:\n" + traceback.format_exc())
            ok = False
        finally:
            self.pub_done.publish(Bool(data=ok))
            with self._lock:
                self._busy = False


    def _call_movej(self, pos, vel, acc, t=0.0) -> bool:
        req = MoveJoint.Request()
        req.pos = [float(x) for x in pos]
        req.vel = float(vel)
        req.acc = float(acc)
        req.time = float(t)
        req.radius = 0.0
        req.mode = 0  # 기본 모드

        # 필드가 있을 때만 세팅 (환경별 msg 생성 차이 대비)
        # srv 변수 오류 CamelCase 가 아녔음
        if hasattr(req, "blend_type"):
            req.blend_type = 0
        if hasattr(req, "sync_type"):
            req.sync_type = 0

        future = self._cli_move_joint.call_async(req)

        start = time.time()
        while rclpy.ok() and not future.done():
            if (time.time() - start) > self.service_timeout_sec:
                raise TimeoutError(f"move_joint timeout ({self.service_timeout_sec}s)")
            time.sleep(0.01)

        resp = future.result()
        if resp is None:
            raise RuntimeError("move_joint got no response")

        if not resp.success:
            self.get_logger().error("move_joint returned success=False")
        return bool(resp.success)

    def _shoot_motion(self) -> bool:
        vel = self.vel
        acc = self.acc
        J_READY = [-4., -24.82, -122.52, 175.12, -57.42, 90.]
        # J_D = [-4., -24.82, -122.52, 175.12, -57.42, 90.]
        J_SALUTE1 = [-4., -24.82, -122.52, 170.12, -57.42, 90.]
        self.get_logger().info("Robot: move to READY")
        if not self._call_movej(J_READY, vel=vel, acc=acc):
            return False
        for i in range(3):

            self.get_logger().info(f"[{i+1}/3] Robot: SHOOT")
            if not self._call_movej(J_SALUTE1, vel=100, acc=150): return False

            self.get_logger().info("Robot: back to READY")
            if not self._call_movej(J_READY, vel=100, acc=150):
                return False
        
        self.get_logger().info("Robot: move to READY")
        if not self._call_movej(J_READY, vel=vel, acc=acc):
            return False
        return True

def main():
    rclpy.init()
    node = ShootRunner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
