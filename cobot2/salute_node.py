#!/usr/bin/env python3
import time
import threading
import traceback

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from dsr_msgs2.srv import MoveJoint


class SaluteRunner(Node):
    def __init__(self):
        super().__init__("salute_runner")

        # ---- params ----
        self.declare_parameter("trigger_topic", "/salute_trigger")
        self.declare_parameter("done_topic", "/salute_done")
        self.declare_parameter("vel", 30.0)
        self.declare_parameter("acc", 50.0)
        self.declare_parameter("cooldown_sec", 2.0)

        # ✅ 네 환경: /dsr01 네임스페이스
        self.declare_parameter("robot_ns", "/dsr01")
        self.declare_parameter("service_timeout_sec", 20.0)

        self.trigger_topic = self.get_parameter("trigger_topic").value
        self.done_topic = self.get_parameter("done_topic").value
        self.vel = float(self.get_parameter("vel").value)
        self.acc = float(self.get_parameter("acc").value)
        self.cooldown_sec = float(self.get_parameter("cooldown_sec").value)

        # ✅ robot_ns 읽어서 멤버로 저장 (이번 에러 해결 포인트)
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
            f"Subscribed: {self.trigger_topic} (Bool). True -> salute\n"
            f"Publish done: {self.done_topic} (Bool)\n"
            f"Using service: {self._srv_move_joint_name}"
        )

        # 서비스 준비될 때까지 대기
        while not self._cli_move_joint.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f"Waiting for service: {self._srv_move_joint_name} ...")

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
        ok = False
        try:
            ok = self._salute_motion()
        except Exception:
            self.get_logger().error("Salute failed:\n" + traceback.format_exc())
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
        if hasattr(req, "blendType"):
            req.blendType = 0
        if hasattr(req, "syncType"):
            req.syncType = 0

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

    def _salute_motion(self) -> bool:
        vel = self.vel
        acc = self.acc

        J_READY = [0, 0, 90, -90, 0, 0]
        J_SALUTE1 = [-90, 90, 0, -90, 0, 90]
        J_SALUTE2 = [-90, 90, -120, -90, 0, 90]

        self.get_logger().info("Robot: move to READY")
        if not self._call_movej(J_READY, vel=vel, acc=acc):
            return False

        self.get_logger().info("Robot: SALUTE")
        if not self._call_movej(J_SALUTE1, vel=vel, acc=acc): return False
        time.sleep(1.0)
        if not self._call_movej(J_SALUTE2, vel=50, acc=acc): return False
        time.sleep(2.0)

        self.get_logger().info("Robot: back to READY")
        if not self._call_movej(J_READY, vel=vel, acc=acc):
            return False

        return True


def main():
    rclpy.init()
    node = SaluteRunner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
