import json
import time
import sys
import importlib

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

TOPIC_NAME = "/is_dambae"
SERVICE_NAME = "/get_is_dambae"


class DambaeRobotNode(Node):
    def __init__(self):
        super().__init__("sign_motion_node", namespace=ROBOT_ID)

        # ---- DR_init / DSR_ROBOT2 통일 로딩 ----
        try:
            dr = importlib.import_module("dsr_common2.imp.DR_init")
        except Exception:
            dr = importlib.import_module("DR_init")

        # 여러 버전 호환용으로 다 넣어줌
        for k, v in [
            ("__dsr__id", ROBOT_ID),
            ("__dsr__model", ROBOT_MODEL),
            ("__dsr__node", self),
            ("_robot_id", ROBOT_ID),
            ("_robot_model", ROBOT_MODEL),
            ("g_node", self),
        ]:
            try:
                setattr(dr, k, v)
            except Exception:
                pass

        sys.modules["DR_init"] = dr  # DSR_ROBOT2가 보는 DR_init 강제

        for modname in ("dsr_common2.imp.DSR_ROBOT2", "DSR_ROBOT2"):
            if modname in sys.modules:
                del sys.modules[modname]

        try:
            DSR = importlib.import_module("dsr_common2.imp.DSR_ROBOT2")
        except Exception:
            DSR = importlib.import_module("DSR_ROBOT2")

        try:
            posx = DSR.posx
        except Exception:
            from DR_common2 import posx  # fallback

        self.posx = posx
        self.movej = getattr(DSR, "movej", None)
        self.get_current_posj = getattr(DSR, "get_current_posj", None)
        self.movel = DSR.movel
        self.get_current_posx = DSR.get_current_posx
        self.set_velx = DSR.set_velx
        self.set_accx = DSR.set_accx
        self.wait = DSR.wait

        # state
        self._last_is_dambae = False
        self._last_stamp_ns = 0
        self._pending = False

        self.create_subscription(Bool, TOPIC_NAME, self._on_topic, 10)
        self.create_service(Trigger, SERVICE_NAME, self._on_service)

        # motion params
        self.ud_motion = 30.0
        self.lr_motion = 40.0
        self.vel_mm_s = 300.0
        self.acc_mm_s2 = 250.0
        self.radius_mm = 0.0

        self.get_logger().info(f"✅ Subscribing: {TOPIC_NAME}")
        self.get_logger().info(f"✅ Service ready: {SERVICE_NAME}")

    def _on_topic(self, msg: Bool):
        self._last_is_dambae = bool(msg.data)
        self._last_stamp_ns = time.time_ns()
        self._pending = True
        self.get_logger().info(f"Topic update: is_dambae={self._last_is_dambae}")

    def _on_service(self, request, response):
        payload = {"is_dambae": self._last_is_dambae, "stamp_ns": int(self._last_stamp_ns)}
        response.success = True
        response.message = json.dumps(payload, ensure_ascii=False)
        return response

    def run_pending_motion_if_any(self):
        """메인 루프에서 호출: 콜백 밖에서 로봇을 움직인다 (중첩 spin 방지)."""
        if not self._pending:
            return
        self._pending = False

        try:
            try:
                self.set_velx(self.vel_mm_s)
                self.set_accx(self.acc_mm_s2)
            except Exception:
                pass

            if self._last_is_dambae:
                self.get_logger().info("✅ '담배'(True) -> UP/DOWN 5s")
                self._move_up_down()
            else:
                self.get_logger().info("❌ NOT '담배'(False) -> LEFT/RIGHT 5s")
                self._move_left_right()

        except Exception as e:
            self.get_logger().error(f"Motion failed: {e}")

    def _get_center_pose(self):
        cur = self.get_current_posx()
        pose = cur[0] if isinstance(cur, (tuple, list)) and len(cur) == 2 else cur
        return pose[:6]

    def _move_left_right(self, duration_sec: float):
        x, y, z, rx, ry, rz = self._get_center_pose()
        left = self.posx(x, y - self.lr_motion, z, rx, ry, rz)
        right = self.posx(x, y + self.lr_motion, z, rx, ry, rz)
        center = self.posx(x, y, z, rx, ry, rz)

        for _ in range(2):  # 왕복 2번
            self.movel(left, vel=self.vel_mm_s, acc=self.acc_mm_s2, radius=self.radius_mm)
            self.movel(right, vel=self.vel_mm_s, acc=self.acc_mm_s2, radius=self.radius_mm)


        self.movel(center, vel=self.vel_mm_s, acc=self.acc_mm_s2, radius=self.radius_mm)
        self.wait(0.1)

    def _move_up_down(self, duration_sec: float):
        x, y, z, rx, ry, rz = self._get_center_pose()
        up = self.posx(x, y, z + self.ud_motion, rx, ry, rz)
        down = self.posx(x, y, z - self.ud_motion, rx, ry, rz)
        center = self.posx(x, y, z, rx, ry, rz)

        for _ in range(2):  # 왕복 2번
            self.movel(up, vel=self.vel_mm_s, acc=self.acc_mm_s2, radius=self.radius_mm)
            self.movel(down, vel=self.vel_mm_s, acc=self.acc_mm_s2, radius=self.radius_mm)


        self.movel(center, vel=self.vel_mm_s, acc=self.acc_mm_s2, radius=self.radius_mm)
        self.wait(0.1)


def main():
    rclpy.init()
    node = DambaeRobotNode()
    try:
        while rclpy.ok():
            # 메시지만 짧게 처리
            rclpy.spin_once(node, timeout_sec=0.1)
            # 콜백 밖에서 로봇 동작 실행
            node.run_pending_motion_if_any()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
