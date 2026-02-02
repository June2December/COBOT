# tcp_follow_node v1.000 2026-02-02
# [이번 버전에서 수정된 사항]
# - (기능구현) base Y/Z 축 절대 리미트 추가: limit_base_y_min/max_mm, limit_base_z_min/max_mm 범위 밖으로 나가려는 vy/vz를 방향별로 컷
# - (유지) startup movej(main()에서 executor.spin 이전 1회 실행), settle/필터리셋, Y/Z speedl 추종(EMA/deadzone/clamp/target_lost) 유지
# - (유지) 디버그 posx 퍼블리시/로그 구조 유지

from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool

import DR_init

# ==========================================================
# ROBOT constants
# ==========================================================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
ROBOT_TOOL = "Tool Weight"
ROBOT_TCP = "GripperDA"

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


def initialize_robot(node: Node):
    """main()에서 노드 준비 후 1회만 호출."""
    import DSR_ROBOT2 as dr

    try:
        dr.set_robot_mode(dr.ROBOT_MODE_AUTONOMOUS)
    except Exception:
        pass

    dr.set_tool(ROBOT_TOOL)
    dr.set_tcp(ROBOT_TCP)
    return dr


# ==========================================================
# Follow params
# ==========================================================
@dataclass
class FollowParamsYZ:
    vy_mm_s_per_error: float
    vz_mm_s_per_error: float
    vmax_y_mm_s: float
    vmax_z_mm_s: float
    deadzone_error_norm: float
    filter_alpha: float
    y_sign: float
    z_sign: float


class RobotInterface:
    def __init__(self, node: Node, *, dry_run: bool):
        self._node = node
        self._dry_run = dry_run
        self._dr = None

    def set_dr(self, dr) -> None:
        self._dr = dr
        if not hasattr(self._dr, "speedl"):
            raise AttributeError("DSR_ROBOT2 missing speedl()")

    def movej_startup(self, joints_deg: List[float], *, vel: float, acc: float) -> None:
        if self._dry_run or self._dr is None:
            return
        self._dr.movej(joints_deg, vel=vel, acc=acc)

    def speedl(
        self,
        vel_6: Tuple[float, float, float, float, float, float],
        *,
        acc: float,
        time_s: float,
    ) -> None:
        if self._dry_run or self._dr is None:
            return

        try:
            self._dr.speedl(list(vel_6), acc, time_s)
            return
        except TypeError:
            pass

        try:
            self._dr.speedl(vel=list(vel_6), acc=acc, time=time_s)
            return
        except TypeError:
            pass

        acc6 = [float(acc)] * 6
        self._dr.speedl(list(vel_6), acc6, time_s)

    def get_current_posx(self):
        if self._dry_run or self._dr is None:
            return None
        if not hasattr(self._dr, "get_current_posx"):
            return None
        try:
            out = self._dr.get_current_posx()
            if isinstance(out, (list, tuple)) and len(out) > 0:
                if isinstance(out[0], (list, tuple)) and len(out[0]) >= 6:
                    return list(out[0])[:6]
                if len(out) >= 6 and isinstance(out[0], (int, float)):
                    return list(out)[:6]
            return None
        except Exception:
            return None


class TcpFollowNode(Node):
    def __init__(self) -> None:
        super().__init__("tcp_follow_node", namespace=ROBOT_ID)

        # ---- Params
        self.declare_parameter("dry_run", False)

        self.declare_parameter("startup_movej_enable", True)
        self.declare_parameter("startup_movej_joints_deg", [0.0, 0.0, -90.0, 90.0, 0.0, 180.0])
        self.declare_parameter("startup_movej_vel", 60.0)
        self.declare_parameter("startup_movej_acc", 60.0)
        self.declare_parameter("startup_settle_sec", 0.8)

        self.declare_parameter("command_rate_hz", 40.0)
        self.declare_parameter("target_lost_timeout_sec", 0.5)
        self.declare_parameter("speedl_acc", 300.0)
        self.declare_parameter("speedl_time_scale", 1.2)

        self.declare_parameter("vy_mm_s_per_error", 320.0)
        self.declare_parameter("vz_mm_s_per_error", 320.0)
        self.declare_parameter("vmax_y_mm_s", 400.0)
        self.declare_parameter("vmax_z_mm_s", 400.0)
        self.declare_parameter("deadzone_error_norm", 0.02)
        self.declare_parameter("filter_alpha", 0.45)
        self.declare_parameter("y_sign", -1.0)
        self.declare_parameter("z_sign", -1.0)

        self.declare_parameter("error_topic", "/follow/error_norm")
        self.declare_parameter("enable_topic", "/follow/enable")
        self.declare_parameter("follow_enable_default", True)


        # ---- base Y/Z absolute limits (NEW)
        self.declare_parameter("limit_base_y_enable", True)
        self.declare_parameter("limit_base_y_min_mm", -300.0)
        self.declare_parameter("limit_base_y_max_mm", 300.0)

        self.declare_parameter("limit_base_z_enable", True)
        self.declare_parameter("limit_base_z_min_mm", 400.0)
        self.declare_parameter("limit_base_z_max_mm", 600.0)

        self.declare_parameter("limit_base_yz_poll_hz", 10.0)  # posx 샘플링 주기

        # posx monitor
        self.declare_parameter("debug_posx_enable", True)
        self.declare_parameter("debug_posx_rate_hz", 10.0)
        self.declare_parameter("debug_posx_pub", True)
        self.declare_parameter("debug_posx_topic", "/follow/posx_debug")
        self.declare_parameter("debug_dposx_topic", "/follow/dposx_debug")

        # ---- Read params
        self._dry_run: bool = bool(self.get_parameter("dry_run").value)

        self._startup_movej_enable: bool = bool(self.get_parameter("startup_movej_enable").value)
        self._startup_movej_joints_deg: List[float] = list(self.get_parameter("startup_movej_joints_deg").value)
        self._startup_movej_vel: float = float(self.get_parameter("startup_movej_vel").value)
        self._startup_movej_acc: float = float(self.get_parameter("startup_movej_acc").value)
        self._startup_settle_sec: float = float(self.get_parameter("startup_settle_sec").value)

        self._enable_topic: str = str(self.get_parameter("enable_topic").value)
        self._follow_enabled: bool = bool(self.get_parameter("follow_enable_default").value)
        self._enable_lock = threading.Lock()


        self._command_rate_hz: float = float(self.get_parameter("command_rate_hz").value)
        self._target_lost_timeout_sec: float = float(self.get_parameter("target_lost_timeout_sec").value)
        self._speedl_acc: float = float(self.get_parameter("speedl_acc").value)
        self._speedl_time_scale: float = float(self.get_parameter("speedl_time_scale").value)

        self._params = FollowParamsYZ(
            vy_mm_s_per_error=float(self.get_parameter("vy_mm_s_per_error").value),
            vz_mm_s_per_error=float(self.get_parameter("vz_mm_s_per_error").value),
            vmax_y_mm_s=float(self.get_parameter("vmax_y_mm_s").value),
            vmax_z_mm_s=float(self.get_parameter("vmax_z_mm_s").value),
            deadzone_error_norm=float(self.get_parameter("deadzone_error_norm").value),
            filter_alpha=float(self.get_parameter("filter_alpha").value),
            y_sign=float(self.get_parameter("y_sign").value),
            z_sign=float(self.get_parameter("z_sign").value),
        )

        self._error_topic: str = str(self.get_parameter("error_topic").value)

        # limits
        self._limit_y_enable: bool = bool(self.get_parameter("limit_base_y_enable").value)
        self._limit_y_min: float = float(self.get_parameter("limit_base_y_min_mm").value)
        self._limit_y_max: float = float(self.get_parameter("limit_base_y_max_mm").value)

        self._limit_z_enable: bool = bool(self.get_parameter("limit_base_z_enable").value)
        self._limit_z_min: float = float(self.get_parameter("limit_base_z_min_mm").value)
        self._limit_z_max: float = float(self.get_parameter("limit_base_z_max_mm").value)

        self._limit_poll_hz: float = float(self.get_parameter("limit_base_yz_poll_hz").value)

        self._dbg_posx_enable: bool = bool(self.get_parameter("debug_posx_enable").value)
        self._dbg_posx_rate_hz: float = float(self.get_parameter("debug_posx_rate_hz").value)
        self._dbg_posx_pub: bool = bool(self.get_parameter("debug_posx_pub").value)
        self._dbg_posx_topic: str = str(self.get_parameter("debug_posx_topic").value)
        self._dbg_dposx_topic: str = str(self.get_parameter("debug_dposx_topic").value)

        self._robot = RobotInterface(self, dry_run=self._dry_run)

        self._latest_error_norm: Optional[Tuple[float, float]] = None
        self._latest_error_time_sec: float = 0.0
        self._err_lock = threading.Lock()

        self._filt_ex: float = 0.0
        self._filt_ey: float = 0.0
        self._have_filter: bool = False

        self._startup_done: bool = False

        self._posx_prev: Optional[List[float]] = None
        self._posx_prev_t: Optional[float] = None

        # limit runtime state
        self._lim_lock = threading.Lock()
        self._y_latest: Optional[float] = None
        self._z_latest: Optional[float] = None
        self._last_poll_t: float = 0.0
        self._warn_t: float = 0.0

        self.create_subscription(Float32MultiArray, self._error_topic, self._on_error_norm, 10)

        self.create_subscription(Bool, self._enable_topic, self._on_enable, 10)

        if self._dbg_posx_pub:
            self._pub_posx = self.create_publisher(Float32MultiArray, self._dbg_posx_topic, 10)
            self._pub_dposx = self.create_publisher(Float32MultiArray, self._dbg_dposx_topic, 10)
        else:
            self._pub_posx = None
            self._pub_dposx = None

    def set_dr(self, dr) -> None:
        self._robot.set_dr(dr)

    def destroy_node(self):
        super().destroy_node()

    def _on_error_norm(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 2:
            return
        ex, ey = float(msg.data[0]), float(msg.data[1])
        with self._err_lock:
            self._latest_error_norm = (ex, ey)
            self._latest_error_time_sec = time.time()

    def _target_alive(self) -> bool:
        with self._err_lock:
            if self._latest_error_norm is None:
                return False
            return (time.time() - self._latest_error_time_sec) <= self._target_lost_timeout_sec

    def _get_latest_error(self) -> Optional[Tuple[float, float]]:
        with self._err_lock:
            return self._latest_error_norm
    
    def _on_enable(self, msg: Bool) -> None:
        with self._enable_lock:
            self._follow_enabled = bool(msg.data)


    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return lo if v < lo else hi if v > hi else v

    def _ema(self, prev: float, cur: float, alpha: float) -> float:
        return (1.0 - alpha) * prev + alpha * cur

    def finalize_startup_after_movej(self) -> None:
        if self._startup_done:
            return

        if self._startup_settle_sec > 0.0:
            time.sleep(self._startup_settle_sec)

        with self._err_lock:
            self._latest_error_norm = None
            self._latest_error_time_sec = 0.0
        self._have_filter = False
        self._filt_ex = 0.0
        self._filt_ey = 0.0

        # prime Y/Z latest
        if self._limit_y_enable or self._limit_z_enable:
            posx = self._robot.get_current_posx()
            if posx is not None and len(posx) >= 3:
                with self._lim_lock:
                    self._y_latest = float(posx[1])
                    self._z_latest = float(posx[2])
                    self._last_poll_t = time.time()
        self._startup_done = True

    def _poll_base_yz_if_needed(self) -> None:
        if not (self._limit_y_enable or self._limit_z_enable):
            return
        if self._limit_poll_hz <= 0.0:
            return

        now = time.time()
        dt_need = 1.0 / max(self._limit_poll_hz, 1e-3)

        with self._lim_lock:
            last = self._last_poll_t
        if (now - last) < dt_need:
            return

        posx = self._robot.get_current_posx()
        if posx is None or len(posx) < 3:
            return

        y = float(posx[1])
        z = float(posx[2])
        with self._lim_lock:
            self._y_latest = y
            self._z_latest = z
            self._last_poll_t = now

    def _apply_base_yz_limits(self, vy: float, vz: float) -> Tuple[float, float]:
        if not (self._limit_y_enable or self._limit_z_enable):
            return vy, vz

        with self._lim_lock:
            y = self._y_latest
            z = self._z_latest

        # Y: 상한 이상에서 +방향으로 더 가려하면 컷
        if self._limit_y_enable and (y is not None):
            if (y >= self._limit_y_max) and (vy > 0.0):
                vy = 0.0
            if (y <= self._limit_y_min) and (vy < 0.0):
                vy = 0.0

        # Z: 상한 이상에서 +방향으로 더 가려하면 컷
        if self._limit_z_enable and (z is not None):
            if (z >= self._limit_z_max) and (vz > 0.0):
                vz = 0.0
            if (z <= self._limit_z_min) and (vz < 0.0):
                vz = 0.0

        return vy, vz

    def _posx_monitor_loop(self):
        dt = 1.0 / max(self._dbg_posx_rate_hz, 1.0)
        while rclpy.ok():
            if not self._dbg_posx_enable:
                time.sleep(dt)
                continue

            posx = self._robot.get_current_posx()
            now = time.time()

            if posx is not None:
                if self._pub_posx is not None:
                    m = Float32MultiArray()
                    m.data = [float(v) for v in posx]
                    self._pub_posx.publish(m)

                if self._posx_prev is not None and self._posx_prev_t is not None:
                    d = [posx[i] - self._posx_prev[i] for i in range(6)]
                    dt_sec = max(1e-6, now - self._posx_prev_t)

                    if self._pub_dposx is not None:
                        md = Float32MultiArray()
                        md.data = [float(v) for v in d]
                        self._pub_dposx.publish(md)

                self._posx_prev = posx
                self._posx_prev_t = now

            time.sleep(dt)

    def spin_speedl_loop(self) -> None:
        dt = 1.0 / max(self._command_rate_hz, 1.0)
        cmd_time = dt * max(self._speedl_time_scale, 1.0)

        if self._dbg_posx_enable:
            threading.Thread(target=self._posx_monitor_loop, daemon=True).start()

        while rclpy.ok():
            with self._enable_lock:
                enabled = self._follow_enabled

            if not enabled:
                self._robot.speedl((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), acc=self._speedl_acc, time_s=cmd_time)
                time.sleep(dt)
                continue

            if not self._startup_done:
                self._robot.speedl((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), acc=self._speedl_acc, time_s=cmd_time)
                time.sleep(dt)
                continue

            self._poll_base_yz_if_needed()

            if not self._target_alive():
                self._robot.speedl((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), acc=self._speedl_acc, time_s=cmd_time)
                time.sleep(dt)
                continue

            e = self._get_latest_error()
            if e is None:
                self._robot.speedl((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), acc=self._speedl_acc, time_s=cmd_time)
                time.sleep(dt)
                continue

            ex, ey = e

            if abs(ex) < self._params.deadzone_error_norm:
                ex = 0.0
            if abs(ey) < self._params.deadzone_error_norm:
                ey = 0.0

            if not self._have_filter:
                self._filt_ex, self._filt_ey = ex, ey
                self._have_filter = True
            else:
                self._filt_ex = self._ema(self._filt_ex, ex, self._params.filter_alpha)
                self._filt_ey = self._ema(self._filt_ey, ey, self._params.filter_alpha)

            fx, fy = self._filt_ex, self._filt_ey

            vy = self._params.y_sign * (self._params.vy_mm_s_per_error * fx)
            vz = self._params.z_sign * (self._params.vz_mm_s_per_error * fy)

            vy = self._clamp(vy, -self._params.vmax_y_mm_s, self._params.vmax_y_mm_s)
            vz = self._clamp(vz, -self._params.vmax_z_mm_s, self._params.vmax_z_mm_s)

            # ✅ base Y/Z 절대 리미트 적용
            vy, vz = self._apply_base_yz_limits(vy, vz)

            self._robot.speedl((0.0, vy, vz, 0.0, 0.0, 0.0), acc=self._speedl_acc, time_s=cmd_time)
            time.sleep(dt)

def main(args=None) -> None:
    rclpy.init(args=args)

    follow_node = TcpFollowNode()

    dsr_node = rclpy.create_node("dsr_internal_worker", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node

    try:
        dr = initialize_robot(follow_node)
        follow_node.set_dr(dr)

        if bool(follow_node.get_parameter("startup_movej_enable").value):
            joints = list(follow_node.get_parameter("startup_movej_joints_deg").value)
            vel = float(follow_node.get_parameter("startup_movej_vel").value)
            acc = float(follow_node.get_parameter("startup_movej_acc").value)
            follow_node._robot.movej_startup(joints, vel=vel, acc=acc)

        follow_node.finalize_startup_after_movej()

    except Exception as e:
        follow_node.destroy_node()
        dsr_node.destroy_node()
        rclpy.shutdown()
        return

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(follow_node)
    executor.add_node(dsr_node)

    t = threading.Thread(target=follow_node.spin_speedl_loop, daemon=True)
    t.start()

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            follow_node.destroy_node()
            dsr_node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()
