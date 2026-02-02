# tcp_follow_node v1.100 2026-02-02
# [이번 버전에서 수정된 사항]
# - (기능구현) 반응성(예민함) 상향 기본값 적용: vy/vz gain 및 vmax 상향, filter_alpha 상향, deadzone 소폭 하향
# - (기능구현) B(=posx ry) 회전 추종 추가: ex 기반으로 speedl의 wy(각속도) 생성/클램프 후 적용(enable_b_rotation)
# - (유지) startup movej(main()에서 executor.spin 이전 1회 실행), settle/필터리셋, Y/Z speedl 추종 및 base Y/Z 절대 리미트 유지
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
from std_msgs.msg import Float32MultiArray

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
# Label logger (console + per-label file)
# ==========================================================
@dataclass
class LabelLogConfig:
    log_dir: str
    log_console_enable: bool
    label_files_enable: bool
    console_labels: List[str]
    flush_each_write: bool = True


class LabelLogger:
    """
    라벨별 파일 저장 + (옵션) 콘솔 출력(화이트리스트).

    - 파일: <log_dir>/<label>.log
    - 라벨: WARN / ERROR / POSE / DPOS
    """

    def __init__(self, ros_logger, cfg: LabelLogConfig):
        self._ros_logger = ros_logger
        self._cfg = cfg
        self._files: Dict[str, object] = {}
        os.makedirs(self._cfg.log_dir, exist_ok=True)
        self._console_set = set([s.upper() for s in (cfg.console_labels or [])])

    def _get_fp(self, label: str):
        if label in self._files:
            return self._files[label]
        path = os.path.join(self._cfg.log_dir, f"{label.lower()}.log")
        fp = open(path, "a", buffering=1)
        self._files[label] = fp
        return fp

    def close(self):
        for fp in self._files.values():
            try:
                fp.close()
            except Exception:
                pass
        self._files.clear()

    def _emit(self, label: str, msg: str, level: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts} [{label}] {msg}"

        if self._cfg.log_console_enable and (label.upper() in self._console_set):
            if level == "error":
                self._ros_logger.error(line)
            elif level == "warn":
                self._ros_logger.warn(line)
            else:
                self._ros_logger.info(line)

        if self._cfg.label_files_enable:
            fp = self._get_fp(label)
            fp.write(line + "\n")
            if self._cfg.flush_each_write:
                try:
                    fp.flush()
                except Exception:
                    pass

    def warn(self, msg: str):
        self._emit("WARN", msg, "warn")

    def error(self, msg: str):
        self._emit("ERROR", msg, "error")

    def pose(self, msg: str):
        self._emit("POSE", msg, "info")

    def dpos(self, msg: str):
        self._emit("DPOS", msg, "info")


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

        # ---- responsiveness tuned defaults (추천 1번 반영)
        self.declare_parameter("vy_mm_s_per_error", 320.0)
        self.declare_parameter("vz_mm_s_per_error", 320.0)
        self.declare_parameter("vmax_y_mm_s", 400.0)
        self.declare_parameter("vmax_z_mm_s", 400.0)
        self.declare_parameter("deadzone_error_norm", 0.015)
        self.declare_parameter("filter_alpha", 0.45)
        self.declare_parameter("y_sign", -1.0)
        self.declare_parameter("z_sign", -1.0)

        self.declare_parameter("error_topic", "/follow/error_norm")

        # ---- base Y/Z absolute limits
        self.declare_parameter("limit_base_y_enable", True)
        self.declare_parameter("limit_base_y_min_mm", -350.0)
        self.declare_parameter("limit_base_y_max_mm", 300.0)

        self.declare_parameter("limit_base_z_enable", True)
        self.declare_parameter("limit_base_z_min_mm", 400.0)
        self.declare_parameter("limit_base_z_max_mm", 600.0)

        self.declare_parameter("limit_base_yz_poll_hz", 10.0)  # posx 샘플링 주기

        # ---- (NEW) B(=posx ry) rotation tracking
        self.declare_parameter("enable_b_rotation", True)
        self.declare_parameter("wb_deg_s_per_error", 12.0)
        self.declare_parameter("wmax_b_deg_s", 18.0)
        self.declare_parameter("b_sign", 1.0)

        # posx monitor
        self.declare_parameter("debug_posx_enable", True)
        self.declare_parameter("debug_posx_rate_hz", 10.0)
        self.declare_parameter("debug_posx_pub", True)
        self.declare_parameter("debug_posx_topic", "/follow/posx_debug")
        self.declare_parameter("debug_dposx_topic", "/follow/dposx_debug")

        # label logger
        self.declare_parameter("log_dir", "/home/gom/logs")
        self.declare_parameter("log_console_enable", True)
        self.declare_parameter("label_files_enable", True)
        self.declare_parameter("console_labels", ["WARN", "ERROR"])

        # ---- Read params
        self._dry_run: bool = bool(self.get_parameter("dry_run").value)

        self._startup_movej_enable: bool = bool(self.get_parameter("startup_movej_enable").value)
        self._startup_movej_joints_deg: List[float] = list(self.get_parameter("startup_movej_joints_deg").value)
        self._startup_movej_vel: float = float(self.get_parameter("startup_movej_vel").value)
        self._startup_movej_acc: float = float(self.get_parameter("startup_movej_acc").value)
        self._startup_settle_sec: float = float(self.get_parameter("startup_settle_sec").value)

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

        # B rotation tracking
        self._enable_b_rotation: bool = bool(self.get_parameter("enable_b_rotation").value)
        self._wb_deg_s_per_error: float = float(self.get_parameter("wb_deg_s_per_error").value)
        self._wmax_b_deg_s: float = float(self.get_parameter("wmax_b_deg_s").value)
        self._b_sign: float = float(self.get_parameter("b_sign").value)

        self._dbg_posx_enable: bool = bool(self.get_parameter("debug_posx_enable").value)
        self._dbg_posx_rate_hz: float = float(self.get_parameter("debug_posx_rate_hz").value)
        self._dbg_posx_pub: bool = bool(self.get_parameter("debug_posx_pub").value)
        self._dbg_posx_topic: str = str(self.get_parameter("debug_posx_topic").value)
        self._dbg_dposx_topic: str = str(self.get_parameter("debug_dposx_topic").value)

        console_labels = [str(x).upper() for x in list(self.get_parameter("console_labels").value)]
        log_cfg = LabelLogConfig(
            log_dir=str(self.get_parameter("log_dir").value),
            log_console_enable=bool(self.get_parameter("log_console_enable").value),
            label_files_enable=bool(self.get_parameter("label_files_enable").value),
            console_labels=console_labels,
        )
        self._llog = LabelLogger(self.get_logger(), log_cfg)

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

        if self._dbg_posx_pub:
            self._pub_posx = self.create_publisher(Float32MultiArray, self._dbg_posx_topic, 10)
            self._pub_dposx = self.create_publisher(Float32MultiArray, self._dbg_dposx_topic, 10)
        else:
            self._pub_posx = None
            self._pub_dposx = None

        # sanity
        if self._limit_y_enable and (self._limit_y_min >= self._limit_y_max):
            self._llog.warn(
                f"base_y_limit invalid: min({self._limit_y_min:.1f}) >= max({self._limit_y_max:.1f}) -> disabled"
            )
            self._limit_y_enable = False
        if self._limit_z_enable and (self._limit_z_min >= self._limit_z_max):
            self._llog.warn(
                f"base_z_limit invalid: min({self._limit_z_min:.1f}) >= max({self._limit_z_max:.1f}) -> disabled"
            )
            self._limit_z_enable = False

    def set_dr(self, dr) -> None:
        self._robot.set_dr(dr)

    def destroy_node(self):
        try:
            self._llog.close()
        except Exception:
            pass
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

            if self._limit_y_enable:
                self._llog.warn(
                    f"base_y_limit ON(abs): range=[{self._limit_y_min:.1f}, {self._limit_y_max:.1f}] mm"
                )
            if self._limit_z_enable:
                self._llog.warn(
                    f"base_z_limit ON(abs): range=[{self._limit_z_min:.1f}, {self._limit_z_max:.1f}] mm"
                )

        if self._enable_b_rotation:
            self._llog.warn(
                f"b_rotation ON: wb={self._wb_deg_s_per_error:.1f} deg/s per err, "
                f"wmax={self._wmax_b_deg_s:.1f} deg/s, b_sign={self._b_sign:+.0f}"
            )

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

    def _throttled_limit_warn(self, msg: str, period_sec: float = 1.0) -> None:
        now = time.time()
        with self._lim_lock:
            if (now - self._warn_t) < period_sec:
                return
            self._warn_t = now
        self._llog.warn(msg)

    def _apply_base_yz_limits(self, vy: float, vz: float) -> Tuple[float, float]:
        if not (self._limit_y_enable or self._limit_z_enable):
            return vy, vz

        with self._lim_lock:
            y = self._y_latest
            z = self._z_latest

        if self._limit_y_enable and (y is not None):
            if (y >= self._limit_y_max) and (vy > 0.0):
                self._throttled_limit_warn(
                    f"base_y_limit hit: y={y:.1f} >= {self._limit_y_max:.1f} -> vy cut"
                )
                vy = 0.0
            if (y <= self._limit_y_min) and (vy < 0.0):
                self._throttled_limit_warn(
                    f"base_y_limit hit: y={y:.1f} <= {self._limit_y_min:.1f} -> vy cut"
                )
                vy = 0.0

        if self._limit_z_enable and (z is not None):
            if (z >= self._limit_z_max) and (vz > 0.0):
                self._throttled_limit_warn(
                    f"base_z_limit hit: z={z:.1f} >= {self._limit_z_max:.1f} -> vz cut"
                )
                vz = 0.0
            if (z <= self._limit_z_min) and (vz < 0.0):
                self._throttled_limit_warn(
                    f"base_z_limit hit: z={z:.1f} <= {self._limit_z_min:.1f} -> vz cut"
                )
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

                    self._llog.pose(
                        f"pos=[{posx[0]:.2f},{posx[1]:.2f},{posx[2]:.2f},{posx[3]:.2f},{posx[4]:.2f},{posx[5]:.2f}]"
                    )
                    self._llog.dpos(
                        f"d=[{d[0]:+.2f},{d[1]:+.2f},{d[2]:+.2f},{d[3]:+.2f},{d[4]:+.2f},{d[5]:+.2f}] dt_ms={dt_sec*1000:.0f}"
                    )

                self._posx_prev = posx
                self._posx_prev_t = now

            time.sleep(dt)

    def spin_speedl_loop(self) -> None:
        dt = 1.0 / max(self._command_rate_hz, 1.0)
        cmd_time = dt * max(self._speedl_time_scale, 1.0)

        if self._dbg_posx_enable:
            threading.Thread(target=self._posx_monitor_loop, daemon=True).start()

        while rclpy.ok():
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

            # ---- (NEW) B rotation tracking (ry) uses ex
            wy = 0.0
            if self._enable_b_rotation:
                wy = self._b_sign * (self._wb_deg_s_per_error * fx)
                wy = self._clamp(wy, -self._wmax_b_deg_s, self._wmax_b_deg_s)

            # base Y/Z 절대 리미트 적용
            vy, vz = self._apply_base_yz_limits(vy, vz)

            self._robot.speedl((0.0, vy, vz, 0.0, wy, 0.0), acc=self._speedl_acc, time_s=cmd_time)
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
        try:
            follow_node._llog.error(f"robot init/startup failed: {e}")
        except Exception:
            pass
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
