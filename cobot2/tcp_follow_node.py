# tcp_follow_node v0.700 2026-02-02
# [이번 버전에서 수정된 사항]
# - (버그수정) spin 스레드에서 movej 호출 시 rclpy spin 재진입(ValueError: generator already executing) 발생 -> startup movej를 main()에서 executor.spin() 전에 1회 실행하도록 변경
# - (기능제거) auto_sign_tune 관련 파라미터/로직 제거(튜닝 스레드/샘플링/로그 포함)
# - (기능구현) startup_settle_sec(이동 후 안정화 대기) 추가 + startup 이후 error/필터 상태 리셋
# - (유지) 기존 Y/Z speedl 추종 로직(EMA/deadzone/clamp/target_lost 정지) 및 디버그 posx 퍼블리시/로그 구조 유지

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

    # 일부 환경/모드에서 set_robot_mode가 거부될 수 있어 예외 무시
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
    console_labels: List[str]  # 콘솔 출력 허용 라벨(화이트리스트)
    flush_each_write: bool = True


class LabelLogger:
    """
    라벨별 파일 저장 + (옵션) 콘솔 출력(화이트리스트).

    - 파일: <log_dir>/<label>.log  (label 소문자)
    - 라벨: WARN / ERROR / POSE / DPOS  (AUTO_SIGN 제거됨)
    - 콘솔: console_labels에 포함된 라벨만 출력
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
        fp = open(path, "a", buffering=1)  # line-buffered
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

        # 콘솔 출력 (화이트리스트 라벨만)
        if self._cfg.log_console_enable and (label.upper() in self._console_set):
            if level == "error":
                self._ros_logger.error(line)
            elif level == "warn":
                self._ros_logger.warn(line)
            else:
                self._ros_logger.info(line)

        # 라벨별 파일 저장
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
        """
        주의: DSR_ROBOT2.movej는 내부에서 spin_until_future_complete를 사용하므로
        executor.spin()이 이미 돌아가는 상태(다른 스레드 포함)에서 호출하면
        'generator already executing'가 날 수 있다.
        -> main()에서 executor.spin() 전에만 호출하도록 설계.
        """
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

        # DSR_ROBOT2 speedl 시그니처 호환 (환경별 차이 대응)
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
        """Return current TCP pose [x,y,z,rx,ry,rz] if available, else None."""
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
        self.declare_parameter("startup_movej_joints_deg", [0.0, 0.0, 90.0, -90.0, 0.0, 0.0])
        self.declare_parameter("startup_movej_vel", 60.0)
        self.declare_parameter("startup_movej_acc", 60.0)
        self.declare_parameter("startup_settle_sec", 0.8)  # movej 후 안정화 대기

        self.declare_parameter("command_rate_hz", 40.0)
        self.declare_parameter("target_lost_timeout_sec", 0.5)
        self.declare_parameter("speedl_acc", 300.0)
        self.declare_parameter("speedl_time_scale", 1.2)

        # BASE Y/Z tracking (vx=0)
        self.declare_parameter("vy_mm_s_per_error", 180.0)  # ex -> vy
        self.declare_parameter("vz_mm_s_per_error", 180.0)  # ey -> vz
        self.declare_parameter("vmax_y_mm_s", 250.0)
        self.declare_parameter("vmax_z_mm_s", 250.0)
        self.declare_parameter("deadzone_error_norm", 0.02)
        self.declare_parameter("filter_alpha", 0.25)
        self.declare_parameter("y_sign", 1.0)
        self.declare_parameter("z_sign", -1.0)

        self.declare_parameter("error_topic", "/follow/error_norm")

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

        # ---- State
        self._robot = RobotInterface(self, dry_run=self._dry_run)

        self._latest_error_norm: Optional[Tuple[float, float]] = None
        self._latest_error_time_sec: float = 0.0
        self._err_lock = threading.Lock()

        self._filt_ex: float = 0.0
        self._filt_ey: float = 0.0
        self._have_filter: bool = False

        # startup: movej는 main()에서 수행, 여기서는 완료 플래그/리셋만 담당
        self._startup_done: bool = False

        # posx monitor state
        self._posx_prev: Optional[List[float]] = None
        self._posx_prev_t: Optional[float] = None

        # ---- I/O
        self.create_subscription(Float32MultiArray, self._error_topic, self._on_error_norm, 10)

        if self._dbg_posx_pub:
            self._pub_posx = self.create_publisher(Float32MultiArray, self._dbg_posx_topic, 10)
            self._pub_dposx = self.create_publisher(Float32MultiArray, self._dbg_dposx_topic, 10)
        else:
            self._pub_posx = None
            self._pub_dposx = None

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
        """
        main()에서 startup movej가 끝난 직후 호출.
        - settle 대기
        - error/필터 상태 리셋
        - startup_done=True
        """
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

        self._startup_done = True

    # -----------------------------
    # posx monitor (POSE/DPOS) -> 파일+토픽만 (콘솔은 whitelist로 자동 차단)
    # -----------------------------
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

    # -----------------------------
    # speedl loop
    # -----------------------------
    def spin_speedl_loop(self) -> None:
        dt = 1.0 / max(self._command_rate_hz, 1.0)
        cmd_time = dt * max(self._speedl_time_scale, 1.0)

        if self._dbg_posx_enable:
            threading.Thread(target=self._posx_monitor_loop, daemon=True).start()

        while rclpy.ok():
            # startup 완료 전엔 절대 추종하지 않음
            if not self._startup_done:
                self._robot.speedl((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), acc=self._speedl_acc, time_s=cmd_time)
                time.sleep(dt)
                continue

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

        # ✅ 핵심: startup movej는 executor.spin() 전에만 1회 호출
        if bool(follow_node.get_parameter("startup_movej_enable").value):
            joints = list(follow_node.get_parameter("startup_movej_joints_deg").value)
            vel = float(follow_node.get_parameter("startup_movej_vel").value)
            acc = float(follow_node.get_parameter("startup_movej_acc").value)
            follow_node._robot.movej_startup(joints, vel=vel, acc=acc)

        # movej 이후 settle/리셋/스타트 완료
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

    # ✅ startup 완료 후에 추종 스레드 시작
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
