# tcp_follow_node v0.200 2026-01-28
# [이번 버전에서 수정된 사항]
# - (기능구현) DSR 통신 전용 노드(dsr_internal_worker) 분리 + MultiThreadedExecutor로 함께 spin 하도록 main() 재구성
# - (기능구현) DR_init.__dsr__node 를 dsr_internal_worker로 지정하여 DSR_ROBOT2의 service client 생성/통신 안정화
# - (기능구현) TcpFollowNode.set_dr()/RobotInterface.set_dr() 추가로 initialize_robot() 결과(dr)를 follow 노드에 주입
# - (유지) 노드 시작 시 1회 startup movej([0,0,90,0,90,0]) 후 추종 시작
# - (유지/주석) 추종 튜닝은 gains/limits(yaw_deg_per_error, max_delta_rotation_deg, deadzone_error_norm, yaw_sign/pitch_sign)만 바꾸면 됨

"""
TCP Follow Node (eye-in-hand, pose-step servo) for Doosan (DSR)

구조:
- TcpFollowNode: /follow/error_norm 구독 -> Δpose 생성 -> dr.movej/dr.movel 호출
- dsr_internal_worker(Node): DSR_ROBOT2가 ROS2 client/service를 만들 때 사용할 전용 노드
- MultiThreadedExecutor: 두 노드를 동시에 spin하여 블로킹 호출로 콜백이 멈추는 문제 완화

Tuning guide:
- yaw_deg_per_error / pitch_deg_per_error : 화면 오차 -> 회전(deg) 민감도
- max_delta_rotation_deg                  : 1 step당 최대 회전량(deg)
- deadzone_error_norm                     : 작은 오차 무시(떨림 감소)
- yaw_sign / pitch_sign                   : 좌우/상하 방향 반대로 나오면 ±1만 뒤집기

추후 수정할 것:
B/C 회전 방향·게인 튜닝
RealSense depth 붙여서 Z축까지
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Float32MultiArray

import DR_init

# ==========================================================
# ROBOT 상수 (사용자 규칙: 파람/상수 정의 바로 뒤 DR_init 세팅 1회)
# ==========================================================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
ROBOT_TOOL = "Tool Weight"
ROBOT_TCP = "GripperDA"

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


def initialize_robot(node: Node):
    """사용자 규칙: main()에서 노드 생성 후 1회만 호출."""
    # ✅ DSR_ROBOT2 import 위치는 여기(함수 내부)로 고정
    import DSR_ROBOT2 as dr

    node.get_logger().info("#" * 50)
    node.get_logger().info("Initializing robot with the following settings:")
    node.get_logger().info(f"ROBOT_ID: {ROBOT_ID}")
    node.get_logger().info(f"ROBOT_MODEL: {ROBOT_MODEL}")
    node.get_logger().info(f"ROBOT_TCP: {ROBOT_TCP}")
    node.get_logger().info(f"ROBOT_TOOL: {ROBOT_TOOL}")
    node.get_logger().info("#" * 50)

    # (필요 시) 모드 설정
    try:
        dr.set_robot_mode(dr.ROBOT_MODE_AUTONOMOUS)
    except Exception:
        pass

    # tool/tcp 1회 설정
    dr.set_tool(ROBOT_TOOL)
    dr.set_tcp(ROBOT_TCP)

    return dr


@dataclass
class FollowGains:
    yaw_deg_per_error: float
    pitch_deg_per_error: float
    x_mm_per_error: float
    y_mm_per_error: float


@dataclass
class FollowLimits:
    max_delta_translation_mm: float
    max_delta_rotation_deg: float
    deadzone_error_norm: float


class RobotInterface:
    """
    dr(DSR_ROBOT2) 핸들 주입 방식:
    - main()에서 initialize_robot()로 dr을 만든 뒤 set_dr(dr)로 주입
    """

    def __init__(self, node: Node, *, dry_run: bool):
        self._node = node
        self._dry_run = dry_run

        self._dr = None
        self._mv_mod_REL = None
        self._DR_TOOL = None

        if self._dry_run:
            self._node.get_logger().warn("[TCP_FOLLOW] dry_run=True (로봇 미구동, 로그만 출력)")

    def set_dr(self, dr) -> None:
        self._dr = dr

        # DSR 상수명은 환경/버전에 따라 다를 수 있어 우선순위로 흡수
        self._mv_mod_REL = getattr(self._dr, "DR_MV_MOD_REL", None) or getattr(self._dr, "mv_mod_REL", None)
        self._DR_TOOL = getattr(self._dr, "DR_TOOL", None)

        if self._mv_mod_REL is None:
            raise AttributeError("DSR_ROBOT2 missing REL motion modifier constant (DR_MV_MOD_REL/mv_mod_REL)")
        if self._DR_TOOL is None:
            raise AttributeError("DSR_ROBOT2 missing DR_TOOL constant")

        self._node.get_logger().info("[TCP_FOLLOW] DSR interface ready (dr injected)")

    def movej_startup(self, joints_deg: List[float], *, vel: float, acc: float) -> None:
        self._node.get_logger().info(
            f"[TCP_FOLLOW] startup movej joints(deg)={joints_deg} (vel={vel:.1f}, acc={acc:.1f})"
        )
        if self._dry_run or self._dr is None:
            return
        self._dr.movej(joints_deg, vel=vel, acc=acc)

    def movel_rel_tool(
        self,
        delta_pose_mm_deg: Tuple[float, float, float, float, float, float],
        *,
        vel: float,
        acc: float,
    ) -> None:
        dx_mm, dy_mm, dz_mm, droll_deg, dpitch_deg, dyaw_deg = delta_pose_mm_deg

        self._node.get_logger().info(
            "[TCP_FOLLOW] movel REL TOOL Δpose = "
            f"[{dx_mm:.2f}, {dy_mm:.2f}, {dz_mm:.2f}, {droll_deg:.2f}, {dpitch_deg:.2f}, {dyaw_deg:.2f}] "
            f"(vel={vel:.1f}, acc={acc:.1f})"
        )

        if self._dry_run or self._dr is None:
            return

        self._dr.movel(
            [dx_mm, dy_mm, dz_mm, droll_deg, dpitch_deg, dyaw_deg],
            vel=vel,
            acc=acc,
            mod=self._mv_mod_REL,
            ref=self._DR_TOOL,
        )


class TcpFollowNode(Node):
    def __init__(self) -> None:
        super().__init__("tcp_follow_node", namespace=ROBOT_ID)

        # ---- Params
        self.declare_parameter("dry_run", False)

        # startup movej
        self.declare_parameter("startup_movej_enable", True)
        self.declare_parameter("startup_movej_joints_deg", [0.0, 0.0, 90.0, 0.0, 90.0, 0.0])
        self.declare_parameter("startup_movej_vel", 60.0)
        self.declare_parameter("startup_movej_acc", 60.0)

        # follow loop
        self.declare_parameter("control_loop_hz", 15.0)
        self.declare_parameter("target_lost_timeout_sec", 0.5)
        self.declare_parameter("vel", 60.0)
        self.declare_parameter("acc", 60.0)

        # tuning knobs (여기만 바꾸면 됨)
        self.declare_parameter("yaw_deg_per_error", 6.0)
        self.declare_parameter("pitch_deg_per_error", 6.0)
        self.declare_parameter("x_mm_per_error", 0.0)
        self.declare_parameter("y_mm_per_error", 0.0)

        self.declare_parameter("max_delta_translation_mm", 3.0)
        self.declare_parameter("max_delta_rotation_deg", 2.0)
        self.declare_parameter("deadzone_error_norm", 0.03)
        self.declare_parameter("yaw_sign", -1.0)
        self.declare_parameter("pitch_sign", -1.0)

        self.declare_parameter("error_topic", "/follow/error_norm")

        # ---- Read params
        self._dry_run: bool = bool(self.get_parameter("dry_run").value)

        self._startup_movej_enable: bool = bool(self.get_parameter("startup_movej_enable").value)
        self._startup_movej_joints_deg: List[float] = list(self.get_parameter("startup_movej_joints_deg").value)
        self._startup_movej_vel: float = float(self.get_parameter("startup_movej_vel").value)
        self._startup_movej_acc: float = float(self.get_parameter("startup_movej_acc").value)

        self._control_loop_hz: float = float(self.get_parameter("control_loop_hz").value)
        self._target_lost_timeout_sec: float = float(self.get_parameter("target_lost_timeout_sec").value)

        self._vel: float = float(self.get_parameter("vel").value)
        self._acc: float = float(self.get_parameter("acc").value)

        self._yaw_sign: float = float(self.get_parameter("yaw_sign").value)
        self._pitch_sign: float = float(self.get_parameter("pitch_sign").value)

        self._gains = FollowGains(
            yaw_deg_per_error=float(self.get_parameter("yaw_deg_per_error").value),
            pitch_deg_per_error=float(self.get_parameter("pitch_deg_per_error").value),
            x_mm_per_error=float(self.get_parameter("x_mm_per_error").value),
            y_mm_per_error=float(self.get_parameter("y_mm_per_error").value),
        )
        self._limits = FollowLimits(
            max_delta_translation_mm=float(self.get_parameter("max_delta_translation_mm").value),
            max_delta_rotation_deg=float(self.get_parameter("max_delta_rotation_deg").value),
            deadzone_error_norm=float(self.get_parameter("deadzone_error_norm").value),
        )

        self._error_topic: str = str(self.get_parameter("error_topic").value)

        # ---- Runtime state
        self._latest_error_norm: Optional[Tuple[float, float]] = None
        self._latest_error_time_sec: float = 0.0
        self._startup_done: bool = False

        # ---- Robot interface (dr 주입 전까지는 호출 시 dry_run처럼 동작)
        self._robot = RobotInterface(self, dry_run=self._dry_run)

        # ---- ROS I/O
        self.create_subscription(Float32MultiArray, self._error_topic, self._on_error_norm, 10)

        # ---- Timer
        timer_period_sec = 1.0 / max(self._control_loop_hz, 1.0)
        self.create_timer(timer_period_sec, self._on_control_timer)

        self.get_logger().info(
            "[TCP_FOLLOW] ready "
            f"(robot_id={ROBOT_ID}, model={ROBOT_MODEL}, tcp={ROBOT_TCP}, tool={ROBOT_TOOL}, dry_run={self._dry_run}, "
            f"hz={self._control_loop_hz:.1f}, lost_timeout={self._target_lost_timeout_sec:.2f}s, "
            f"vel={self._vel:.1f}, acc={self._acc:.1f}, topic={self._error_topic})"
        )

    def set_dr(self, dr) -> None:
        self._robot.set_dr(dr)

    def _run_startup_once(self) -> None:
        if self._startup_done:
            return
        self._startup_done = True

        if not self._startup_movej_enable:
            self.get_logger().info("[TCP_FOLLOW] startup movej disabled")
            return

        self._robot.movej_startup(
            self._startup_movej_joints_deg,
            vel=self._startup_movej_vel,
            acc=self._startup_movej_acc,
        )

    def _on_error_norm(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 2:
            self.get_logger().warn("[TCP_FOLLOW] error_norm msg.data must be [error_x_norm, error_y_norm]")
            return

        self._latest_error_norm = (float(msg.data[0]), float(msg.data[1]))
        self._latest_error_time_sec = time.time()

    def _on_control_timer(self) -> None:
        # (1) startup movej 1회
        self._run_startup_once()

        # (2) 추종 입력 없으면 대기
        if self._latest_error_norm is None:
            return

        now_sec = time.time()
        if (now_sec - self._latest_error_time_sec) > self._target_lost_timeout_sec:
            self._latest_error_norm = None
            self.get_logger().warn("[TCP_FOLLOW] target lost -> stop following")
            return

        error_x_norm, error_y_norm = self._latest_error_norm

        # deadzone (튜닝 포인트)
        if abs(error_x_norm) < self._limits.deadzone_error_norm:
            error_x_norm = 0.0
        if abs(error_y_norm) < self._limits.deadzone_error_norm:
            error_y_norm = 0.0
        if error_x_norm == 0.0 and error_y_norm == 0.0:
            return

        # mapping (튜닝 포인트: gains/limits/sign)
        delta_yaw_deg = self._yaw_sign * (self._gains.yaw_deg_per_error * error_x_norm)
        delta_pitch_deg = self._pitch_sign * (self._gains.pitch_deg_per_error * error_y_norm)

        delta_x_mm = self._gains.x_mm_per_error * error_x_norm
        delta_y_mm = self._gains.y_mm_per_error * error_y_norm

        # clamp (튜닝 포인트)
        delta_x_mm = self._clamp(delta_x_mm, -self._limits.max_delta_translation_mm, self._limits.max_delta_translation_mm)
        delta_y_mm = self._clamp(delta_y_mm, -self._limits.max_delta_translation_mm, self._limits.max_delta_translation_mm)

        delta_yaw_deg = self._clamp(delta_yaw_deg, -self._limits.max_delta_rotation_deg, self._limits.max_delta_rotation_deg)
        delta_pitch_deg = self._clamp(delta_pitch_deg, -self._limits.max_delta_rotation_deg, self._limits.max_delta_rotation_deg)

        self._robot.movel_rel_tool(
            (delta_x_mm, delta_y_mm, 0.0, 0.0, delta_pitch_deg, delta_yaw_deg),
            vel=self._vel,
            acc=self._acc,
        )

    @staticmethod
    def _clamp(value: float, min_value: float, max_value: float) -> float:
        if value < min_value:
            return min_value
        if value > max_value:
            return max_value
        return value


def main(args=None) -> None:
    rclpy.init(args=args)

    follow_node = TcpFollowNode()

    # =========================================================
    # DSR 통신(ROS2 client/service) 전용 노드
    # =========================================================
    dsr_node = rclpy.create_node("dsr_internal_worker", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node

    # dr 초기화 1회 + follow 노드에 주입
    dr = initialize_robot(follow_node)
    follow_node.set_dr(dr)

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(follow_node)
    executor.add_node(dsr_node)

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
