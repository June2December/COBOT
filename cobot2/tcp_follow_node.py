# tcp_follow_node v0.000 2026-01-28
# [이번 버전에서 수정된 사항]
# - (기능구현) YOLO/Tracker에서 계산된 정규화 화면 오차(error_x_norm, error_y_norm)를 받아 TCP를 REL movel로 추종하는 ROS2 노드 뼈대 추가
# - (기능구현) TOOL 기준(ref=DR_TOOL) 증분 포즈(Δx, Δy, Δyaw, Δpitch) 생성 + 데드존/클램프/주기제한/타겟 유실 타임아웃 포함
# - (유지) 실제 로봇 SDK 호출부는 RobotInterface로 분리하여 기존 방식(movel API) 그대로 연결 가능

"""
TCP Follow Node (eye-in-hand, pose-step servo)

- 입력: 추종 대상의 화면 중심 오차 (정규화) -> error_x_norm, error_y_norm
  * error_x_norm: +면 화면 오른쪽(타겟이 오른쪽)
  * error_y_norm: +면 화면 아래쪽(타겟이 아래)

- 출력(동작): TOOL 기준 REL movel로 TCP를 조금씩 움직여 타겟이 화면 중앙으로 오도록 유도
  * Δyaw/Δpitch(회전) 위주로 센터링을 먼저 안정화
  * 옵션으로 Δx/Δy(평면 이동)도 함께 사용 가능

- 안전장치:
  * deadzone: 작은 오차는 무시
  * clamp: 한 번에 움직이는 이동(mm)/회전(deg) 상한
  * rate limit: 제어 루프 주기 제한
  * lost timeout: 입력이 끊기면 추종 중단
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


# =========================
# 설정 데이터 구조
# =========================
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


# =========================
# 로봇 인터페이스 (기존 API 그대로 붙이기)
# =========================
class RobotInterface:
    """
    프로젝트에서 쓰는 기존 로봇 제어 모듈(DSR 등)의 movel 호출을 여기로 모아 연결.
    - movel([x,y,z,rx,ry,rz], vel=, acc=, mod=mv_mod_REL/ABS, ref=BASE/TOOL)
    """

    def __init__(self, node: Node, *, dry_run: bool):
        self._node = node
        self._dry_run = dry_run

        # TODO: 여기서 기존 방식대로 로봇 SDK import/초기화
        # 예) import DR_init, from DSR_ROBOT2 import movel, ...
        self._movel = None  # 실제 movel 함수 핸들

        if self._dry_run:
            self._node.get_logger().warn("[TCP_FOLLOW] dry_run=True (로봇 미구동, 로그만 출력)")

    def movel_rel_tool(
        self,
        delta_pose_mm_deg: Tuple[float, float, float, float, float, float],
        *,
        vel: float,
        acc: float,
    ) -> None:
        dx_mm, dy_mm, dz_mm, droll_deg, dpitch_deg, dyaw_deg = delta_pose_mm_deg

        if self._dry_run or self._movel is None:
            self._node.get_logger().info(
                "[TCP_FOLLOW] movel REL TOOL Δpose = "
                f"[{dx_mm:.2f}, {dy_mm:.2f}, {dz_mm:.2f}, {droll_deg:.2f}, {dpitch_deg:.2f}, {dyaw_deg:.2f}] "
                f"(vel={vel:.1f}, acc={acc:.1f})"
            )
            return

        # TODO: 기존 사용 방식 그대로
        # self._movel([dx_mm, dy_mm, dz_mm, droll_deg, dpitch_deg, dyaw_deg],
        #            vel=vel, acc=acc, mod=mv_mod_REL, ref=DR_TOOL)
        raise NotImplementedError("여기에 프로젝트 로봇 movel 호출을 연결하세요.")


# =========================
# 메인 노드
# =========================
class TcpFollowNode(Node):
    def __init__(self) -> None:
        super().__init__("tcp_follow_node")

        # ---- Parameters
        self.declare_parameter("dry_run", True)

        self.declare_parameter("control_loop_hz", 15.0)
        self.declare_parameter("target_lost_timeout_sec", 0.5)

        self.declare_parameter("vel", 60.0)
        self.declare_parameter("acc", 60.0)

        # Gains (기본은 회전 위주: 센터링부터)
        self.declare_parameter("yaw_deg_per_error", 6.0)
        self.declare_parameter("pitch_deg_per_error", 6.0)
        self.declare_parameter("x_mm_per_error", 0.0)
        self.declare_parameter("y_mm_per_error", 0.0)

        # Limits
        self.declare_parameter("max_delta_translation_mm", 3.0)
        self.declare_parameter("max_delta_rotation_deg", 2.0)
        self.declare_parameter("deadzone_error_norm", 0.03)

        # Input topic: Float32MultiArray [error_x_norm, error_y_norm]
        self.declare_parameter("error_topic", "/follow/error_norm")

        self._dry_run: bool = bool(self.get_parameter("dry_run").value)

        self._control_loop_hz: float = float(self.get_parameter("control_loop_hz").value)
        self._target_lost_timeout_sec: float = float(self.get_parameter("target_lost_timeout_sec").value)

        self._vel: float = float(self.get_parameter("vel").value)
        self._acc: float = float(self.get_parameter("acc").value)

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

        # Robot interface
        self._robot = RobotInterface(self, dry_run=self._dry_run)

        # Subscriptions
        self.create_subscription(Float32MultiArray, self._error_topic, self._on_error_norm, 10)

        # Control loop timer
        timer_period_sec = 1.0 / max(self._control_loop_hz, 1.0)
        self.create_timer(timer_period_sec, self._on_control_timer)

        self.get_logger().info(
            "[TCP_FOLLOW] ready "
            f"(hz={self._control_loop_hz:.1f}, lost_timeout={self._target_lost_timeout_sec:.2f}s, "
            f"vel={self._vel:.1f}, acc={self._acc:.1f}, topic={self._error_topic})"
        )

    def _on_error_norm(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 2:
            self.get_logger().warn("[TCP_FOLLOW] error_norm msg.data must be [error_x_norm, error_y_norm]")
            return

        error_x_norm = float(msg.data[0])
        error_y_norm = float(msg.data[1])

        self._latest_error_norm = (error_x_norm, error_y_norm)
        self._latest_error_time_sec = time.time()

    def _on_control_timer(self) -> None:
        # 타겟 유실
        if self._latest_error_norm is None:
            return

        now_sec = time.time()
        if (now_sec - self._latest_error_time_sec) > self._target_lost_timeout_sec:
            # 유실 시 추종 중단(정지 개념). REL movel을 안 보내는 게 안전.
            self._latest_error_norm = None
            self.get_logger().warn("[TCP_FOLLOW] target lost -> stop following")
            return

        error_x_norm, error_y_norm = self._latest_error_norm

        # deadzone
        if abs(error_x_norm) < self._limits.deadzone_error_norm:
            error_x_norm = 0.0
        if abs(error_y_norm) < self._limits.deadzone_error_norm:
            error_y_norm = 0.0

        if error_x_norm == 0.0 and error_y_norm == 0.0:
            return

        # ---- Mapping (TOOL 기준 REL)
        # 화면 오른쪽(+ex) -> yaw를 (-)로 돌려 타겟을 중앙으로 끌어오기 (부호는 장착 방향에 따라 바뀔 수 있음)
        delta_yaw_deg = -self._gains.yaw_deg_per_error * error_x_norm
        delta_pitch_deg = -self._gains.pitch_deg_per_error * error_y_norm

        # 옵션: 평면 이동도 쓰고 싶으면 gains를 0보다 크게
        delta_x_mm = self._gains.x_mm_per_error * error_x_norm
        delta_y_mm = self._gains.y_mm_per_error * error_y_norm

        # clamp
        delta_x_mm = self._clamp(delta_x_mm, -self._limits.max_delta_translation_mm, self._limits.max_delta_translation_mm)
        delta_y_mm = self._clamp(delta_y_mm, -self._limits.max_delta_translation_mm, self._limits.max_delta_translation_mm)

        delta_yaw_deg = self._clamp(delta_yaw_deg, -self._limits.max_delta_rotation_deg, self._limits.max_delta_rotation_deg)
        delta_pitch_deg = self._clamp(delta_pitch_deg, -self._limits.max_delta_rotation_deg, self._limits.max_delta_rotation_deg)

        # dz, roll은 기본 0 (필요 시 확장)
        delta_z_mm = 0.0
        delta_roll_deg = 0.0

        self._robot.movel_rel_tool(
            (delta_x_mm, delta_y_mm, delta_z_mm, delta_roll_deg, delta_pitch_deg, delta_yaw_deg),
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


def main() -> None:
    rclpy.init()
    node = TcpFollowNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
