# virtual_webcam_follow_with_dsr.launch v0.200 2026-01-28
# [이번 버전에서 수정된 사항]
# - (기능구현) dsr_bringup2 에뮬레이터/RViz 브링업(dsr_bringup2_rviz.launch.py) 포함
# - (기능구현) 웹캠 YOLO 노드 + tcp_follow_node를 동일 런치에서 동시 실행
# - (유지) mode/host를 LaunchArgument로 노출하여 virtual/real 전환 및 대상 호스트 변경 가능

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    dsr_bringup2_dir = get_package_share_directory("dsr_bringup2")

    mode_arg = DeclareLaunchArgument("mode", default_value="virtual")
    host_arg = DeclareLaunchArgument("host", default_value="127.0.0.1")

    # (A) 두산 로봇 에뮬레이터/RViz 실행
    dsr_simulator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(dsr_bringup2_dir, "launch", "dsr_bringup2_rviz.launch.py")
        ),
        launch_arguments={
            "mode": LaunchConfiguration("mode"),
            "host": LaunchConfiguration("host"),
            "model": "m0609",
            "port": "12345",
        }.items(),
    )

    # (B) Webcam YOLO -> /follow/error_norm
    yolo_webcam = Node(
        package="cobot2",
        executable="yolo_webcam_node",
        name="yolo_webcam_node",
        output="screen",
        parameters=[
            {
                "camera_index": 0,
                "frame_width": 640,
                "frame_height": 480,
                "model": "yolo11n.pt",
                "imgsz": 640,
                "target_class_name": "person",
                "min_confidence": 0.6,
                "publish_topic": "/follow/error_norm",
                "control_hz": 15.0,
                "show_debug": True,
                "print_error": False,
            }
        ],
    )

    # (C) /follow/error_norm -> TCP REL movel (두산 에뮬레이터로 연결할 계획이면 dry_run=False로 전환)
    tcp_follow = Node(
        package="cobot2",
        executable="tcp_follow_node",
        name="tcp_follow_node",
        output="screen",
        parameters=[
            {
                # 에뮬레이터로 실제 movel을 보내려면 False로 전환 + RobotInterface 연결 필요
                "dry_run": False,
                "robot_id": "dsr01",
                "robot_model": "m0609",
                "startup_movej_enable": True,
                "startup_movej_joints_deg": [0.0, 0.0, 90.0, 0.0, 90.0, 0.0],
                "startup_movej_vel": 60.0,
                "startup_movej_acc": 60.0,
                "error_topic": "/follow/error_norm",
                "control_loop_hz": 15.0,
                "target_lost_timeout_sec": 0.5,
                "vel": 60.0,
                "acc": 60.0,
                "yaw_deg_per_error": 6.0,
                "pitch_deg_per_error": 6.0,
                "x_mm_per_error": 0.0,
                "y_mm_per_error": 0.0,
                "max_delta_translation_mm": 3.0,
                "max_delta_rotation_deg": 2.0,
                "deadzone_error_norm": 0.03,
                "yaw_sign": -1.0,
                "pitch_sign": -1.0,
            }
        ],
    )
    delayed_nodes = TimerAction(period=10.0, actions=[yolo_webcam, tcp_follow])

    return LaunchDescription([mode_arg, host_arg, dsr_simulator, delayed_nodes])

