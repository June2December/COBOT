# virtual_follow_with_dsr_no_realsense.launch v0.510 2026-01-29
# [이번 버전에서 수정된 사항]
# - (기능구현) RealSense bringup 제외 (센서 드라이버는 별도 터미널에서 구동)
# - (기능구현) dsr_bringup2 브링업 후 tcp_follow_node(초기 movej) -> yolo_camera_node 순서 실행

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    dsr_bringup2_dir = get_package_share_directory("dsr_bringup2")

    # mode_arg = DeclareLaunchArgument("mode", default_value="virtual")
    # host_arg = DeclareLaunchArgument("host", default_value="127.0.0.1")

    mode_arg = DeclareLaunchArgument("mode", default_value="real")
    host_arg = DeclareLaunchArgument("host", default_value="192.168.1.100")

    bringup_delay_arg = DeclareLaunchArgument("bringup_delay_sec", default_value="10.0")
    follow_to_yolo_delay_arg = DeclareLaunchArgument("follow_to_yolo_delay_sec", default_value="8.0")

    # RealSense 토픽만 넘겨받아 YOLO가 구독 (RealSense는 외부에서 이미 실행 중이어야 함)
    image_topic_arg = DeclareLaunchArgument("image_topic", default_value="/camera/camera/color/image_raw")

    # (A) Doosan emulator + RViz
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

    # (B) tcp_follow_node 먼저 실행(초기 movej 수행)
    tcp_follow = Node(
        package="cobot2",
        executable="tcp_follow_node",
        name="tcp_follow_node",
        output="screen",
        parameters=[
            {
                "dry_run": False,

                "startup_movej_enable": True,
                "startup_movej_vel": 50.0,
                "startup_movej_acc": 20.0,
                "startup_settle_sec": 7.0,

                "error_topic": "/follow/error_norm",
                "control_loop_hz": 15.0,
                "target_lost_timeout_sec": 0.5,

                "vel": 180.0,
                "acc": 20.0,

                "enable_translation": True,
                "enable_rotation": False,

                # "x_mm_per_error": 6.0,
                # "y_mm_per_error": 6.0,
                # "yaw_deg_per_error": 6.0,
                # "pitch_deg_per_error": 6.0,

                "max_delta_translation_mm": 5.0,
                "max_delta_rotation_deg": 1.0,
                "deadzone_error_norm": 0.03,

                # "x_sign": 1.0,
                # "y_sign": -1.0,
                # "yaw_sign": 1.0,
                # "pitch_sign": 1.0,
            }
        ],
    )

    # (C) yolo_camera_node 나중 실행(추종 시작)
    yolo_camera = Node(
        package="cobot2",
        executable="yolo_camera_node",
        name="yolo_camera_node",
        output="screen",
        parameters=[
            {
                # "image_topic": LaunchConfiguration("image_topic"),
                "target_class_name": "person",
                "min_confidence": 0.6,
                "publish_topic": "/follow/error_norm",
                "show_debug": True,
            }
        ],
    )

    delayed_chain = TimerAction(
        period=LaunchConfiguration("bringup_delay_sec"),
        actions=[
            tcp_follow,
            TimerAction(
                period=LaunchConfiguration("follow_to_yolo_delay_sec"),
                actions=[yolo_camera],
            ),
        ],
    )

    return LaunchDescription(
        [
            mode_arg,
            host_arg,
            bringup_delay_arg,
            follow_to_yolo_delay_arg,
            image_topic_arg,
            dsr_simulator,
            delayed_chain,
        ]
    )
