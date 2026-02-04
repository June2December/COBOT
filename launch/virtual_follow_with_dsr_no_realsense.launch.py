# virtual_follow_with_dsr_no_realsense.launch v0.610 2026-02-04
# [이번 버전에서 수정된 사항]
# - (기능구현) follow_ui_node 추가: 왼쪽 annotated 영상(/follow/annotated_image), 오른쪽 이벤트 로그(/follow/ui_event)
# - (기능구현) UI raw fallback 토픽을 launch 인자 image_topic으로 연동
# - (유지) RealSense bringup 제외 / dsr_bringup2 후 tcp_follow_node -> yolo_camera_node 순서 실행 유지

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

    # RealSense 토픽만 넘겨받아 YOLO/UI가 구독 (RealSense는 외부에서 이미 실행 중이어야 함)
    image_topic_arg = DeclareLaunchArgument("image_topic", default_value="/camera/camera/color/image_raw")

    # (A) Doosan bringup + RViz
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
                # "control_loop_hz": 15.0,
                "target_lost_timeout_sec": 0.5,

                # "vel": 180.0,
                # "acc": 20.0,

                "enable_translation": True,
                "enable_rotation": False,

                # "max_delta_translation_mm": 5.0,
                # "max_delta_rotation_deg": 1.0,
                # "deadzone_error_norm": 0.03,
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
                "image_topic": LaunchConfiguration("image_topic"),
                "target_class_name": "person",
                "min_confidence": 0.6,
                "publish_topic": "/follow/error_norm",
                "show_debug": False,
            }
        ],
    )

    # (D) UI 노드: annotated 영상 + 로그 이벤트 표시
    follow_ui = Node(
        package="cobot2",
        executable="follow_ui_node",
        name="follow_ui_node",
        output="screen",
        parameters=[
            {
                "annotated_image_topic": "/follow/annotated_image",
                "ui_event_topic": "/follow/ui_event",
                # annotated 끊기면 raw로 fallback (Day/Night 토픽은 launch 인자만 바꿔주면 됨)
                "raw_image_topic": LaunchConfiguration("image_topic"),
            }
        ],
    )

    # bringup -> tcp_follow -> yolo -> ui
    delayed_chain = TimerAction(
        period=LaunchConfiguration("bringup_delay_sec"),
        actions=[
            tcp_follow,
            TimerAction(
                period=LaunchConfiguration("follow_to_yolo_delay_sec"),
                actions=[
                    yolo_camera,
                    # UI는 yolo와 동시에 떠도 되지만, 토픽 생성 타이밍 이슈 줄이려면 살짝 딜레이 주는 게 깔끔함
                    TimerAction(period=0.5, actions=[follow_ui]),
                ],
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
