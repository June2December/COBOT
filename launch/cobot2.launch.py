from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    REAL_SWITCH = True

    REAL = {"mode": "real", "host": "192.168.1.100", "port": "12345", "model": "m0609"}
    VIRTUAL = {"mode": "virtual", "host": "127.0.0.1", "port": "12345", "model": "m0609"}
    launch_args = (REAL if REAL_SWITCH else VIRTUAL)

    dsr_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("dsr_bringup2"),
                "launch",
                "dsr_bringup2_rviz.launch.py",
            )
        ),
        launch_arguments=launch_args.items(),
    )

    # --- Perception (YOLO) ---
    yolo_camera_node = Node(
        package="cobot2",
        executable="yolo_camera",
        name="yolo_camera_node",
        output="screen",
        parameters=[{
            "image_topic": "/camera/camera/color/image_raw",
            "show_debug": False,

            # UI outputs
            "publish_annotated": True,
            "annotated_topic": "/follow/annotated_image",
            "ui_event_topic": "/follow/ui_event",

            # lock_done topic
            "lock_done_topic": "/follow/lock_done",
        }],
    )

    # --- DB Logger (SQLite + snapshots) ---
    follow_logger_node = Node(
        package="cobot2",
        executable="follow_logger_node",
        name="follow_logger_node",
        output="screen",
        parameters=[{
            "ui_event_topic": "/follow/ui_event",
            "annotated_image_topic": "/follow/annotated_image",
            "log_root_dir": os.path.join(os.path.expanduser("~"), "logs", "follow_runs"),
            "snapshot_enable": True,
        }],
    )

    # --- UI (Live + History) ---
    follow_ui_node = Node(
        package="cobot2",
        executable="follow_ui_node",
        name="follow_ui_node",
        output="screen",
        parameters=[{
            "annotated_image_topic": "/follow/annotated_image",
            "ui_event_topic": "/follow/ui_event",
        }],
    )

    # --- Authentication Action Server ---
    auth_action_server = Node(
        package="cobot2",
        executable="auth_action",
        name="auth_action_server",
        output="screen",
    )

    # --- Motion nodes ---
    salute_node = Node(
        package="cobot2",
        executable="salute",
        name="salute_node",
        output="screen",
    )

    shoot_node = Node(
        package="cobot2",
        executable="shoot",
        name="shoot_node",
        output="screen",
    )

    # --- Follow control ---
    tcp_follow_node = Node(
        package="cobot2",
        executable="tcp_follow",
        name="tcp_follow_node",
        output="screen",
        parameters=[{
            "error_topic": "/follow/error_norm",
            "enable_topic": "/follow/enable",
        }],
    )

    # --- Orchestrator ---
    orchestrator_node = Node(
        package="cobot2",
        executable="orchestrator",
        name="orchestrator_node",
        output="screen",
    )

    return LaunchDescription([
        dsr_launch,

        TimerAction(period=10.0, actions=[yolo_camera_node]),
        # yolo 이후에 logger/ui 붙이기
        TimerAction(period=12.0, actions=[follow_logger_node, follow_ui_node]),

        TimerAction(period=13.0, actions=[auth_action_server, salute_node, shoot_node]),
        TimerAction(period=16.0, actions=[tcp_follow_node]),
        TimerAction(period=20.0, actions=[orchestrator_node]),
    ])
