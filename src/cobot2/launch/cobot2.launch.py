from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("realsense2_camera"),
                "launch",
                "rs_launch.py",
            )
        ),
        launch_arguments={
            "depth_module.depth_profile": "640x480x30",
            "rgb_camera.color_profile": "640x480x30",
            "initial_reset": "true",
            "align_depth.enable": "true",
            "enable_rgbd": "true",
            "pointcloud.enable": "true",
        }.items(),
    )
    # --- Doosan (alias: real) ---
    dsr_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("dsr_bringup2"),
                "launch",
                "dsr_bringup2_rviz.launch.py",
            )
        ),
        launch_arguments={
            "mode": "real",
            "host": "192.168.1.100",
            "port": "12345",
            "model": "m0609",
        }.items(),
    )

    return LaunchDescription(
        [
            # --- Hardware bringup ---
            realsense_launch,
            dsr_launch,

            # --- Perception ---
            Node(
                package="cobot2",
                executable="yolo_camera",
                name="yolo_camera_node",
                output="screen",
                # parameters=[{}],  # 필요 시 추가
            ),

            # --- Follow control ---
            Node(
                package="cobot2",
                executable="tcp_follow",
                name="tcp_follow_node",
                output="screen",
                parameters=[
                    {
                        "error_topic": "/follow/error_norm",
                        "enable_topic": "/follow/enable",
                        "follow_enable_default": True,
                    }
                ],
            ),

            # --- Orchestrator ---
            Node(
                package="cobot2",
                executable="orchestrator",
                name="orchestrator_node",
                output="screen",
            ),

            # --- Authentication Action Server ---
            Node(
                package="cobot2",
                executable="auth_action",
                name="auth_action_server",
                output="screen",
            ),

            # --- Motion nodes ---
            Node(
                package="cobot2",
                executable="salute",
                name="salute_node",
                output="screen",
            ),
            Node(
                package="cobot2",
                executable="shoot",
                name="shoot_node",
                output="screen",
            ),
        ]
    )
