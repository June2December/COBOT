from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        # yolo camera node
        Node(
            package="cobot2",
            executable="yolo_camera",
            name="yolo_camera_node",
            output="screen",
            parameters=[
                # 나중에 추가? Ex) conf 값이나 그 yolo 실행 시 필요한 토픽 이름?
            ]
        ),

        # tcp_follow
        Node(
            package="cobot2",
            executable="tcp_follow",
            name="tcp_follow_node",
            output="screen",
            parameters=[{
                "error_topic": "/follow/error_norm",
                "enable_topic": "/follow/enable",
                "follow_enable_default": True,
            }],
        ),


        # Orchestrator
        Node(
            package="cobot2",
            executable="orchestrator",
            name="orchestrator",
            output="screen",
        ),

        #  Authentication
        Node(
            package="cobot2",
            executable="auth_action",
            name="auth_action_server",
            output="screen",
        ),
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

    ])
