# virtual_webcam_follow.launch v0.100 2026-01-28
# [이번 버전에서 수정된 사항]
# - (기능구현) 웹캠 YOLO 노드 + tcp_follow_node(dry_run) 동시 실행 런치 추가

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
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

    tcp_follow = Node(
        package="cobot2",
        executable="tcp_follow_node",
        name="tcp_follow_node",
        output="screen",
        parameters=[
            {
                "dry_run": True,  # ★ 가상 확인: movel 로그만 출력
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

    return LaunchDescription([yolo_webcam, tcp_follow])
