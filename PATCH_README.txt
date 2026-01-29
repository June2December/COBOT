이 ZIP은 cobot2 패키지의 최소 수정 패치입니다.
- setup.py: console_scripts 정리(sign_check 제거), sign_motion/sign_topic만 남김
- sign_topic_node.py: 패키지 import(.STT, .sign_check)로 수정 + .env 로딩 안정화
- sign_motion_node.py: DSR_ROBOT2 지연 import(노드 생성 후 DR_init.__dsr__node 세팅)로 g_node None 오류 해결

적용:
1) ~/cobot_ws/src 에서 cobot2 폴더와 병합(덮어쓰기)하세요.
2) cd ~/cobot_ws && colcon build --symlink-install
3) source install/setup.bash
4) ros2 run cobot2 sign_motion
5) ros2 run cobot2 sign_topic
