import os
import cv2
import time
import rclpy
from ultralytics import YOLO

# ==============================================================================
# [설정] 사용자 정의 파라미터
# ==============================================================================
# 1. 카메라 설정 (여기를 변경하세요!)
DEVICE_NUMBER = 1  # 0, 1, 2, 4 등 리얼센스가 연결된 번호 입력

# 2. 로봇 설정
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

# 3. 모델 경로
model_path = '/home/jaylee/cobot_ws/src/cobot_mine/people.v1i.coco/runs/segment/people_result5/weights/best.pt'

# 4. 락온(Lock-on) 및 조준 설정
LOCK_CONF_HIGH = 0.9      # 락온 시작 정확도
MAINTAIN_CONF_LOW = 0.25  # 락온 유지 정확도
AIM_THRESHOLD = 30        # 조준 허용 오차 (픽셀)
KP = 0.1                  # 모터 제어 게인
FIRE_COOLDOWN = 1.0       # 발사 쿨타임

# 5. 경계 모드(Patrol) 설정
PATROL_LIMIT = 90.0       
PATROL_SPEED_VAL = 5.0    

# ==============================================================================
# [함수] 로봇 제어 (시뮬레이션)
# ==============================================================================
def move_robot(vel_x, vel_y):
    """로봇 회전 명령 (나중에 실제 두산 로봇 함수로 교체)"""
    if vel_x == 0 and vel_y == 0: return
    print(f"🤖 로봇 이동 >> x:{vel_x:.2f}, y:{vel_y:.2f}")

def fire_gun():
    """총 발사 명령"""
    print("🔥🔥🔥 [탕!] 발사!!! 🔥🔥🔥")

# ==============================================================================
# [메인] 실행 로직
# ==============================================================================
def main(args=None):
    # 1. ROS2 노드 초기화 (로봇 제어를 위해 유지)
    rclpy.init(args=args)
    node = rclpy.create_node("auto_aim_system", namespace=ROBOT_ID)
    
    # 2. 모델 로드
    print("⏳ YOLO 모델을 불러오는 중입니다...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {model_path}")
        print(f"에러 내용: {e}")
        return

    # 3. 카메라 연결 (요청하신 부분)
    print(f"📷 현재 선택된 device number는 {DEVICE_NUMBER}입니다.")
    cap = cv2.VideoCapture(DEVICE_NUMBER)
    
    # 해상도 설정 (D435i 안정성 확보)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"❌ 카메라 {DEVICE_NUMBER}번을 열 수 없습니다. 번호를 확인해주세요.")
        return

    print("✅ 시스템 시작! (종료: q, 리셋: r)")

    # 상태 변수 초기화
    LOCKED_TARGET_ID = None
    last_fire_time = 0
    patrol_angle = 0.0
    patrol_direction = 1.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 비디오 프레임을 읽을 수 없습니다.")
                break

            h, w, _ = frame.shape
            center_x, center_y = w // 2, h // 2

            # -----------------------------------------------------------------
            # AI 추론
            # -----------------------------------------------------------------
            results = model.track(source=frame, conf=MAINTAIN_CONF_LOW, persist=True, tracker="bytetrack.yaml", verbose=False)
            annotated_frame = results[0].plot(boxes=False, labels=False)
            
            current_target_box = None 

            # -----------------------------------------------------------------
            # 타겟 선별 (락온)
            # -----------------------------------------------------------------
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                confs = results[0].boxes.conf.cpu().numpy()

                for i, box_id in enumerate(ids):
                    # 새 타겟 찾기
                    if LOCKED_TARGET_ID is None:
                        if confs[i] >= LOCK_CONF_HIGH:
                            LOCKED_TARGET_ID = box_id
                            current_target_box = boxes[i]
                            print(f"🎯 타겟 락온! (ID: {box_id})")
                            break 
                    # 기존 타겟 추적
                    else:
                        if box_id == LOCKED_TARGET_ID:
                            current_target_box = boxes[i]
                            break

            # -----------------------------------------------------------------
            # 행동 결정 (추적 vs 경계)
            # -----------------------------------------------------------------
            if current_target_box is not None:
                # [추적 모드]
                box_cx, box_cy, _, box_h = current_target_box
                obj_x = int(box_cx)
                obj_y = int((box_cy - box_h / 2) + (box_h * 0.4)) # 가슴 상단 조준
                
                error_x = obj_x - center_x
                error_y = obj_y - center_y

                # 시각화
                cv2.line(annotated_frame, (center_x, center_y), (obj_x, obj_y), (0, 255, 255), 2)
                cv2.putText(annotated_frame, f"LOCKED (ID:{LOCKED_TARGET_ID})", (obj_x, obj_y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 발사 로직
                if abs(error_x) < AIM_THRESHOLD and abs(error_y) < AIM_THRESHOLD:
                    if time.time() - last_fire_time > FIRE_COOLDOWN:
                        cv2.circle(annotated_frame, (center_x, center_y), AIM_THRESHOLD, (0, 0, 255), 3)
                        fire_gun()
                        last_fire_time = time.time()
                    else:
                        cv2.circle(annotated_frame, (center_x, center_y), AIM_THRESHOLD, (0, 255, 0), 2)
                        wait_time = FIRE_COOLDOWN - (time.time() - last_fire_time)
                        cv2.putText(annotated_frame, f"RELOAD.. {wait_time:.1f}s", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    move_robot(0, 0)
                else:
                    cv2.circle(annotated_frame, (center_x, center_y), AIM_THRESHOLD, (0, 0, 255), 2)
                    move_robot(error_x * KP, error_y * KP)

            else:
                # [경계 모드]
                if LOCKED_TARGET_ID is not None:
                    cv2.putText(annotated_frame, "LOST TARGET... SEARCHING", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    move_robot(0, 0)
                else:
                    patrol_angle += (PATROL_SPEED_VAL * 0.5) * patrol_direction 
                    if patrol_angle > PATROL_LIMIT: patrol_direction = -1.0
                    elif patrol_angle < -PATROL_LIMIT: patrol_direction = 1.0
                    
                    move_robot(PATROL_SPEED_VAL * patrol_direction, 0)

                    cv2.putText(annotated_frame, "MODE: PATROL SCAN", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                    # 경계 바 그리기
                    bar_len = w // 3
                    bar_pos = int(center_x + (patrol_angle / PATROL_LIMIT) * (bar_len))
                    cv2.rectangle(annotated_frame, (center_x - bar_len, h-50), (center_x + bar_len, h-30), (100, 100, 100), 2)
                    cv2.circle(annotated_frame, (bar_pos, h-40), 10, (0, 255, 255), -1)

            # UI 십자가
            cv2.line(annotated_frame, (center_x - 20, center_y), (center_x + 20, center_y), (255, 255, 255), 1)
            cv2.line(annotated_frame, (center_x, center_y - 20), (center_x, center_y + 20), (255, 255, 255), 1)

            cv2.imshow("Auto Aiming System", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):       
                break
            elif key == ord('r'):     
                print("🔄 타겟 리셋")
                LOCKED_TARGET_ID = None

    finally:
        # 종료 시 정리
        cap.release()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == "__main__":
    main()