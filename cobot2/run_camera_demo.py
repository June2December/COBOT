import os
import cv2
import time
import datetime
import rclpy
import threading
import numpy as np # [추가] 행렬 연산을 위해 필요
from ultralytics import YOLO

# ==============================================================================
# [설정] 사용자 정의 파라미터
# ==============================================================================
Day_CAM = 1      
Night_CAM = 3    

Day_Time = (7, 30, 0)    
Night_Time = (17, 44, 0) 

ROBOT_ID = "dsr01"
model_path = '/home/jaylee/cobot_ws/src/cobot_mine/people.v1i.coco/runs/segment/people_result5/weights/best.pt'

LOCK_CONF_HIGH = 0.93      
MAINTAIN_CONF_LOW = 0.4  
AIM_THRESHOLD = 20        
KP = 0.1                  
FIRE_COOLDOWN = 1.0       
PATROL_LIMIT = 90.0       
PATROL_SPEED_VAL = 5.0    

# ==============================================================================
# [클래스] 딜레이 방지용 카메라
# ==============================================================================
class NoDelayCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.width = 1280
        self.height = 720
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.running = False
        self.latest_frame = None
        
        ret, frame = self.cap.read()
        if ret:
            self.latest_frame = frame
        else:
            print(f"⚠️ V4L2 실패. 기본 모드로 재시도합니다... (CAM {src})")
            self.cap.release()
            self.cap = cv2.VideoCapture(src)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def start(self):
        if self.running: return self
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.latest_frame = frame

    def read(self):
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join()
        self.cap.release()

# ==============================================================================
# [함수] 헬퍼
# ==============================================================================
def check_is_daytime():
    now = datetime.datetime.now().time()
    start = datetime.time(*Day_Time)
    end = datetime.time(*Night_Time)
    if start < end:
        return start <= now < end
    else:
        return now >= start or now < end

def move_robot(vel_x, vel_y):
    if vel_x == 0 and vel_y == 0: return
    # print(f"🤖 로봇 이동 >> x:{vel_x:.2f}, y:{vel_y:.2f}")

def fire_gun():
    print("🔥🔥🔥 발사!!! 🔥🔥🔥")

# ==============================================================================
# [메인] 실행 로직
# ==============================================================================
def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("auto_aim_system", namespace=ROBOT_ID)
    
    print("YOLO Segmentation 모델 로드 중...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return

    is_day = check_is_daytime()
    current_cam_id = Day_CAM if is_day else Night_CAM
    
    print(f"🚀 시스템 시작 (CAM {current_cam_id}) - Mask Tracking Mode")
    cam = NoDelayCamera(current_cam_id).start()
    
    window_name = "Auto Aiming System (Mask Mode)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    LOCKED_TARGET_ID = None
    last_fire_time = 0
    patrol_angle = 0.0
    patrol_direction = 1.0
    last_time_check = time.time()
    prev_frame_time = 0 

    try:
        while True:
            if time.time() - last_time_check > 1.0:
                check_now_is_day = check_is_daytime()
                if check_now_is_day != is_day:
                    print(f"\n🔄 시간 변경 감지! 카메라 전환 중...")
                    is_day = check_now_is_day
                    target_cam_id = Day_CAM if is_day else Night_CAM
                    
                    cam.stop()
                    time.sleep(0.5)
                    cam = NoDelayCamera(target_cam_id).start()
                    
                    LOCKED_TARGET_ID = None 
                    print(f"✅ 카메라 {target_cam_id}번으로 전환 완료")
                last_time_check = time.time()

            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            h, w, _ = frame.shape
            center_x, center_y = w // 2, h // 2

            # 3. AI 추론 (retina_masks=True로 마스크 품질 향상)
            results = model.track(source=frame, 
                                  conf=MAINTAIN_CONF_LOW, 
                                  persist=True, 
                                  tracker="bytetrack.yaml", 
                                  verbose=False,
                                  retina_masks=True) 
            
            # 기본 박스 그리기 대신 마스크는 아래 로직에서 직접 그림
            annotated_frame = frame.copy() 
            
            target_mask_contour = None
            target_box_info = None

            # 4. 타겟 처리
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                confs = results[0].boxes.conf.cpu().numpy()
                
                # [중요] 마스크 데이터 확인
                masks = results[0].masks

                for i, box_id in enumerate(ids):
                    # 락온 로직
                    if LOCKED_TARGET_ID is None:
                        if confs[i] >= LOCK_CONF_HIGH:
                            LOCKED_TARGET_ID = box_id
                            break 
                    
                    # 락온된 타겟 찾기
                    if box_id == LOCKED_TARGET_ID:
                        target_box_info = boxes[i] # 백업용(마스크 없을 때 대비)
                        
                        # 마스크가 존재하면 윤곽선 가져오기
                        if masks is not None:
                            # masks.xy는 폴리곤 좌표 리스트입니다.
                            try:
                                target_mask_contour = masks.xy[i]
                            except:
                                target_mask_contour = None
                        break

            # 5. 행동 결정 (Mask 기반 조준)
            obj_x, obj_y = None, None

            if LOCKED_TARGET_ID is not None:
                # ---------------------------------------------------------
                # [A] 마스크(윤곽선)가 있는 경우 -> 무게 중심(Centroid) 사용
                # ---------------------------------------------------------
                if target_mask_contour is not None and len(target_mask_contour) > 0:
                    # 윤곽선 그리기 (녹색)
                    cnt = target_mask_contour.astype(np.int32)
                    cv2.polylines(annotated_frame, [cnt], True, (0, 255, 0), 2)
                    
                    # 모멘트(Moments) 계산하여 무게 중심 찾기
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00']) # 무게 중심 X
                        cy = int(M['m01'] / M['m00']) # 무게 중심 Y
                        
                        # 조준점 보정: 무게 중심은 '배꼽' 쯤이므로, 가슴쪽으로 약간 올림
                        # 윤곽선의 가장 높은 점(top_y)을 찾아서 비율로 조정
                        top_y = np.min(cnt[:, 1])
                        height_span = cy - top_y
                        
                        obj_x = cx
                        obj_y = int(cy - (height_span * 0.3)) # 중심에서 위로 30% 이동
                        
                        cv2.circle(annotated_frame, (cx, cy), 5, (255, 0, 0), -1) # 실제 무게중심(파란점)
                
                # ---------------------------------------------------------
                # [B] 마스크 실패 시 -> 기존 박스 방식 백업 사용
                # ---------------------------------------------------------
                elif target_box_info is not None:
                    box_cx, box_cy, _, box_h = target_box_info
                    obj_x = int(box_cx)
                    obj_y = int((box_cy - box_h / 2) + (box_h * 0.4))
                    cv2.putText(annotated_frame, "MASK FAILED - BOX MODE", (obj_x, obj_y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)


            else:
                # [타겟 없음 - 경계 모드]
                if LOCKED_TARGET_ID is not None:
                    move_robot(0, 0) # 타겟 놓쳤을 때 잠시 정지
                else:
                    patrol_angle += (PATROL_SPEED_VAL * 0.5) * patrol_direction 
                    if patrol_angle > PATROL_LIMIT: patrol_direction = -1.0
                    elif patrol_angle < -PATROL_LIMIT: patrol_direction = 1.0
                    move_robot(PATROL_SPEED_VAL * patrol_direction, 0)

            # FPS 및 UI 정보
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            mode_text = "DAY" if is_day else "NIGHT"
            cv2.putText(annotated_frame, f"{mode_text} FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.line(annotated_frame, (center_x - 20, center_y), (center_x + 20, center_y), (255, 255, 255), 1)
            cv2.line(annotated_frame, (center_x, center_y - 20), (center_x, center_y + 20), (255, 255, 255), 1)

            cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'): LOCKED_TARGET_ID = None

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == "__main__":
    main()