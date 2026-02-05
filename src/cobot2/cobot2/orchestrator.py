# 날짜별 문어/답어 딕셔너리(WEEKLY_PHRASES) 자동 선택,
# lock_done IDLE gate, follow 타이밍( AUTH 중 ON / salute·shoot 직전 OFF / done 후 ON ) 유지

# use for annotations(hint) for debug error
from __future__ import annotations

import datetime
import time
from zoneinfo import ZoneInfo

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from std_msgs.msg import String, Bool

# usrr defined interface / Authentication
from cobot2_interfaces.action import Auth

WEEKLY_PHRASES: dict[str, tuple[str, str]] = {
    "2026-02-05": ("iphone", "galaxy"),
    "2026-02-06": ("빨강", "파랑"),
    "2026-02-07": ("고양이", "강아지"),
    "2026-02-08": ("아메리카노", "라떼"),
    "2026-02-09": ("사과", "바나나"),
    "2026-02-10": ("해", "달"),
    "2026-02-11": ("봄", "가을"),
}

class OrchestratorNode(Node):
    def __init__(self):
        super().__init__("orchestrator_node")
        self._cbg = ReentrantCallbackGroup()
        
        # start, status : for test
        self.declare_parameter("start_topic", "/orchestrator/start") 
        self.declare_parameter("status_topic", "/orchestrator/status") # orche 상태 출력용
        # for motion nodes (topic trigger)
        self.declare_parameter("salute_trigger_topic", "/salute_trigger")
        self.declare_parameter("salute_done_topic", "/salute_done")
        self.declare_parameter("shoot_trigger_topic", "/shoot_trigger")
        self.declare_parameter("shoot_done_topic", "/shoot_done")
        self.declare_parameter("follow_enable_topic", "/follow/enable")

        # (추가1) salute_node의 stt sub/pub param ============================================
        # 암구호 일치
        self.declare_parameter("salute_accord_topic", "/salute_accord_topic")
        self.declare_parameter("salute_accord_out_topic", "/orchestrator/salute_accord_topic")
        # 들은 정보 ex) '중위 김준혁'
        self.declare_parameter("salute_heard_topic", "/salute_heard_text")
        self.declare_parameter("salute_heard_out_topic", "/orchestrator/salute_heard_text")
        # ==================================================================================

        # for node
        self.declare_parameter("auth_action_name", "/auth_action")
        self.declare_parameter("auth_attempts", 3)

        # trigger 기반 시작
        self.declare_parameter("trigger_topic", "/orchestrator/trigger")

        # lock_done 기반 자동 시작
        self.declare_parameter("lock_done_topic", "/follow/lock_done")
        self.declare_parameter("lock_done_debounce_sec", 2.0)

        # 암구호 default
        self.declare_parameter("challenge_text", "아이폰")
        self.declare_parameter("expected_text", "갤럭시")

        # DB/로그용 단계 이벤트 토픽 (로거가 타임스탬프를 찍음)
        self.declare_parameter("ui_event_topic", "/follow/ui_event")
        #########################################################
        self._start_topic = self.get_parameter("start_topic").value
        self._status_topic = self.get_parameter("status_topic").value
        self._auth_action_name = self.get_parameter("auth_action_name").value
        self._auth_attempts = int(self.get_parameter("auth_attempts").value)

        self._trigger_topic = self.get_parameter("trigger_topic").value
        # self._challenge = self.get_parameter("challenge_text").value
        # self._expected = self.get_parameter("expected_text").value

        # lock_done 으로 암구호 시퀀스로 넘어가는 단계
        self._lock_done_topic = self.get_parameter("lock_done_topic").value
        self._lock_done_debounce_sec = float(self.get_parameter("lock_done_debounce_sec").value)
        self._last_lock_done_start_t: float = 0.0

        self._salute_trigger_topic = self.get_parameter("salute_trigger_topic").value
        self._salute_done_topic = self.get_parameter("salute_done_topic").value
        self._shoot_trigger_topic = self.get_parameter("shoot_trigger_topic").value
        self._shoot_done_topic = self.get_parameter("shoot_done_topic").value
        self._follow_enable_topic = self.get_parameter("follow_enable_topic").value
        
        # (추가2) salute_node stt sub/pub self =============================================
        # 암구호 일치
        self._salute_accord_topic = self.get_parameter("salute_accord_topic").value
        self._salute_accord_out_topic = self.get_parameter("salute_accord_out_topic").value
        self._last_salute_accord_text = ""
        # 들은 정보
        self._salute_heard_topic = self.get_parameter("salute_heard_topic").value
        self._salute_heard_out_topic = self.get_parameter("salute_heard_out_topic").value
        self._last_salute_heard_text = ""
        # =================================================================================
        self._challenge_default = str(self.get_parameter("challenge_text").value)
        self._expected_default = str(self.get_parameter("expected_text").value)

        # 런타임 적용값(시작 시 딕셔너리로 갱신)
        self._challenge = self._challenge_default
        self._expected = self._expected_default

        # state machine
        self._state = "IDLE"  # IDLE | AUTH | SALUTE_WAIT | SHOOT_WAIT
        self._busy = False
        self._attempts = 0

        self._pub_status = self.create_publisher(String, self._status_topic, 10) # orche 상태 출력용
        self._pub_ui_event = self.create_publisher(String, self._ui_event_topic, 10)
        self._sub_start = self.create_subscription(String,
                                                   self._start_topic,
                                                   self._on_start,
                                                   10,
                                                   callback_group=self._cbg)
        
        self._sub_trigger = self.create_subscription(Bool,
                                                self._trigger_topic,
                                                self._on_trigger,
                                                10,
                                                callback_group=self._cbg)
        # lock_done latched-like QoS (publisher와 동일하게)
        self._qos_lock_done = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        # lock_done 자동 트리거 구독
        self._sub_lock_done = self.create_subscription(
            Bool, self._lock_done_topic, self._on_lock_done, self._qos_lock_done, callback_group=self._cbg)
        
        self._pub_follow_enable = self.create_publisher(Bool, self._follow_enable_topic, 10)        
        # salute/shoot trigger publishers
        self._pub_salute_trigger = self.create_publisher(Bool, self._salute_trigger_topic, 10)
        self._pub_shoot_trigger = self.create_publisher(Bool, self._shoot_trigger_topic, 10)

        # (추가3) salute_node stt republish ============================================================
        # 암구호 일치
        self._pub_salute_accord_text = self.create_publisher(String, self._salute_accord_out_topic, 10)
        # 들은 정보
        self._pub_salute_heard_text = self.create_publisher(String, self._salute_heard_out_topic, 10)
        # =============================================================================================
        
        # salute/shoot done subscribers
        self._sub_salute_done = self.create_subscription(
            Bool, self._salute_done_topic, self._on_salute_done, 10, callback_group=self._cbg
        )
        self._sub_shoot_done = self.create_subscription(
            Bool, self._shoot_done_topic, self._on_shoot_done, 10, callback_group=self._cbg
        )

        # (추가4) salute_node stt sub create ==============================================================
        # 암구호 일치
        self._sub_salute_accord_text = self.create_subscription(
            String, self._salute_accord_topic, self._on_salute_accord_text, 10, callback_group=self._cbg
        )
        # 들은 정보
        self._sub_salute_heard_text = self.create_subscription(
            String, self._salute_heard_topic, self._on_salute_heard_text, 10, callback_group=self._cbg
        )
        # ================================================================================================
        self._auth = ActionClient(self, Auth, self._auth_action_name, callback_group=self._cbg)
        self._set_status("Ready...")
        self._emit_event("System initiate")

    def _emit_event(self, text: str) -> None:
        try:
            self._pub_ui_event.publish(String(data=str(text)))
        except Exception:
            return
    # -------------------------
    # Phrase selection (dict)
    # -------------------------
    @staticmethod
    def _today_key() -> str:
        now = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
        return now.strftime("%Y-%m-%d")

    def _refresh_phrase_for_today(self, *, force_log: bool = False) -> None:
        key = self._today_key()

        if key in WEEKLY_PHRASES:
            ch, ex = WEEKLY_PHRASES[key]
            ch = str(ch).strip()
            ex = str(ex).strip()
            if ch and ex:
                changed = (ch != self._challenge) or (ex != self._expected)
                self._challenge, self._expected = ch, ex
                if force_log or changed:
                    self.get_logger().info(
                        f"[PHRASE] date={key} challenge='{self._challenge}' expected='{self._expected}'"
                    )
                    self._emit_event(f"오늘의 문어: {self._challenge} / 답어: {self._expected}")
                return

        # fallback
        changed = (self._challenge != self._challenge_default) or (self._expected != self._expected_default)
        self._challenge, self._expected = self._challenge_default, self._expected_default
        if force_log or changed:
            self.get_logger().info(
                f"[PHRASE] date={key} not found -> fallback challenge='{self._challenge}' expected='{self._expected}'"
            )
            self._emit_event(f"오늘의 문어: {self._challenge} / 답어: {self._expected}")

    # helper functions
    #########################################################
    def _on_start(self, msg:String):
        if self._busy:
            self._set_status("Busy, ignore start")
            self._emit_event("Progressing, ignore start")
            return

        txt = msg.data.strip()
        if txt:
            self._expected = txt  # 디버그로만 override
            self._emit_event("manual start(load CEOI sucessfully)")
        else:
            self._emit_event("manual start(failed load CEOI)")
        self._start_sequence()

    def _start_sequence(self):
        self._refresh_phrase_for_today(force_log=False)
        self._busy = True
        self._state = "AUTH"
        self._attempts = 0

        if not str(self._expected).strip():
            self._set_status("Expected text is empty")
            self._emit_event("Expected text is empty")
            self._busy = False
            self._state = "IDLE"
            return

        self._set_status("Start !!!")
        self.get_logger().info(f"challenge='{self._challenge}', expected='{self._expected}'")
        self._emit_event(f"challenge='{self._challenge}', expected='{self._expected}'")
        self._set_follow_enable(True)
        self._emit_event("Keep tracking...")
        self._try_auth()

    # lock_done 자동 시작
    def _on_lock_done(self, msg: Bool):
        # True일 때만 시작
        if not msg.data:
            return
        if self._state != "IDLE":
            return
        # 중복 시작 방지
        if self._busy:
            return

        # 디바운스 (락온 완료 True가 연속으로 들어오는 경우 대비)
        now = time.time()
        if (now - self._last_lock_done_start_t) < self._lock_done_debounce_sec:
            return
        self._last_lock_done_start_t = now

        self._set_status("LOCK_DONE -> start AUTH")
        self._emit_event("LOCK_DONE -> start AUTH")
        self._start_sequence()

    def _on_trigger(self, msg: Bool):
        # True일 때만 시작
        if not msg.data:
            return

        # 중복 시작 방지
        if self._busy:
            self._set_status("Busy, ignoring trigger")
            self._emit_event("processing, ignore start")
            return
        self._emit_event("start Authentication")
        self._start_sequence()


    def _set_status(self, txt):
        msg = String()
        msg.data = txt
        self._pub_status.publish(msg)

    def _on_salute_done(self, msg: Bool):
        if self._state != "SALUTE_WAIT":
            return
        self._set_status(f"Salute done: {'OK' if msg.data else 'FAIL'}")
        self._set_follow_enable(True)
        self._emit_event("tracking restart")
        self._state = "IDLE"
        self._busy = False

    def _on_shoot_done(self, msg: Bool):
        if self._state != "SHOOT_WAIT":
            return
        self._set_status(f"Shoot done: {'OK' if msg.data else 'FAIL'}")
        self._emit_event("shoot done")
        self._set_follow_enable(True)
        self._emit_event("tracking restart")
        self._state = "IDLE"
        self._busy = False
    
    # (추가5) salute_node stt def ======================================================
    # 암구호 일치
    def _on_salute_accord_text(self, msg: String):
        text = (msg.data or "").strip()
        self._last_salute_accord_text = text
        out = String()
        out.data = text
        self._pub_salute_accord_text.publish(String(data=text))
    # 들은 정보
    def _on_salute_heard_text(self, msg: String):
        text = (msg.data or "").strip()
        self._last_salute_heard_text = text
        out = String()
        out.data = text
        self._pub_salute_heard_text.publish(String(data=text))
    # ==================================================================================
    def _try_auth(self):
        self._attempts += 1
        self._set_status(f"Authentication : {self._attempts} / {self._auth_attempts}")
        self._emit_event(f"Authentication : {self._attempts} / {self._auth_attempts}")
        ok = False
        for i in range(5):
            if self._auth.wait_for_server(timeout_sec=1.0):
                ok = True
                break
            self._set_status(f"Waiting auth server... {i+1}/{5}")

        if not ok:
            self._set_status("Auth server not available")
            self._emit_event("Auth server not available")
            self._set_follow_enable(True)
            self._busy = False
            self._state = "IDLE"
            return
        
        goal = Auth.Goal()
        goal.challenge = self._challenge
        goal.expected = self._expected

        future = self._auth.send_goal_async(goal, feedback_callback=self._auth_feedback)
        future.add_done_callback(self._auth_goal_response)

    def _auth_goal_response(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self._set_status("Auth goal rejected")
            self._emit_event("Auth goal rejected")
            self._set_follow_enable(True)
            self._busy = False
            self._state = "IDLE"
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._auth_result)

    def _auth_result(self, future):
        result = future.result().result
        self.get_logger().info(
            f"auth result: success={result.success} \
                heard='{result.heard_text}' \
                    code={result.code} \
                        reason='{result.reason}'"
        )

        ok_txt = "success" if result.success else "fail"
        heard = (result.heard_text or "").strip()
        if heard:
            self._emit_event(f"Auth {self._attempts} result: {ok_txt} (heard: '{heard}')")
        else:
            self._emit_event(f"Auth {self._attempts} result: {ok_txt}")

        if result.success:
            self._set_status("Auth success -> SALUTE")
            self._state = "SALUTE_WAIT"
            self._set_follow_enable(False)  # SALUTE, tracking 끄고 하기
            self._emit_event("stop tracking")
            self._emit_event("salute executing")
            self._pub_salute_trigger.publish(Bool(data=True))
            return

        if self._attempts < self._auth_attempts:
            self._emit_event("try Atuh again")
            self._try_auth()
        else:
            self._set_status("Auth failed 3 times -> SHOOT")
            self._state = "SHOOT_WAIT"
            self._set_follow_enable(False)  # SHOOT, tracking 끄고 하기
            self._emit_event("stop tracking")
            self._emit_event("shoot executing")
            self._pub_shoot_trigger.publish(Bool(data=True))
            return
        
    def _auth_feedback(self, fb_msg):
        fb = fb_msg.feedback
        self._set_status(f"Auth mode: {fb.mode}")
    
    def _set_follow_enable(self, enabled: bool) -> None:
        self._pub_follow_enable.publish(Bool(data=bool(enabled)))


def main(args=None):
    rclpy.init(args=args)
    node = OrchestratorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        