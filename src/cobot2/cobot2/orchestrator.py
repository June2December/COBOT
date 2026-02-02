"""
현재 목표: 암구호를 최대 3회 묻고 성공적으로 답어를 받으면 경례하고
        3번 이내에 성공적 답변 듣지 못하면 더미동작 수행하는 거 해야지

1. 암구호 시퀀스 최대 3회 호출
        # Goal challenge(문어), expected(답어)
        ---
        # Result success(성공여부), heard_text(거수자의 대답), reason(성공 여부 판명 근거), code(세부 판명 code)
        ---
        # Feedback mode(현재 시퀀스에서 어느 단계인지)
    - success=True : 경례
    - success=Falsae : 더미 동작
"""

# use for annotations(hint) for debug error
from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient

from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger

# usrr defined interface / Authentication
from cobot2_interfaces.action import Auth

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
        
        # for node
        self.declare_parameter("auth_action_name", "/auth_action")
        self.declare_parameter("auth_attempts", 3)

        # trigger 기반 시작
        self.declare_parameter("trigger_topic", "/orchestrator/trigger")

        # 오늘의 암구호(문어/답어)는 여기서 주입받도록(launch에서 바꾸기 쉬움)
        self.declare_parameter("challenge_text", "아이폰")
        self.declare_parameter("expected_text", "갤럭시")

        #########################################################

        self._start_topic = self.get_parameter("start_topic").value
        self._status_topic = self.get_parameter("status_topic").value
        self._auth_action_name = self.get_parameter("auth_action_name").value
        self._auth_attempts = int(self.get_parameter("auth_attempts").value)

        self._trigger_topic = self.get_parameter("trigger_topic").value
        self._challenge = self.get_parameter("challenge_text").value
        self._expected = self.get_parameter("expected_text").value


        self._salute_trigger_topic = self.get_parameter("salute_trigger_topic").value
        self._salute_done_topic = self.get_parameter("salute_done_topic").value
        self._shoot_trigger_topic = self.get_parameter("shoot_trigger_topic").value
        self._shoot_done_topic = self.get_parameter("shoot_done_topic").value
        self._follow_enable_topic = self.get_parameter("follow_enable_topic").value

        # state machine
        self._state = "IDLE"  # IDLE | AUTH | SALUTE_WAIT | SHOOT_WAIT


        self._busy = False
        self._attempts = 0

        self._pub_status = self.create_publisher(String, self._status_topic, 10) # orche 상태 출력용
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
        
        self._pub_follow_enable = self.create_publisher(Bool, self._follow_enable_topic, 10)        
        # salute/shoot trigger publishers
        self._pub_salute_trigger = self.create_publisher(Bool, self._salute_trigger_topic, 10)
        self._pub_shoot_trigger = self.create_publisher(Bool, self._shoot_trigger_topic, 10)

        # salute/shoot done subscribers
        self._sub_salute_done = self.create_subscription(
            Bool, self._salute_done_topic, self._on_salute_done, 10, callback_group=self._cbg
        )
        self._sub_shoot_done = self.create_subscription(
            Bool, self._shoot_done_topic, self._on_shoot_done, 10, callback_group=self._cbg
        )

        self._auth = ActionClient(self, Auth, self._auth_action_name, callback_group=self._cbg)
        self._set_status("Ready")

    # helper functions
    #########################################################
    def _on_start(self, msg:String):
        if self._busy:
            self._set_status("Busy, ignoring start")
            return

        txt = msg.data.strip()
        if txt:
            self._expected = txt  # 디버그로만 override

        self._start_sequence()

    def _start_sequence(self):
        self._busy = True
        self._state = "AUTH"
        self._attempts = 0

        # 파라미터로 이미 self._challenge / self._expected가 세팅되어 있다고 가정합니다.
        if not str(self._expected).strip():
            self._set_status("Expected text is empty")
            self._busy = False
            self._state = "IDLE"
            return

        self._set_status("Start !!!")
        self.get_logger().info(f"challenge='{self._challenge}', expected='{self._expected}'")
        self._set_follow_enable(False)
        self._try_auth()


    def _on_trigger(self, msg: Bool):
        # True일 때만 시작
        if not msg.data:
            return

        # 중복 시작 방지
        if self._busy:
            self._set_status("Busy, ignoring trigger")
            return

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
        self._state = "IDLE"
        self._busy = False

    def _on_shoot_done(self, msg: Bool):
        if self._state != "SHOOT_WAIT":
            return
        self._set_status(f"Shoot done: {'OK' if msg.data else 'FAIL'}")
        self._set_follow_enable(True)
        self._state = "IDLE"
        self._busy = False

    def _try_auth(self):
        self._attempts += 1
        self._set_status(f"Authentication : {self._attempts} / {self._auth_attempts}")

        if not self._auth.wait_for_server(timeout_sec=1.0):
            self._set_status("Auth server not available")
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
            self._busy = False
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

        if result.success:
            self._set_status("Auth success -> SALUTE")
            self._state = "SALUTE_WAIT"
            self._pub_salute_trigger.publish(Bool(data=True))
            return

        if self._attempts < self._auth_attempts:
            self._try_auth()
        else:
            self._set_status("Auth failed 3 times -> SHOOT")
            self._state = "SHOOT_WAIT"
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

        
