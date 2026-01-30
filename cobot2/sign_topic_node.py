import os
from dotenv import load_dotenv

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

from .STT import STT
from .sign_check import is_only_dambae

TOPIC_NAME = "/is_dambae"


class VoiceToTopic(Node):
    def __init__(self, stt: STT):
        super().__init__("sign_topic_node_once")
        self.pub = self.create_publisher(Bool, TOPIC_NAME, 10)
        self.stt = stt

    def run_once_and_publish(self):
        text = self.stt.speech2text()
        is_d = is_only_dambae(text)

        msg = Bool()
        msg.data = bool(is_d)
        self.pub.publish(msg)

        self.get_logger().info(f"STT='{text}' -> /is_dambae={msg.data}")


def main():
    load_dotenv(".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Export it or put it in .env")

    stt = STT(api_key)
    stt.duration = 1.2

    rclpy.init()
    node = VoiceToTopic(stt=stt)

    try:
        node.run_once_and_publish()

        for _ in range(5):
            rclpy.spin_once(node, timeout_sec=0.1)

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

