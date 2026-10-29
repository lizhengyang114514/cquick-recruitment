import rclpy
from rclpy.node import Node

from cquick_interfaces.msg import TargetPosition


class TargetSubscriber(Node):
    def __init__(self):
        super().__init__("target_subscriber")
        self.create_subscription(
            TargetPosition, "/target_position", self.on_msg, 10)
        self.count = 0
        self.get_logger().info("等待 /target_position 消息 ...")

    def on_msg(self, msg):
        self.count += 1
        if msg.detected:
            self.get_logger().info(
                "#" + str(self.count) + " detected=True  u=" + str(round(msg.u, 1)) +
                "  v=" + str(round(msg.v, 1)) + "  area=" + str(round(msg.area, 1)))
        else:
            self.get_logger().info(
                "#" + str(self.count) + " detected=False  未检测到目标（u/v/area 为默认值 0）")

        if self.count == 10:
            self.get_logger().info("已收到 10 条消息，验收要求满足")


def main(args=None):
    rclpy.init(args=args)
    node = TargetSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
