import os

import cv2
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node

from cquick_interfaces.msg import TargetPosition

from target_detector.detector import annotate, detect, list_images


class TargetPublisher(Node):
    def __init__(self):
        super().__init__("target_publisher")

        self.declare_parameter("image_dir", "")
        self.declare_parameter("use_camera", False)
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("fps", 1.0)

        self.declare_parameter("h_min", 5)
        self.declare_parameter("s_min", 110)
        self.declare_parameter("v_min", 90)
        self.declare_parameter("h_max", 22)
        self.declare_parameter("s_max", 255)
        self.declare_parameter("v_max", 255)
        self.declare_parameter("min_area", 2000.0)

        self.pub = self.create_publisher(TargetPosition, "/target_position", 10)

        self.use_camera = self.get_parameter("use_camera").value
        self.camera_index = self.get_parameter("camera_index").value

        self.images = []
        self.index = 0
        self.cap = None

        if self.use_camera:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                self.get_logger().warn("摄像头打不开，改用测试图片")
                self.use_camera = False
                self.cap = None

        if not self.use_camera:
            folder = self.get_parameter("image_dir").value
            if not folder:
                folder = os.path.join(
                    get_package_share_directory("target_detector"), "test_images")
            self.images = list_images(folder)
            if not self.images:
                self.get_logger().error("没有找到测试图片: " + folder)
            else:
                self.get_logger().info("找到 " + str(len(self.images)) + " 张测试图: " + folder)

        fps = float(self.get_parameter("fps").value)
        self.timer = self.create_timer(1.0 / max(fps, 0.1), self.on_timer)
        self.count = 0

    def read_frame(self):
        if self.use_camera:
            ok, frame = self.cap.read()
            if not ok:
                return None, "camera"
            return frame, "camera/" + str(self.camera_index)

        if not self.images:
            return None, "no-image"

        path = self.images[self.index % len(self.images)]
        self.index += 1
        return cv2.imread(path), os.path.basename(path)

    def on_timer(self):
        frame, source = self.read_frame()

        msg = TargetPosition()
        if frame is None:
            msg.detected = False
            msg.u = 0.0
            msg.v = 0.0
            msg.area = 0.0
            self.pub.publish(msg)
            self.get_logger().warn("读不到图像(" + source + ")，发布 detected=false")
            return

        frame = cv2.resize(frame, (600, int(frame.shape[0] * 600 / frame.shape[1])))

        result = detect(
            frame,
            int(self.get_parameter("h_min").value),
            int(self.get_parameter("s_min").value),
            int(self.get_parameter("v_min").value),
            int(self.get_parameter("h_max").value),
            int(self.get_parameter("s_max").value),
            int(self.get_parameter("v_max").value),
            float(self.get_parameter("min_area").value),
        )

        msg.detected = bool(result["detected"])
        msg.u = float(result["u"])
        msg.v = float(result["v"])
        msg.area = float(result["area"])
        self.pub.publish(msg)
        self.count += 1

        if result["detected"]:
            self.get_logger().info(
                "#" + str(self.count) + " " + source + " detected=True u=" +
                str(round(msg.u, 1)) + " v=" + str(round(msg.v, 1)) +
                " area=" + str(round(msg.area, 1)))
        else:
            self.get_logger().info(
                "#" + str(self.count) + " " + source + " detected=False（未检测到目标）")

        if not self.use_camera:
            cv2.imwrite("/tmp/target_last.png", annotate(frame, result))

    def destroy_node(self):
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TargetPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
