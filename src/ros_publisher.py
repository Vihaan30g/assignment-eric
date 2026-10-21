"""ROS2 stop-signal publisher with an explicit console-only fallback."""
import logging

LOGGER = logging.getLogger(__name__)


class StopPublisher:
    def __init__(self, topic="stop_signal", no_ros=False):
        self._last_value = None
        self._rclpy = None
        self._node = None
        self._publisher = None
        self._owns_context = False
        if no_ros:
            LOGGER.info("ROS publishing disabled; using console fallback")
            return
        try:
            import rclpy
            from std_msgs.msg import Int32
        except ImportError:
            LOGGER.warning("ROS2 message packages are unavailable; using console fallback")
            return
        self._rclpy = rclpy
        self._int32_type = Int32
        if not rclpy.ok():
            rclpy.init()
            self._owns_context = True
        self._node = rclpy.create_node("stop_signal_publisher")
        self._publisher = self._node.create_publisher(Int32, topic, 10)

    def publish(self, value):
        value = int(bool(value))
        if self._publisher is not None:
            message = self._int32_type()
            message.data = value
            self._publisher.publish(message)
        elif value != self._last_value:
            LOGGER.info("stop_signal=%d (console fallback)", value)
        self._last_value = value

    def close(self):
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._owns_context and self._rclpy is not None and self._rclpy.ok():
            self._rclpy.shutdown()
            self._owns_context = False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="stop_signal")
    parser.add_argument("--no-ros", action="store_true")
    args = parser.parse_args()
    publisher = StopPublisher(args.topic, args.no_ros)
    try:
        publisher.publish(0)
    finally:
        publisher.close()
