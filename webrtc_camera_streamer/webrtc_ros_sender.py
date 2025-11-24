#!/usr/bin/env python3
"""
ROS2 WebRTC Camera Streamer
Receives ROS2 camera images from TOPIC and streams via WebRTC
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import asyncio
import time
import logging
from av import VideoFrame
from .lib.WebRTCSender import WebRTCSender

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Server configuration
SERVER_PORT = 3001
STUN_TURN_HOST = 'gcs.iotocean.org'
STUN_TURN_PORT = 3478
CLIENT_NAME = 'HuskyROS'
FPS = 30.0

#room_id = 'husky1'

# Shared frame buffer
current_frame = None

def ros2_frame_generator(frame_count, elapsed_time):
    """Frame generator function for WebRTCSender that provides ROS2 camera frames"""
    global current_frame

    # Return black frame if no camera data available
    if current_frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        av_frame = VideoFrame.from_ndarray(frame, format='rgb24')
    else:
        # Use current frame (already in RGB format)
        av_frame = current_frame

    return av_frame


class ROS2WebRTCSender(Node):
    def __init__(self):
        super().__init__('webrtc_camera_streamer')

        self.declare_parameter('room_id', 'husky1')
        self.declare_parameter('camera_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('server_host', 'gcs.iotocean.org')
        room_id = self.get_parameter('room_id').value
        topic = self.get_parameter('camera_topic').value
        server_host = self.get_parameter('server_host').value

        # ROS2 setup
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            topic,
            self.image_callback,
            10)

        # FPS control
        self.last_frame_time = 0
        self.frame_interval = 1.0 / FPS

        # WebRTC sender setup
        self.webrtc_sender = WebRTCSender(
            signal_server_host=server_host,
            signal_server_port=SERVER_PORT,
            stun_turn_host=STUN_TURN_HOST,
            stun_turn_port=STUN_TURN_PORT,
            room_id=room_id,
            client_name=CLIENT_NAME,
            target_fps=FPS,
            frame_generator_func=ros2_frame_generator
        )

        self.get_logger().info('🚀 ROS2 WebRTC Camera Streamer initialized')
        self.get_logger().info(f'📡 Listening on {topic}')

    def set_frame(self, cv_image):
        global current_frame

        """Set current frame for streaming"""
        # Direct BGR to RGB conversion
        frame = cv_image.copy()
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update global frame (atomic operation)
        current_frame = VideoFrame.from_ndarray(frame, format='rgb24')

    def image_callback(self, msg):
        """Callback for ROS2 image messages"""
        current_time = time.time()

        # Only update frame if enough time has passed (FPS control)
        if current_time - self.last_frame_time >= self.frame_interval:
            try:
                # Convert ROS2 Image message to OpenCV image
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                cv_image = cv2.resize(cv_image, (1920, 1080))

                # Update the frame for WebRTC streaming
                self.set_frame(cv_image)
                self.last_frame_time = current_time

            except Exception as e:
                self.get_logger().error(f'Failed to convert ROS2 image: {e}')

    async def start_webrtc(self):
        """Start the WebRTC sender using the WebRTCSender library"""
        await self.webrtc_sender.start()

def main(args=None):
    rclpy.init(args=args)
    
    node = ROS2WebRTCSender()

    # Start WebRTC in background
    import threading

    def run_webrtc():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(node.start_webrtc())
    
    webrtc_thread = threading.Thread(target=run_webrtc, daemon=True)
    webrtc_thread.start()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
