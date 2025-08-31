# ROS2 WebRTC Camera Streamer

A ROS2 package that receives camera images from ROS2 topics and streams them via WebRTC to web browsers or other WebRTC clients.

## Features

- 📹 Subscribes to ROS2 camera image topics
- 🌐 Streams video via WebRTC protocol  
- 🎯 Real-time video transmission with frame counter overlay
- 🔧 Configurable server endpoints and camera topics
- 📊 FPS monitoring and logging
- 🎥 Automatic image resizing to 1920x1080 HD

## Dependencies

- ROS2 (tested with Humble)
- OpenCV (`cv_bridge`)
- aiortc (WebRTC implementation)
- python-socketio
- numpy

## Installation

1. Create and navigate to your ROS2 workspace:
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

2. Clone or copy this package to your workspace

3. Build the package:
```bash
cd ~/ros2_ws
colcon build --packages-select webrtc_camera_streamer
source install/setup.bash
```

4. Install Python dependencies:
```bash
pip3 install aiortc python-socketio[asyncio] aiohttp opencv-python
```

## Usage

### Running the WebRTC Sender

Start the ROS2 WebRTC camera streamer:

```bash
ros2 run webrtc_camera_streamer webrtc_ros_sender
```

The node will:
1. Subscribe to `/argus/ar0234_front_left/image_raw` topic
2. Connect to the WebRTC signaling server at `gcs.iotocean.org:3001`
3. Create room `device1/camera` for streaming
4. Wait for WebRTC clients to connect and receive video stream

### Configuration

Edit the configuration variables in `webrtc_ros_sender.py`:

```python
# Server configuration
SERVER_HOST = 'gcs.iotocean.org'
SERVER_PORT = 3001
STUN_TURN_HOST = 'gcs.iotocean.org' 
STUN_TURN_PORT = 3478
ROOM_ID = 'device1/camera'
CLIENT_NAME = 'ROS2CameraSender'
```

### Camera Topic

The default camera topic is `/argus/ar0234_front_left/image_raw`. To change it, modify the subscription in the `__init__` method:

```python
self.subscription = self.create_subscription(
    Image,
    '/your/camera/topic',  # Change this
    self.image_callback,
    10)
```

## Video Stream Features

- **Frame Counter**: Displays current frame number on the right side
- **Topic Info**: Shows the ROS2 topic name being streamed
- **HD Resolution**: Automatically scales images to 1920x1080
- **FPS Monitoring**: Logs frame rate every 30 frames
- **Fallback Display**: Shows "Waiting for ROS2 camera data..." when no camera data is available

## Viewing the Stream

1. Open a web browser and navigate to your WebRTC server web interface
2. Join the room `device1/camera`
3. The ROS2 camera stream should appear in the remote video element

## Troubleshooting

### No Video Stream
- Check that the camera topic is publishing: `ros2 topic echo /argus/ar0234_front_left/image_raw`
- Verify WebRTC server is running and accessible
- Check network connectivity to STUN/TURN servers

### ROS2 Topic Issues
- List available topics: `ros2 topic list`
- Check topic type: `ros2 topic info /argus/ar0234_front_left/image_raw`
- Ensure topic publishes `sensor_msgs/Image` messages

### WebRTC Connection Issues  
- Check server logs for ICE candidate generation
- Verify STUN/TURN server configuration
- Test with local network first before trying over internet

## Architecture

```
ROS2 Camera → Image Topic → ROS2 Node → WebRTC Track → Signaling Server → Web Browser
```

The system creates a bridge between ROS2's image transport and WebRTC's real-time communication protocol, enabling ROS2 camera streams to be viewed in standard web browsers without additional plugins.