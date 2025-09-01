#!/usr/bin/env python3
"""
ROS2 WebRTC Camera Streamer
Receives ROS2 camera images from /argus/ar0234_front_left/image_raw and streams via WebRTC
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import asyncio
import threading
import time
import logging
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCConfiguration, RTCIceServer
import socketio
from av import VideoFrame

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Server configuration
SERVER_HOST = 'gcs.iotocean.org'
SERVER_PORT = 3001
STUN_TURN_HOST = 'gcs.iotocean.org'
STUN_TURN_PORT = 3478
SERVER_URL = f'http://{SERVER_HOST}:{SERVER_PORT}'
ROOM_ID = 'husky/camera'
CLIENT_NAME = 'Husky'
TOPIC = '/camera/camera/color/image_raw'

class ROS2ImageVideoStreamTrack(VideoStreamTrack):
    """Custom video stream track that sends ROS2 camera images"""
    
    def __init__(self):
        super().__init__()
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_time = time.time()
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
    def set_frame(self, cv_image):
        """Set the current frame to be sent"""
        
        if self.frame_lock.acquire(blocking=False):
            self.current_frame = cv_image.copy()
            self.frame_lock.release()
        
    def create_frame_with_info(self):
        """Create frame with ROS2 info overlay"""
        with self.frame_lock:
            if self.current_frame is None:
                # Create default frame if no camera data available
                frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                cv2.putText(frame, 'Waiting for ROS2 camera data...', (50, 500), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            else:
                # Resize camera image to 1920x1080 if needed
                frame = cv2.resize(self.current_frame, (1920, 1080))
        
        # Add frame counter and info on the right side
        frame_text = f'Frame: {self.frame_count}'
        
        # Calculate text size for positioning
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 3
        (text_width, text_height), baseline = cv2.getTextSize(frame_text, font, font_scale, thickness)
        
        # Position text on the right side of the image
        x_pos = 1920 - text_width - 50  # 50 pixels from right edge
        y_pos = 100  # 100 pixels from top
        
        # Add white background rectangle for better visibility
        cv2.rectangle(frame, 
                     (x_pos - 10, y_pos - text_height - 10), 
                     (x_pos + text_width + 10, y_pos + baseline + 10), 
                     (255, 255, 255), -1)
        
        # Add frame counter text
        cv2.putText(frame, frame_text, (x_pos, y_pos), 
                   font, font_scale, (0, 0, 0), thickness)
        
        # Add ROS2 topic info
        topic_text = TOPIC
        topic_y_pos = y_pos + 80
        (topic_width, _), _ = cv2.getTextSize(topic_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        topic_x_pos = 1920 - topic_width - 50
        cv2.putText(frame, topic_text, (topic_x_pos, topic_y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame
        
    async def recv(self):
        pts, time_base = await self.next_timestamp()
        
        # Create frame with ROS2 camera data
        #frame = self.create_frame_with_info()
        frame = None
        #if self.frame_lock.acquire(blocking=False):
        with self.frame_lock:
            frame = self.current_frame
        #    self.frame_lock.release()
        #else:
        #    return
        self.frame_count += 1
        
        # Debug: Log frame generation and calculate FPS every 30 frames
        if self.frame_count % 30 == 0:
            current_time = time.time()
            elapsed = current_time - self.last_fps_time
            fps = 30.0 / elapsed if elapsed > 0 else 0
            total_elapsed = current_time - self.start_time
            avg_fps = self.frame_count / total_elapsed if total_elapsed > 0 else 0
            
            logger.info(f"📹 ROS2 Frame {self.frame_count} (PTS: {pts}) | Current FPS: {fps:.1f} | Avg FPS: {avg_fps:.1f}")
            self.last_fps_time = current_time
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create VideoFrame
        av_frame = VideoFrame.from_ndarray(frame, format='rgb24')
        av_frame.pts = pts
        av_frame.time_base = time_base

        return av_frame

class ROS2WebRTCSender(Node):
    def __init__(self):
        super().__init__('webrtc_camera_streamer')
        
        # ROS2 setup
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            TOPIC,
            self.image_callback,
            10)
        
        # WebRTC setup
        self.sio = socketio.AsyncClient()
        self.ice_servers = []
        self.peer_connections = {}
        self.client_id = None
        self.video_track = ROS2ImageVideoStreamTrack()
        
        self.get_logger().info('🚀 ROS2 WebRTC Camera Streamer initialized')
        self.get_logger().info(f'📡 Listening on '+TOPIC)
        
        # Start WebRTC in separate thread
        self.webrtc_thread = threading.Thread(target=self.run_webrtc_async)
        self.webrtc_thread.daemon = True
        self.webrtc_thread.start()
        
        self.setup_socket_handlers()
    
    def image_callback(self, msg):
        """Callback for ROS2 image messages"""
        try:
            # Convert ROS2 Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Update the video track with new frame
            self.video_track.set_frame(cv_image)
                            
        except Exception as e:
            self.get_logger().error(f'Failed to convert ROS2 image: {e}')
    
    def run_webrtc_async(self):
        """Run WebRTC asyncio loop in separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.start_webrtc())
    
    def setup_socket_handlers(self):
        @self.sio.event
        async def connect():
            logger.info("Connected to WebRTC signaling server")
            await self.sio.emit('join-room', {
                'roomId': self.actual_room_id,
                'clientName': CLIENT_NAME
            })
        
        @self.sio.event
        async def disconnect():
            logger.info("Disconnected from WebRTC signaling server")
        
        @self.sio.event
        async def connected(data):
            self.client_id = data['clientId']
            logger.info(f"WebRTC Client ID: {self.client_id}")
        
        @self.sio.on('room-joined')
        async def room_joined(data):
            logger.info(f"Joined WebRTC room: {data['roomName']} ({data['roomId']})")
        
        @self.sio.on('client-joined')
        async def client_joined(data):
            logger.info(f"WebRTC client joined: {data['clientName']}")
            await self.create_peer_connection(data['clientId'])
        
        @self.sio.on('room-clients')
        async def room_clients(data):
            logger.info(f"Existing WebRTC clients in room: {len(data['clients'])}")
            for client in data['clients']:
                await self.create_peer_connection(client['clientId'])
        
        @self.sio.event
        async def offer(data):
            await self.handle_offer(data)
        
        @self.sio.event
        async def answer(data):
            await self.handle_answer(data)
        
        @self.sio.on('ice-candidate')
        async def ice_candidate(data):
            await self.handle_ice_candidate(data)
        
        @self.sio.on('client-left')
        async def client_left(data):
            logger.info(f"WebRTC client left: {data['clientName']}")
            client_id = data['clientId']
            if client_id in self.peer_connections:
                await self.peer_connections[client_id].close()
                del self.peer_connections[client_id]
        
        @self.sio.event
        async def error(data):
            logger.error(f"WebRTC server error: {data['message']}")
    
    async def load_ice_servers(self):
        """Load ICE servers configuration"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{SERVER_URL}/api/ice-servers") as resp:
                    data = await resp.json()
                    self.ice_servers = data['iceServers']
                    logger.info(f"Loaded {len(self.ice_servers)} ICE servers")
                    logger.info(f"ICE servers: {self.ice_servers}")
        except Exception as e:
            logger.error(f"Failed to load ICE servers: {e}")
            # Fallback configuration
            self.ice_servers = [
                {"urls": [f"stun:{STUN_TURN_HOST}:{STUN_TURN_PORT}"]},
                {"urls": [f"turn:{STUN_TURN_HOST}:{STUN_TURN_PORT}"], "username": "keti", "credential": "keti123"}
            ]
    
    async def create_peer_connection(self, client_id):
        """Create a new peer connection"""
        if client_id in self.peer_connections:
            return self.peer_connections[client_id]
        
        # Convert ICE server dictionaries to RTCIceServer objects
        ice_servers = []
        for server in self.ice_servers:
            if isinstance(server, dict):
                ice_servers.append(RTCIceServer(urls=server["urls"]))
            else:
                ice_servers.append(server)
        
        pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))
        self.peer_connections[client_id] = pc
        
        @pc.on("connectionstatechange")
        async def on_connection_state_change():
            logger.info(f"WebRTC peer connection with {client_id} state: {pc.connectionState}")
            if pc.connectionState == "connected":
                logger.info(f"🎥 ROS2 camera stream flowing to {client_id}")
        
        @pc.on("iceconnectionstatechange")
        async def on_ice_connection_state_change():
            logger.info(f"🧊 ICE connection state with {client_id}: {pc.iceConnectionState}")
        
        # Add ROS2 video track
        pc.addTrack(self.video_track)
        logger.info(f"Added ROS2 camera video track to peer connection {client_id}")
        
        # Create and send offer
        await self.create_offer(client_id)
        
        return pc
    
    async def create_offer(self, client_id):
        """Create and send offer to client"""
        pc = self.peer_connections[client_id]
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        
        # Wait for ICE gathering to complete
        while pc.iceGatheringState != "complete":
            await asyncio.sleep(0.1)
                
        await self.sio.emit('offer', {
            'targetClientId': client_id,
            'offer': {
                'type': pc.localDescription.type,
                'sdp': pc.localDescription.sdp
            }
        })
        logger.info(f"Sent WebRTC offer to {client_id}")
    
    async def handle_offer(self, data):
        """Handle incoming offer"""
        client_id = data['fromClientId']
        offer = data['offer']
        
        if client_id not in self.peer_connections:
            await self.create_peer_connection_without_offer(client_id)
        
        pc = self.peer_connections[client_id]
        
        try:
            await pc.setRemoteDescription(RTCSessionDescription(
                sdp=offer['sdp'],
                type=offer['type']
            ))
            
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            
            await self.sio.emit('answer', {
                'targetClientId': client_id,
                'answer': {
                    'type': pc.setLocalDescription.type,
                    'sdp': pc.setLocalDescription.sdp
                }
            })
            logger.info(f"Sent WebRTC answer to {client_id}")
            
        except Exception as e:
            logger.error(f"Failed to handle WebRTC offer from {client_id}: {e}")
    
    async def handle_answer(self, data):
        """Handle incoming answer"""
        client_id = data['fromClientId']
        answer = data['answer']
        
        if client_id not in self.peer_connections:
            logger.error(f"No peer connection for {client_id}")
            return
        
        pc = self.peer_connections[client_id]
        
        try:
            await pc.setRemoteDescription(RTCSessionDescription(
                sdp=answer['sdp'],
                type=answer['type']
            ))
            logger.info(f"Set remote description from {client_id}")
        except Exception as e:
            logger.error(f"Failed to handle WebRTC answer from {client_id}: {e}")
    
    async def handle_ice_candidate(self, data):
        """Handle ICE candidate"""
        client_id = data['fromClientId']
        candidate_data = data['candidate']
        
        if client_id not in self.peer_connections:
            return
        
        pc = self.peer_connections[client_id]
        
        try:
            if candidate_data.get('candidate') == '':
                # End-of-candidates indicator
                await pc.addIceCandidate(None)
                logger.info(f"Added end-of-candidates indicator from {client_id}")
            else:
                # Parse the candidate string manually
                candidate_line = candidate_data['candidate']
                logger.info(f"Processing ICE candidate: {candidate_line}")
                
                # Parse candidate line: "candidate:foundation component protocol priority ip port typ type ..."
                parts = candidate_line.split()
                if len(parts) >= 8:
                    foundation = parts[0].split(':')[1]  # Remove "candidate:" prefix
                    component = int(parts[1])
                    protocol = parts[2].lower()
                    priority = int(parts[3])
                    ip = parts[4]
                    port = int(parts[5])
                    candidate_type = parts[7]  # after "typ"
                    
                    # Handle related address/port for reflexive/relay candidates
                    related_address = None
                    related_port = None
                    if 'raddr' in parts:
                        raddr_idx = parts.index('raddr')
                        if raddr_idx + 1 < len(parts):
                            related_address = parts[raddr_idx + 1]
                    if 'rport' in parts:
                        rport_idx = parts.index('rport')
                        if rport_idx + 1 < len(parts):
                            related_port = int(parts[rport_idx + 1])
                    
                    from aiortc import RTCIceCandidate
                    candidate = RTCIceCandidate(
                        component=component,
                        foundation=foundation,
                        ip=ip,
                        port=port,
                        priority=priority,
                        protocol=protocol,
                        type=candidate_type,
                        relatedAddress=related_address,
                        relatedPort=related_port,
                        sdpMid=candidate_data.get('sdpMid'),
                        sdpMLineIndex=candidate_data.get('sdpMLineIndex')
                    )
                    
                    await pc.addIceCandidate(candidate)
                    logger.info(f"✅ Added ICE candidate from {client_id}: {candidate_type} {ip}:{port}")
                    
                    if candidate_type == "host":
                        logger.info(f"🏠 HOST candidate added - direct connection possible")
                else:
                    logger.warning(f"Invalid candidate format from {client_id}: {candidate_line}")
            
        except Exception as e:
            logger.error(f"Failed to add ICE candidate from {client_id}: {e}")
            logger.error(f"Candidate data: {candidate_data}")
            # Continue anyway - connection might still work
    
    async def create_peer_connection_without_offer(self, client_id):
        """Create peer connection without sending offer"""
        if client_id in self.peer_connections:
            return self.peer_connections[client_id]
        
        ice_servers = []
        for server in self.ice_servers:
            if isinstance(server, dict):
                ice_servers.append(RTCIceServer(urls=server["urls"]))
            else:
                ice_servers.append(server)
        
        pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))
        self.peer_connections[client_id] = pc
        
        pc.addTrack(self.video_track)
        logger.info(f"Added ROS2 camera video track to peer connection {client_id} (no offer)")
        
        return pc
    
    async def ensure_room_exists(self):
        """Ensure the WebRTC room exists"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{SERVER_URL}/api/rooms", 
                                      json={
                                          "roomId": ROOM_ID,
                                          "name": "Husky", 
                                          "maxClients": 20
                                      }) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        logger.info(f"Created WebRTC room: {data['name']} (ID: {data['roomId']})")
                    elif resp.status == 409:
                        data = await resp.json()
                        logger.info(f"WebRTC room already exists: {data['name']} (ID: {data['roomId']})")
                    
                    return ROOM_ID
                    
        except Exception as e:
            logger.error(f"Failed to ensure WebRTC room exists: {e}")
            return ROOM_ID
    
    async def start_webrtc(self):
        """Start the WebRTC sender"""
        logger.info("🌐 Starting WebRTC sender for ROS2 camera stream...")
        
        # Load ICE servers
        await self.load_ice_servers()
        
        # Ensure the room exists
        self.actual_room_id = await self.ensure_room_exists()
        
        # Connect to signaling server
        await self.sio.connect(SERVER_URL)
        
        try:
            # Keep running
            await self.sio.wait()
        except Exception as e:
            logger.error(f"WebRTC error: {e}")
        finally:
            # Cleanup
            for pc in self.peer_connections.values():
                await pc.close()
            await self.sio.disconnect()

def main(args=None):
    rclpy.init(args=args)
    
    node = ROS2WebRTCSender()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
