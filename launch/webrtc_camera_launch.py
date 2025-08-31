#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_topic',
            default_value='/argus/ar0234_front_left/image_raw',
            description='ROS2 camera image topic to stream via WebRTC'
        ),
        
        DeclareLaunchArgument(
            'server_host',
            default_value='gcs.iotocean.org',
            description='WebRTC signaling server hostname'
        ),
        
        DeclareLaunchArgument(
            'server_port',
            default_value='3001',
            description='WebRTC signaling server port'
        ),
        
        DeclareLaunchArgument(
            'room_id',
            default_value='device1/camera',
            description='WebRTC room ID for streaming'
        ),
        
        Node(
            package='webrtc_camera_streamer',
            executable='webrtc_ros_sender',
            name='webrtc_camera_streamer_node',
            output='screen',
            parameters=[{
                'camera_topic': LaunchConfiguration('camera_topic'),
                'server_host': LaunchConfiguration('server_host'),
                'server_port': LaunchConfiguration('server_port'),
                'room_id': LaunchConfiguration('room_id'),
            }],
            remappings=[
                ('image_raw', LaunchConfiguration('camera_topic')),
            ]
        )
    ])