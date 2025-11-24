from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py
import subprocess
from pathlib import Path

class build_py(_build_py):
    def run(self):
        # 빌드 전에 external 파일 동기화
        '''root = Path(__file__).parent
        script = root / "scripts" / "update_external_lib.py"
        print(f"[webrtc_camera_streamer] Updating WebRTCSender.py via {script}")
        subprocess.check_call(["python3", str(script)], cwd=root)'''

        # 원래 build_py 동작
        super().run()

package_name = 'webrtc_camera_streamer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/webrtc_camera_launch.py']),
    ],
    install_requires=[
        'setuptools',
        'aiortc',
        'python-socketio[asyncio]',
        'aiohttp',
        'numpy<2',  # Pin to NumPy 1.x for cv_bridge compatibility
        'opencv-python>=4.5.0,<4.6.0' # Compatible with NumPy 1.x        
    ],
    cmdclass={
        'build_py': build_py,  # ⬅️ 여기서 우리가 custom build를 씌움
    },
    zip_safe=True,
    maintainer='sjlee',
    maintainer_email='sjlee88@keti.re.kr',
    description='ROS2 WebRTC Camera Streamer - Streams ROS2 camera images via WebRTC',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'webrtc_ros_sender = webrtc_camera_streamer.webrtc_ros_sender:main',
            'webrtc_ros_sender_gstreamer = webrtc_camera_streamer.webrtc_ros_sender_gstreamer:main',
        ],
    },
)
