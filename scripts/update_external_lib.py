# scripts/update_external_lib.py
import pathlib
import urllib.request

COMMIT = "db4a3f79ba2d12235774eef315b8ed8a64641417"

RAW_URL = f"https://raw.githubusercontent.com/IoTKETI/BrowserlessWebRTC/{COMMIT}/clients/python/WebRTCSender.py"

token = "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"

req = urllib.request.Request(RAW_URL)
# GitHub는 Authorization 헤더로 토큰 받음
req.add_header("Authorization", f"Bearer {token}")

root = pathlib.Path(__file__).resolve().parent.parent  # 패키지 루트
lib_dir = root / "webrtc_camera_streamer" / "lib"
lib_dir.mkdir(parents=True, exist_ok=True)

target = lib_dir / "WebRTCSender.py"
print(f"Downloading {RAW_URL} -> {target}")
with urllib.request.urlopen(req) as resp, open(target, "wb") as f:
    f.write(resp.read())
print("Done.")