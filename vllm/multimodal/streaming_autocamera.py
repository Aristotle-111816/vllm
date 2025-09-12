import os
import time
import threading
import queue
import cv2
import base64
from openai import OpenAI

# ===== Config =====
FPS = 1  # Target FPS
FRAME_W, FRAME_H = 320, 320
VIDEO_URL = "https://814f2cb40416.ngrok-free.app/video_feed"  # Refresh every time
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "token-abc123")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
MODEL = os.getenv("VLLM_MODEL", "/root/autodl-tmp/MiniCPM-o-2_6")
STOP_TOKENS = [151645, 151643]
# ==================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class FrameProducer(threading.Thread):
    """Capture video frames in real-time (only keep the latest one in queue)"""
    def __init__(self, out_q: queue.Queue, fps: int, video_url: str):
        super().__init__(daemon=True)
        self.q = out_q
        self.fps = fps
        self.video_url = video_url
        self._stop = threading.Event()
        self._idx = 0

    def run(self):
        cap = cv2.VideoCapture(self.video_url)
        if not cap.isOpened():
            print("[ERROR] Cannot open video stream")
            return

        # Try to reduce buffer size to avoid frame lag
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        interval = 1.0 / self.fps
        next_t = time.perf_counter()

        while not self._stop.is_set():
            # Discard old frames, keep only the latest one
            for _ in range(5):
                cap.grab()
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from video stream")
                continue

            frame_resized = cv2.resize(frame, (FRAME_W, FRAME_H))

            # If queue is full, discard the old frame
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put((self._idx, frame_resized))
            self._idx += 1

            # Control capture rate
            next_t += interval
            sleep_t = next_t - time.perf_counter()
            if sleep_t > 0:
                time.sleep(sleep_t)

        cap.release()

    def stop(self):
        self._stop.set()


def frame_to_base64(frame) -> str:
    _, buf = cv2.imencode(".jpg", frame)
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")


def request_inference(image_b64: str, prompt: str):
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_b64}},
            ],
        }],
        extra_body={"stop_token_ids": STOP_TOKENS},
        max_tokens=256,
    )
    return resp


def save_debug_frame(frame, idx: int):
    filename = os.path.join(SCRIPT_DIR, f"frame_{idx:04d}.jpg")
    cv2.imwrite(filename, frame)
    print(f"[DEBUG] Saved frame to {filename}")


def main():
    q = queue.Queue(maxsize=1)
    producer = FrameProducer(q, FPS, VIDEO_URL)
    producer.start()

    print(f"Stream running at {FPS} fps.")
    print("Type a question for inference, or 'q' to quit.")

    try:
        while True:
            prompt = input("> ").strip()
            if not prompt:
                continue
            if prompt.lower() == "q":
                break

            try:
                idx, latest_frame = q.get(timeout=2.0)
            except queue.Empty:
                print("No frames yet. Try again in a moment.")
                continue

            print(f"Frames captured so far: {idx + 1}, current frame index: {idx}")

            # Save debug frame if needed
            # save_debug_frame(latest_frame, idx)

            try:
                img_b64 = frame_to_base64(latest_frame)
                resp = request_inference(img_b64, prompt)
                print(resp.choices[0].message.content)
            except Exception as e:
                print("Request failed:", e)

    finally:
        producer.stop()
        producer.join(timeout=1.0)
        print("Bye.")


if __name__ == "__main__":
    main()
