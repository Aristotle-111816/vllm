import argparse
import os
from typing import Optional

import cv2
import numpy as np
import librosa
from PIL import Image
from transformers import AutoTokenizer

from openai import OpenAI


def extract_frames_per_second(video_path: str, max_seconds: int = 10) -> list[Image.Image]:
    """Extract 1 frame/sec with OpenCV, fallback to imageio if OpenCV fails."""
    frames: list[Image.Image] = []

    # First try OpenCV
    cap = cv2.VideoCapture(video_path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if total > 0 and fps > 0:
            seconds = min(int(np.ceil(total / fps)), max_seconds)
            for s in range(seconds):
                idx = int(round(s * fps))
                idx = min(max(idx, 0), total - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
    finally:
        cap.release()

    if frames:
        return frames

    # Fallback: imageio (requires imageio and imageio-ffmpeg)
    try:
        import imageio as iio
        reader = iio.get_reader(video_path, format="ffmpeg")
        meta = reader.get_meta_data()
        fps = float(meta.get("fps", 0.0))
        duration = float(meta.get("duration", 0.0))
        if fps <= 0 and duration > 0:
            # Try to estimate fps from nframes/duration
            nframes = int(meta.get("nframes", 0))
            fps = (nframes / duration) if duration > 0 else 0.0
        if fps <= 0 and duration <= 0:
            reader.close()
            return frames
        seconds = max_seconds if duration <= 0 else min(int(np.ceil(duration)), max_seconds)
        for s in range(seconds):
            idx = int(round(s * fps))
            try:
                frame = reader.get_data(idx)
            except Exception:
                break
            frames.append(Image.fromarray(frame))
        reader.close()
    except Exception:
        # Keep frames as empty
        pass

    return frames


def extract_audio_1s_chunks_from_video(video_path: str, sr: int = 16000, max_seconds: int = 10) -> list[np.ndarray]:
    try:
        wav, _ = librosa.load(video_path, sr=sr, mono=True)
    except Exception:
        return []
    one_sec = int(sr)
    num = min(len(wav) // one_sec, max_seconds)
    return [wav[i * one_sec:(i + 1) * one_sec].astype(np.float32) for i in range(num)]


def extract_audio_1s_chunks_from_file(audio_path: str, sr: int = 16000, max_seconds: int = 10) -> list[np.ndarray]:
    try:
        wav, _ = librosa.load(audio_path, sr=sr, mono=True)
    except Exception:
        return []
    one_sec = int(sr)
    num = min(len(wav) // one_sec, max_seconds)
    return [wav[i * one_sec:(i + 1) * one_sec].astype(np.float32) for i in range(num)]


def pil_to_base64_jpeg(img: Image.Image) -> str:
    import io, base64
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def audio_to_wav_base64(audio_1s: np.ndarray, sr: int = 16000) -> str:
    import io, base64, wave, struct
    # Clamp and convert float32 [-1,1] to int16 PCM
    a = np.asarray(audio_1s, dtype=np.float32).clip(-1.0, 1.0)
    pcm16 = (a * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return b64


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=os.path.join(os.path.dirname(__file__), "oceans_240.mp4"))
    parser.add_argument("--api_base", type=str, default=os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1"))
    parser.add_argument("--api_key", type=str, default=os.environ.get("VLLM_API_KEY", "token-abc123"))
    parser.add_argument("--model", type=str, default=os.environ.get("VLLM_MODEL", "/root/autodl-tmp/MiniCPM-o-2_6"))
    parser.add_argument("--session", type=str, default="ocean-demo")
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--audio", type=str, default=None,
                        help="Optional standalone audio file; overrides video audio extraction")
    args = parser.parse_args()

    # Prepare tokenizer only for chat template if needed (optional)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # OpenAI-compatible client
    client = OpenAI(api_key=args.api_key, base_url=args.api_base)

    # 1) Extract 1 frame/sec and 1s audio chunks
    frames = extract_frames_per_second(args.video, max_seconds=args.seconds)
    if args.audio:
        audios = extract_audio_1s_chunks_from_file(args.audio, sr=16000, max_seconds=args.seconds)
    else:
        audios = extract_audio_1s_chunks_from_video(args.video, sr=16000, max_seconds=args.seconds)
    if len(frames) == 0:
        print("No frames extracted. Please check video path/codec. If running in headless env, install imageio & imageio-ffmpeg.")
        return
    if len(audios) == 0:
        print("Warning: no audio chunks extracted. Proceeding with image-only prefill/decode.")
        n = min(len(frames), args.seconds)
    else:
        n = min(len(frames), len(audios), args.seconds)

    # Helper to build stable UUIDs for images and audio
    def img_uuid(i: int) -> str:
        return f"{args.session}-img-{i:06d}"

    def aud_uuid(i: int) -> str:
        return f"{args.session}-aud-{i:06d}"

    # 2) Prefill loop: for i in [0..9), each time send one frame + 1s audio, with max_tokens=0
    for i in range(n):
        # Build a minimal prompt with both image and audio placeholders
        parts = [
            {"type": "text", "text": "(<image>./</image>)(<audio>./</audio>)"},
            {"type": "image_url", "image_url": {"url": pil_to_base64_jpeg(frames[i])}},
        ]
        if i < len(audios):
            aud_b64 = audio_to_wav_base64(audios[i], sr=16000)
            parts.append({
                "type": "input_audio",
                "input_audio": {"data": aud_b64, "format": "wav"}
            })
        messages = [{"role": "user", "content": parts}]

        # Prefill-only request
        resp = client.chat.completions.create(
            model=args.model,
            messages=messages,
            max_tokens=0,  # prefill only
            stream=False,
        )
        print(f"[prefill] {i+1}/{n} status=ok id={resp.id if hasattr(resp,'id') else 'n/a'}")

    # 3) Decode once with all pairs (image+audio). Build k pairs of placeholders and append a question.
    k = n
    content_text = "".join(["(<image>./</image>)(<audio>./</audio>)" for _ in range(k)])
    content_text += "\nPlease describe the ocean scene concisely."
    decode_parts = [{"type": "text", "text": content_text}]
    for i in range(k):
        decode_parts.append({
            "type": "image_url",
            "image_url": {"url": pil_to_base64_jpeg(frames[i])},
        })
    for i in range(min(k, len(audios))):
        aud_b64 = audio_to_wav_base64(audios[i], sr=16000)
        decode_parts.append({
            "type": "input_audio",
            "input_audio": {"data": aud_b64, "format": "wav"}
        })
    messages = [{"role": "user", "content": decode_parts}]

    decode = client.chat.completions.create(
        model=args.model,
        messages=messages,
        max_tokens=512,
        stream=False,
    )
    text = decode.choices[0].message.content if decode.choices else ""
    print("===== Generated =====")
    print(text)


if __name__ == "__main__":
    main()



