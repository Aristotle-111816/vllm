import argparse
import base64
import io
import json
import math
from typing import Optional, List

import cv2
import numpy as np
from PIL import Image
from openai import OpenAI
import librosa
import wave
import subprocess



def _build_image_uuid(session_id: str, frame_index: int) -> str:
    """Build a stable UUID for image cache reuse."""
    return f"{session_id}-img-{frame_index:06d}"



def _pil_to_data_url_jpeg(img: Image.Image, *, quality: int = 95) -> str:
    """Encode PIL image to deterministic JPEG data URL.

    Rationale:
    - Deterministic encoding improves cache hit rate on the server side
      when content hashing is used to identify multi-modal items.
    """
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(
        buf,
        format="JPEG",
        quality=quality,
        optimize=False,
        progressive=False,
        subsampling=0,
    )
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"



def build_online_messages(*, user_text: Optional[str], image_data_urls: List[str]) -> list:
    """Build OpenAI Chat messages payload with optional text and images.

    Structure aligns with OpenAI Chat Completions API and vLLM OpenAI server.
    """
    content: list = []
    if user_text:
        content.append({"type": "text", "text": user_text})
    for url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return [{"role": "user", "content": content}]


def build_extra_body(
    *,
    stop_token_ids: Optional[List[int]] = None,
    mm_processor_kwargs: Optional[dict] = None,
    multi_modal_uuids: Optional[dict] = None,
) -> dict:
    """Build extra_body for vLLM OpenAI server.

    Notes:
    - `multi_modal_uuids` support requires server-side propagation to engine.
    - `mm_processor_kwargs` forwards processor overrides to the HF processor.
    """
    extra: dict = {}
    if stop_token_ids:
        extra["stop_token_ids"] = stop_token_ids
    if mm_processor_kwargs:
        extra["mm_processor_kwargs"] = mm_processor_kwargs
    if multi_modal_uuids:
        extra["multi_modal_uuids"] = multi_modal_uuids
    return extra



def extract_video_frames_1fps(
    video_path: str,
    *,
    max_seconds: Optional[int] = None,
    fps_override: Optional[float] = None,
) -> List[Image.Image]:
    """Extract one frame per second from a local video file.

    Args:
        video_path: Path to the input video file.
        max_seconds: Optional cap on extracted seconds/frames.
        fps_override: Override FPS read from metadata if provided.

    Returns:
        A list of PIL.Image frames sampled at 1 fps.
    """
    frames: List[Image.Image] = []
    cap = cv2.VideoCapture(video_path)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return frames

        fps = float(fps_override) if fps_override is not None else float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            return frames

        duration_sec = total_frames / fps
        seconds_to_cover = int(math.ceil(duration_sec))
        target_count = seconds_to_cover if max_seconds is None else min(seconds_to_cover, max_seconds)
        if target_count <= 0:
            return frames

        for s in range(target_count):
            idx = int(round(s * fps))
            if idx >= total_frames:
                idx = total_frames - 1
            if idx < 0:
                idx = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        return frames
    finally:
        cap.release()



def _split_audio_into_1s(wav: np.ndarray, sr: int) -> List[np.ndarray]:
    """Split mono float32 audio into 1-second chunks."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if sr <= 0 or wav.size == 0:
        return []
    one_sec = int(sr)
    num_full = wav.size // one_sec
    return [wav[i * one_sec:(i + 1) * one_sec] for i in range(num_full)]


def extract_audio_slices_from_video(video_path: str, target_sr: int = 16000) -> List[np.ndarray]:
    """Extract mono audio from video and split into 1-second slices."""
    try:
        wav, _ = librosa.load(video_path, sr=target_sr, mono=True)
        chunks = _split_audio_into_1s(wav, target_sr)
        if chunks:
            return chunks
    except Exception:
        pass

    # Fallback: extract audio using ffmpeg to handle more codecs.
    return extract_audio_slices_from_video_ffmpeg(video_path, target_sr)


def extract_audio_slices_from_video_ffmpeg(video_path: str, target_sr: int = 16000) -> List[np.ndarray]:
    """Extract 1-second mono audio from video using ffmpeg (PCM16 WAV via stdout).

    Requirements:
    - ffmpeg must be available in PATH.
    """
    cmd = [
        "ffmpeg", "-i", video_path,
        "-f", "wav", "-ac", "1", "-ar", str(target_sr), "-vn",
        "-hide_banner", "-loglevel", "error", "-",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        data, _ = proc.communicate()
        if proc.returncode != 0 or not data:
            return []
        buf = io.BytesIO(data)
        with wave.open(buf, 'rb') as wf:
            sr = wf.getframerate()
            nframes = wf.getnframes()
            sampwidth = wf.getsampwidth()
            nch = wf.getnchannels()
            pcm = wf.readframes(nframes)
        if sampwidth != 2:
            return []  # Expect PCM16
        arr = np.frombuffer(pcm, dtype=np.int16)
        if nch > 1:
            arr = arr.reshape(-1, nch).mean(axis=1).astype(np.int16)
        wav = (arr.astype(np.float32) / 32767.0)
        return _split_audio_into_1s(wav, sr)
    except Exception:
        return []


def extract_audio_slices_from_file(audio_path: str, target_sr: int = 16000) -> List[np.ndarray]:
    """Extract mono audio from an audio file and split into 1-second slices."""
    try:
        wav, _ = librosa.load(audio_path, sr=target_sr, mono=True)
        return _split_audio_into_1s(wav, target_sr)
    except Exception:
        return []


def _audio_to_wav_base64(audio_1s: np.ndarray, sr: int = 16000) -> str:
    """Encode a 1-second mono float32 audio array into base64 WAV (PCM16).

    Implementation details:
    - Convert float32 in [-1, 1] to int16 PCM.
    - Use Python's built-in wave module to produce standard PCM WAV.
    """
    # Clamp and scale to int16
    x = np.clip(audio_1s, -1.0, 1.0)
    x = (x * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(x.tobytes())
    return base64.b64encode(buf.getvalue()).decode('utf-8')



def online_streaming_preill(
    *,
    client: OpenAI,
    model: str,
    frame_pil: Image.Image,
    session_id: str,
    frame_index: int,
    mm_processor_kwargs: Optional[dict] = None,
    stop_token_ids: Optional[List[int]] = None,
    skip_uuids: bool = False,
) -> None:
    """Perform a prefill-only request for a single frame via OpenAI Chat API.

    Behavior:
    - Build a one-image message (no text).
    - Use max_tokens=0 to avoid generation; server will pre-process and encode
      multi-modal inputs, enabling cache reuse.
    - Provide a stable UUID through `extra_body.multi_modal_uuids` for cache keys.
    """
    data_url = _pil_to_data_url_jpeg(frame_pil)
    messages = build_online_messages(user_text=None, image_data_urls=[data_url])

    uuid = _build_image_uuid(session_id, frame_index)
    extra_body = build_extra_body(
        stop_token_ids=stop_token_ids,
        mm_processor_kwargs=mm_processor_kwargs,
        multi_modal_uuids=None if skip_uuids else {"image": [uuid]},
    )

    client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=0,
        extra_body=extra_body,
    )

def online_streaming_prefill_av_chunk(
    *,
    client: OpenAI,
    model: str,
    frame_pil: Image.Image,
    audio_1s: np.ndarray,
    session_id: str,
    chunk_index: int,
    sr: int = 16000,
    mm_processor_kwargs: Optional[dict] = None,
    stop_token_ids: Optional[List[int]] = None,
    skip_uuids: bool = False,
) -> None:
    """Prefill a single audio-visual chunk via OpenAI Chat API.

    Behavior:
    - Compose content with one image and one 1-second audio segment.
    - Use max_tokens=0 to avoid generation and populate caches only.
    - Optionally propagate stable UUIDs for both modalities.
    """
    img_url = _pil_to_data_url_jpeg(frame_pil)
    aud_b64 = _audio_to_wav_base64(audio_1s, sr=sr)
    parts: list = [
        {"type": "image_url", "image_url": {"url": img_url}},
        {"type": "input_audio", "input_audio": {"data": aud_b64, "format": "wav"}},
    ]
    messages = [{"role": "user", "content": parts}]

    img_uuid = _build_image_uuid(session_id, chunk_index)
    aud_uuid = f"{session_id}-aud-{chunk_index:06d}"
    mm_uuids = None if skip_uuids else {"image": [img_uuid], "audio": [aud_uuid]}

    extra_body = build_extra_body(
        stop_token_ids=stop_token_ids,
        mm_processor_kwargs=mm_processor_kwargs,
        multi_modal_uuids=mm_uuids,
    )

    client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=0,
        extra_body=extra_body,
    )


def online_streaming_decode(
    *,
    client: OpenAI,
    model: str,
    frames: List[Image.Image],
    session_id: str,
    question: str,
    max_tokens: int = 512,
    stop_token_ids: Optional[List[int]] = None,
    mm_processor_kwargs: Optional[dict] = None,
    stream: bool = False,
    skip_uuids: bool = False,
) -> str:
    """Trigger decoding using historical frames + a question via Chat API.

    Returns:
        Assistant text response. If `stream=True`, concatenates streamed deltas.
    """
    if not frames:
        return ""

    # Build explicit placeholder text to align with multimodal positions,
    # then append the actual image items. This mirrors the pattern used in
    # the offline demo to ensure placeholder count matches embeddings.
    n = len(frames)
    content_text = "".join(["(<image>./</image>)" for _ in range(n)])
    if question:
        content_text += "\n" + question
    parts: list = [{"type": "text", "text": content_text}]
    for f in frames:
        parts.append({"type": "image_url", "image_url": {"url": _pil_to_data_url_jpeg(f)}})
    messages = [{"role": "user", "content": parts}]

    uuids = [_build_image_uuid(session_id, i) for i in range(len(frames))]
    extra_body = build_extra_body(
        stop_token_ids=stop_token_ids,
        mm_processor_kwargs=mm_processor_kwargs,
        multi_modal_uuids=None if skip_uuids else {"image": uuids},
    )

    if stream:
        chunks: List[str] = []
        with client.chat.completions.stream(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            extra_body=extra_body,
        ) as s:
            for event in s:
                delta = getattr(event, "delta", None)
                if delta and getattr(delta, "content", None):
                    chunks.append(delta.content)
        return "".join(chunks)

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    choice = resp.choices[0].message
    return getattr(choice, "content", "") or ""

def online_streaming_decode_av(
    *,
    client: OpenAI,
    model: str,
    frames: List[Image.Image],
    audios: List[np.ndarray],
    session_id: str,
    question: str,
    max_tokens: int = 512,
    stop_token_ids: Optional[List[int]] = None,
    mm_processor_kwargs: Optional[dict] = None,
    sr: int = 16000,
    stream: bool = False,
    skip_uuids: bool = False,
) -> str:
    """Trigger decoding using interleaved audio+image chunks and a question.

    Behavior:
    - Build a message combining optional user text and interleaved pairs of
      (input_audio, image_url) for the first N pairs, where N is the min
      length of frames and audios.
    - Return assistant text (or streamed deltas if enabled).
    """
    n = min(len(frames), len(audios))
    if n == 0:
        return ""

    # Build explicit interleaved placeholders, then append items in the
    # same modality-block order (images first, then audios), which matches
    # the offline demo and avoids mask/embeds mismatch.
    content_text = "".join(["(<image>./</image>)(<audio>./</audio>)" for _ in range(n)])
    if question:
        content_text += "\n" + question
    parts: list = [{"type": "text", "text": content_text}]
    # Append all images first
    for i in range(n):
        parts.append({"type": "image_url", "image_url": {"url": _pil_to_data_url_jpeg(frames[i])}})
    # Append all audios next
    for i in range(n):
        parts.append({"type": "input_audio", "input_audio": {"data": _audio_to_wav_base64(audios[i], sr=sr), "format": "wav"}})
    messages = [{"role": "user", "content": parts}]

    mm_uuids = None
    if not skip_uuids:
        img_uuids = [_build_image_uuid(session_id, i) for i in range(n)]
        aud_uuids = [f"{session_id}-aud-{i:06d}" for i in range(n)]
        mm_uuids = {"image": img_uuids, "audio": aud_uuids}

    extra_body = build_extra_body(
        stop_token_ids=stop_token_ids,
        mm_processor_kwargs=mm_processor_kwargs,
        multi_modal_uuids=mm_uuids,
    )

    if stream:
        chunks: List[str] = []
        with client.chat.completions.stream(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            extra_body=extra_body,
        ) as s:
            for event in s:
                delta = getattr(event, "delta", None)
                if delta and getattr(delta, "content", None):
                    chunks.append(delta.content)
        return "".join(chunks)

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    choice = resp.choices[0].message
    return getattr(choice, "content", "") or ""


def _parse_int_list_csv(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    try:
        return [int(x) for x in s.split(",") if x.strip()]
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Online streaming prefill (max_tokens=0) + decode via OpenAI Chat API",
    )
    parser.add_argument("--api-base", type=str, required=True, help="OpenAI base URL, e.g., http://localhost:8000/v1")
    parser.add_argument("--api-key", type=str, required=True, help="API key for the server")
    parser.add_argument("--model", type=str, required=True, help="Model path or HF ID registered by server")
    parser.add_argument("--video", type=str, required=True, help="Local video path")
    parser.add_argument("--audio", type=str, default=None, help="Optional standalone audio file; overrides video audio extraction")
    parser.add_argument("--session", type=str, default="session-0928", help="Session id for stable UUIDs")
    parser.add_argument("--max-seconds", type=int, default=16, help="Max seconds/frames to extract at 1 fps")
    parser.add_argument("--fps", type=float, default=None, help="Optional FPS override")
    parser.add_argument("--stop-token-ids", type=str, default=None, help="Comma-separated stop token ids")
    parser.add_argument("--mm-processor-kwargs", type=str, default=None, help="JSON string for mm processor kwargs")
    parser.add_argument("--question", type=str, default="", help="Question to ask at decode time")
    parser.add_argument("--skip-uuids", action="store_true", help="Do not send multi_modal_uuids in extra_body")
    parser.add_argument("--decode-max-images-per-call", type=int, default=0, help="If >0, only use the last K frames at decode time (image-only mode)")
    parser.add_argument("--decode-max-pairs", type=int, default=0, help="If >0, only use the last K (audio+image) pairs at decode time")
    parser.add_argument("--sr", type=int, default=16000, help="Target audio sample rate for extraction")
    parser.add_argument("--stream", action="store_true", help="Stream decode responses")

    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key, base_url=args.api_base)
    stop_ids = _parse_int_list_csv(args.stop_token_ids)
    mm_kwargs = None
    if args.mm_processor_kwargs:
        try:
            mm_kwargs = json.loads(args.mm_processor_kwargs)
        except Exception:
            mm_kwargs = None

    frames = extract_video_frames_1fps(
        args.video,
        max_seconds=args.max_seconds,
        fps_override=args.fps,
    )
    print(f"[info] extracted frames: {len(frames)}")

    # Extract audio and decide AV vs image-only mode
    audios = (extract_audio_slices_from_file(args.audio, target_sr=args.sr)
              if args.audio else extract_audio_slices_from_video(args.video, target_sr=args.sr))
    print(f"[info] extracted audio seconds: {len(audios)}")

    if audios:
        # AV prefill: pair each 1s audio slice with the matching frame (by second)
        pair_n = min(len(frames), len(audios))
        for i in range(pair_n):
            online_streaming_prefill_av_chunk(
                client=client,
                model=args.model,
                frame_pil=frames[i],
                audio_1s=audios[i],
                session_id=args.session,
                chunk_index=i,
                sr=args.sr,
                mm_processor_kwargs=mm_kwargs,
                stop_token_ids=stop_ids,
                skip_uuids=args.skip_uuids,
            )
            print(f"[prefill][av] committed pair {i+1}/{pair_n} (max_tokens=0)")
    else:
        # Image-only prefill: one frame per request with max_tokens=0
        for i, fr in enumerate(frames):
            online_streaming_preill(
                client=client,
                model=args.model,
                frame_pil=fr,
                session_id=args.session,
                frame_index=i,
                mm_processor_kwargs=mm_kwargs,
                stop_token_ids=stop_ids,
                skip_uuids=args.skip_uuids,
            )
            print(f"[prefill] committed frame {i+1}/{len(frames)} (max_tokens=0)")

    # Trigger decode when needed
    if args.question:
        if audios:
            # AV decode
            if args.decode_max_pairs and args.decode_max_pairs > 0:
                n = min(len(frames), len(audios))
                k = min(args.decode_max_pairs, n)
                frames_for_decode = frames[-k:]
                audios_for_decode = audios[-k:]
            else:
                n = min(len(frames), len(audios))
                frames_for_decode = frames[:n]
                audios_for_decode = audios[:n]
            text = online_streaming_decode_av(
                client=client,
                model=args.model,
                frames=frames_for_decode,
                audios=audios_for_decode,
                session_id=args.session,
                question=args.question,
                max_tokens=512,
                stop_token_ids=stop_ids,
                mm_processor_kwargs=mm_kwargs,
                sr=args.sr,
                stream=args.stream,
                skip_uuids=args.skip_uuids,
            )
        else:
            # Image-only decode
            if args.decode_max_images_per_call and args.decode_max_images_per_call > 0:
                frames_for_decode = frames[-args.decode_max_images_per_call:]
                print(f"[decode] using last {len(frames_for_decode)} frames (cap={args.decode_max_images_per_call})")
            else:
                frames_for_decode = frames
            text = online_streaming_decode(
                client=client,
                model=args.model,
                frames=frames_for_decode,
                session_id=args.session,
                question=args.question,
                max_tokens=512,
                stop_token_ids=stop_ids,
                mm_processor_kwargs=mm_kwargs,
                stream=args.stream,
                skip_uuids=args.skip_uuids,
            )
        print("===== Generated =====")
        print(text)
    else:
        print("[info] decode skipped (no question provided)")


if __name__ == "__main__":
    main()


