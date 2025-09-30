import argparse
import base64
import io
import threading
import time
import traceback
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
import librosa
import wave

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

from openai import OpenAI

TARGET_SR = 16000
MAX_DECODE_PAIRS = 64
ENABLE_PREFILL = False
MAX_BUFFERED_PAIRS = 128
PREFILL_EVERY_N_PAIRS = 1
PREFILL_MIN_INTERVAL_S = 0.0

GEN_LOCK = threading.Lock()

def _pil_to_data_url_jpeg(img: Image.Image, *, quality: int = 95) -> str:
    """Encode PIL image to deterministic JPEG data URL for cache stability."""
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


def _audio_to_wav_base64(audio_1s: np.ndarray, sr: int = 16000) -> str:
    """Encode a 1-second mono float32 array to base64 WAV (PCM16)."""
    x = np.asarray(audio_1s, dtype=np.float32).reshape(-1)
    x = np.clip(x, -1.0, 1.0)
    pcm16 = (x * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm16.tobytes())
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _decode_image_b64(b64_data: str) -> Image.Image:
    """Decode base64-encoded image (webp/jpeg) into a RGB PIL image."""
    raw = base64.b64decode(b64_data)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _decode_wav_b64_to_float32(b64_data: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """Decode base64 WAV bytes to mono float32 waveform at target_sr."""
    wav_bytes = base64.b64decode(b64_data)
    y, sr = librosa.load(io.BytesIO(wav_bytes), sr=None, mono=True)
    if sr != target_sr and sr is not None and sr > 0:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return np.asarray(y, dtype=np.float32)


class StreamingVideoSessionOpenAI:
    """Session wrapper managing multimodal prefill and decode using OpenAI API.

    This class encapsulates the logic to build OpenAI Chat Completions requests
    for both prefill (max_tokens=0) and final decode (generation). It maintains
    historical frames and audio slices for each session.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        session_id: str,
        *,
        stop_token_ids: Optional[List[int]] = None,
        mm_processor_kwargs: Optional[dict] = None,
        sr: int = TARGET_SR,
    ):
        self.client = client
        self.model = model
        self.sid = session_id
        self.stop_token_ids = stop_token_ids
        self.mm_processor_kwargs = mm_processor_kwargs
        self.sr = sr

        self.frames: List[Image.Image] = []
        self.audio_slices: List[np.ndarray] = []

    @staticmethod
    def _build_image_uuid(session_id: str, frame_index: int) -> str:
        return f"{session_id}-img-{frame_index:06d}"

    @staticmethod
    def _build_audio_uuid(session_id: str, chunk_index: int) -> str:
        return f"{session_id}-aud-{chunk_index:06d}"

    def streaming_prefill_frame(self, frame_pil: Image.Image, frame_index: int) -> None:
        """Prefill a single frame via OpenAI Chat API (max_tokens=0)."""
        frame_pil = frame_pil.convert("RGB")
        # Keep local history for robust final decode
        self.frames.append(frame_pil)
        img_url = _pil_to_data_url_jpeg(frame_pil)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_url}},
            ],
        }]

        img_uuid = self._build_image_uuid(self.sid, frame_index)
        extra_body: dict = {}
        if self.stop_token_ids:
            extra_body["stop_token_ids"] = self.stop_token_ids
        if self.mm_processor_kwargs:
            extra_body["mm_processor_kwargs"] = self.mm_processor_kwargs
        extra_body["multi_modal_uuids"] = {"image": [img_uuid]}

        self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=0,
            extra_body=extra_body,
        )

    def streaming_prefill_av_chunk(
        self,
        frame_pil: Image.Image,
        audio_1s: np.ndarray,
        chunk_index: int,
    ) -> None:
        """Prefill a single AV chunk (frame + 1s audio) via OpenAI Chat API."""
        frame_pil = frame_pil.convert("RGB")
        # Keep local history for robust final decode
        self.frames.append(frame_pil)
        self.audio_slices.append(np.asarray(audio_1s, dtype=np.float32))

        img_url = _pil_to_data_url_jpeg(frame_pil)
        aud_b64 = _audio_to_wav_base64(self.audio_slices[-1], sr=self.sr)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_url}},
                {"type": "input_audio", "input_audio": {"data": aud_b64, "format": "wav"}},
            ],
        }]

        img_uuid = self._build_image_uuid(self.sid, chunk_index)
        aud_uuid = self._build_audio_uuid(self.sid, chunk_index)
        extra_body: dict = {}
        if self.stop_token_ids:
            extra_body["stop_token_ids"] = self.stop_token_ids
        if self.mm_processor_kwargs:
            extra_body["mm_processor_kwargs"] = self.mm_processor_kwargs
        extra_body["multi_modal_uuids"] = {"image": [img_uuid], "audio": [aud_uuid]}

        self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=0,
            extra_body=extra_body,
        )

    def decode_images_segment(
        self,
        frames: List[Image.Image],
        uuids: List[str],
        question: str,
        *,
        max_tokens: int = 512,
    ) -> str:
        """Decode using a subset of frames with explicit UUID alignment."""
        if not frames:
            return ""
        content_text = "".join(["(<image>./</image>)" for _ in range(len(frames))])
        if question:
            content_text += "\n" + question
        parts: List[dict] = [{"type": "text", "text": content_text}]
        for f in frames:
            parts.append({"type": "image_url", "image_url": {"url": _pil_to_data_url_jpeg(f)}})
        messages = [{"role": "user", "content": parts}]

        extra_body: dict = {}
        if self.stop_token_ids:
            extra_body["stop_token_ids"] = self.stop_token_ids
        if self.mm_processor_kwargs:
            extra_body["mm_processor_kwargs"] = self.mm_processor_kwargs
        extra_body["multi_modal_uuids"] = {"image": uuids}

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        choice = resp.choices[0].message
        return getattr(choice, "content", "") or ""

    def decode_av_segment(
        self,
        frames: List[Image.Image],
        audios: List[np.ndarray],
        img_uuids: List[str],
        aud_uuids: List[str],
        question: str,
        *,
        max_tokens: int = 512,
    ) -> str:
        """Decode using interleaved AV chunks with explicit UUID alignment."""
        n = min(len(frames), len(audios))
        if n == 0:
            return ""
        content_text = "".join(["(<image>./</image>)(<audio>./</audio>)" for _ in range(n)])
        if question:
            content_text += "\n" + question
        parts: List[dict] = [{"type": "text", "text": content_text}]
        for i in range(n):
            parts.append({"type": "image_url", "image_url": {"url": _pil_to_data_url_jpeg(frames[i])}})
        for i in range(n):
            parts.append({"type": "input_audio", "input_audio": {"data": _audio_to_wav_base64(audios[i], sr=self.sr), "format": "wav"}})
        messages = [{"role": "user", "content": parts}]

        extra_body: dict = {}
        if self.stop_token_ids:
            extra_body["stop_token_ids"] = self.stop_token_ids
        if self.mm_processor_kwargs:
            extra_body["mm_processor_kwargs"] = self.mm_processor_kwargs
        extra_body["multi_modal_uuids"] = {"image": img_uuids, "audio": aud_uuids}

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        choice = resp.choices[0].message
        return getattr(choice, "content", "") or ""


class SessionState:
    """Per-UID state: pair frames and 1s audio slices; perform optional prefill."""

    def __init__(self, session: StreamingVideoSessionOpenAI):
        self.session = session
        self.lock = threading.Lock()
        self.audio_buf = np.zeros((0,), dtype=np.float32)
        self.frame_queue: List[Image.Image] = []
        self.paired = 0
        self._last_prefill_ts = 0.0
        self._prefill_counter = 0

    def _try_pair_and_prefill(self):
        one_sec = TARGET_SR
        while self.audio_buf.shape[0] >= one_sec and len(self.frame_queue) > 0:
            audio_1s = self.audio_buf[:one_sec]
            self.audio_buf = self.audio_buf[one_sec:]
            frame = self.frame_queue.pop(0)

            do_prefill = False
            if ENABLE_PREFILL:
                self._prefill_counter += 1
                if self._prefill_counter >= max(1, PREFILL_EVERY_N_PAIRS):
                    now = time.time()
                    if (now - self._last_prefill_ts) >= max(0.0, PREFILL_MIN_INTERVAL_S):
                        do_prefill = True
                        self._prefill_counter = 0
                        self._last_prefill_ts = now

            if do_prefill:
                try:
                    with GEN_LOCK:
                        self.session.streaming_prefill_av_chunk(
                            frame, audio_1s, chunk_index=self.paired
                        )
                except Exception as e:
                    try:
                        print(f"[prefill][error] uid={self.session.sid} err={e}", flush=True)
                    except Exception:
                        pass
                    try:
                        self.session.frames.append(frame.convert("RGB"))
                        self.session.audio_slices.append(audio_1s)
                    except Exception:
                        pass
                self.paired += 1
            else:
                try:
                    self.session.frames.append(frame.convert("RGB"))
                    self.session.audio_slices.append(audio_1s)
                except Exception as e:
                    try:
                        print(f"[buffer][error] uid={self.session.sid} err={e}", flush=True)
                    except Exception:
                        pass
                self.paired += 1

            if len(self.session.frames) > MAX_BUFFERED_PAIRS:
                self.session.frames = self.session.frames[-MAX_BUFFERED_PAIRS:]
            if len(self.session.audio_slices) > MAX_BUFFERED_PAIRS:
                self.session.audio_slices = self.session.audio_slices[-MAX_BUFFERED_PAIRS:]
            try:
                print(
                    f"[pair] uid={self.session.sid} paired_idx={self.paired} "
                    f"buf_sec={self.audio_buf.shape[0]/TARGET_SR:.2f} "
                    f"frames_q={len(self.frame_queue)}",
                    flush=True,
                )
            except Exception:
                pass

    def ingest_image(self, img: Image.Image):
        with self.lock:
            self.frame_queue.append(img)
            self._try_pair_and_prefill()

    def ingest_audio(self, wav_f32: np.ndarray):
        if wav_f32.size == 0:
            return
        with self.lock:
            self.audio_buf = np.concatenate([self.audio_buf, wav_f32])
            self._try_pair_and_prefill()


class SessionManager:
    """Thread-safe manager for per-UID SessionState instances."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        *,
        stop_token_ids: Optional[List[int]] = None,
        mm_processor_kwargs: Optional[dict] = None,
        sr: int = TARGET_SR,
    ):
        self.client = client
        self.model = model
        self.stop_token_ids = stop_token_ids
        self.mm_processor_kwargs = mm_processor_kwargs
        self.sr = sr
        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def get(self, uid: Optional[str]) -> SessionState:
        if not uid:
            raise HTTPException(status_code=400, detail="Missing uid header")
        with self._lock:
            st = self._sessions.get(uid)
            if st is None:
                sess = StreamingVideoSessionOpenAI(
                    self.client,
                    self.model,
                    session_id=uid,
                    stop_token_ids=self.stop_token_ids,
                    mm_processor_kwargs=self.mm_processor_kwargs,
                    sr=self.sr,
                )
                st = SessionState(sess)
                self._sessions[uid] = st
            return st

    def reset(self, uid: Optional[str]):
        if not uid:
            return
        with self._lock:
            if uid in self._sessions:
                del self._sessions[uid]


class OnlineStreamingHTTPServer:
    """HTTP server bridging frontend streaming AV to OpenAI Chat API backend."""

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model_id: str,
        *,
        enable_prefill: bool = ENABLE_PREFILL,
        prefill_every_n: int = PREFILL_EVERY_N_PAIRS,
        prefill_min_interval_s: float = PREFILL_MIN_INTERVAL_S,
        max_buffered_pairs: int = MAX_BUFFERED_PAIRS,
        max_decode_pairs: int = MAX_DECODE_PAIRS,
        stop_token_ids: Optional[List[int]] = None,
        mm_processor_kwargs: Optional[dict] = None,
        sr: int = TARGET_SR,
    ):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model_id

        self.enable_prefill = bool(enable_prefill)
        self.prefill_every_n_pairs = max(1, int(prefill_every_n))
        self.prefill_min_interval_s = max(0.0, float(prefill_min_interval_s))
        self.max_buffered_pairs = max(1, int(max_buffered_pairs))
        self.max_decode_pairs = max(1, int(max_decode_pairs))
        self.stop_token_ids = stop_token_ids
        self.mm_processor_kwargs = mm_processor_kwargs
        self.sr = sr

        global ENABLE_PREFILL, PREFILL_EVERY_N_PAIRS, PREFILL_MIN_INTERVAL_S, MAX_BUFFERED_PAIRS, MAX_DECODE_PAIRS, TARGET_SR
        ENABLE_PREFILL = self.enable_prefill
        PREFILL_EVERY_N_PAIRS = self.prefill_every_n_pairs
        PREFILL_MIN_INTERVAL_S = self.prefill_min_interval_s
        MAX_BUFFERED_PAIRS = self.max_buffered_pairs
        MAX_DECODE_PAIRS = self.max_decode_pairs
        TARGET_SR = self.sr

        self.mgr = SessionManager(
            self.client,
            self.model,
            stop_token_ids=self.stop_token_ids,
            mm_processor_kwargs=self.mm_processor_kwargs,
            sr=self.sr,
        )
        self.app = FastAPI()
        self._register_routes()

    def _register_routes(self):
        app = self.app
        mgr = self.mgr

        @app.post("/api/v1/stream")
        async def stream_endpoint(request: Request, uid: Optional[str] = Header(None)):
            """Accept incremental image/audio payloads from the frontend client."""
            body = await request.json()
            try:
                content = body["messages"][0]["content"][0]
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid message format")

            st = mgr.get(uid)
            ctype = content.get("type")

            if ctype == "image_data":
                img_b64 = content["image_data"]["data"]
                try:
                    print(f"[recv][img] uid={uid} b64_len={len(img_b64)}", flush=True)
                except Exception:
                    pass
                img = _decode_image_b64(img_b64)
                st.ingest_image(img)
                return JSONResponse({"status": "ok", "paired": st.paired})

            if ctype == "input_audio":
                a_b64 = content["input_audio"]["data"]
                try:
                    print(f"[recv][aud] uid={uid} b64_len={len(a_b64)}", flush=True)
                except Exception:
                    pass
                wav = _decode_wav_b64_to_float32(a_b64, target_sr=TARGET_SR)
                st.ingest_audio(wav)
                return JSONResponse({"status": "ok", "paired": st.paired})

            raise HTTPException(status_code=400, detail=f"Unsupported content type: {ctype}")

        @app.post("/api/v1/generate")
        async def generate_endpoint(payload: dict, uid: Optional[str] = Header(None)):
            """Trigger final text generation using cached paired AV chunks."""
            st = mgr.get(uid)
            question = payload.get("question") or "Please provide a brief summary of the audio content."
            try:
                print(f"[generate] uid={uid} question_len={len(question)}", flush=True)
            except Exception:
                pass

            try:
                with st.lock:
                    n_img = len(st.session.frames)
                    n_aud = len(st.session.audio_slices)
                    n = min(n_img, n_aud)
                    if n == 0 and n_img > 0:
                        # Image-only mode: always use the most recent K frames
                        n_use = min(n_img, MAX_DECODE_PAIRS)
                        frames_seg = list(st.session.frames[-n_use:])
                        start_idx = max(0, n_img - n_use)
                        img_uuids = [
                            st.session._build_image_uuid(st.session.sid, start_idx + i)
                            for i in range(n_use)
                        ]
                        with GEN_LOCK:
                            text = st.session.decode_images_segment(
                                frames_seg,
                                img_uuids,
                                question,
                                max_tokens=512,
                            )
                        return JSONResponse({"text": text})

                    if n == 0:
                        return JSONResponse({"text": "", "detail": "no paired AV chunks"})

                    # Always use the most recent K pairs to decode
                    n_use = min(n, MAX_DECODE_PAIRS)
                    # Select last n_use items from both modalities to ensure alignment
                    frames_seg = list(st.session.frames[-n_use:])
                    audios_seg = [wav.astype(np.float32) for wav in st.session.audio_slices[-n_use:]]
                    # Absolute indices for the selected pairs: [paired - n_use, ..., paired - 1]
                    abs_start = max(0, st.paired - n_use)
                    img_uuids = [
                        st.session._build_image_uuid(st.session.sid, abs_start + i)
                        for i in range(n_use)
                    ]
                    aud_uuids = [
                        st.session._build_audio_uuid(st.session.sid, abs_start + i)
                        for i in range(n_use)
                    ]

                    with GEN_LOCK:
                        text = st.session.decode_av_segment(
                            frames_seg,
                            audios_seg,
                            img_uuids,
                            aud_uuids,
                            question,
                            max_tokens=512,
                        )
                    return JSONResponse({"text": text})
            except Exception as e:
                try:
                    print(f"[generate][error] uid={uid} err={e}\n{traceback.format_exc()}", flush=True)
                except Exception:
                    pass
                return JSONResponse({"error": str(e)}, status_code=500)

        @app.post("/api/v1/reset")
        async def reset_endpoint(payload: dict, uid: Optional[str] = Header(None)):
            """Reset and remove session state for the given UID."""
            mgr.reset(uid)
            return JSONResponse({"status": "reset"})

        @app.get("/api/v1/status")
        async def status_endpoint(uid: Optional[str] = Header(None)):
            """Return current session statistics for the given UID."""
            st = mgr.get(uid)
            with st.lock:
                n_img = len(st.session.frames)
                n_aud = len(st.session.audio_slices)
                buf_sec = st.audio_buf.shape[0] / TARGET_SR
                q = len(st.frame_queue)
            return JSONResponse({
                "paired": st.paired,
                "frames_cached": n_img,
                "audio_slices_cached": n_aud,
                "audio_buffer_sec": round(buf_sec, 3),
                "frames_in_queue": q,
            })


def build_app(
    api_base: str,
    api_key: str,
    model_id: str,
    *,
    enable_prefill: bool = ENABLE_PREFILL,
    prefill_every_n: int = PREFILL_EVERY_N_PAIRS,
    prefill_min_interval_s: float = PREFILL_MIN_INTERVAL_S,
    max_buffered_pairs: int = MAX_BUFFERED_PAIRS,
    max_decode_pairs: int = MAX_DECODE_PAIRS,
    stop_token_ids: Optional[List[int]] = None,
    mm_processor_kwargs: Optional[dict] = None,
    sr: int = TARGET_SR,
) -> FastAPI:
    server = OnlineStreamingHTTPServer(
        api_base=api_base,
        api_key=api_key,
        model_id=model_id,
        enable_prefill=enable_prefill,
        prefill_every_n=prefill_every_n,
        prefill_min_interval_s=prefill_min_interval_s,
        max_buffered_pairs=max_buffered_pairs,
        max_decode_pairs=max_decode_pairs,
        stop_token_ids=stop_token_ids,
        mm_processor_kwargs=mm_processor_kwargs,
        sr=sr,
    )
    return server.app


def _parse_int_list_csv(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    try:
        return [int(x) for x in s.split(",") if x.strip()]
    except Exception:
        return None


def main():
    """Entry point for OpenAI-backed streaming AV HTTP server."""
    parser = argparse.ArgumentParser(description="HTTP uplink server for online streaming AV via OpenAI Chat API")
    parser.add_argument("--api-base", type=str, required=True, help="OpenAI base URL, e.g., http://localhost:8000/v1")
    parser.add_argument("--api-key", type=str, required=True, help="API key for the server")
    parser.add_argument("--model", type=str, required=True, help="Model path or HF ID registered by server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=32551)
    parser.add_argument("--sr", type=int, default=TARGET_SR, help="Target audio sample rate for processing")
    # Prefill/Decode knobs
    parser.add_argument("--enable-prefill", action="store_true", help="Enable streaming prefill on paired AV chunks")
    parser.add_argument("--prefill-every-n", type=int, default=PREFILL_EVERY_N_PAIRS, help="Prefill every N pairs")
    parser.add_argument("--prefill-min-interval", type=float, default=PREFILL_MIN_INTERVAL_S, help="Min seconds between prefill calls")
    parser.add_argument("--max-buffered-nuse", type=int, default=MAX_BUFFERED_PAIRS, help="Cap buffered AV pairs history")
    parser.add_argument("--max-decode-uuidnum", type=int, default=MAX_DECODE_PAIRS, help="Cap pairs used in final decode")
    # Extra decode controls
    parser.add_argument("--stop-token-ids", type=str, default=None, help="Comma-separated stop token ids")
    parser.add_argument("--mm-processor-kwargs", type=str, default=None, help="JSON string for mm processor kwargs")

    args = parser.parse_args()

    stop_ids = _parse_int_list_csv(args.stop_token_ids)
    mm_kwargs = None
    if args.mm_processor_kwargs:
        try:
            import json
            mm_kwargs = json.loads(args.mm_processor_kwargs)
        except Exception:
            mm_kwargs = None

    app = build_app(
        api_base=args.api_base,
        api_key=args.api_key,
        model_id=args.model,
        enable_prefill=args.enable_prefill,
        prefill_every_n=args.prefill_every_n,
        prefill_min_interval_s=args.prefill_min_interval,
        max_buffered_pairs=args.max_buffered_nuse,
        max_decode_pairs=args.max_decode_uuidnum,
        stop_token_ids=stop_ids,
        mm_processor_kwargs=mm_kwargs,
        sr=int(args.sr),
    )
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()



