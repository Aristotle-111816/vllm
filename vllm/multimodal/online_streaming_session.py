import threading
import time
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from fastapi import HTTPException
from openai import OpenAI

# Media helpers imported from modality modules to avoid duplication
from vllm.multimodal.image import pil_to_data_url_jpeg as _pil_to_data_url_jpeg
from vllm.multimodal.audio import audio_to_wav_base64 as _audio_to_wav_base64


# Runtime knobs (kept global for backward-compatible behavior with callers)
TARGET_SR = 16000
MAX_DECODE_PAIRS = 64
ENABLE_PREFILL = False
MAX_BUFFERED_PAIRS = 128
PREFILL_EVERY_N_PAIRS = 1
PREFILL_MIN_INTERVAL_S = 0.0


# Global generation lock to serialize backend decode calls
GEN_LOCK = threading.Lock()


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
        content_text = "".join(
            ["(<image>./</image>)" for _ in range(len(frames))]
        )
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
        content_text = "".join(
            ["(<image>./</image>)(<audio>./</audio>)" for _ in range(n)]
        )
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
                except Exception:
                    try:
                        print(f"[buffer][error] uid={self.session.sid} err=buffer", flush=True)
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



