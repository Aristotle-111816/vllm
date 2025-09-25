"""Streaming session/state management for audio-video inputs.

This module intentionally excludes HTTP/base64 concerns. It operates on
decoded media only (PIL.Image and numpy.ndarray). Heavy dependencies are
imported lazily within methods to avoid import-time overhead.
"""

from dataclasses import dataclass
import threading
import time
from typing import List





@dataclass
class StreamingConfig:
    """Runtime configuration for streaming behavior.

    Attributes:
        target_sr: Target sampling rate for audio seconds pairing.
        max_decode_pairs: Max number of AV pairs to use at final decode.
        enable_prefill: Whether to perform streaming prefill during pairing.
        max_buffered_pairs: Max history length kept in memory per stream.
        prefill_every_n_pairs: Prefill cadence (every N pairs) when enabled.
        prefill_min_interval_s: Minimum seconds between prefill operations.
    """

    target_sr: int = 16000
    max_decode_pairs: int = 64
    enable_prefill: bool = False
    max_buffered_pairs: int = 128
    prefill_every_n_pairs: int = 1
    prefill_min_interval_s: float = 0.0

# Serialize engine generate calls across threads for timestampsafety
GEN_LOCK = threading.Lock()
class StreamingVideoSession:
    """Session for frame-by-frame AV prefill and final generation.

    This class stores decoded frames and 1s audio slices, provides prefill
    methods to reuse multimodal cache, and supports final text generation
    with either image-only or audio-video content.
    """

    def __init__(self, llm: object, tokenizer: object, session_id: str,
                 cfg: StreamingConfig):
        self.llm = llm
        self.tok = tokenizer
        self.sid = session_id
        self.cfg = cfg
        self.frames: List[object] = []
        self.audio_slices: List[object] = []

    @staticmethod
    def _build_uuid(session_id: str, frame_index: int) -> str:
        return f"{session_id}-img-{frame_index:06d}"

    @staticmethod
    def _build_audio_uuid(session_id: str, chunk_index: int) -> str:
        return f"{session_id}-aud-{chunk_index:06d}"

    @staticmethod
    def _split_audio_into_1s(wav: "object", sr: int) -> List["object"]:
        import numpy as np

        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if sr <= 0 or wav.size == 0:
            return []
        one_sec = int(sr)
        num_full = wav.size // one_sec
        return [wav[i * one_sec:(i + 1) * one_sec] for i in range(num_full)]

    def load_audio_slices_from_file(self, audio_path: str) -> int:
        """Load audio and slice into 1-second chunks at cfg.target_sr."""
        import librosa

        try:
            wav, _ = librosa.load(audio_path, sr=self.cfg.target_sr, mono=True)
            self.audio_slices = self._split_audio_into_1s(
                wav, self.cfg.target_sr)
        except Exception:
            self.audio_slices = []
        return len(self.audio_slices)

    def streaming_prefill_frame(self, frame_pil: "object", frame_index: int):
        from PIL import Image
        from vllm import SamplingParams

        frame_pil = Image.Image.convert(frame_pil, "RGB")
        self.frames.append(frame_pil)

        placeholder = "(<image>./</image>)"
        messages = [{"role": "user", "content": placeholder}]
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        uuid = self._build_uuid(self.sid, frame_index)
        payload = {
            "prompt": prompt,
            "multi_modal_data": {"image": [frame_pil]},
            "multi_modal_uuids": {"image": [uuid]},
        }
        sp = SamplingParams(max_tokens=1)
        self.llm.generate(payload, sampling_params=sp)

    def streaming_prefill_av_chunk(self, frame_pil: "object",
                                   audio_1s: "object", chunk_index: int,
                                   sr: int = 16000):
        from PIL import Image
        import numpy as np
        from vllm import SamplingParams

        frame_pil = Image.Image.convert(frame_pil, "RGB")
        self.frames.append(frame_pil)
        self.audio_slices.append(audio_1s)

        content = "(<image>./</image>)(<audio>./</audio>)"
        messages = [{"role": "user", "content": content}]
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        img_uuid = self._build_uuid(self.sid, chunk_index)
        aud_uuid = self._build_audio_uuid(self.sid, chunk_index)

        payload = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": [frame_pil],
                "audio": [(np.asarray(audio_1s, dtype=np.float32), sr)],
            },
            "multi_modal_uuids": {
                "image": [img_uuid],
                "audio": [aud_uuid],
            },
        }
        sp = SamplingParams(max_tokens=1, temperature=0.0, top_p=1.0)
        self.llm.generate(payload, sampling_params=sp)

    def streaming_generate(self, question: str, max_tokens: int = 512) -> str:
        if not self.frames:
            return ""
        n = len(self.frames)
        placeholder = "(<image>./</image>)"
        prefix = placeholder * n
        content = prefix + ("\n" + question if question else "")
        messages = [{"role": "user", "content": content}]
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        uuids = [self._build_uuid(self.sid, i) for i in range(n)]
        payload = {
            "prompt": prompt,
            "multi_modal_data": {"image": list(self.frames)},
            "multi_modal_uuids": {"image": uuids},
        }

        stop_tokens = ['<|im_end|>', '<|endoftext|>']
        stop_ids = [self.tok.convert_tokens_to_ids(i) for i in stop_tokens]
        from vllm import SamplingParams

        sp = SamplingParams(
            stop_token_ids=stop_ids,
            temperature=0.7,
            top_p=0.8,
            max_tokens=max_tokens,
        )
        out = self.llm.generate(payload, sampling_params=sp)
        return out[0].outputs[0].text

    def streaming_generate_av(self, question: str, max_tokens: int = 512,
                               sr: int = 16000) -> str:
        import numpy as np

        n_img = len(self.frames)
        n_aud = len(self.audio_slices)
        n = min(n_img, n_aud)
        if n == 0:
            return ""

        pairs = "".join(
            "(<image>./</image>)(<audio>./</audio>)" for _ in range(n)
        )
        content = pairs + ("\n" + question if question else "")
        messages = [{"role": "user", "content": content}]
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        img_uuids = [self._build_uuid(self.sid, i) for i in range(n)]
        aud_uuids = [self._build_audio_uuid(self.sid, i) for i in range(n)]

        payload = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": list(self.frames[:n]),
                "audio": [
                    (np.asarray(wav, dtype=np.float32), sr)
                    for wav in self.audio_slices[:n]
                ],
            },
            "multi_modal_uuids": {
                "image": img_uuids,
                "audio": aud_uuids,
            },
        }

        stop_tokens = ['<|im_end|>', '<|endoftext|>']
        stop_ids = [self.tok.convert_tokens_to_ids(i) for i in stop_tokens]
        from vllm import SamplingParams

        sp = SamplingParams(
            stop_token_ids=stop_ids,
            temperature=0.7,
            top_p=0.8,
            max_tokens=max_tokens,
        )
        out = self.llm.generate(payload, sampling_params=sp)
        return out[0].outputs[0].text


class SessionState:
    """State machine pairing audio seconds with frames, with prefill.

    This class performs AV pairing at 1-second granularity, optional prefill
    throttling, and history bounding based on the provided configuration.
    """

    def __init__(self, session: StreamingVideoSession, cfg: StreamingConfig):
        self.session = session
        self.cfg = cfg
        self.lock = threading.Lock()
        self.audio_buf = None  # Lazily initialized (numpy array)
        self.frame_queue: List[object] = []
        self.paired = 0
        self._last_prefill_ts = 0.0
        self._prefill_counter = 0

    def _ensure_audio_buf(self):
        if self.audio_buf is None:
            import numpy as np

            self.audio_buf = np.zeros((0,), dtype=np.float32)

    def _try_pair_and_prefill(self):
        import numpy as np

        one_sec = self.cfg.target_sr
        while self.audio_buf.shape[0] >= one_sec and len(self.frame_queue) > 0:
            audio_1s = self.audio_buf[:one_sec]
            self.audio_buf = self.audio_buf[one_sec:]
            frame = self.frame_queue.pop(0)

            do_prefill = False
            if self.cfg.enable_prefill:
                self._prefill_counter += 1
                if self._prefill_counter >= max(1, self.cfg.prefill_every_n_pairs):
                    now = time.time()
                    if ((now - self._last_prefill_ts) >=
                            max(0.0, self.cfg.prefill_min_interval_s)):
                        do_prefill = True
                        self._prefill_counter = 0
                        self._last_prefill_ts = now

            if do_prefill:
                try:
                    with GEN_LOCK:
                        self.session.streaming_prefill_av_chunk(
                            frame, audio_1s, chunk_index=self.paired,
                            sr=self.cfg.target_sr,
                        )
                except Exception:
                    try:
                        from PIL import Image

                        self.session.frames.append(
                            Image.Image.convert(frame, "RGB"))
                        self.session.audio_slices.append(np.asarray(audio_1s))
                    except Exception:
                        pass
                self.paired += 1
            else:
                try:
                    from PIL import Image

                    self.session.frames.append(
                        Image.Image.convert(frame, "RGB"))
                    self.session.audio_slices.append(np.asarray(audio_1s))
                except Exception:
                    pass
                self.paired += 1

            if len(self.session.frames) > self.cfg.max_buffered_pairs:
                self.session.frames = self.session.frames[-self.cfg
                                                          .max_buffered_pairs:]
            if len(self.session.audio_slices) > self.cfg.max_buffered_pairs:
                self.session.audio_slices = (
                    self.session.audio_slices[-self.cfg.max_buffered_pairs:])
            try:
                print(
                    f"[pair] uid={self.session.sid} paired_idx={self.paired} "
                    f"buf_sec={self.audio_buf.shape[0]/self.cfg.target_sr:.2f} "
                    f"frames_q={len(self.frame_queue)}",
                    flush=True,
                )
            except Exception:
                pass

    def ingest_image(self, img: "object"):
        with self.lock:
            self.frame_queue.append(img)
            if self.audio_buf is None:
                self._ensure_audio_buf()
            self._try_pair_and_prefill()

    def ingest_audio(self, wav_f32: "object"):
        if getattr(wav_f32, "size", 0) == 0:
            return
        import numpy as np

        with self.lock:
            if self.audio_buf is None:
                self._ensure_audio_buf()
            self.audio_buf = np.concatenate([self.audio_buf, wav_f32])
            self._try_pair_and_prefill()


class SessionManager:
    """Manager for per-UID SessionState instances."""

    def __init__(self, llm: object, tok: object, cfg: StreamingConfig):
        self.llm = llm
        self.tok = tok
        self.cfg = cfg
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def get(self, uid: str) -> SessionState:
        if not uid:
            raise ValueError("Missing uid header")
        with self._lock:
            st = self._sessions.get(uid)
            if st is None:
                sess = StreamingVideoSession(self.llm, self.tok, uid, self.cfg)
                st = SessionState(sess, self.cfg)
                self._sessions[uid] = st
            return st

    def reset(self, uid: str):
        with self._lock:
            if uid in self._sessions:
                del self._sessions[uid]


