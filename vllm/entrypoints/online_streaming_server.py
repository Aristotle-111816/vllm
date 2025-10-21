import argparse
import traceback
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn
from openai import OpenAI

from vllm.multimodal.image import (
    decode_image_b64 as _decode_image_b64,
)
from vllm.multimodal.audio import (
    decode_wav_b64_to_float32 as _decode_wav_b64_to_float32,
)
import vllm.multimodal.online_streaming_session as ss
from vllm.multimodal.online_streaming_session import SessionManager, GEN_LOCK


class OnlineStreamingHTTPServer:
    """HTTP server bridging frontend streaming AV to OpenAI Chat API backend."""

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model_id: str,
        *,
        enable_prefill: bool = ss.ENABLE_PREFILL,
        prefill_every_n: int = ss.PREFILL_EVERY_N_PAIRS,
        prefill_min_interval_s: float = ss.PREFILL_MIN_INTERVAL_S,
        max_buffered_pairs: int = ss.MAX_BUFFERED_PAIRS,
        max_decode_pairs: int = ss.MAX_DECODE_PAIRS,
        stop_token_ids: Optional[List[int]] = None,
        mm_processor_kwargs: Optional[dict] = None,
        sr: int = ss.TARGET_SR,
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

        # Sync knobs to session module (preserve previous behavior)
        ss.ENABLE_PREFILL = self.enable_prefill
        ss.PREFILL_EVERY_N_PAIRS = self.prefill_every_n_pairs
        ss.PREFILL_MIN_INTERVAL_S = self.prefill_min_interval_s
        ss.MAX_BUFFERED_PAIRS = self.max_buffered_pairs
        ss.MAX_DECODE_PAIRS = self.max_decode_pairs
        ss.TARGET_SR = self.sr

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
                wav = _decode_wav_b64_to_float32(a_b64, target_sr=ss.TARGET_SR)
                st.ingest_audio(wav)
                return JSONResponse({"status": "ok", "paired": st.paired})

            raise HTTPException(status_code=400, detail=f"Unsupported content type: {ctype}")

        @app.post("/api/v1/generate")
        async def generate_endpoint(payload: dict, uid: Optional[str] = Header(None)):
            """Trigger final text generation using cached paired AV chunks."""
            st = mgr.get(uid)
            question = payload.get("question") or (
                "Please provide a brief summary of the audio content."
            )
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
                        n_use = min(n_img, ss.MAX_DECODE_PAIRS)
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

                    n_use = min(n, ss.MAX_DECODE_PAIRS)
                    frames_seg = list(st.session.frames[-n_use:])
                    audios_seg = [
                        wav.astype(np.float32)
                        for wav in st.session.audio_slices[-n_use:]
                    ]
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
                    print(
                        f"[generate][error] uid={uid} err={e}\n"
                        f"{traceback.format_exc()}",
                        flush=True,
                    )
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
                buf_sec = st.audio_buf.shape[0] / ss.TARGET_SR
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
    enable_prefill: bool = ss.ENABLE_PREFILL,
    prefill_every_n: int = ss.PREFILL_EVERY_N_PAIRS,
    prefill_min_interval_s: float = ss.PREFILL_MIN_INTERVAL_S,
    max_buffered_pairs: int = ss.MAX_BUFFERED_PAIRS,
    max_decode_pairs: int = ss.MAX_DECODE_PAIRS,
    stop_token_ids: Optional[List[int]] = None,
    mm_processor_kwargs: Optional[dict] = None,
    sr: int = ss.TARGET_SR,
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
    parser = argparse.ArgumentParser(
        description=(
            "HTTP uplink server for online streaming AV via OpenAI Chat API"
        )
    )
    parser.add_argument(
        "--api-base",
        type=str,
        required=True,
        help="OpenAI base URL, e.g., http://localhost:8000/v1",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="API key for the server",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path or HF ID registered by server",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=32551)
    parser.add_argument("--sr", type=int, default=ss.TARGET_SR, help="Target audio sample rate for processing")
    # Prefill/Decode knobs
    parser.add_argument(
        "--enable-prefill",
        action="store_true",
        help="Enable streaming prefill on paired AV chunks",
    )
    parser.add_argument(
        "--prefill-every-n",
        type=int,
        default=ss.PREFILL_EVERY_N_PAIRS,
        help="Prefill every N pairs",
    )
    parser.add_argument(
        "--prefill-min-interval",
        type=float,
        default=ss.PREFILL_MIN_INTERVAL_S,
        help="Min seconds between prefill calls",
    )
    parser.add_argument(
        "--max-buffered-nuse",
        type=int,
        default=ss.MAX_BUFFERED_PAIRS,
        help="Cap buffered AV pairs history",
    )
    parser.add_argument(
        "--max-decode-uuidnum",
        type=int,
        default=ss.MAX_DECODE_PAIRS,
        help="Cap pairs used in final decode",
    )
    # Extra decode controls
    parser.add_argument(
        "--stop-token-ids",
        type=str,
        default=None,
        help="Comma-separated stop token ids",
    )
    parser.add_argument(
        "--mm-processor-kwargs",
        type=str,
        default=None,
        help="JSON string for mm processor kwargs",
    )

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



