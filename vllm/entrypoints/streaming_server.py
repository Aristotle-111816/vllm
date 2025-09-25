import argparse
import traceback
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from vllm.multimodal.streaming import (
    StreamingConfig,
    SessionManager,
    GEN_LOCK,
)
from vllm.multimodal.image import _decode_image_b64
from vllm.multimodal.audio import _decode_wav_b64_to_float32


class StreamingHTTPServer:
    """HTTP server wrapping FastAPI routes for AV streaming."""

    def __init__(self, model_id: str, cfg: StreamingConfig):
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.llm = LLM(
            model=model_id,
            max_model_len=32768,
            trust_remote_code=True,
            disable_mm_preprocessor_cache=False,
            limit_mm_per_prompt={"image": 256},
        )
        self.mgr = SessionManager(self.llm, self.tok, cfg)
        self.app = FastAPI()
        self._register_routes()

    def _register_routes(self):
        app = self.app
        mgr = self.mgr
        cfg = self.cfg

        @app.post("/api/v1/stream")
        async def stream_endpoint(request: Request, uid: Optional[str] = Header(None)):
            body = await request.json()
            try:
                content = body["messages"][0]["content"][0]
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid message format")

            st = mgr.get(uid)
            ctype = content.get("type")

            if ctype == "image_data":
                img_b64 = content["image_data"]["data"]
                img = _decode_image_b64(img_b64)
                st.ingest_image(img)
                return JSONResponse({"status": "ok", "paired": st.paired})

            if ctype == "input_audio":
                a_b64 = content["input_audio"]["data"]
                wav = _decode_wav_b64_to_float32(a_b64, target_sr=cfg.target_sr)
                st.ingest_audio(wav)
                return JSONResponse({"status": "ok", "paired": st.paired})

            raise HTTPException(status_code=400, detail=f"Unsupported content type: {ctype}")

        @app.post("/api/v1/generate")
        async def generate_endpoint(payload: dict, uid: Optional[str] = Header(None)):
            st = mgr.get(uid)
            question = payload.get("question") or (
                "Please provide a brief summary of the audio content."
            )

            try:
                with st.lock:
                    n_img = len(st.session.frames)
                    n_aud = len(st.session.audio_slices)
                    n = min(n_img, n_aud)
                    if n == 0:
                        return JSONResponse({"text": "", "detail": "no paired AV chunks"})
                    n_use = min(n, cfg.max_decode_pairs)
                    start = max(0, n - n_use)

                    pairs = "".join(
                        "(<image>./</image>)(<audio>./</audio>)" for _ in range(n_use)
                    )
                    content = pairs + ("\n" + question if question else "")
                    messages = [{"role": "user", "content": content}]
                    prompt = st.session.tok.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    img_uuids = [
                        st.session._build_uuid(st.session.sid, (st.paired - n) + start + i)
                        for i in range(n_use)
                    ]
                    aud_uuids = [
                        st.session._build_audio_uuid(st.session.sid, (st.paired - n) + start + i)
                        for i in range(n_use)
                    ]

                    payload_gen = {
                        "prompt": prompt,
                        "multi_modal_data": {
                            "image": list(st.session.frames[start:n]),
                            "audio": [
                                (wav, cfg.target_sr) for wav in st.session.audio_slices[start:n]
                            ],
                        },
                        "multi_modal_uuids": {
                            "image": img_uuids,
                            "audio": aud_uuids,
                        },
                    }

                    stop_tokens = ['<|im_end|>', '<|endoftext|>']
                    stop_ids = [st.session.tok.convert_tokens_to_ids(i) for i in stop_tokens]
                    sp = SamplingParams(
                        stop_token_ids=stop_ids,
                        temperature=0.7,
                        top_p=0.8,
                        max_tokens=512,
                    )
                    with GEN_LOCK:
                        out = st.session.llm.generate(payload_gen, sampling_params=sp)
                    text = out[0].outputs[0].text
                    return JSONResponse({"text": text})
            except Exception as e:
                return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)

        @app.post("/api/v1/reset")
        async def reset_endpoint(payload: dict, uid: Optional[str] = Header(None)):
            mgr.reset(uid)
            return JSONResponse({"status": "reset"})

        @app.get("/api/v1/status")
        async def status_endpoint(uid: Optional[str] = Header(None)):
            st = mgr.get(uid)
            with st.lock:
                n_img = len(st.session.frames)
                n_aud = len(st.session.audio_slices)
                buf_sec = 0.0
                if st.audio_buf is not None:
                    buf_sec = st.audio_buf.shape[0] / cfg.target_sr
                q = len(st.frame_queue)
            return JSONResponse({
                "paired": st.paired,
                "frames_cached": n_img,
                "audio_slices_cached": n_aud,
                "audio_buffer_sec": round(buf_sec, 3),
                "frames_in_queue": q,
            })


def build_app(model_id: str, cfg_overrides: dict | None = None) -> FastAPI:
    cfg = StreamingConfig()
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    server = StreamingHTTPServer(model_id, cfg)
    return server.app


def main():
    parser = argparse.ArgumentParser(description="HTTP streaming server for AV")
    parser.add_argument("--model", type=str, required=True, help="HF model ID or local path")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=32550)
    parser.add_argument("--enable-prefill", action="store_true")
    parser.add_argument("--prefill-every-n", type=int, default=1)
    parser.add_argument("--prefill-min-interval", type=float, default=0.0)
    parser.add_argument("--max-buffered-nuse", type=int, default=128)
    parser.add_argument("--max-decode-uuidnum", type=int, default=64)
    parser.add_argument("--target-sr", type=int, default=16000)
    args = parser.parse_args()

    cfg = StreamingConfig(
        target_sr=args.target_sr,
        max_decode_pairs=args.max_decode_uuidnum,
        enable_prefill=bool(args.enable_prefill),
        max_buffered_pairs=args.max_buffered_nuse,
        prefill_every_n_pairs=max(1, int(args.prefill_every_n)),
        prefill_min_interval_s=max(0.0, float(args.prefill_min_interval)),
    )
    app = StreamingHTTPServer(args.model, cfg).app
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()


