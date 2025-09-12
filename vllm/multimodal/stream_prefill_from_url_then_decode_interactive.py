import os
import sys
import time
import threading
import queue
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.video import start_url_stream


def bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


class FrameProducer(threading.Thread):
    """Wrapper thread that delegates to shared URL streaming helpers."""

    def __init__(self, out_q: queue.Queue, fps: int, video_url: str,
                 frame_size: tuple[int, int]):
        super().__init__(daemon=True)
        self.q = out_q
        self.fps = fps
        self.video_url = video_url
        self.frame_size = frame_size
        self._stop = threading.Event()
        self._idx = 0

    def run(self):
        # Reuse start_url_stream helper from video.py
        t, q = start_url_stream(self.video_url, self.fps,
                                self.frame_size[0], self.frame_size[1],
                                queue_size=1)
        try:
            while not self._stop.is_set():
                try:
                    idx, frame = q.get(timeout=0.1)
                    if self.q.full():
                        try:
                            self.q.get_nowait()
                        except queue.Empty:
                            pass
                    self.q.put((idx, frame))
                except queue.Empty:
                    pass
        finally:
            t.stop()

    def stop(self):
        self._stop.set()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_url", type=str, required=True,
                        help="Public/Local URL of MJPEG/HTTP video stream.")
    parser.add_argument("--fps", type=int, default=int(os.environ.get("STREAM_FPS", "1")))
    parser.add_argument("--width", type=int, default=int(os.environ.get("FRAME_W", "320")))
    parser.add_argument("--height", type=int, default=int(os.environ.get("FRAME_H", "320")))
    parser.add_argument("--model", type=str, default=os.environ.get("MODEL_NAME", "/root/autodl-tmp/MiniCPM-o-2_6"))
    parser.add_argument("--max_model_len", type=int, default=int(os.environ.get("MAX_MODEL_LEN", "8192")))
    parser.add_argument("--session", type=str, default=os.environ.get("SESSION_ID", "sessA"))
    parser.add_argument("--prefill_max_tokens", type=int, default=int(os.environ.get("PREFILL_MAX_TOKENS", "1")))
    parser.add_argument("--window_size", type=int, default=0,
                        help="Optional sliding window over tail frames; 0 disables.")
    parser.add_argument("--gpu_util", type=float, default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.9")))
    args, _ = parser.parse_known_args()

    os.environ.setdefault("VLLM_USE_V1", "1")
    os.environ.setdefault("VLLM_GPU_MEMORY_UTILIZATION", str(args.gpu_util))

    # Init tokenizer & LLM
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        disable_mm_preprocessor_cache=False,
    )

    # Build model-aware multimodal processor
    tg = llm.llm_engine.get_tokenizer_group()
    mm_processor = MULTIMODAL_REGISTRY.create_processor(
        llm.llm_engine.model_config,
        tokenizer=tg,
    )

    # Stop ids
    stop_tokens = ['<|im_end|>', '<|endoftext|>']
    stop_token_ids = []
    for tok in stop_tokens:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                stop_token_ids.append(tid)
        except Exception:
            pass
    if getattr(tokenizer, 'eos_token_id', None) is not None:
        stop_token_ids.append(int(tokenizer.eos_token_id))

    # Frame producer
    q: "queue.Queue[tuple[int, np.ndarray]]" = queue.Queue(maxsize=1)
    producer = FrameProducer(q, args.fps, args.video_url, (args.width, args.height))
    producer.start()

    print(f"Streaming from {args.video_url} at {args.fps} fps.")
    print("Type a question to trigger decode; 'q' to quit. Prefill runs continuously.")

    frames: List[Image.Image] = []
    window_size: Optional[int] = args.window_size if args.window_size > 0 else None

    try:
        while True:
            # Try to fetch the freshest frame and prefill
            try:
                idx, frame_bgr = q.get(timeout=1.0)
                frame_pil = bgr_to_pil(frame_bgr)
                frames.append(frame_pil)
                payload = mm_processor.build_streaming_images_prefill_payload(
                    images=frames,
                    session_id=args.session,
                    window_size=window_size,
                )

                # Wrap prompt with chat template
                eff_images = len(payload["multi_modal_data"]["image"])
                messages = [{"role": "user", "content": "(<image>./</image>)" * eff_images}]
                payload["prompt"] = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)

                prefill_params = SamplingParams(
                    max_tokens=max(1, args.prefill_max_tokens),
                    temperature=0.0,
                    top_p=1.0,
                    stop_token_ids=stop_token_ids,
                )
                _ = llm.generate([payload], sampling_params=prefill_params)
                print(f"[prefill] frames={len(frames)} last_idx={idx}")
            except queue.Empty:
                pass

            # Non-blocking check for user input
            if sys.stdin in select_readable(0.01):
                line = sys.stdin.readline()
                if not line:
                    continue
                prompt = line.strip()
                if not prompt:
                    continue
                if prompt.lower() == 'q':
                    break

                # Build decode payload with current window
                decode_payload = mm_processor.build_streaming_images_decode_payload(
                    images=frames,
                    user_text=prompt,
                    session_id=args.session,
                    window_size=window_size,
                )
                eff_images = len(decode_payload["multi_modal_data"]["image"])
                content = ("(<image>./</image>)" * eff_images) + "\n" + prompt
                messages = [{"role": "user", "content": content}]
                decode_payload["prompt"] = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)

                decode_params = SamplingParams(
                    stop_token_ids=stop_token_ids,
                    temperature=0.2,
                    top_p=0.95,
                    max_tokens=512,
                )
                outputs = llm.generate([decode_payload], sampling_params=decode_params)
                print("==== Generated ====")
                print(outputs[0].outputs[0].text)

    finally:
        producer.stop()
        producer.join(timeout=1.0)
        print("Bye.")


def select_readable(timeout_sec: float):
    """Return a list of readable fds; used for non-blocking stdin poll."""
    import select
    rlist, _, _ = select.select([sys.stdin], [], [], timeout_sec)
    return rlist


if __name__ == "__main__":
    sys.exit(main())


