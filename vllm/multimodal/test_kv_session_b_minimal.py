import os
import sys
import time
import argparse
from typing import List

import cv2
import numpy as np
from PIL import Image
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams


def bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str)
    parser.add_argument("--seconds_to_prefill", "-s", type=int, default=5)
    parser.add_argument("--model", type=str, default=os.environ.get("MODEL_NAME", "/root/autodl-tmp/MiniCPM-o-2_6"))
    parser.add_argument("--session", type=str, default=os.environ.get("SESSION_ID", "sessA"))
    parser.add_argument("--max_model_len", type=int, default=int(os.environ.get("MAX_MODEL_LEN", "8192")))
    parser.add_argument("--prefill_max_tokens", type=int, default=int(os.environ.get("PREFILL_MAX_TOKENS", "1")))
    parser.add_argument("--window_size", type=int, default=0)
    parser.add_argument("--question", "-q", type=str, default="Please describe what happens across these frames.")
    args, _ = parser.parse_known_args()

    os.environ.setdefault("VLLM_USE_V1", "1")

    # Init LLM
    _ = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, max_model_len=args.max_model_len, trust_remote_code=True,
              disable_mm_preprocessor_cache=False)

    window_size = args.window_size if args.window_size > 0 else None
    llm.llm_engine.kv_session_begin(args.session, window_size=window_size)

    # read 1fps
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(fps), 1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_steps = min(args.seconds_to_prefill, int(total / step) if total > 0 else args.seconds_to_prefill)

    for sec in range(max_steps):
        cap.set(cv2.CAP_PROP_POS_FRAMES, sec * step)
        ok, frame = cap.read()
        if not ok:
            break
        pil = bgr_to_pil(frame)
        llm.llm_engine.kv_session_append_images(args.session, [pil], prefill_max_tokens=args.prefill_max_tokens)
        print(f"[kv-session prefill] sec={sec}")

    cap.release()

    # decode
    llm.llm_engine.kv_session_decode(args.session, args.question,
                                     sampling_params=SamplingParams(max_tokens=512, temperature=0.2, top_p=0.95))

    # Poll until all requests finished; track decode request text progressively
    decode_req_id = f"{args.session}-decode"
    final_text = ""
    t0 = time.time()
    timeout_s = 180
    while llm.llm_engine.has_unfinished_requests():
        step_out = llm.llm_engine.step()
        if step_out:
            for ro in step_out:
                if getattr(ro, "request_id", None) == decode_req_id:
                    try:
                        if ro.outputs and ro.outputs[0].text is not None:
                            final_text = ro.outputs[0].text
                    except Exception:
                        pass
        if time.time() - t0 > timeout_s:
            print("[WARN] decode timeout waiting for completion")
            break
        time.sleep(0.02)

    print("==== Generated ====")
    print(final_text if final_text else "<empty>")


if __name__ == "__main__":
    sys.exit(main())


