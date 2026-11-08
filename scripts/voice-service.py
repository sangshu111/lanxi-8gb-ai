"""按需语音服务：用的时候启动，空闲自动退出（不占显存）

环境变量：
  GSV_REPO      GPT-SoVITS 仓库根目录
  GSV_REF_WAV   参考音频
  GSV_REF_TEXT  参考音频文字
  FTLANG_CACHE  fast_langdetect 缓存目录

接口：
  POST /say    {"text":"...", "semitones":0, "speed":0.97} -> {"ok":true,"path":"..."}
  GET  /health -> {"ok":true,"idle_left":123}

启动：
  python voice-service.py --idle 300 --port 9880
"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.environ.get("GSV_REPO", "")
REF_WAV = os.environ.get("GSV_REF_WAV", "")
REF_TEXT = os.environ.get("GSV_REF_TEXT", "")
OUT_DIR = os.environ.get("LANXI_OUT", "./voice_out")

LAST_ACTIVE = time.time()
IDLE = 300
LOCK = threading.Lock()
TTS_INSTANCE = {"tts": None}


def get_tts():
    if TTS_INSTANCE["tts"] is None:
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

        cfg = TTS_Config(os.path.join(REPO, "GPT_SoVITS", "configs", "tts_infer.yaml"))
        cfg.device = "cuda"
        cfg.is_half = True
        cfg.version = "v2Pro"
        TTS_INSTANCE["tts"] = TTS(cfg)
    return TTS_INSTANCE["tts"]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        global LAST_ACTIVE
        if self.path.startswith("/health"):
            LAST_ACTIVE = time.time()
            self._json(200, {"ok": True, "loaded": TTS_INSTANCE["tts"] is not None,
                             "idle_left": int(IDLE)})
            return
        self._json(404, {"ok": False})

    def do_POST(self):
        global LAST_ACTIVE
        if not self.path.startswith("/say"):
            return self._json(404, {"ok": False})
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._json(400, {"ok": False, "error": f"bad json {e}"})

        text = (req.get("text") or "").strip()
        if not text:
            return self._json(400, {"ok": False, "error": "empty text"})

        import numpy as np
        import soundfile as sf
        import sys as _sys

        _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from importlib import import_module
        mod = import_module("voice-say".replace("-", "_")) if False else None

        t0 = time.time()
        try:
            with LOCK:
                tts = get_tts()
                payload = {
                    "text": text, "text_lang": "zh",
                    "ref_audio_path": REF_WAV, "prompt_text": REF_TEXT, "prompt_lang": "zh",
                    "top_k": 16, "top_p": 1.0, "temperature": 1.0,
                    "text_split_method": "cut0", "batch_size": 1,
                    "speed_factor": float(req.get("speed", 0.97)),
                    "ref_free": False, "seed": int(req.get("seed", 5)),
                    "parallel_infer": True, "repetition_penalty": 1.3,
                }
                got = None
                for sr, audio in tts.run(payload):
                    got = (sr, np.asarray(audio, dtype=np.float32))
                    break
            if not got:
                return self._json(500, {"ok": False, "error": "no audio"})

            sr, y = got
            y = y * (0.95 / (float(np.abs(y).max()) or 1.0))
            os.makedirs(OUT_DIR, exist_ok=True)
            out = os.path.join(OUT_DIR, f"tts_{int(time.time()*1000)}.wav")
            sf.write(out, y, sr)
            LAST_ACTIVE = time.time()
            self._json(200, {"ok": True, "path": os.path.abspath(out),
                             "elapsed": round(time.time() - t0, 2)})
        except Exception as e:
            self._json(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})


def idle_watchdog():
    while True:
        time.sleep(2)
        if time.time() - LAST_ACTIVE > IDLE:
            print(f"[voice] 空闲 {IDLE}s，退出释放显存", flush=True)
            os._exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--idle", type=int, default=300)
    ap.add_argument("--port", type=int, default=9880)
    args = ap.parse_args()
    IDLE = args.idle

    if not REPO or not REF_WAV or not REF_TEXT:
        raise SystemExit("请设置 GSV_REPO / GSV_REF_WAV / GSV_REF_TEXT")

    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))
    os.chdir(REPO)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    print("[voice] 预热模型...", flush=True)
    t0 = time.time()
    get_tts()
    print(f"[voice] 就绪 {time.time()-t0:.1f}s，监听 127.0.0.1:{args.port}，空闲 {IDLE}s 退出", flush=True)
    LAST_ACTIVE = time.time()
    threading.Thread(target=idle_watchdog, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
