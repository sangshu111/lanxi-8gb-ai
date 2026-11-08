"""语音合成 CLI（GPT-SoVITS v2Pro + 后期链）

环境变量：
  GSV_REPO      GPT-SoVITS 仓库根目录
  GSV_REF_WAV   参考音频（4~8 秒，必须是你想模仿的声音）
  GSV_REF_TEXT  参考音频对应的文字（用 Whisper 转写，必填）
  FTLANG_CACHE  fast_langdetect 缓存目录（需含 lid.176.bin）

用法：
  python voice-say.py --text "主人，你回来啦。" --out out.wav
"""
import argparse
import os

REPO = os.environ.get("GSV_REPO", "")
REF_WAV = os.environ.get("GSV_REF_WAV", "")
REF_TEXT = os.environ.get("GSV_REF_TEXT", "")


def denoise(y, sr, strength=0.7):
    """STFT 谱减降噪，压掉气声/磨砂感"""
    import numpy as np
    import librosa
    from scipy.ndimage import uniform_filter

    S = librosa.stft(y, n_fft=1024, hop_length=256)
    mag, phase = np.abs(S), np.angle(S)
    noise = np.percentile(mag, 12, axis=1, keepdims=True)
    mask = np.clip((mag - strength * noise) / np.maximum(mag, 1e-8), 0.0, 1.0)
    mask = uniform_filter(mask, size=(3, 3))
    return librosa.istft(mag * mask * np.exp(1j * phase), hop_length=256, length=len(y))


def compress(y, sr, thresh_db=-20.0, ratio=3.0, attack_ms=4, release_ms=120, makeup_db=5.0):
    """动态压缩：解决"有气无力"（只做峰值归一是不够的）"""
    import numpy as np

    env = np.abs(y)
    a_att = float(np.exp(-1.0 / (sr * attack_ms / 1000.0)))
    a_rel = float(np.exp(-1.0 / (sr * release_ms / 1000.0)))
    out = np.zeros_like(env)
    prev = 0.0
    for i, v in enumerate(env):
        c = a_att if v > prev else a_rel
        prev = c * prev + (1 - c) * v
        out[i] = prev
    e_db = 20 * np.log10(np.maximum(out, 1e-6))
    over = np.maximum(0.0, e_db - thresh_db)
    gain_db = -over * (1 - 1 / ratio) + makeup_db
    return (y * (10 ** (gain_db / 20))).astype(np.float32)


def polish(y, sr, semitones=0.0, target_dbfs=-16.0, shelf_db=1.2, shelf_hz=3200, lpf_hz=11000):
    """变调（保时长）+ EQ + RMS 归一 + 软限幅"""
    import numpy as np
    import librosa

    if semitones:
        # 注意：必须用相位声码器，重采样会同时把时长缩短
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
    spec = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
    shelf = 1.0 + (10 ** (shelf_db / 20) - 1.0) / (1.0 + (shelf_hz / np.maximum(freqs, 1)) ** 2)
    lpf = 1.0 / (1.0 + (freqs / lpf_hz) ** 4)
    y = np.fft.irfft(spec * shelf * lpf, n=len(y)).astype(np.float32)
    rms = float(np.sqrt((y ** 2).mean()))
    y = y * (10 ** (target_dbfs / 20) / max(rms, 1e-9))
    y = np.tanh(y * 1.15) / np.tanh(1.15)     # 软限幅，硬削波会把响度压回去
    peak = float(np.abs(y).max())
    if peak > 0.95:
        y = y * (0.95 / peak)
    return y.astype(np.float32)


def main():
    import sys
    import time
    import numpy as np
    import soundfile as sf

    if not REPO or not os.path.isdir(REPO):
        raise SystemExit("请设置 GSV_REPO 指向 GPT-SoVITS 仓库")
    if not REF_WAV or not REF_TEXT:
        raise SystemExit("请设置 GSV_REF_WAV 和 GSV_REF_TEXT")

    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))
    os.chdir(REPO)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--semitones", type=float, default=0.0)
    ap.add_argument("--speed", type=float, default=0.97)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

    t0 = time.time()
    cfg = TTS_Config(os.path.join(REPO, "GPT_SoVITS", "configs", "tts_infer.yaml"))
    cfg.device = "cuda"
    cfg.is_half = True
    cfg.version = "v2Pro"          # 比 v2 清楚得多
    tts = TTS(cfg)
    print(f"模型加载 {time.time()-t0:.1f}s")

    req = {
        "text": args.text, "text_lang": "zh",
        "ref_audio_path": REF_WAV,
        "prompt_text": REF_TEXT,   # 不给这行会沙哑 + 大舌头
        "prompt_lang": "zh",
        "top_k": 16, "top_p": 1.0, "temperature": 1.0,
        "text_split_method": "cut0",   # cut5 会让句尾逐段下沉
        "batch_size": 1, "speed_factor": args.speed,
        "ref_free": False, "seed": args.seed,
        "parallel_infer": True, "repetition_penalty": 1.3,
    }
    for sr, audio in tts.run(req):
        y = np.asarray(audio, dtype=np.float32)
        y = compress(denoise(y, sr), sr)
        y = polish(y, sr, semitones=args.semitones)
        sf.write(args.out, y, sr)
        print(f"saved {args.out}  {len(y)/sr:.2f}s  rms={20*np.log10(max(float(np.sqrt((y**2).mean())),1e-9)):.1f}dB")
        break


if __name__ == "__main__":
    main()
