"""出图：SDXL + 角色 LoRA（8GB 显存优化版）

环境变量：
  LANXI_CKPT  底模 .safetensors 路径（必填）
  LANXI_LORA  LoRA 路径（可选）
  LANXI_OUT   输出目录（默认 ./output）
"""
import os

CKPT = os.environ.get("LANXI_CKPT", "")
LORA = os.environ.get("LANXI_LORA", "")
OUT_DIR = os.environ.get("LANXI_OUT", "./output")

PROMPT = ("clean manga style, cel shading, crisp lineart, matte, "
          "1girl, solo, navy blue hair, cyan hair tips, ahoge, whale fin ears, "
          "white lace maid headdress, deep blue eyes, "
          "white victorian maid dress, white apron, white opaque tights, "
          "white background, full body, masterpiece, best quality")
NEG = ("glossy, shiny skin, oily, bloom, airbrushed, 3d, painterly, plastic, "
       "room, furniture, colored background, multiple views, "
       "bad anatomy, extra legs, bad hands, text, watermark, lowres, blurry")


def main():
    import time
    import torch
    from diffusers import StableDiffusionXLPipeline

    if not CKPT or not os.path.exists(CKPT):
        raise SystemExit("请设置 LANXI_CKPT 指向底模 .safetensors")
    os.makedirs(OUT_DIR, exist_ok=True)

    t0 = time.time()
    pipe = StableDiffusionXLPipeline.from_single_file(
        CKPT, torch_dtype=torch.float16, use_safetensors=True)
    # 8GB 显存必须开这三个，否则溢出到共享内存：25 秒会变 320 秒
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.set_progress_bar_config(disable=True)

    if LORA and os.path.exists(LORA):
        pipe.load_lora_weights(LORA, adapter_name="char")
        pipe.set_adapters(["char"], adapter_weights=[0.7])
    print(f"加载 {time.time()-t0:.1f}s", flush=True)

    g = torch.Generator("cpu").manual_seed(42)
    t = time.time()
    img = pipe(prompt=PROMPT, negative_prompt=NEG, num_inference_steps=28,
               guidance_scale=6.5, width=832, height=1216, generator=g).images[0]
    p = os.path.join(OUT_DIR, "out.png")
    img.save(p)
    print(f"出图 {time.time()-t:.1f}s -> {p}")
    print(f"显存峰值 {torch.cuda.max_memory_allocated()/2**30:.2f} GB")


if __name__ == "__main__":
    main()
