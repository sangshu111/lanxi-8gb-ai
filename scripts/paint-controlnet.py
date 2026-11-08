"""出图：SDXL + LoRA + ControlNet（姿势可控）

环境变量：
  LANXI_CKPT     底模 .safetensors
  LANXI_LORA     LoRA .safetensors
  LANXI_CN_DIR   ControlNet 模型目录（含 config.json + diffusion_pytorch_model.safetensors）
  LANXI_POSE     骨架图路径（OpenPose 彩色骨架）
  LANXI_OUT      输出目录
"""
import os

CKPT = os.environ.get("LANXI_CKPT", "")
LORA = os.environ.get("LANXI_LORA", "")
CN_DIR = os.environ.get("LANXI_CN_DIR", "")
POSE = os.environ.get("LANXI_POSE", "")
OUT_DIR = os.environ.get("LANXI_OUT", "./output")
W, H = 768, 1120

PROMPT = ("kneeling on one knee, clean manga style, cel shading, crisp lineart, matte, "
          "1girl, solo, navy blue hair, cyan hair tips, ahoge, whale fin ears, "
          "white lace maid headdress, deep blue eyes, "
          "white victorian maid dress, white apron, white opaque tights, "
          "single whale tail fin behind skirt, white background, full body, masterpiece")
NEG = ("standing, straight legs, frame, border, colored background, room, furniture, "
       "glossy, shiny skin, oily, bloom, 3d, painterly, plastic, "
       "red eyes, big eyes, missing ears, no ahoge, multiple views, "
       "whole whale, plush, extra legs, bad anatomy, bad hands, text, watermark, lowres")


def main():
    import time
    import torch
    from PIL import Image
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

    for name, val in (("LANXI_CKPT", CKPT), ("LANXI_CN_DIR", CN_DIR), ("LANXI_POSE", POSE)):
        if not val or not os.path.exists(val):
            raise SystemExit(f"请设置 {name}")
    os.makedirs(OUT_DIR, exist_ok=True)

    t0 = time.time()
    cn = ControlNetModel.from_pretrained(CN_DIR, torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_single_file(
        CKPT, controlnet=cn, torch_dtype=torch.float16, use_safetensors=True)
    # ControlNet + SDXL 在 8GB 上必须用 sequential offload，否则直接 OOM
    pipe.enable_sequential_cpu_offload()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.set_progress_bar_config(disable=True)
    if LORA and os.path.exists(LORA):
        pipe.load_lora_weights(LORA, adapter_name="char")
        pipe.set_adapters(["char"], adapter_weights=[0.66])
    print(f"加载 {time.time()-t0:.1f}s", flush=True)

    pose = Image.open(POSE).convert("RGB").resize((W, H))
    g = torch.Generator("cpu").manual_seed(42)
    t = time.time()
    img = pipe(prompt=PROMPT, negative_prompt=NEG, image=pose,
               controlnet_conditioning_scale=0.95, num_inference_steps=26,
               guidance_scale=6.5, width=W, height=H, generator=g).images[0]
    p = os.path.join(OUT_DIR, "out_controlnet.png")
    img.save(p)
    print(f"出图 {time.time()-t:.1f}s -> {p}")
    print(f"显存峰值 {torch.cuda.max_memory_allocated()/2**30:.2f} GB")


if __name__ == "__main__":
    main()
