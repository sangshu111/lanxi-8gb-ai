# 蓝汐 AI 角色管线（Lanxi AI Pipeline）

<p align="center">![封面](assets/cover.png)</p>

> 在一台 **8GB 显存笔记本**上，从零搭出「会说话 + 会画画 + 有人格」的 AI 角色。
> 全部本地运行，零 API 费用。踩过的坑、实测数据、可复现脚本都在这里。

![状态](https://img.shields.io/badge/status-work--in--progress-yellow)
![显存](https://img.shields.io/badge/VRAM-8GB-blue)
![成本](https://img.shields.io/badge/API%20cost-0-green)

---

## 这个项目能做什么

| 能力 | 实现 | 实测性能 |
|---|---|---|
| **角色出图** | SDXL + 自训 LoRA + ControlNet | 25 秒/张（无 CN）· 2~4 分钟/张（带 CN） |
| **角色说话** | GPT-SoVITS v2Pro 零样本 + 后期链 | **1.5 秒/句**（常驻）· 34 秒（冷启动） |
| **姿势控制** | OpenPose ControlNet + 自绘骨架 | 任意姿势，画风不变 |
| **人格对话** | 本地桥接 + 大模型 + 提示词工程 | 实时 |
| **消息发送** | QQ 官方 Bot / NapCat | 语音、图片都能发 |

**没有用任何付费 API**——出图和语音全在本地显卡上跑。

---

## 硬件与实测资源占用

测试机：**神舟战神 S9**（i7-13620H / RTX 5070 Laptop **8GB** / 16GB DDR5）

| 任务 | 显存 | 内存 | 耗时 |
|---|---|---|---|
| SDXL 出图（832×1216，28 步） | **5.4 GB** | 4.2 GB | 25 秒 |
| SDXL + ControlNet（768×1120，26 步） | **7.8 GB** | 6.5 GB | 2~4 分钟 |
| GPT-SoVITS 语音（常驻） | **1.5 GB** | 4.2 GB | 1.5 秒/句 |
| 同时跑语音 + 出图 | ~7 GB | ~7 GB | 勉强够，建议错开 |

> **关键结论**：8GB 显存是硬瓶颈。出图必须开 CPU offload，否则溢出到共享内存，**25 秒变 320 秒**。

---

## 架构

```
┌─────────────────────────────────────────────────────┐
│                    输入层                            │
│   QQ 消息 / 命令行 / HTTP 请求                        │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│                  调度层（桥接）                        │
│   解析意图 → 人格提示词 → 调用大模型 → 分发任务         │
└───────┬─────────────────────────────────┬───────────┘
        ↓                                 ↓
┌───────────────────┐            ┌────────────────────┐
│   语音子系统       │            │    出图子系统       │
│ GPT-SoVITS v2Pro  │            │ SDXL + LoRA        │
│ 谱减降噪 + 压缩    │            │ + ControlNet 姿势   │
│ RMS 归一到 -16dBFS │            │ CPU offload 省显存  │
│ 按需启动 / 空闲退出 │            │ 文生图 / 图生图      │
└───────────────────┘            └────────────────────┘
        ↓                                 ↓
┌─────────────────────────────────────────────────────┐
│                   输出层                             │
│   QQ 语音消息 / 图片消息 / 本地文件                    │
└─────────────────────────────────────────────────────┘
```

---

## 快速开始

### 1. 环境准备

```bash
# Python 3.10+，推荐用 venv
python -m venv venv
venv\Scripts\activate

# 关键：RTX 50 系（Blackwell / sm_120）必须用 cu128
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> 国内下载慢？用 [阿里云镜像](https://mirrors.aliyun.com/pytorch-wheels/cu128/) 直接下 whl 再本地安装。

### 2. 出图

见 [docs/01-出图.md](docs/01-出图.md)

```bash
python scripts/paint-sdxl-8gb.py     # 基础出图
python scripts/paint-controlnet.py   # 姿势控制
```

### 3. 语音

见 [docs/02-语音.md](docs/02-语音.md)

```bash
# 常驻服务（推荐）
python scripts/voice-service.py --idle 300

# 或按需调用
node scripts/voice-client.mjs --voice shenhe --text "主人，你回来啦。"
```

### 4. 踩坑记录

**强烈建议先读** [docs/03-踩坑记录.md](docs/03-踩坑记录.md) —— 里面记了 20+ 个版本冲突和不报错的坑。

---

## 实测数据

### 出图质量对比

| 方案 | 角色还原 | 画风 | 备注 |
|---|---|---|---|
| 纯提示词（无 LoRA） | 60% | 底模风格 | 变形严重 |
| LoRA 0.85 + txt2img | 85% | 偏柔光 | 尾巴会变成整条鲸鱼 |
| **LoRA 0.66 + ControlNet** | **85%** | **干净漫画风** | 姿势可控 |
| LoRA 0.72 + img2img 0.45 | 90% | 最接近参考图 | 姿势难改 |

### 语音质量演进

| 版本 | 问题 | 解决 |
|---|---|---|
| v1 零样本 + 无参考文本 | 沙哑、大舌头 | ✅ 补 `prompt_text`（Whisper 转写） |
| v2 加了文本 | 咬字黏连 | ✅ 换 `v2Pro` 模型 |
| v3 换 v2Pro | 太慢（变调导致加速） | ✅ 用 `librosa.pitch_shift` 保时长 |
| v4 修好时长 | 有气无力（RMS -24dB） | ✅ 压缩 + RMS 归一到 -16dBFS |
| **v5 定稿** | — | 1.5 秒/句，响度稳定 |

---

## 目录结构

```
lanxi-8gb-ai/
├── README.md
├── docs/
│   ├── 01-出图.md          出图完整流程（LoRA 训练 + ControlNet）
│   ├── 02-语音.md          语音完整流程（数据准备 + 后期链）
│   ├── 03-踩坑记录.md      20+ 个坑
│   └── 04-发布指南.md      怎么发 GitHub / 做视频
├── scripts/
│   ├── paint-sdxl-8gb.py   8GB 显存优化出图
│   ├── paint-controlnet.py ControlNet 姿势控制
│   ├── paint-final.py      定稿参数
│   ├── pose-skeleton.py    自绘 OpenPose 骨架
│   ├── voice-say.py        语音合成 CLI
│   ├── voice-service.py    按需启动的语音服务
│   ├── voice-client.mjs    客户端（自动拉起服务）
│   ├── qq-send-voice.mjs   发语音到 QQ
│   └── qq-send-image.mjs   发图片到 QQ
└── assets/                 示例图（自行添加）
```

---

## 关键参数速查

### 出图（定稿）

```python
lora_weight   = 0.66
steps         = 26
cfg           = 6.5
size          = 768x1120
controlnet    = 0.95
# 必须开：
pipe.enable_sequential_cpu_offload()
pipe.enable_vae_slicing(); pipe.enable_vae_tiling()
```

### 语音（定稿）

```python
version           = "v2Pro"
text_split_method = "cut0"     # 整句合成，避免句尾下沉
speed_factor      = 0.97
ref_free          = False      # 必须给 prompt_text
# 后期链：
denoise → compress(threshold -20dB, ratio 3) → pitch_shift(保时长) → RMS 归一 -16dBFS
```

---

## 已知限制（诚实版）

| 问题 | 原因 | 状态 |
|---|---|---|
| 尾巴有时变成整条鲸鱼 | 训练集里两种形态混在一起 | ❌ 需重训 |
| 背景控制不稳 | LoRA 训练时背景不统一 | ❌ 需重训 |
| ControlNet 下画风偏软 | 8GB 无法同时强控姿势和画风 | ⚠️ 折中 |
| 提示词超 77 token 被截断 | CLIP 限制 | ⚠️ 需精简 |
| 语音不像"真人" | 零样本上限 | ⚠️ 需微调 |

**下一步**：用 GPT 批量生成标准参考图（白底、统一画风、只画尾鳍），重训 LoRA v2。

---

## 成本

| 项 | 费用 |
|---|---|
| 本地推理 | **¥0** |
| LoRA 训练（租 3080Ti） | 约 **¥2 / 次** |
| 语音训练（可选） | 约 **¥1.5 / 次** |
| 总计（一次性） | **< ¥10** |

对比：商业 API 出图约 ¥0.3~1.2/张。

---

## 致谢

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — 语音合成
- [sd-scripts](https://github.com/kohya-ss/sd-scripts) — LoRA 训练
- [xinsir/controlnet-union-sdxl-1.0](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0) — ControlNet
- [Illustrious-XL](https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0) — 底模

## License

MIT（脚本部分）。模型权重遵循各自原始协议，**请勿用于商业倒卖**。

## 本地出图教程（新增）

完整教程见 [`docs/08-本地出图完整教程-SDXL与骨架.md`](docs/08-本地出图完整教程-SDXL与骨架.md)，配套视频：[`assets/tutorial-本地出图.mp4`](assets/tutorial-本地出图.mp4)（117 秒）。

要点速览：

- **两条路**：她自己 → 本地 SDXL + LoRA（最像，免费，约 2.5 分钟）；别的角色 → NovelAI（20~40 秒，28 步内 0 花费）
- **必须用骨架**：不加 ControlNet openpose 骨架，约 1/4 概率出「角色设定集」多姿势拼图
- **提示词只有 77 token**：超了从后往前截断，丝袜/尾巴会消失，所以固定外观块要短、场景词自己写
- **调参先快后慢**：先 `--steps 6 --w 512 --h 768` 试构图（24 秒），满意再跑 26 步 768×1120
- **常用参数**：`--lora 0.72`（0.66 偏淡）、`--cfg 6.5`、`--seed` 固定才能复现

## AI 文生视频教程（新增）

文档：[`docs/09-AI文生视频实战教程.md`](docs/09-AI文生视频实战教程.md) ｜ 视频：[`assets/tutorial-AI文生视频.mp4`](assets/tutorial-AI文生视频.mp4)（106 秒）

核心结论：

- **提示词不是描述画面，是分镜单**：主体 + 动作 + 场景 + 镜头 + 光线 + 风格 + 参数
- 十个新手九个漏「镜头」和「光线」——这两项决定像不像视频
- **最强用法是图生视频**：先用 SDXL/NovelAI 出定妆图锁长相，视频只写「怎么动、怎么拍」
- 一条 5 秒只写一个动作；人物一致性靠参考图，不靠提示词
- 平台顺序：即梦（免费够用）→ 可灵（运动强）→ RunningHub（Wan2.2 工作流，我们账上有 4476 分）
