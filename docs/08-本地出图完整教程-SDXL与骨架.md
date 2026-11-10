# 本地出图完整教程（SDXL + LoRA + ControlNet 骨架）

> 环境：Windows 11 ｜ RTX 5070 Laptop **8 GB 显存** ｜ 16 GB 内存
> 目标：不花一分钱，在自己电脑上把「大肥鱼」画出来，脸尽量像。

---

## 一、为什么分成「本地」和「在线」两条路

| | 本地 SDXL | NovelAI（在线） |
|---|---|---|
| 训练过她吗 | ✅ 有 LoRA，脸最像 | ❌ 没训练，画出来是别人 |
| 速度 | 加载 55 s + 生成 100~150 s | 20~40 s |
| 花钱 | 0 | 28 步以内 0 |
| 显存 | 8 GB 够用 | 不用 |

**结论：画她自己走本地；画别的角色走 NovelAI。**

---

## 二、环境准备

```
Python      E:\ai\gsv\venv\Scripts\python.exe
torch       2.7.1+cu128
diffusers   0.31.0
```

模型文件：

```
底模    E:\ai\models\checkpoints\Illustrious-XL-v0.1.safetensors
LoRA    E:\ai\models\loras\lanxi_v1.safetensors
骨架    E:\ai\models\controlnet\openpose
```

8 GB 显存能跑，靠的是这三行：

```python
pipe.enable_sequential_cpu_offload()   # 逐层搬到 CPU，省显存
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
```

---

## 三、最简用法

```powershell
& "E:\ai\gsv\venv\Scripts\python.exe" "E:\ai\gsv\paint_local.py" `
  --prompt "solo, single character, sitting at a cafe table, holding a cup, warm lighting"
```

输出最后一行是 JSON：

```json
{"ok":true,"file":"...\\local-2026-11-09T20-28-53-2402.png","seed":2402,
 "size":[768,1120],"load_s":54.4,"gen_s":138.3,"pose":true,"lora":0.72}
```

`file` 就是图片路径，直接用。

---

## 四、参数表

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--prompt` | 必填 | **只写场景和动作**，外观交给 LoRA |
| `--pose` | `cn_poses\sk2_sit.png` | 骨架图，决定姿势；不想要就传 `--pose ""` |
| `--w --h` | 768×1120 | 竖构图比例接近 2:3 |
| `--seed` | 随机 | 固定 seed 才能复现同一张 |
| `--lora` | 0.72 | 0.66 偏淡，0.8 以上容易过拟合 |
| `--steps` | 26 | 26~28 够用，再多收益很小 |
| `--cfg` | 6.5 | 7 以上开始发硬，5.5 偏软 |
| `--neg` | 内置 | 追加负面词才需要传 |

---

## 五、提示词怎么写（最容易踩坑的一节）

### 5.1 CLIP 只吃 77 个 token

超过 77 会被**从后往前截断**——尾巴、丝袜这类写在最后的词直接消失。

**所以脚本里的固定身份块（STYLE）必须短，并且把关键特征放前面：**

```python
STYLE = ("lanxi, solo, navy blue hair, cyan hair tips, ahoge, whale fin ears, "
         "deep blue eyes, white lace maid headdress, white apron, blue maid dress, "
         "white opaque tights, white background, cel shading, crisp lineart")
```

实测 **71 token**，加上你的场景词也不会超。

### 5.2 你自己只写场景

```
✅ solo, single character, sitting at a cafe table, holding a cup, warm sunlight
❌ 蓝发、鲸鱼耳、白围裙、蓝裙子、白丝袜……（LoRA 已经记住了，写了反而挤爆）
```

### 5.3 风格词

```
cel shading        赛璐璐上色
crisp lineart      细线
matte              哑光不要油光
saturated colors   颜色更浓（发色变淡时加）
```

---

## 六、骨架 ControlNet（关键技巧）

### 6.1 为什么必须用

**不用骨架时**，模型有大约 1/4 概率给你一张「角色设定集」——一张图里画六七个不同姿势的小人。

**用了骨架**，一张图只出一个姿势，构图可控。

### 6.2 现成的骨架图

```
E:\ai\gsv\out\cn_poses\
  sk2_sit.png      坐着（默认）
  sk2_kneel.png    跪坐
  sk2_lying.png    躺
  pose_sit.png     另一版坐姿
```

### 6.3 怎么用

```powershell
& $py "E:\ai\gsv\paint_local.py" `
  --prompt "solo, standing, hands folded, looking at viewer" `
  --pose "E:\ai\gsv\out\cn_poses\sk2_sit.png" `
  --seed 2402
```

### 6.4 骨架强度

脚本里 `controlnet_conditioning_scale = 0.9`。

| 值 | 效果 |
|---|---|
| 1.0 | 姿势完全照搬，可能显得僵 |
| **0.9** | **默认，推荐** |
| 0.7 | 姿势只作参考，模型自由发挥多一点 |

### 6.5 自己做骨架

任何 OpenPose 骨架图都行（黑底彩色火柴人，17 个关键点）。可以用：

- ControlNet 官方示例图
- 或者用 `openpose` 模型从别的照片里提取

**骨架图必须是纯火柴人，不能带原图**，否则会被当成参考图。

---

## 七、常见问题对照表

| 症状 | 原因 | 解法 |
|---|---|---|
| 一张图里好几个小人 | 没用骨架 | 加 `--pose`，负面词加 `character sheet, multiple characters` |
| 发色变浅、变灰 | LoRA 权重太低 | `--lora 0.72`（默认） |
| 尾巴 / 丝袜没了 | 提示词超 77 token 被截断 | 缩短 prompt，别重复描述外观 |
| 脸偏幼、偏圆 | cfg 低 + 缺成熟感 | 提高 cfg 到 7，或加 `mature` |
| 同一张画不出第二次 | 没固定种子 | `--seed 2402` |
| 出图特别慢 | 显存不够在换页 | 关掉别的吃显存程序（浏览器硬件加速最常见） |
| 背景是灰的 | 没指定背景 | STYLE 里已带 `white background`，想要场景就写场景词 |

---

## 八、耗时参考（RTX 5070 Laptop 8 GB）

```
模型加载      53~58 s   （每次运行都要，没法省）
生成 26 步    100~150 s
生成 4 步      24 s     （调构图时用这个快速试）
```

**调参建议**：先用 `--steps 6 --w 512 --h 768` 快速试构图和姿势，确定了再跑完整参数。

---

## 九、完整示例

```powershell
$py = "E:\ai\gsv\venv\Scripts\python.exe"

# 快速试构图（24 秒一张）
& $py "E:\ai\gsv\paint_local.py" `
  --prompt "solo, single character, sitting, holding a cup, cafe interior" `
  --w 512 --h 768 --steps 6 --seed 2402

# 满意后出正式图（约 2.5 分钟）
& $py "E:\ai\gsv\paint_local.py" `
  --prompt "solo, single character, sitting, holding a cup, cafe interior" `
  --seed 2402
```

---

## 十、和 QQ 机器人怎么接

机器人不直接调 `paint_local.py`，统一走路由：

```powershell
node "E:\deepseek\qq-bridge\tools\draw.mjs" "<英文 tag>" --who local
```

- `--who local` → 本地 SDXL
- `--who other` → NovelAI
- 不传 `--who` → 看 `var\draw-engine.txt` 里写的是 `local` 还是 `nai`

**默认引擎切换只改一个字：**

```
E:\deepseek\qq-bridge\var\draw-engine.txt
   local    ←→    nai
```

---

## 附：文件清单

```
E:\ai\gsv\paint_local.py              本地出图 CLI
E:\ai\gsv\out\local\                  出图目录
E:\ai\gsv\out\cn_poses\               骨架图
E:\deepseek\qq-bridge\tools\draw.mjs  路由（本地 / NovelAI）
E:\deepseek\qq-bridge\var\draw-engine.txt  默认引擎
```
