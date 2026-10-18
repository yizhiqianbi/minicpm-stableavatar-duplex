# MiniCPM-o + StableAvatar-1.3B 训练指南

这个仓库的训练建议分成三条线：

```text
StableAvatar LoRA：训练数字人的身份、嘴型和动作
MiniCPM-o LoRA：训练回答风格、listen/speak 和打断行为
联合实验：冻结两个底座，只增加状态/历史条件 adapter
```

不要一开始做端到端联合训练。StableAvatar 的视频扩散过程和 MiniCPM-o 的对话/音频过程没有直接的可用梯度接口，先分开训练更容易定位收益。

## 一、StableAvatar LoRA

StableAvatar 官方仓库提供 Wan2.1-1.3B 的训练和 LoRA 路径，官方 1.3B 训练设置使用过 4 张 A100 80GB。参考：[StableAvatar](https://github.com/Francis-Rings/StableAvatar)。

### 数据目录

官方训练器读取样本目录路径列表，而不是本项目的双工 JSON manifest。建议目录如下：

```text
talking_face_data/
├── rec/speech/00001/
│   ├── sub_clip.mp4
│   ├── audio.wav
│   ├── images/frame_0.png ...
│   ├── face_masks/frame_0.png ...
│   └── lip_masks/frame_0.png ...
├── vec/speech/00001/
└── square/speech/00001/

talking_face_data/video_rec_path.txt
talking_face_data/video_vec_path.txt
talking_face_data/video_square_path.txt
```

`audio.wav` 是数字人目标音频，不能放用户麦克风音频。视频、音频、帧和 mask 必须时间对齐。全双工数据中的 `user_audio` 应放在 sidecar manifest 中，供 MiniCPM-o 或 ASR 使用。

单一身份建议先准备 30–60 分钟干净素材，正式实验再扩展到 2–5 小时。切片可以从 3–10 秒开始，覆盖不同语速、停顿、表情和轻微头部运动；验证集按视频或场景划分，不能随机切帧后混入训练集。

### Smoke test

先用少量样本验证路径、显存和 checkpoint：

```bash
cd minicpm-stableavatar-duplex

STABLEAVATAR_ROOT=/data/StableAvatar \
DATA_ROOT=/data/StableAvatar/talking_face_data \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MAX_STEPS=20 \
MAX_TRAIN_SAMPLES=8 \
bash scripts/train_stableavatar_lora.sh
```

### LoRA baseline

当前脚本默认使用 rec/vec 混合分辨率训练：

```bash
STABLEAVATAR_ROOT=/data/StableAvatar \
DATA_ROOT=/data/StableAvatar/talking_face_data \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
LORA_RANK=32 \
NETWORK_ALPHA=16 \
VIDEO_FRAMES=81 \
MIXED_PRECISION=bf16 \
MAX_STEPS=1000 \
bash scripts/train_stableavatar_lora.sh
```

如果已经下载官方 StableAvatar transformer，可以指定：

```bash
TRANSFORMER_PATH=/data/StableAvatar/checkpoints/StableAvatar-1.3B/transformer3d-rec-vec.pt
```

这样才是在 StableAvatar 权重上做 LoRA finetune；如果不指定，则是在 Wan2.1-1.3B 上训练 adapter。正式实验建议显式设置 `TRANSFORMER_PATH`，避免把 Wan2.1 adapter 误当成 StableAvatar finetune。

需要注意当前仓库的一个加载边界：官方 `inference.py` 主要接收完整 `--transformer_path`，而 LoRA trainer 保存的 `lora-checkpoint-*.pt` 不能直接作为推理 transformer 使用。训练完成后还需要二选一：

1. 在推理端创建与 trainer 相同的 LoRA 网络，加载 LoRA state，再 merge/unload 后导出完整 transformer；
2. 给 StableAvatar 推理脚本增加 LoRA 加载参数，并在 adapter 中显式加载 LoRA。

当前 `StableAvatarBackend` 还没有自动 merge LoRA 的逻辑，因此训练输出不能直接填入 `stableavatar_transformer_path`。当前包装器只对 rec/vec 提供 LoRA 入口；square 数据需要按上游 full trainer 单独处理。

建议实验顺序：

```text
rank=16，500 steps
rank=32，1,000 steps
rank=32，2,000 steps
```

固定身份图、提示词、验证音频和随机种子，比较身份相似度、唇形同步、视频质量、推理时延和音画偏差。

## 二、MiniCPM-o 训练

当前仓库里的 MiniCPM-o 模块是推理适配器，支持 mock、Realtime WebSocket 和本地 MiniCPMODuplex；它还不是完整的 MiniCPM-o 训练器。MiniCPM-o 的全双工流程按音频 chunk 工作，并输出 listen/speak 决策、文本和目标音频。[Duplex 文档](https://github.com/OpenBMB/MiniCPM-o-Demo/blob/main/docs/en/architecture/duplex.md)

MiniCPM-o 4.5 的普通视觉/文本 LoRA 路径和 Omni 全双工路径不能直接等同。当前更稳妥的研究做法是：先冻结 Omni 模型和 TTS，只训练一个外部 duplex policy/state head；如果要改语言行为，再用与具体 checkpoint 兼容的文本/视觉 LoRA。官方文档列出了 LLaMA-Factory 和 SWIFT 等训练框架，但需要逐项确认连续音频、speech output、TDM/KV streaming 是否被目标 checkpoint 的训练代码支持。[MiniCPM-o 4.5 文档](https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/minicpm_o4dot5_en.md)

训练数据应保留时间结构：

```json
{
  "session_id": "s001",
  "identity_id": "avatar_01",
  "user_audio": "user_0001.wav",
  "target_audio": "target_0001.wav",
  "transcript": "用户说话文本",
  "response_text": "数字人回复文本",
  "events": [
    {"start": 0.0, "end": 0.8, "role": "user", "state": "speak"},
    {"start": 0.8, "end": 1.2, "role": "avatar", "state": "backchannel"},
    {"start": 1.2, "end": 3.0, "role": "avatar", "state": "speak"}
  ],
  "interrupt": false
}
```

至少标注：

```text
listen
speak
backchannel
interrupt
end_of_turn
silence
overlap
```

推荐训练顺序：

1. 先训练文本回答风格。
2. 再训练 listen/speak 决策。
3. 再训练打断、backchannel 和 end-of-turn。
4. 最后再做声音风格或音频输出适配。

第一版应冻结音频编码器和 TTS，只对语言模型或状态头做 LoRA。OpenBMB 的 finetune 文档提供了 JSON 对话格式、LoRA 和 DeepSpeed 的基本范式；MiniCPM-o 4.5 需要根据具体 checkpoint 核对 LLaMA-Factory、SWIFT 或官方训练代码的支持情况。[MiniCPM finetune](https://github.com/OpenBMB/MiniCPM-V/blob/main/finetune/readme.md) · [MiniCPM-o 4.5](https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/minicpm_o4dot5_en.md)

不要把用户音频直接写入 StableAvatar 的 `audio.wav`。用户音频用于理解和状态判断，`target_audio` 才是驱动数字人嘴型的音频。

## 三、联合实验

建议至少保留以下对照：

```text
S0：原始 MiniCPM-o + 原始 StableAvatar
S1：MiniCPM-o LoRA + 原始 StableAvatar
S2：原始 MiniCPM-o + StableAvatar LoRA
S3：MiniCPM-o LoRA + StableAvatar LoRA
S4：S3 + state/history adapter
S5：S4 + user-audio overlap conditioning
```

联合阶段不要直接对 MP4 反向传播。先把 MiniCPM-o 输出的 target audio、state 和 history 保存成 sidecar，再训练 StableAvatar 的状态条件 adapter。这样可以清楚区分提升来自对话模型、视频身份、双工状态还是用户音频建模。

## 四、8 卡安排

建议先分阶段使用 8 卡，而不是同时训练两个底座：

```text
StableAvatar：先用 4 卡复现实验，再扩到 8 卡
MiniCPM-o LoRA：2–4 卡做行为实验
剩余 GPU：验证、推理和数据预处理
```

一个实际流程可以是：

```text
第 1 步：StableAvatar 20 steps smoke
第 2 步：StableAvatar LoRA baseline
第 3 步：整理 MiniCPM-o 双工标注
第 4 步：MiniCPM-o LoRA / 状态头
第 5 步：联合消融和系统评测
```

## 五、评测指标

### Avatar

```text
身份相似度
唇形同步
视频质量
音画偏差
长时间漂移
```

### Duplex brain

```text
listen/speak F1
interrupt latency
backchannel precision
WER
response appropriateness
```

### 系统

```text
TTFA：首段目标音频延时
TTFC：首个视频 block 延时
RTF：生成时间 / 视频播放时间
interrupt-stop P95
A/V sync
queue backlog
```

第一版研究结果最好报告四个维度：

```text
质量：identity / lip-sync / video quality
交互：listen/speak / interrupt / backchannel
实时：TTFA / TTFC / RTF
稳定：长时漂移 / queue backlog / A/V skew
```

## 六、当前仓库边界

目前可以直接运行的是：

- mock brain + mock avatar 的全双工 baseline；
- MiniCPM-o 推理适配器；
- StableAvatar 官方 `inference.py` 适配器；
- StableAvatar rec/vec LoRA 训练包装器；
- 双工 sidecar manifest 示例。

目前还需要继续实现的是：

- MiniCPM-o 4.5 的 checkpoint-specific LoRA 训练入口；
- 常驻 StableAvatar worker；
- target audio delta 到视频窗口的流式调度；
- 状态/history 条件 adapter；
- WebRTC 或 WebCodecs 视频输出。
