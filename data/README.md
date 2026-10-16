# StableAvatar 数据准备

这个目录只保存数据说明和样例 manifest；真实视频、音频、帧和 mask 放在 StableAvatar checkout 外部。训练入口是 [`../scripts/train_stableavatar_lora.sh`](../scripts/train_stableavatar_lora.sh)，它调用 StableAvatar 官方的 `train_1B_rec_vec_lora.py`，不会复制或修改上游训练代码。

## 先跑官方兼容 baseline

StableAvatar 训练器读取三个“样本目录路径列表”，而不是本项目的 JSON manifest。每个样本目录至少包含：

```text
talking_face_data/
├── rec/speech/00001/
│   ├── sub_clip.mp4
│   ├── audio.wav
│   ├── images/frame_0.png ...
│   ├── face_masks/frame_0.png ...
│   └── lip_masks/frame_0.png ...
├── vec/speech/00001/      # 与 rec 相同
└── square/speech/00001/   # 与 rec 相同
```

将目录绝对路径分别写入：

```text
talking_face_data/video_rec_path.txt
talking_face_data/video_vec_path.txt
talking_face_data/video_square_path.txt
```

音频应为清晰的单人语音，视频背景尽量固定；`sub_clip.mp4`、`audio.wav` 和 `images/` 必须时间对齐。官方预处理脚本可从视频抽音频，从视频帧生成 `images/`，并用 `lip_mask_extractor.py` 生成嘴唇 mask。脸部 mask 可按上游仓库推荐的 StableAnimator 流程生成。完整目录要求以 [StableAvatar README 的训练说明](https://github.com/Francis-Rings/StableAvatar#-model-training) 为准。

最小 baseline（混合 480×832/832×480）命令：

```bash
cd h3_duplex_avatar
STABLEAVATAR_ROOT=/data/StableAvatar \
DATA_ROOT=/data/StableAvatar/talking_face_data \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MAX_STEPS=20 MAX_TRAIN_SAMPLES=8 \
bash scripts/train_stableavatar_lora.sh
```

正式训练前先去掉 `MAX_STEPS` 和 `MAX_TRAIN_SAMPLES`。当前上游公开的 1B LoRA trainer 是混合 `rec_vec` 版本；512×512 数据请直接运行上游 `train_1B_square.py` 全量 trainer，或先按你的实验需要改写一个 square LoRA trainer，不要在本包装器中假装它是 LoRA。若已有发布的 `StableAvatar-1.3B` transformer 权重，可设置 `TRANSFORMER_PATH=/data/.../transformer3d-rec-vec.pt` 做 LoRA finetuning；留空则是在 Wan2.1-1.3B 上训练一个 LoRA adapter。`LORA_RANK=32` 是本项目起步值，正式实验应在日志中记录 rank、alpha、帧数、GPU 数和 checkpoint。

## 全双工 sidecar manifest

[`stableavatar_duplex_manifest.example.json`](stableavatar_duplex_manifest.example.json) 是研究数据的 sidecar，不会被官方 StableAvatar trainer 直接读取。它把官方目标视频/目标音频样本与用户音频、交互事件和状态标签关联起来：

```json
{
  "video": ".../sub_clip.mp4",
  "target_audio": ".../audio.wav",
  "user_audio": ".../user.wav",
  "events": [
    {"start_sec": 0.0, "end_sec": 0.4, "role": "user", "kind": "speech"},
    {"start_sec": 0.4, "end_sec": 1.0, "role": "target", "kind": "backchannel"}
  ],
  "state_labels": ["listen", "backchannel", "speak"]
}
```

这里要明确区分：

- `target_audio` 是数字人要说的音频，驱动 StableAvatar 的嘴型；
- `user_audio` 是用户麦克风，供 MiniCPM-o/ASR 和后续双工适配器使用；
- `events` 可以允许两个角色时间段重叠，不能为了方便预处理成互斥的单人语音；
- `state_labels` 是研究字段，当前官方 LoRA trainer 会忽略它。

第一阶段只用 `target_audio` 复现官方模型，建立身份、唇形同步和推理时延 baseline。第二阶段再根据 sidecar 训练用户状态/历史条件适配器；不要把用户音频误写到 `audio.wav`，否则监督目标会让模型把用户声音当成数字人声音。

## 训练入口的限制

`train_stableavatar_lora.sh` 是可运行的 StableAvatar 官方 LoRA 包装器，不是全双工模型训练器。它不会自动把 MiniCPM-o 的事件标签注入 StableAvatar，也不会实现因果 KV cache。后续研究改动应保持这条 baseline 不变，另加状态条件 adapter 和消融实验，便于比较：

```text
StableAvatar LoRA
StableAvatar LoRA + user audio（无 state）
StableAvatar LoRA + user audio + state/history（方法）
```
