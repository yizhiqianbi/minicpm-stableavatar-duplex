# MiniCPM-o + StableAvatar 全双工数字人 baseline

这个研究原型把全双工数字人拆成三个可替换模块：MiniCPM-o 持续接收用户音频并决定 `listen` 或 `speak`，输出文本和数字人的目标语音；StableAvatar-1.3B 用目标语音驱动参考人像生成视频；`DuplexSession` 管理 listen/thinking/speaking/interrupted/stopped 状态、turn 号和旧结果丢弃。

当前默认是 `mock + mock`，不需要下载模型即可验证 WebSocket、双工状态和中断路径。真实 StableAvatar 适配器通过官方 `inference.py` 子进程懒加载，每个 block 生成一个 MP4；官方输出是无声视频，适配器会保存目标 WAV，并在有 ffmpeg 时把它 mux 回公开 MP4。

## 本地运行

```bash
cd h3_duplex_avatar
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
PYTHONPATH=src python -m duplex_avatar.server --config configs/default.yaml
```

另一个终端运行协议 smoke test：

```bash
PYTHONPATH=src python scripts/ws_smoke.py
```

客户端先发送：

```json
{"type":"start","prompt":"请简短、自然地回答用户。","identity_image_path":"/data/avatar/reference.png"}
```

之后发送 16 kHz、单声道、PCM16 的 `audio` 消息。服务端会返回 `state`、`brain`、`interrupt`、`block_ready` 和 `error` 事件。`block_ready` 中的 `video_url` 和 `audio_url` 是相对 `/artifacts` 的 URL。用户在数字人说话时继续发送音频，会取消未播放的生成任务、递增 `turn_id`，再开始新一轮。

也可以直接查看：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/sessions/demo
```

## 接入真实模型

### StableAvatar-1.3B

StableAvatar 官方仓库是 [Francis-Rings/StableAvatar](https://github.com/Francis-Rings/StableAvatar)。它基于 Wan2.1-1.3B，输入参考图和 driving audio，支持 rec/vec/square 训练与推理。先在独立环境安装上游依赖并下载 Wan2.1、StableAvatar transformer 和 wav2vec checkpoint，然后修改 `configs/default.yaml`：

```yaml
brain:
  name: mock                 # 先用 mock 验证协议
avatar:
  name: stableavatar
  stableavatar_root: /data/StableAvatar
  stableavatar_pretrained_model_path: checkpoints/Wan2.1-Fun-V1.1-1.3B-InP
  stableavatar_transformer_path: checkpoints/StableAvatar-1.3B/transformer3d-rec-vec.pt
  stableavatar_wav2vec_path: checkpoints/wav2vec2-base-960h
```

启动时必须在 `start` 消息中提供服务器可读的 `identity_image_path`。StableAvatar 每个 block 会重新启动一次官方脚本，适合先做正确性和时延 baseline；模型常驻 worker、增量 VAE 和真正 causal frame stream 是后续优化方向。

### MiniCPM-o

MiniCPM-o 的公开全双工接口以约一秒的音频单元返回 listen、文本和音频增量。开发阶段可以把 `brain.name` 改成 `remote`，设置一个兼容 `/v1/realtime` 的 WebSocket endpoint；本地模型则使用 `local` 并填写 `model_path`：

```yaml
brain:
  name: remote            # mock | remote | local
  endpoint: https://your-minicpm-gateway.example
  remote_mode: audio
  prompt: "You are a warm, concise voice assistant."
```

或：

```yaml
brain:
  name: local
  model_path: /data/models/MiniCPM-o-4_5
```

本项目不把 MiniCPM-o 的目标音频误当成麦克风输入：用户音频写入 `user_audio_*.wav`，模型返回的 24 kHz float32 音频写入 `target_audio_*.wav`，只有后者会送给 StableAvatar。

## LoRA 训练入口

`[scripts/train_stableavatar_lora.sh](scripts/train_stableavatar_lora.sh)` 是官方 StableAvatar 1B rec/vec LoRA 训练器的包装器。它支持 4/8 卡、LoRA rank/alpha、短 smoke 训练和已发布 StableAvatar transformer 的 finetune：

```bash
STABLEAVATAR_ROOT=/data/StableAvatar \
DATA_ROOT=/data/StableAvatar/talking_face_data \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MAX_STEPS=20 MAX_TRAIN_SAMPLES=8 \
bash scripts/train_stableavatar_lora.sh
```

数据目录、路径列表和全双工 sidecar manifest 见 [data/README.md](data/README.md)。官方 LoRA trainer 只学习目标数字人的视频/音频监督，不会自动学习 MiniCPM-o 的 listen/speak 状态；状态和用户音频适配器应作为后续研究变量单独加入。

## 当前边界

这是 blockwise/offline baseline，不是已经完成的因果实时视频模型。一次响应要等 MiniCPM-o 给出目标音频、StableAvatar 完成一个视频 block 后才能 `block_ready`；因此应测 TTFC、打断停止延迟、音画同步和长时漂移，再逐步把 StableAvatar 替换为常驻 worker、滑窗 latent/VAE 和 WebRTC/WebCodecs 输出。

参考：

- [StableAvatar 官方仓库](https://github.com/Francis-Rings/StableAvatar)
- [MiniCPM-o Duplex Architecture](https://github.com/OpenBMB/MiniCPM-o-Demo/blob/main/docs/en/architecture/duplex.md)
- [MiniCPM-o Realtime API](https://github.com/OpenBMB/MiniCPM-o-Demo/blob/main/docs-app/content/docs/en/realtime-api/overview.md)
