from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np
from PIL import Image
import soundfile as sf

from duplex_avatar.artifacts import write_video, write_wav
from duplex_avatar.schemas import GeneratedBlock, GenerationRequest


class H3DiffSynthBackend:
    """Adapter for the official DiffSynth-Studio MiniMax-H3 Ref2VA pipeline.

    The adapter intentionally emits one complete generated block. The current
    upstream H3 pipeline denoises and decodes a whole clip before returning, so
    this is chunked pseudo-streaming until a causal DiT/VAE implementation is
    added.
    """

    name = "h3-diffsynth"

    def __init__(self, artifacts_dir: str | Path, config):
        self.artifacts_dir = Path(artifacts_dir)
        self.config = config
        self.pipe = None
        self._load_error: str | None = None

    async def health(self) -> dict:
        return {
            "ok": self.pipe is not None,
            "backend": self.name,
            "mode": "offline_block",
            "loaded": self.pipe is not None,
            "error": self._load_error,
        }

    def _load(self) -> None:
        if self.pipe is not None:
            return
        try:
            import torch
            from diffsynth.core.loader import ModelConfig
            from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline

            variant = self.config.variant
            if variant != "Ref2VA":
                raise ValueError("The starter currently wires Ref2VA only; add FL2VA separately with its own conditioning contract.")
            model_id = self.config.model_id
            base = self.config.model_base_path
            vram_config = {
                "offload_dtype": torch.bfloat16,
                "offload_device": "cpu",
                "onload_dtype": torch.bfloat16,
                "onload_device": "cpu",
                "preparing_dtype": torch.bfloat16,
                "preparing_device": "cuda",
                "computation_dtype": torch.bfloat16,
                "computation_device": "cuda",
            }
            model_configs = [
                ModelConfig(model_id=model_id, origin_file_pattern=f"{variant}/text_encoder/model*.safetensors", local_model_path=base, **vram_config),
                ModelConfig(model_id=model_id, origin_file_pattern=f"{variant}/transformer/model*.safetensors", local_model_path=base, **vram_config),
                ModelConfig(model_id=model_id, origin_file_pattern=f"{variant}/video_vae/source/model.safetensors", local_model_path=base, **vram_config),
                ModelConfig(model_id=model_id, origin_file_pattern=f"{variant}/audio_vae/model.safetensors", local_model_path=base, **vram_config),
            ]
            processor = ModelConfig(model_id=model_id, origin_file_pattern=f"{variant}/processor/", local_model_path=base)
            self.pipe = MiniMaxH3Pipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=model_configs,
                processor_config=processor,
                vram_limit=max(1.0, torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 5),
            )
        except Exception as exc:  # fail at request time with a readable health/error
            self._load_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(
                "Could not load DiffSynth-Studio MiniMax-H3. Install requirements-h3.txt, "
                "use a CUDA host, and check model_base_path/model_id. " + self._load_error
            ) from exc

    @staticmethod
    def _load_audio(path: str, target_rate: int = 32000) -> tuple[object, int]:
        import torch
        samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        if samples.shape[1] == 1:
            samples = np.repeat(samples, 2, axis=1)
        if sample_rate != target_rate:
            source_t = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
            target_t = np.linspace(0.0, 1.0, num=round(len(samples) * target_rate / sample_rate), endpoint=False)
            samples = np.stack(
                [np.interp(target_t, source_t, samples[:, channel]) for channel in range(samples.shape[1])],
                axis=1,
            )
            sample_rate = target_rate
        samples = torch.from_numpy(samples.T.copy())
        return samples, sample_rate

    async def generate(self, request: GenerationRequest, cancel_event: asyncio.Event) -> GeneratedBlock:
        if cancel_event.is_set():
            raise asyncio.CancelledError
        started = time.perf_counter()
        await asyncio.to_thread(self._load)
        if cancel_event.is_set():
            raise asyncio.CancelledError

        # DiffSynth's pipeline is synchronous and returns (list[PIL.Image], audio tensor).
        references = []
        if request.identity_image_path:
            references.append({"type": "image", "image": Image.open(request.identity_image_path).convert("RGB")})
        if request.user_audio_path:
            target_rate = int(getattr(getattr(self.pipe, "audio_vae", None), "sample_rate", 32000))
            audio, sample_rate = await asyncio.to_thread(self._load_audio, request.user_audio_path, target_rate)
            references.append({"type": "audio", "audio": audio, "sample_rate": sample_rate})

        result = await asyncio.to_thread(
            self.pipe,
            prompt=request.prompt,
            height=request.height,
            width=request.width,
            num_frames=request.num_frames,
            num_inference_steps=request.num_inference_steps,
            seed=request.seed + request.block_index,
            cfg_scale=request.cfg_scale,
            references=references or None,
        )
        if cancel_event.is_set():
            raise asyncio.CancelledError
        frames, audio = result
        out_dir = self.artifacts_dir / request.session_id / f"turn_{request.turn_id:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        video_path = write_video(out_dir / f"block_{request.block_index:04d}.mp4", frames, request.fps)
        audio_np = audio.detach().float().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
        output_rate = int(getattr(getattr(self.pipe, "audio_vae", None), "sample_rate", 32000))
        audio_path = write_wav(out_dir / f"block_{request.block_index:04d}.wav", audio_np, output_rate)
        return GeneratedBlock(
            session_id=request.session_id,
            turn_id=request.turn_id,
            block_index=request.block_index,
            video_path=video_path,
            audio_path=audio_path,
            frames=len(frames),
            fps=request.fps,
            latency_ms=round((time.perf_counter() - started) * 1000),
        )

