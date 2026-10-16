from __future__ import annotations

"""StableAvatar-1.3B adapter.

The official StableAvatar repository exposes inference through ``inference.py``
and imports a sizeable Wan/diffusers stack.  Keeping that stack in a child
process gives the websocket server a small, importable baseline and makes
model loading lazy: no StableAvatar code or weights are touched until the
first ``generate`` request.
"""

import asyncio
import os
import selectors
import shutil
import subprocess
import time
from pathlib import Path

from duplex_avatar.schemas import GeneratedBlock, GenerationRequest


class StableAvatarBackend:
    """Run StableAvatar's Wan2.1-1.3B inference script per video block.

    StableAvatar itself writes a silent ``video_without_audio.mp4``. The
    adapter keeps a WAV artifact and muxes it into a public MP4 when ffmpeg is
    available. ``target_audio_path`` is preferred; ``user_audio_path`` is a
    compatibility fallback for the first baseline before MiniCPM-o/TTS is
    connected.
    """

    name = "stableavatar"

    def __init__(self, artifacts_dir: str | Path, config):
        self.artifacts_dir = Path(artifacts_dir)
        self.config = config
        self._validated = False
        self._load_error: str | None = None

    def _resolve(self, value: str | None) -> Path | None:
        if value is None:
            return None
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return Path(self.config.stableavatar_root).expanduser() / path

    def _validate(self) -> None:
        if self._validated:
            return
        root = Path(self.config.stableavatar_root).expanduser()
        script = self._resolve(self.config.stableavatar_inference_script)
        if not root.exists():
            raise FileNotFoundError(
                f"StableAvatar root does not exist: {root}. Clone the official "
                "StableAvatar repository and set avatar.stableavatar_root."
            )
        if script is None or not script.is_file():
            raise FileNotFoundError(f"StableAvatar inference script does not exist: {script}")
        required = {
            "pretrained_model": self._resolve(self.config.stableavatar_pretrained_model_path),
            "transformer": self._resolve(self.config.stableavatar_transformer_path),
            "wav2vec": self._resolve(self.config.stableavatar_wav2vec_path),
        }
        missing = [f"{key}={path}" for key, path in required.items() if path is None or not path.exists()]
        if missing:
            raise FileNotFoundError(
                "StableAvatar model paths are missing: " + ", ".join(missing) +
                ". Download the checkpoint layout described in the StableAvatar README."
            )
        self._validated = True

    async def health(self) -> dict:
        """Report configuration without importing torch or loading weights."""
        try:
            self._validate()
            self._load_error = None
            configured = True
        except Exception as exc:  # health must remain useful on CPU/dev hosts
            configured = False
            self._load_error = f"{type(exc).__name__}: {exc}"
        return {
            "ok": configured,
            "backend": self.name,
            "mode": "offline_block",
            "loaded": self._validated,
            "lazy": True,
            "model_root": str(Path(self.config.stableavatar_root).expanduser()),
            "error": self._load_error,
        }

    def _command(self, request: GenerationRequest, audio_path: Path, output_dir: Path) -> list[str]:
        root = Path(self.config.stableavatar_root).expanduser()
        config_path = self._resolve(self.config.stableavatar_config_path)
        pretrained = self._resolve(self.config.stableavatar_pretrained_model_path)
        transformer = self._resolve(self.config.stableavatar_transformer_path)
        wav2vec = self._resolve(self.config.stableavatar_wav2vec_path)
        script = self._resolve(self.config.stableavatar_inference_script)
        if request.identity_image_path is None:
            raise ValueError("StableAvatar requires GenerationRequest.identity_image_path")
        identity = Path(request.identity_image_path).expanduser()
        if not identity.is_file():
            raise FileNotFoundError(f"identity image does not exist: {identity}")
        if not audio_path.is_file():
            raise FileNotFoundError(f"target audio does not exist: {audio_path}")

        # Keep every option as an argv item.  Prompts may contain spaces or
        # non-ASCII text and must never be interpolated into a shell command.
        return [
            str(self.config.stableavatar_python), str(script),
            f"--config_path={config_path}",
            f"--pretrained_model_name_or_path={pretrained}",
            f"--transformer_path={transformer}",
            f"--pretrained_wav2vec_path={wav2vec}",
            f"--validation_reference_path={identity}",
            f"--validation_driven_audio_path={audio_path}",
            f"--output_dir={output_dir}",
            f"--validation_prompts={request.prompt}",
            f"--seed={request.seed + request.block_index}",
            f"--ulysses_degree={int(self.config.stableavatar_ulysses_degree)}",
            f"--ring_degree={int(self.config.stableavatar_ring_degree)}",
            f"--motion_frame={int(self.config.stableavatar_motion_frame)}",
            f"--sample_steps={int(self.config.stableavatar_sample_steps)}",
            f"--width={int(request.width)}",
            f"--height={int(request.height)}",
            f"--overlap_window_length={int(self.config.stableavatar_overlap_window_length)}",
            f"--clip_sample_n_frames={int(self.config.stableavatar_clip_sample_n_frames or request.num_frames)}",
            f"--GPU_memory_mode={self.config.stableavatar_gpu_memory_mode}",
            f"--sample_text_guide_scale={float(self.config.stableavatar_text_guide_scale)}",
            f"--sample_audio_guide_scale={float(self.config.stableavatar_audio_guide_scale)}",
        ]

    @staticmethod
    def _find_video(output_dir: Path) -> Path:
        preferred = output_dir / "video_without_audio.mp4"
        if preferred.is_file():
            return preferred
        candidates = sorted(output_dir.glob("*.mp4"))
        if candidates:
            return candidates[0]
        raise FileNotFoundError(
            f"StableAvatar finished without an MP4 in {output_dir}; inspect its logs for inference errors"
        )

    @staticmethod
    def _mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> Path:
        """Mux target audio when ffmpeg is available, preserving a silent fallback."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            return video_path
        command = [
            ffmpeg, "-y", "-loglevel", "error", "-i", str(video_path),
            "-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-shortest", "-movflags", "+faststart",
            str(output_path),
        ]
        try:
            subprocess.run(command, check=True)
        except (OSError, subprocess.CalledProcessError):
            return video_path
        return output_path

    @staticmethod
    def _run_process(command: list[str], cwd: Path, env: dict[str, str], cancel_event: asyncio.Event) -> None:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output: list[str] = []
        selector = selectors.DefaultSelector()
        if process.stdout is not None:
            selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while process.poll() is None:
                if cancel_event.is_set():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise asyncio.CancelledError
                for key, _ in selector.select(timeout=0.1):
                    line = key.fileobj.readline()
                    if line:
                        output.append(line.rstrip())
            if process.stdout is not None:
                output.extend(line.rstrip() for line in process.stdout.readlines())
            if process.returncode != 0:
                tail = "\n".join(output[-40:])
                raise RuntimeError(f"StableAvatar inference failed ({process.returncode}):\n{tail}")
        finally:
            selector.close()
            if process.poll() is None:
                process.kill()
                process.wait()

    async def generate(self, request: GenerationRequest, cancel_event: asyncio.Event) -> GeneratedBlock:
        if cancel_event.is_set():
            raise asyncio.CancelledError
        started = time.perf_counter()
        try:
            # This is the only point where paths are checked.  No model import
            # happens in the API process and the subprocess loads weights only
            # for an actual generation request.
            await asyncio.to_thread(self._validate)
            audio_value = request.target_audio_path or request.user_audio_path
            if audio_value is None:
                raise ValueError("StableAvatar requires target_audio_path (or user_audio_path for baseline)")
            audio_path = Path(audio_value).expanduser()
            turn_dir = self.artifacts_dir / request.session_id / f"turn_{request.turn_id:04d}"
            # Keep the subprocess output in a hidden per-block directory so
            # its fixed ``video_without_audio.mp4`` name cannot collide with
            # another request.  The public artifacts are copied to turn_dir,
            # matching the URL convention already used by session.py.
            output_dir = turn_dir / f".stableavatar_{request.block_index:04d}"
            output_dir.mkdir(parents=True, exist_ok=True)
            command = self._command(request, audio_path, output_dir)
            env = os.environ.copy()
            devices = self.config.stableavatar_cuda_visible_devices
            if devices:
                env["CUDA_VISIBLE_DEVICES"] = str(devices)
            await asyncio.to_thread(
                self._run_process,
                command,
                Path(self.config.stableavatar_root).expanduser(),
                env,
                cancel_event,
            )
            if cancel_event.is_set():
                raise asyncio.CancelledError
            video_path = self._find_video(output_dir)
            # StableAvatar intentionally emits silent video.  Keep a stable
            # per-block WAV URL for the existing client, and also copy the
            # video to a predictable artifact name.
            block_video = turn_dir / f"block_{request.block_index:04d}.mp4"
            shutil.copy2(video_path, block_video)
            block_audio = turn_dir / f"block_{request.block_index:04d}.wav"
            shutil.copy2(audio_path, block_audio)
            muxed_video = turn_dir / f"block_{request.block_index:04d}_muxed.mp4"
            public_video = self._mux_audio(block_video, block_audio, muxed_video)
            return GeneratedBlock(
                session_id=request.session_id,
                turn_id=request.turn_id,
                block_index=request.block_index,
                video_path=str(public_video),
                audio_path=str(block_audio),
                frames=request.num_frames,
                fps=request.fps,
                latency_ms=round((time.perf_counter() - started) * 1000),
                mode="stableavatar_offline_block",
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            raise

