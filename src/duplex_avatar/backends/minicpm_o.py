"""MiniCPM-o full-duplex adapter.

The avatar renderer and the conversational model have different lifecycles.  This
module keeps the MiniCPM-o side deliberately small and transport agnostic so the
session scheduler can feed one audio unit (normally one second) at a time and
consume ``listen``/``text``/``audio`` events.

Three modes are supported:

``mock``
    No model or network dependency.  Useful for exercising the duplex protocol.
``remote``
    The public MiniCPM-o Realtime WebSocket protocol (``/v1/realtime``).
``local``
    Lazy loading of the official ``MiniCPMODuplex`` implementation.  The model
    package is optional and is only imported after the first ``start`` call.

The adapter does not render avatar video.  Audio events are float32 mono PCM at
24 kHz (the MiniCPM-o Realtime API format), ready to be handed to a
StableAvatar/H3 rendering backend.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np


MiniCPMEventKind = Literal["listen", "text", "audio", "error", "session"]


@dataclass(slots=True)
class MiniCPMEvent:
    """One output branch emitted by MiniCPM-o.

    ``audio_pcm`` is little-endian float32 mono PCM.  Keeping bytes in the
    event avoids requiring the caller to import numpy and makes the event safe
    to put on an asyncio queue.  ``as_message`` is useful for WebSocket clients.
    """

    kind: MiniCPMEventKind
    text: str = ""
    audio_pcm: bytes | None = None
    audio_sample_rate: int = 24_000
    audio_format: str = "float32"
    end_of_turn: bool = False
    response_id: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    raw: dict[str, Any] | None = None

    def as_message(self) -> dict[str, Any]:
        """Serialize to the local session event shape."""

        message: dict[str, Any] = {
            "type": "minicpm_output",
            "kind": self.kind,
            "text": self.text,
            "sample_rate": self.audio_sample_rate,
            "audio_format": self.audio_format,
            "end_of_turn": self.end_of_turn,
            "metrics": self.metrics,
        }
        if self.response_id:
            message["response_id"] = self.response_id
        if self.audio_pcm is not None:
            message["audio_b64"] = base64.b64encode(self.audio_pcm).decode("ascii")
        if self.error:
            message["error"] = self.error
        return message


@dataclass(slots=True)
class MiniCPMOConfig:
    """Optional MiniCPM-o adapter settings.

    The fields intentionally mirror the public Realtime API.  A dict or an
    existing application ``BackendConfig`` can also be passed to
    :class:`MiniCPMOBackend`; unknown fields are ignored.
    """

    mode: str = "mock"
    model_path: str | None = None
    remote_url: str = "https://minicpmo45.modelbest.cn"
    remote_mode: str = "audio"
    api_key: str | None = None
    system_prompt: str = "You are a helpful voice assistant."
    ref_audio_path: str | None = None
    tts_ref_audio_path: str | None = None
    force_listen_count: int = 3
    max_new_speak_tokens_per_chunk: int = 20
    decode_mode: str = "greedy"
    length_penalty: float = 1.1
    response_timeout_s: float = 3.0
    response_grace_s: float = 0.08
    mock_text: str = "你好，我在听。"
    mock_audio_ms: int = 480
    device: str = "cuda"
    dtype: str = "bfloat16"
    audio_sample_rate: int = 16_000


def _get(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _config_from(config: Any = None, **overrides: Any) -> MiniCPMOConfig:
    defaults = MiniCPMOConfig()
    fields = {name: _get(config, name, getattr(defaults, name)) for name in MiniCPMOConfig.__dataclass_fields__}
    # The application historically calls this object ``BackendConfig`` and
    # selects implementations with ``name``.  Accept both spellings so the
    # brain can be wired without a config migration.
    configured_name = _get(config, "name", None)
    configured_mode = _get(config, "mode", None)
    if configured_mode is None and configured_name:
        name = str(configured_name).lower().replace("_", "-")
        if name in {"mock", "mock-brain"}:
            fields["mode"] = "mock"
        elif name in {"minicpm", "minicpm-o", "minicpm-remote", "minicpm-local"}:
            fields["mode"] = "local" if name.endswith("local") else "remote"
    if _get(config, "prompt", None) and not _get(config, "system_prompt", None):
        fields["system_prompt"] = _get(config, "prompt")
    # BackendConfig historically calls the model path ``model_base_path``.
    if fields["model_path"] is None:
        fields["model_path"] = _get(config, "model_base_path", None) or _get(config, "minicpm_model_path", None)
    fields["remote_url"] = _get(config, "remote_url", _get(config, "minicpm_remote_url", fields["remote_url"]))
    fields["remote_mode"] = _get(config, "remote_mode", _get(config, "minicpm_remote_mode", fields["remote_mode"]))
    for key, value in overrides.items():
        if value is not None and key in fields:
            fields[key] = value
    return MiniCPMOConfig(**fields)


def _float32_audio(data: bytes | bytearray | memoryview | np.ndarray, *, sample_rate: int = 16_000, channels: int = 1) -> np.ndarray:
    """Normalize an input chunk to a contiguous float32 mono waveform."""

    if isinstance(data, np.ndarray):
        array = np.asarray(data)
        if np.issubdtype(array.dtype, np.integer):
            scale = float(np.iinfo(array.dtype).max or 1)
            array = array.astype(np.float32) / scale
        else:
            array = array.astype(np.float32, copy=False)
    else:
        # The project WebSocket uses signed 16-bit PCM for input chunks.
        if len(data) % 2:
            raise ValueError("PCM16 audio payload must contain an even number of bytes")
        array = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        if array.size % channels:
            raise ValueError("audio payload does not contain complete channel frames")
        array = array.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(array.reshape(-1), dtype=np.float32)


def _audio_event(data: Any, *, sample_rate: int = 24_000, end_of_turn: bool = False, **kwargs: Any) -> MiniCPMEvent | None:
    if data is None:
        return None
    if hasattr(data, "detach"):
        data = data.detach().float().cpu().numpy()
    array = np.asarray(data, dtype=np.float32).reshape(-1)
    return MiniCPMEvent(
        kind="audio",
        audio_pcm=np.ascontiguousarray(array).tobytes(),
        audio_sample_rate=sample_rate,
        end_of_turn=end_of_turn,
        **kwargs,
    )


class MiniCPMOBackend:
    """Full-duplex MiniCPM-o backend with lazy local/remote loading.

    Usage::

        backend = MiniCPMOBackend(config)
        await backend.start(system_prompt="...")
        events = await backend.feed_audio(pcm16_bytes)
        await backend.interrupt()
        await backend.close()

    ``feed_audio`` is serialized because both the local model KV cache and the
    Realtime WebSocket session are stateful.  The renderer can independently
    consume each returned audio event and hand it to StableAvatar.
    """

    name = "minicpm-o"

    def __init__(self, config: Any = None, **overrides: Any):
        self.config = _config_from(config, **overrides)
        self.mode = self.config.mode.lower().replace("_", "-")
        if self.mode in {"minicpm", "minicpm-o", "minicpm-remote"}:
            # Explicitly named MiniCPM backend defaults to remote only when a
            # URL was supplied; otherwise local can be selected in config.
            self.mode = "remote" if self.config.remote_url else "local"
        self._model: Any = None
        self._ws: Any = None
        self._started = False
        self._closed = False
        self._load_error: str | None = None
        self._unit_count = 0
        self._force_listen_remaining = max(0, int(self.config.force_listen_count))
        self._lock = asyncio.Lock()
        # Interrupts must be able to arrive while ``feed_audio`` is waiting
        # for a remote response.  Keep this flag outside the serialization
        # lock; the next unit is also sent with ``force_listen``.
        self._interrupt_requested = asyncio.Event()
        self._remote_session_id: str | None = None
        self._ref_audio: np.ndarray | None = None

    async def health(self) -> dict[str, Any]:
        return {
            "ok": self.mode == "mock" or self._started,
            "backend": self.name,
            "mode": self.mode,
            "loaded": self._model is not None or self._ws is not None,
            "started": self._started,
            "error": self._load_error,
            "audio_input_sample_rate": self.config.audio_sample_rate,
            "audio_output_sample_rate": 24_000,
        }

    async def start(self, *, system_prompt: str | None = None, ref_audio_path: str | None = None, tts_ref_audio_path: str | None = None) -> None:
        """Initialize a local model or remote session; safe to call once per turn."""

        async with self._lock:
            if self._started:
                return
            if system_prompt:
                self.config.system_prompt = system_prompt
            self.config.ref_audio_path = ref_audio_path or self.config.ref_audio_path
            self.config.tts_ref_audio_path = tts_ref_audio_path or self.config.tts_ref_audio_path or self.config.ref_audio_path
            self._force_listen_remaining = max(0, int(self.config.force_listen_count))
            self._closed = False
            try:
                if self.mode == "mock":
                    pass
                elif self.mode == "remote":
                    await self._connect_remote()
                elif self.mode == "local":
                    await asyncio.to_thread(self._load_local)
                    await asyncio.to_thread(self._prepare_local)
                else:
                    raise ValueError(f"unknown MiniCPM-o mode: {self.mode!r}; choose mock, remote, or local")
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"
                raise
            self._started = True

    async def feed_audio(
        self,
        audio: bytes | bytearray | memoryview | np.ndarray,
        *,
        sample_rate: int | None = None,
        channels: int = 1,
        force_listen: bool = False,
        video_frames: Sequence[Any] | None = None,
    ) -> list[MiniCPMEvent]:
        """Feed one input unit and collect output events.

        Input bytes are signed little-endian PCM16, while numpy arrays are
        interpreted as float32 in [-1, 1].  The public Realtime endpoint
        expects float32 PCM16-kHz mono and emits 24-kHz float32 audio.
        """

        if not self._started:
            await self.start()
        async with self._lock:
            if self._closed:
                raise RuntimeError("MiniCPM-o session is closed")
            self._interrupt_requested.clear()
            source_rate = int(sample_rate or self.config.audio_sample_rate)
            waveform = _float32_audio(audio, sample_rate=source_rate, channels=channels)
            # The official backend currently requires 16 kHz.  Do a small
            # dependency-free resample for callers using another microphone rate.
            if source_rate != 16_000:
                waveform = self._resample(waveform, source_rate, 16_000)
            if force_listen:
                self._force_listen_remaining = max(self._force_listen_remaining, 1)
            self._unit_count += 1
            if self.mode == "mock":
                return self._mock_events(waveform)
            if self.mode == "remote":
                return await self._feed_remote(waveform, force_listen=force_listen, video_frames=video_frames)
            return await asyncio.to_thread(self._feed_local, waveform, force_listen, video_frames)

    async def interrupt(self) -> None:
        """Force the conversational model back to listen mode."""

        self._interrupt_requested.set()
        self._force_listen_remaining = max(1, self._force_listen_remaining)
        if self.mode in {"mock", "remote"}:
            return
        if self._model is not None:
            for method_name in ("duplex_set_break", "set_break"):
                method = getattr(self._model, method_name, None)
                if method is not None:
                    # Do not take ``_lock`` here: this method is specifically
                    # called while a generation unit may be in flight.
                    await asyncio.to_thread(method)
                    return

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            self._started = False
            if self._ws is not None:
                with contextlib.suppress(Exception):
                    await self._ws.send(json.dumps({"type": "session.close", "reason": "client_closed"}))
                with contextlib.suppress(Exception):
                    await self._ws.close()
            self._ws = None
            self._remote_session_id = None
            if self._model is not None:
                for method_name in ("duplex_stop", "stop"):
                    method = getattr(self._model, method_name, None)
                    if method is not None:
                        with contextlib.suppress(Exception):
                            await asyncio.to_thread(method)
                        break

    # -- remote Realtime protocol -------------------------------------------------

    def _remote_ws_url(self) -> str:
        url = self.config.remote_url.rstrip("/")
        parsed = urlsplit(url)
        scheme = {"http": "ws", "https": "wss"}.get(parsed.scheme, parsed.scheme)
        path = parsed.path.rstrip("/")
        if not path.endswith("/v1/realtime"):
            path += "/v1/realtime"
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query.setdefault("mode", self.config.remote_mode)
        return urlunsplit((scheme, parsed.netloc, path, urlencode(query), parsed.fragment))

    async def _connect_remote(self) -> None:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError("remote MiniCPM-o mode requires the optional 'websockets' package") from exc
        connect_kwargs: dict[str, Any] = {"max_size": 128 * 1024 * 1024}
        if self.config.api_key:
            # websockets renamed this keyword in v14.  Try the current name
            # first and fall back for the older dependency used by the demo.
            connect_kwargs["additional_headers"] = {"Authorization": f"Bearer {self.config.api_key}"}
        try:
            self._ws = await websockets.connect(self._remote_ws_url(), **connect_kwargs)
        except TypeError:
            if "additional_headers" not in connect_kwargs:
                raise
            connect_kwargs["extra_headers"] = connect_kwargs.pop("additional_headers")
            self._ws = await websockets.connect(self._remote_ws_url(), **connect_kwargs)
        payload: dict[str, Any] = {"system_prompt": self.config.system_prompt}
        if self.config.ref_audio_path:
            payload.setdefault("voice", {})["ref_audio_base64"] = self._read_audio_b64(self.config.ref_audio_path)
        if self.config.tts_ref_audio_path:
            payload.setdefault("voice", {})["tts_ref_audio_base64"] = self._read_audio_b64(self.config.tts_ref_audio_path)
        await self._ws.send(json.dumps({"type": "session.init", "payload": payload}, ensure_ascii=False))
        # queue_done may precede session.created; ignore progress events.
        deadline = time.monotonic() + self.config.response_timeout_s
        while time.monotonic() < deadline:
            event = await asyncio.wait_for(self._recv_json(), max(0.1, deadline - time.monotonic()))
            if event.get("type") == "session.created":
                self._remote_session_id = event.get("session_id")
                return
            if event.get("type") == "error":
                raise RuntimeError(self._remote_error(event))
        raise TimeoutError("timed out waiting for MiniCPM-o session.created")

    async def _feed_remote(self, waveform: np.ndarray, *, force_listen: bool, video_frames: Sequence[Any] | None) -> list[MiniCPMEvent]:
        if self._ws is None:
            raise RuntimeError("MiniCPM-o remote session is not connected")
        input_payload: dict[str, Any] = {
            "audio": base64.b64encode(np.ascontiguousarray(waveform, dtype="<f4").tobytes()).decode("ascii"),
            "force_listen": bool(force_listen or self._force_listen_remaining > 0),
        }
        if self.config.remote_mode == "video" and video_frames:
            input_payload["video_frames"] = [self._frame_b64(frame) for frame in video_frames]
        await self._ws.send(json.dumps({"type": "input.append", "input": input_payload}))
        if self._force_listen_remaining > 0:
            self._force_listen_remaining -= 1
        return await self._collect_remote_events()

    async def _collect_remote_events(self) -> list[MiniCPMEvent]:
        events: list[MiniCPMEvent] = []
        first = True
        while True:
            if self._interrupt_requested.is_set():
                events.append(MiniCPMEvent(kind="listen", metrics={"interrupted": True}))
                break
            timeout = float(self.config.response_timeout_s if first else self.config.response_grace_s)
            deadline = time.monotonic() + timeout
            try:
                # Poll briefly so an interrupt can stop a slow remote response
                # without waiting for the full response timeout.
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    try:
                        event = await asyncio.wait_for(self._recv_json(), min(remaining, 0.1))
                        break
                    except asyncio.TimeoutError:
                        if self._interrupt_requested.is_set():
                            raise
            except asyncio.TimeoutError:
                if first:
                    if self._interrupt_requested.is_set():
                        events.append(MiniCPMEvent(kind="listen", metrics={"interrupted": True}))
                    else:
                        events.append(MiniCPMEvent(kind="error", error="MiniCPM-o response timeout"))
                break
            first = False
            converted = self._convert_remote_event(event)
            if converted is not None:
                events.append(converted)
                # listen marks the end of this one-second unit in full duplex.
                if converted.kind == "listen":
                    break
            if event.get("type") in {"session.closed", "error"}:
                break
        return events

    async def _recv_json(self) -> dict[str, Any]:
        raw = await self._ws.recv()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        event = json.loads(raw)
        if not isinstance(event, dict):
            raise RuntimeError("MiniCPM-o backend event must be a JSON object")
        return event

    def _convert_remote_event(self, event: dict[str, Any]) -> MiniCPMEvent | None:
        if event.get("type") != "response.output.delta":
            if event.get("type") == "error":
                return MiniCPMEvent(kind="error", error=self._remote_error(event), raw=event)
            return None
        kind = event.get("kind")
        metrics = dict(event.get("metrics") or {})
        response_id = event.get("response_id")
        if kind == "listen":
            return MiniCPMEvent(kind="listen", response_id=response_id, metrics=metrics, raw=event)
        if kind == "text":
            return MiniCPMEvent(kind="text", text=str(event.get("text") or ""), response_id=response_id, metrics=metrics, raw=event)
        if kind == "audio":
            try:
                pcm = base64.b64decode(event.get("audio") or "", validate=True)
            except Exception as exc:
                return MiniCPMEvent(kind="error", error=f"invalid MiniCPM-o audio delta: {exc}", raw=event)
            return MiniCPMEvent(kind="audio", audio_pcm=pcm, audio_sample_rate=24_000, response_id=response_id, metrics=metrics, raw=event)
        return None

    @staticmethod
    def _remote_error(event: dict[str, Any]) -> str:
        error = event.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        return str(error or event.get("message") or "MiniCPM-o remote error")

    # -- local PyTorch implementation --------------------------------------------

    def _load_local(self) -> None:
        if self._model is not None:
            return
        if not self.config.model_path:
            raise ValueError("local MiniCPM-o mode requires model_path")
        try:
            from MiniCPMO45.modeling_minicpmo import MiniCPMODuplex
        except ImportError:
            try:
                from modeling_minicpmo import MiniCPMODuplex
            except ImportError as exc:
                raise RuntimeError("install or clone the MiniCPM-o repository so MiniCPMO45.modeling_minicpmo is importable") from exc
        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "attn_implementation": "sdpa",
            "audio_chunk_length": 1,
            "ls_mode": "explicit",
            "n_timesteps": 5,
            "force_listen_count": int(self.config.force_listen_count),
        }
        if self.config.device:
            kwargs["device_map"] = self.config.device
        try:
            self._model = MiniCPMODuplex.from_pretrained(self.config.model_path, **kwargs)
        except TypeError:
            # Some released local checkpoints do not accept ``device_map``;
            # let their own device setup handle placement in that case.
            kwargs.pop("device_map", None)
            self._model = MiniCPMODuplex.from_pretrained(self.config.model_path, **kwargs)
        if hasattr(self._model, "eval"):
            self._model.eval()

    def _prepare_local(self) -> None:
        if self._model is None:
            raise RuntimeError("local MiniCPM-o model is not loaded")
        ref_path = self.config.ref_audio_path or self.config.tts_ref_audio_path
        ref_audio = self._read_audio_array(ref_path, target_rate=16_000) if ref_path else None
        prefix = f"<|im_start|>system\n{self.config.system_prompt}\n<|audio_start|>"
        suffix = "<|audio_end|><|im_end|>"
        method = getattr(self._model, "prepare", None) or getattr(self._model, "duplex_prepare", None)
        if method is None:
            raise RuntimeError("MiniCPM-o local model does not expose prepare/duplex_prepare")
        kwargs = {"prefix_system_prompt": prefix, "suffix_system_prompt": suffix}
        if ref_audio is not None:
            kwargs["ref_audio"] = ref_audio
        if self.config.tts_ref_audio_path:
            kwargs["prompt_wav_path"] = self.config.tts_ref_audio_path
        try:
            method(**kwargs)
        except TypeError:
            # Older local checkpoints accept only the text prompt and path.
            kwargs.pop("ref_audio", None)
            method(**kwargs)
        self._ref_audio = ref_audio

    def _feed_local(self, waveform: np.ndarray, force_listen: bool, video_frames: Sequence[Any] | None) -> list[MiniCPMEvent]:
        if self._model is None:
            raise RuntimeError("MiniCPM-o local model is not loaded")
        if force_listen:
            for method_name in ("duplex_set_break", "set_break"):
                method = getattr(self._model, method_name, None)
                if method is not None:
                    method()
                    break
        prefill = getattr(self._model, "streaming_prefill", None) or getattr(self._model, "duplex_prefill", None)
        generate = getattr(self._model, "streaming_generate", None) or getattr(self._model, "duplex_generate", None)
        if prefill is None or generate is None:
            raise RuntimeError("MiniCPM-o local model does not expose streaming_prefill/streaming_generate")
        kwargs: dict[str, Any] = {"audio_waveform": waveform}
        if video_frames:
            kwargs["frame_list"] = list(video_frames)
        prefill(**kwargs)
        generate_kwargs: dict[str, Any] = {
            "max_new_speak_tokens_per_chunk": int(self.config.max_new_speak_tokens_per_chunk),
            "decode_mode": self.config.decode_mode,
            "length_penalty": float(self.config.length_penalty),
        }
        prompt_wav_path = self.config.tts_ref_audio_path or self.config.ref_audio_path
        if prompt_wav_path:
            generate_kwargs["prompt_wav_path"] = prompt_wav_path
        try:
            result = generate(**generate_kwargs)
        except TypeError:
            # Older snapshots use a smaller generate signature.
            generate_kwargs.pop("length_penalty", None)
            result = generate(**generate_kwargs)
        if not isinstance(result, Mapping):
            raise RuntimeError("MiniCPM-o local streaming_generate must return a mapping")
        metrics = {k: result[k] for k in ("cost_all_ms", "cost_llm_ms", "n_tokens", "current_time") if k in result}
        response_id = result.get("response_id")
        is_listen = bool(result.get("is_listen", False))
        events: list[MiniCPMEvent] = []
        if is_listen:
            events.append(MiniCPMEvent(kind="listen", end_of_turn=bool(result.get("end_of_turn", False)), metrics=metrics, response_id=response_id))
        else:
            text = str(result.get("text") or "")
            if text:
                events.append(MiniCPMEvent(kind="text", text=text, metrics=metrics, response_id=response_id))
            audio = result.get("audio_waveform", result.get("audio_data"))
            converted = _audio_event(audio, sample_rate=24_000, end_of_turn=bool(result.get("end_of_turn", False)), metrics=metrics, response_id=response_id)
            if converted is not None:
                events.append(converted)
        if self._force_listen_remaining > 0:
            self._force_listen_remaining -= 1
        return events

    # -- mock and utilities -------------------------------------------------------

    def _mock_events(self, waveform: np.ndarray) -> list[MiniCPMEvent]:
        if self._force_listen_remaining > 0:
            self._force_listen_remaining -= 1
            return [MiniCPMEvent(kind="listen", metrics={"mock": True, "unit": self._unit_count})]
        n = max(1, round(24_000 * max(1, self.config.mock_audio_ms) / 1000))
        t = np.arange(n, dtype=np.float32) / 24_000.0
        audio = (0.035 * np.sin(2 * np.pi * 220.0 * t)).astype("<f4").tobytes()
        return [
            MiniCPMEvent(kind="text", text=self.config.mock_text, metrics={"mock": True, "unit": self._unit_count}),
            MiniCPMEvent(kind="audio", audio_pcm=audio, audio_sample_rate=24_000, end_of_turn=True, metrics={"mock": True}),
        ]

    @staticmethod
    def _resample(array: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate == target_rate or len(array) == 0:
            return np.ascontiguousarray(array, dtype=np.float32)
        target_len = max(1, round(len(array) * target_rate / source_rate))
        source_t = np.arange(len(array), dtype=np.float64) / source_rate
        target_t = np.arange(target_len, dtype=np.float64) / target_rate
        return np.ascontiguousarray(np.interp(target_t, source_t, array).astype(np.float32))

    @staticmethod
    def _read_audio_array(path: str | Path, target_rate: int = 16_000) -> np.ndarray:
        import soundfile as sf
        samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
        array = samples.mean(axis=1)
        return MiniCPMOBackend._resample(array, int(rate), target_rate)

    @classmethod
    def _read_audio_b64(cls, path: str | Path) -> str:
        array = cls._read_audio_array(path, target_rate=16_000)
        return base64.b64encode(np.ascontiguousarray(array, dtype="<f4").tobytes()).decode("ascii")

    @staticmethod
    def _frame_b64(frame: Any) -> str:
        if isinstance(frame, str):
            return base64.b64encode(Path(frame).read_bytes()).decode("ascii")
        if isinstance(frame, (bytes, bytearray, memoryview)):
            return base64.b64encode(bytes(frame)).decode("ascii")
        raise TypeError("video_frames entries must be paths or JPEG/PNG bytes")


MockMiniCPMOBackend = MiniCPMOBackend


__all__ = ["MiniCPMEvent", "MiniCPMOConfig", "MiniCPMOBackend", "MockMiniCPMOBackend"]
