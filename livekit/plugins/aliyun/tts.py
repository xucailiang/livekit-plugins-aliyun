"""
Alibaba Cloud TTS (Text-to-Speech) Plugin for LiveKit Agents.

This module provides streaming text-to-speech synthesis using Alibaba Cloud's
CosyVoice WebSocket API. It implements the LiveKit TTS interface with full
support for streaming synthesis, connection pooling, and voice interruption handling.

Key Features:
    - Single task streaming: One run-task per stream with multiple continue-task messages
    - Single segment per instance: Compliant with LiveKit's latest framework requirements
    - Server-side sentence splitting: Leverages Alibaba Cloud's automatic text segmentation
    - Connection pooling: Efficient WebSocket connection reuse
    - Voice interruption support: Proper CancelledError handling for agent interruptions

Official Documentation References:
    - Alibaba Cloud CosyVoice WebSocket API:
      https://www.alibabacloud.com/help/en/model-studio/cosyvoice-websocket-api
    - Alibaba Cloud High-concurrency Scenarios:
      https://www.alibabacloud.com/help/en/model-studio/high-concurrency-scenarios
    - LiveKit Agents TTS API:
      https://docs.livekit.io/python/livekit/agents/tts/index.html

Example Usage:
    ```python
    from livekit.plugins.aliyun import TTS

    tts = TTS(
        api_key="your-api-key",
        voice="longcheng",
        model="cosyvoice-v2",
    )

    # Streaming synthesis
    stream = tts.stream()
    stream.push_text("Hello, world!")
    stream.mark_segment_end()

    async for event in stream:
        if isinstance(event, tts.SynthesizedAudio):
            # Process audio data
            pass
    ```

API Protocol Flow (per Alibaba Cloud documentation):
    1. Establish WebSocket connection
    2. Send run-task (generate unique task_id)
    3. Wait for task-started event
    4. Send multiple continue-task messages (same task_id)
    5. Send finish-task (same task_id)
    6. Wait for task-finished event

Important Constraints:
    - task_id must remain consistent throughout the task lifecycle
    - Text submission interval must not exceed 23 seconds
    - Connection cannot be reused after task-failed event
"""

import os
from dataclasses import dataclass
from typing import AsyncIterable, Optional, Dict
import time
import aiohttp
import asyncio
import json

from livekit.agents import tts, APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS, utils

from .log import logger


STREAM_EOS = "EOS"


@dataclass
class TTSOptions:
    """
    Configuration options for Alibaba Cloud TTS synthesis.

    This dataclass holds all configuration parameters for the TTS service,
    including API credentials, voice settings, and audio output parameters.
    It also provides methods to generate WebSocket API instruction payloads.

    Attributes:
        api_key: Alibaba Cloud DashScope API key for authentication.
        model: TTS model to use (e.g., "cosyvoice-v2", "cosyvoice-v1").
        voice: Voice ID for synthesis (e.g., "longcheng", "longhua").
        rate: Speech rate multiplier, range: 0.5-2.0 (1.0 = normal speed).
        volume: Audio volume level, range: 0-100 (50 = standard volume).
        sample_rate: Audio sample rate in Hz (8000, 16000, 22050, 24000, 44100, 48000).
        pitch: Voice pitch multiplier, range: 0.5-2.0 (1.0 = normal pitch).
        min_text_chunk_size: Minimum characters to buffer before sending.
            Set to 0 to send immediately (recommended, leverages server-side splitting).

    Official Documentation:
        https://www.alibabacloud.com/help/en/model-studio/cosyvoice-websocket-api

    Example:
        ```python
        opts = TTSOptions(
            api_key="your-api-key",
            model="cosyvoice-v2",
            voice="longcheng",
            rate=1.0,
            volume=50,
            sample_rate=24000,
        )
        ```
    """
    api_key: str
    model: str
    # 音色
    voice: str
    # 语速，取值范围：0.5~2.0，默认 1.0
    rate: float
    # 合成音频的音量，取值范围：0~100，默认 50
    volume: int
    # 采样率，取值范围：8000, 16000, 22050, 24000, 44100, 48000
    sample_rate: int
    # 音调，取值范围：0.5~2.0，默认 1.0
    pitch: float = 1.0
    # 文本聚合最小字符数，0 表示立即发送（利用服务端分句能力）
    min_text_chunk_size: int = 0

    def get_ws_url(self) -> str:
        return "wss://dashscope.aliyuncs.com/api-ws/v1/inference"

    def get_ws_header(self) -> Dict[str, str]:
        return {
            "Authorization": f"bearer {self.api_key}",
            "X-DashScope-DataInspection": "enable",
        }

    def get_run_task_params(self, task_id: str) -> Dict[str, str]:
        """
        生成 run-task 指令。

        重要：task_id 由外部传入，确保整个任务生命周期使用相同的 task_id。

        - input 字段必须存在，格式为空对象 {}
        - 不要在 run-task 中发送文本
        """
        params = {
            "header": {
                "action": "run-task",
                "task_id": task_id,
                "streaming": "duplex",
            },
            "payload": {
                "task_group": "audio",
                "task": "tts",
                "function": "SpeechSynthesizer",
                "model": self.model,
                "parameters": {
                    "text_type": "PlainText",
                    "voice": self.voice,
                    "format": "pcm",
                    "sample_rate": self.sample_rate,
                    "volume": self.volume,
                    "rate": self.rate,
                    "pitch": self.pitch,
                },
                "input": {},
            },
        }
        return params

    def get_continue_task_params(self, task_id: str, text: str) -> Dict[str, str]:
        """
        生成 continue-task 指令。

        重要：task_id 由外部传入，必须与 run-task 使用相同的 task_id。

        - 只能在收到 task-started 事件后发送
        - 文本提交间隔不能超过 23 秒
        """
        params = {
            "header": {
                "action": "continue-task",
                "task_id": task_id,
                "streaming": "duplex",
            },
            "payload": {
                "input": {
                    "text": text,
                }
            },
        }
        return params

    def get_finish_task_params(self, task_id: str) -> Dict[str, str]:
        """
        生成 finish-task 指令。

        重要：task_id 由外部传入，必须与 run-task 使用相同的 task_id。

        - 必须发送此指令，否则会导致音频不完整、连接超时、计费异常
        - input 字段必须存在，格式为空对象 {}
        """
        params = {
            "header": {
                "action": "finish-task",
                "task_id": task_id,
                "streaming": "duplex",
            },
            "payload": {"input": {}},
        }
        return params


class TTS(tts.TTS):
    """
    Alibaba Cloud TTS (Text-to-Speech) implementation for LiveKit Agents.

    This class provides streaming text-to-speech synthesis using Alibaba Cloud's
    CosyVoice WebSocket API. It manages WebSocket connection pooling and creates
    SynthesizeStream instances for actual synthesis operations.

    The implementation follows Alibaba Cloud's official API protocol:
    - Uses WebSocket duplex streaming for low-latency synthesis
    - Supports connection pooling for efficient resource utilization
    - Leverages server-side sentence splitting for optimal performance

    Attributes:
        _opts: TTSOptions instance containing configuration parameters.
        _pool: ConnectionPool for WebSocket connection management and reuse.
        _session: aiohttp ClientSession for HTTP/WebSocket operations.

    Official Documentation:
        - CosyVoice WebSocket API:
          https://www.alibabacloud.com/help/en/model-studio/cosyvoice-websocket-api
        - High-concurrency Scenarios:
          https://www.alibabacloud.com/help/en/model-studio/high-concurrency-scenarios

    Example:
        ```python
        from livekit.plugins.aliyun import TTS

        # Initialize TTS with custom settings
        tts = TTS(
            api_key="your-dashscope-api-key",
            voice="longcheng",
            model="cosyvoice-v2",
            sample_rate=24000,
        )

        # Create a streaming synthesis session
        stream = tts.stream()
        stream.push_text("你好，世界！")
        stream.mark_segment_end()

        async for event in stream:
            # Handle synthesis events
            pass
        ```

    Note:
        The API key can be provided directly or via the DASHSCOPE_API_KEY
        environment variable.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        sample_rate: int = 24000,
        voice: str = "longcheng",
        model: str = "cosyvoice-v2",
        volume: int = 50,
        rate: float = 1.0,
        pitch: float = 1.0,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: float = 600,
        min_text_chunk_size: int = 0,
    ) -> None:
        """
        Initialize the Alibaba Cloud TTS instance.

        Args:
            api_key: DashScope API key. If not provided, reads from
                DASHSCOPE_API_KEY environment variable.
            sample_rate: Audio sample rate in Hz. Supported values:
                8000, 16000, 22050, 24000, 44100, 48000. Default: 24000.
            voice: Voice ID for synthesis. Default: "longcheng".
            model: TTS model name. Default: "cosyvoice-v2".
            volume: Audio volume level (0-100). Default: 50 (standard volume).
            rate: Speech rate multiplier (0.5-2.0). Default: 1.0.
            pitch: Voice pitch multiplier (0.5-2.0). Default: 1.0.
            http_session: Optional aiohttp ClientSession for connection reuse.
            max_session_duration: Maximum WebSocket session duration in seconds
                before connection refresh. Default: 600 (10 minutes).
            min_text_chunk_size: Minimum text buffer size before sending.
                Set to 0 for immediate sending (recommended). Default: 0.

        Raises:
            ValueError: If api_key is not provided and DASHSCOPE_API_KEY
                environment variable is not set.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,
        )
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY must be set")
        self._session = http_session
        self._opts = TTSOptions(
            model=model,
            api_key=api_key,
            voice=voice,
            volume=volume,
            sample_rate=sample_rate,
            rate=rate,
            pitch=pitch,
            min_text_chunk_size=min_text_chunk_size,
        )
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=max_session_duration,
            mark_refreshed_on_get=True,
        )

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = utils.http_context.http_session()

        return self._session

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        url = self._opts.get_ws_url()
        headers = self._opts.get_ws_header()
        return await asyncio.wait_for(
            session.ws_connect(url, headers=headers),
            timeout=timeout,
        )

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse):
        await ws.close()

    def synthesize(
        self,
        text: str,
    ) -> AsyncIterable[tts.SynthesizedAudio]:
        raise NotImplementedError

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "SynthesizeStream":
        return SynthesizeStream(tts=self, opts=self._opts, conn_options=conn_options)


class SynthesizeStream(tts.SynthesizeStream):
    """
    阿里云 TTS 流式合成实现。

    符合 LiveKit 最新规范：
    - 单实例单 segment
    - 使用 ConnectionPool 复用 WebSocket 连接
    - 正确处理 CancelledError

    - task_id 在整个任务生命周期保持一致
    - 正确的指令顺序：run-task → task-started → continue-task* → finish-task → task-finished
    - sentence-synthesis 与 BINARY 帧一一对应
    """

    def __init__(
        self,
        *,
        tts: TTS,
        opts: TTSOptions,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = opts
        # Task state management events for coordinating send/recv tasks
        # Set when task-started event is received from server
        self._task_started = asyncio.Event()
        # Set when task-finished event is received from server
        self._task_finished = asyncio.Event()

    async def _run(self, emitter: tts.AudioEmitter) -> None:
        """
        主运行方法，协调发送和接收协程。

        生命周期：
        1. 生成唯一 task_id（整个流程共用）
        2. 初始化 emitter 并创建单个 segment
        3. 获取 WebSocket 连接
        4. 启动 send_task 和 recv_task 协程
        5. 等待完成或处理异常
        6. 清理资源
        """
        task_id = utils.shortuuid()
        send_task = None
        recv_task = None
        segment_started = False

        # 初始化 emitter（符合 LiveKit 规范）
        emitter.initialize(
            request_id=task_id,
            sample_rate=self._opts.sample_rate,
            mime_type="audio/pcm",
            stream=True,  # 流式模式
            num_channels=1,
            frame_size_ms=200,
        )

        try:
            async with self._tts._pool.connection(
                timeout=self._conn_options.timeout
            ) as ws:
                # 整个 stream 只创建一个 segment（符合 LiveKit 最新规范）
                segment_id = utils.shortuuid()
                emitter.start_segment(segment_id=segment_id)
                segment_started = True

                # 创建发送和接收任务
                send_task = asyncio.create_task(
                    self._send_task(ws, task_id),
                    name="TTS._send_task"
                )
                recv_task = asyncio.create_task(
                    self._recv_task(ws, emitter),
                    name="TTS._recv_task"
                )

                # 等待两个任务完成
                await asyncio.gather(send_task, recv_task)

        except asyncio.CancelledError:
            # 打断发生，清理资源后重新抛出
            logger.debug("TTS stream cancelled (interrupted)")
            raise

        except Exception as e:
            logger.error(f"TTS stream error: {e}")
            raise

        finally:
            # 确保任务被取消（使用 LiveKit 推荐的方式）
            await utils.aio.gracefully_cancel(send_task, recv_task)

            # 确保 segment 正确结束
            if segment_started:
                emitter.end_segment()

    async def _send_task(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        task_id: str
    ) -> None:
        """
        发送协程：处理 run-task, continue-task, finish-task。

        流程：
        1. 发送 run-task
        2. 等待 task-started 事件
        3. 循环发送 continue-task（从 _input_ch 读取文本）
        4. 发送 finish-task
        """
        send_timeout = self._conn_options.timeout or 20.0

        try:
            # 1. 发送 run-task
            run_task_params = self._opts.get_run_task_params(task_id)
            await asyncio.wait_for(
                ws.send_json(run_task_params),
                timeout=send_timeout
            )
            logger.debug(f"Sent run-task, task_id={task_id}")

            # 2. 等待 task-started 事件
            await asyncio.wait_for(
                self._task_started.wait(),
                timeout=send_timeout
            )
            logger.debug("Received task-started, ready to send text")

            # 3. 循环发送 continue-task
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    continue  # flush 信号，继续处理

                if data:  # 非空文本
                    continue_task_params = self._opts.get_continue_task_params(task_id, data)
                    await asyncio.wait_for(
                        ws.send_json(continue_task_params),
                        timeout=send_timeout
                    )
                    logger.debug(f"Sent continue-task, text length={len(data)}")

            # 4. 发送 finish-task
            finish_task_params = self._opts.get_finish_task_params(task_id)
            await asyncio.wait_for(
                ws.send_json(finish_task_params),
                timeout=send_timeout
            )
            logger.debug("Sent finish-task")

            # 等待 task-finished 事件
            await self._task_finished.wait()

        except asyncio.TimeoutError:
            logger.error(f"TTS send timeout, task_id={task_id}")
            raise
        except asyncio.CancelledError:
            logger.debug("TTS send task cancelled")
            raise

    async def _recv_task(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        emitter: tts.AudioEmitter
    ) -> None:
        """
        接收协程：处理服务端事件和音频数据。

        事件类型：
        - task-started: 任务开始，设置 _task_started 事件
        - result-generated: 合成结果，根据 type 分发处理
          - sentence-begin: 句子开始
          - sentence-synthesis: 音频块就绪，下一帧是 BINARY
          - sentence-end: 句子结束
        - task-finished: 任务完成，设置 _task_finished 事件
        - task-failed: 任务失败，关闭连接并抛出异常

        重要：sentence-synthesis 和 BINARY 帧一一对应
        """
        read_timeout = 10.0  # 读取超时
        expecting_binary = False  # 是否期待 BINARY 帧
        is_first_audio = True
        start_time = time.perf_counter()  # 首字延迟监控起始时间

        try:
            while True:
                try:
                    msg = await asyncio.wait_for(
                        ws.receive(),
                        timeout=read_timeout
                    )
                except asyncio.TimeoutError:
                    logger.error("TTS read timeout (connection may be dead)")
                    break

                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    event = data.get("header", {}).get("event")

                    # 5.1: task-started 事件处理
                    if event == "task-started":
                        logger.debug("Received task-started event")
                        self._task_started.set()

                    # 5.2: result-generated 事件处理
                    elif event == "result-generated":
                        output = data.get("payload", {}).get("output", {})
                        output_type = output.get("type")

                        if output_type == "sentence-begin":
                            original_text = output.get("original_text", "")
                            logger.debug(f"Sentence begin: {original_text[:50]}...")

                        elif output_type == "sentence-synthesis":
                            # 下一帧是对应的音频数据
                            expecting_binary = True

                        elif output_type == "sentence-end":
                            # 记录累计计费字符
                            usage = data.get("payload", {}).get("usage", {})
                            characters = usage.get("characters", 0)
                            logger.debug(f"Sentence end, total characters: {characters}")

                    # 5.4: task-finished 事件处理
                    elif event == "task-finished":
                        logger.debug("Received task-finished event")
                        # 记录最终计费信息
                        usage = data.get("payload", {}).get("usage", {})
                        characters = usage.get("characters", 0)
                        request_uuid = data.get("header", {}).get("attributes", {}).get("request_uuid", "")
                        logger.info(f"TTS task finished, characters={characters}, request_uuid={request_uuid}")
                        self._task_finished.set()
                        break

                    elif event == "task-failed":
                        error_code = data.get("header", {}).get("error_code", "Unknown")
                        error_msg = data.get("header", {}).get("error_message", "Unknown error")
                        logger.error(f"TTS task failed: {error_code} - {error_msg}")
                        # task-failed 后连接不可复用，主动关闭
                        await ws.close()
                        raise Exception(f"TTS task failed: {error_code} - {error_msg}")

                # 5.3: BINARY 帧处理
                # 注意：某些模型/配置下，服务端可能直接发送 BINARY 帧而不先发送 sentence-synthesis
                # 为了兼容性，我们接受所有 BINARY 帧
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    # 5.5: 首字延迟监控 - 记录首个音频帧延迟
                    if is_first_audio:
                        elapsed = time.perf_counter() - start_time
                        logger.info(f"TTS first audio received, latency={elapsed:.3f}s")
                        is_first_audio = False

                    # 记录未配对的 BINARY 帧（用于调试）
                    if not expecting_binary:
                        logger.debug("Received BINARY frame without preceding sentence-synthesis (compatibility mode)")

                    # 推送音频数据
                    emitter.push(data=msg.data)
                    expecting_binary = False  # 重置状态

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {msg.data}")
                    break

                elif msg.type == aiohttp.WSMsgType.CLOSE:
                    logger.warning("WebSocket connection closed by server")
                    break

        except asyncio.CancelledError:
            logger.debug("TTS recv task cancelled")
            raise
