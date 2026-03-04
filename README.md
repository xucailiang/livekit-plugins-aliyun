# livekit-plugins-aliyun

[![PyPI version](https://badge.fury.io/py/livekit-plugins-aliyun.svg)](https://pypi.org/project/livekit-plugins-aliyun/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

[LiveKit Agents](https://github.com/livekit/agents) 阿里云插件，提供 STT、TTS、LLM 三大模块的完整集成。

## 特性

- **语音识别 (STT)** - Paraformer 实时语音识别，支持热词、说话人分离
- **语音合成 (TTS)** - CosyVoice 流式语音合成，低延迟、高质量
- **大语言模型 (LLM)** - 通义千问系列，OpenAI 兼容接口

## 安装

```bash
pip install livekit-plugins-aliyun
```

## 环境变量

```bash
export DASHSCOPE_API_KEY=your_api_key
```

## 快速开始

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import aliyun

async def entry_point(ctx: JobContext):
    await ctx.connect()
    
    session = AgentSession(
        stt=aliyun.STT(),
        tts=aliyun.TTS(),
        llm=aliyun.LLM(),
    )
    
    agent = Agent(instructions="你是一个友好的助手。")
    await session.start(agent=agent, room=ctx.room)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
```

## 模块独立使用

三个模块可以独立使用，按需导入：

```python
# 只使用 STT
from livekit.plugins.aliyun import STT
stt = STT(api_key="...")

# 只使用 TTS
from livekit.plugins.aliyun import TTS
tts = TTS(api_key="...")

# 只使用 LLM
from livekit.plugins.aliyun import LLM
llm = LLM(api_key="...")
```

## STT 配置

```python
from livekit.plugins import aliyun

stt = aliyun.STT(
    model="paraformer-realtime-v2",      # 模型
    language="zh",                        # 语言
    vocabulary_id="vocab_xxx",            # 热词表 ID（可选）
    max_sentence_silence=500,             # 句尾静音检测 (ms)
    disfluency_removal_enabled=False,     # 过滤语气词
    semantic_punctuation_enabled=False,   # 语义断句
)
```

## TTS 配置

```python
from livekit.plugins import aliyun

tts = aliyun.TTS(
    model="cosyvoice-v2",        # 模型
    voice="longxiaochun_v2",     # 音色
    sample_rate=24000,           # 采样率 (8000/16000/22050/24000/44100/48000)
    volume=50,                   # 音量 (0-100)
    rate=1.0,                    # 语速 (0.5-2.0)
    pitch=1.0,                   # 音调 (0.5-2.0)
)
```

### TTS 流式合成

插件实现了阿里云 CosyVoice WebSocket API 的完整流式能力：

- 单任务多文本：一个 stream 只需一次 run-task
- 服务端分句：自动分句，降低首字延迟
- 连接复用：WebSocket 连接池管理

```python
# 流式合成示例
stream = tts.stream()
stream.push_text("你好，")
stream.push_text("世界！")
stream.end_input()

async for event in stream:
    if event.frame:
        # 处理音频数据
        audio_data = event.frame.data
```

## LLM 配置

```python
from livekit.plugins import aliyun

llm = aliyun.LLM(
    model="qwen-plus",           # 模型 (qwen-plus/qwen-max/qwen-turbo)
    temperature=0.7,             # 温度
)
```

## 支持的模型

| 类型 | 模型 | 说明 |
| ---- | ---- | ---- |
| STT  | paraformer-realtime-v2 | 实时语音识别 |
| TTS  | cosyvoice-v2 | 流式语音合成 |
| LLM  | qwen-plus / qwen-max / qwen-turbo | 通义千问系列 |

## 官方文档

- [CosyVoice WebSocket API](https://www.alibabacloud.com/help/en/model-studio/cosyvoice-websocket-api)
- [Paraformer 实时语音识别](https://help.aliyun.com/zh/model-studio/paraformer-real-time-speech-recognition-api)
- [DashScope API](https://help.aliyun.com/zh/model-studio/)

## License

Apache 2.0
