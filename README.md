# livekit-plugins-aliyun 飞舟定制版

[![PyPI version](https://badge.fury.io/py/livekit-plugins-aliyun.svg)](https://pypi.org/project/livekit-plugins-aliyun/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

阿里云服务专用的 [LiveKit Agents](https://github.com/livekit/agents) 插件，提供完整的语音和语言模型集成解决方案。

## ✨ 特性

- 🎤 **语音识别 (STT)** - 支持阿里云Paraformer语音识别服务
- 🗣️ **语音合成 (TTS)** - 支持阿里云CosyVoice文本转语音服务
- 🤖 **大语言模型 (LLM)** - 支持阿里云Qwen系列大模型
- 🔧 **热词功能** - 支持STT热词识别增强
- 📦 **开箱即用** - 完整的 Python 包支持

## 📋 支持的服务

| 服务 | 描述 | 文档链接 |
|------|------|----------|
| TTS | 文本转语音 | [阿里云TTS](https://bailian.console.aliyun.com/model-market?capabilities=%5B%22TTS%22%5D) |
| STT | 语音识别 | [阿里云ASR](https://bailian.console.aliyun.com/model-market?capabilities=%5B%22ASR%22%5D) |
| LLM | 大语言模型 | [阿里云LLM](https://bailian.console.aliyun.com/model-market) |

## 🛠️ 安装

### 使用 pip 安装

```bash
pip install livekit-plugins-aliyun
```

### 从源码安装

```bash
git clone https://github.com/itqdtgcom/livekit-plugins-aliyun.git
cd livekit-plugins-volcengine
pip install -e ./livekit-plugins/livekit-plugins-aliyun
```

### 系统要求

- Python >= 3.9
- LiveKit Agents >= 1.2.9

## ⚙️ 配置

### 环境变量

在使用插件前，请配置以下环境变量：

| 环境变量 | 描述 | 获取方式 |
|----------|------|----------|
| `DASHSCOPE_API_KEY` | DashScope API 密钥 | [阿里云控制台](https://bailian.console.aliyun.com/) |

### .env 文件示例

```bash
# .env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

## 📖 使用指南

### 基础使用

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import aliyun
from dotenv import load_dotenv

async def entry_point(ctx: JobContext):
    agent = Agent(instructions="You are a helpful assistant.")

    session = AgentSession(
        # 语音识别
        stt=aliyun.STT(model="paraformer-realtime-v2"),
        # 语音合成
        tts=aliyun.TTS(model="cosyvoice-v2", voice="longcheng_v2"),
        # 大语言模型
        llm=aliyun.LLM(model="qwen-plus")
    )

    await session.start(agent=agent, room=ctx.room)
    await ctx.connect()

if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
```

### STT热词功能

阿里云STT支持热词功能，可以提高特定词汇的识别准确率：

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import aliyun
from dotenv import load_dotenv

async def entry_point(ctx: JobContext):
    agent = Agent(instructions="You are a helpful assistant.")

    session = AgentSession(
        # 配置热词功能的STT
        stt=aliyun.STT(
            model="paraformer-realtime-v2",
            vocabulary_id="your_vocabulary_id"  # 热词表ID
        ),
        tts=aliyun.TTS(model="cosyvoice-v2", voice="longcheng_v2"),
        llm=aliyun.LLM(model="qwen-plus")
    )

    await session.start(agent=agent, room=ctx.room)
    await ctx.connect()

if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
```

### 高级配置

```python
from livekit.plugins import aliyun

# 自定义TTS配置
tts = aliyun.TTS(
    model="cosyvoice-v2",
    voice="longcheng_v2",  # 语音类型
    speech_rate=1.0,      # 语速 (0.5-2.0)
    pitch_rate=1.0,       # 音调 (0.5-2.0)
    volume=50             # 音量 (0-100)
)

# 自定义LLM配置
llm = aliyun.LLM(
    model="qwen-max",     # 模型名称
    temperature=0.7,      # 温度
    max_tokens=2000       # 最大token数
)

# 自定义STT配置
stt = aliyun.STT(
    model="paraformer-realtime-v2",
    vocabulary_id="your_vocabulary_id",  # 热词表ID
    format="wav",         # 音频格式
    sample_rate=16000     # 采样率
)
```

## 🔧 API 参考

### TTS (文本转语音)

```python
aliyun.TTS(
    model: str = "cosyvoice-v2",      # 模型名称
    voice: str = "longcheng_v2",      # 语音类型
    speech_rate: float = 1.0,        # 语速 (0.5-2.0)
    pitch_rate: float = 1.0,         # 音调 (0.5-2.0)
    volume: int = 50                 # 音量 (0-100)
)
```

### STT (语音识别)

```python
aliyun.STT(
    model: str = "paraformer-realtime-v2",  # 模型名称
    vocabulary_id: str = None,        # 热词表ID
    format: str = "wav",             # 音频格式
    sample_rate: int = 16000         # 采样率
)
```

### LLM (大语言模型)

```python
aliyun.LLM(
    model: str = "qwen-plus",        # 模型名称
    temperature: float = 0.7,        # 温度
    max_tokens: int = 2000           # 最大token数
)
```
# livekit-plugins-aliyun
