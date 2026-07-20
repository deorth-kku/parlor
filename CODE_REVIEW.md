# Parlor 代码审查报告

> **审查日期**: 2026-07-20  
> **审查范围**: 全仓库 (Python 后端 + 前端 JS/HTML/CSS)  
> **项目类型**: 实时多模态 AI 语音+视觉聊天应用

---

## 📊 总体评价

| 维度 | 评分 | 说明 |
|------|------|------|
| **架构设计** | ⭐⭐⭐☆☆ | 功能完整但耦合度高、全局单例多、难以测试 |
| **代码质量** | ⭐⭐⭐☆☆ | 有死代码、竞态条件、缺乏类型提示 |
| **安全性** | ⭐⭐☆☆☆ | 无请求大小限制、潜在提示注入、DoS 风险 |
| **可维护性** | ⭐⭐☆☆☆ | 单文件过大、重复代码多、无模块化 |
| **部署就绪度** | ⭐⭐☆☆☆ | 无健康检查、依赖系统库、无 Docker 支持 |

---

## 🔴 严重问题 (Critical)

### 1. 全局单例模式导致测试/部署困难
**文件**: `src/server.py:32-33`

```python
backend: OpenAICompatibleBackend | None = None
tts_backend = None
```

**问题**:
- 多进程部署时每个 worker 独立初始化，资源浪费
- 热重载时状态不一致
- 单元测试无法隔离、Mock 困难

**修复建议**: 使用 FastAPI `app.state` + `lifespan` 依赖注入
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.backend = await OpenAICompatibleBackend.from_env()
    app.state.tts_backend = await load_tts()
    yield
    await app.state.backend.aclose()

# 使用时
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket, backend: OpenAICompatibleBackend = Depends(get_backend)):
    ...
```

---

### 2. TTS 后端竞态条件 (猴子补丁非线程安全)
**文件**: `src/tts.py:331-333`

```python
def _install_run_options(self) -> None:
    ...
    self.model._create_audio = _create_audio_with_run_options  # 竞态点
```

**问题**: 并发请求会互相覆盖 `_create_audio` 方法，导致错误的运行配置被应用。

**修复**: 使用线程局部存储或请求级会话，避免共享可变状态。

---

### 3. `zipja` 函数死代码 - 日语 TTS 可能损坏
**文件**: `src/tts.py:400-405`

```python
def zipja(s: str):
    half = len(s) // 2
    return s[:half]  # 直接返回前半部分！
    return "".join(a + b for a, b in zip(s[:half], s[half:]))  # 永远不执行
```

**影响**: 日语音素处理逻辑错误，可能导致日语语音质量下降或报错。

**修复**: 删除死代码，实现正确的音素压缩逻辑或移除该函数。

---

### 4. 无请求大小限制 - DoS 漏洞
**文件**: `src/server.py` (WebSocket 端点)

- 无 `max_size` 限制 WebSocket 消息
- Base64 音频/图片载荷可任意大
- 可导致 OOM 崩溃

**修复**:
```python
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    # 限制单条消息 10MB
    data = await ws.receive_text(max_size=10_000_000)
    ...
```

---

### 5. 潜在提示注入风险
**文件**: `src/server.py:127-130`

```python
parts.append({
    "type": "text",
    "text": build_language_instruction(tts_language, tts_voice)
})
```

**问题**: `tts_language`/`tts_voice` 直接插入系统提示词，虽有目录校验但未做转义。

**修复**: 使用严格白名单校验 + 模板化提示词。

---

## 🟠 重要问题 (Major)

### 6. 硬编码模型路径 + 强制联网下载
**文件**: `src/tts.py:117-119`

```python
self._model_path = hf_hub_download("xun/kokoro-v1.1-zh-onnx", "onnx/kokoro-v1.1-zh.onnx")
self._voices_path = hf_hub_download("fastrtc/kokoro-onnx", "voices-v1.0.bin")
self._config_path = hf_hub_download("hexgrad/Kokoro-82M-v1.1-zh", "config.json")
```

**问题**:
- 启动必须联网，无离线模式
- 模型仓库硬编码，无法自定义
- 无本地缓存策略文档

**建议**: 支持环境变量覆盖路径，提供本地模式文档。

---

### 7. 紧耦合、无抽象层
**文件**: `src/server.py`

```python
from openai_backend import OpenAICompatibleBackend
import tts
```
直接依赖具体实现，难以：
- 替换 TTS/LLM 后端
- 编写单元测试 (Mock 困难)
- 扩展新提供商

**建议**: 定义 `Protocol` 接口 + 工厂模式。

---

### 8. 阻塞调用在异步上下文中
**文件**: `src/server.py:160-165`

```python
pcm = await asyncio.get_event_loop().run_in_executor(
    None,  # 使用默认线程池 (无界!)
    lambda s=sentence, lang=language, selected_voice=voice: tts_backend.generate(s, lang, selected_voice),
)
```

**问题**:
- `None` 创建无界 `ThreadPoolExecutor`，高并发下线程耗尽
- TTS 为 CPU/GPU 密集型，应用 `ProcessPoolExecutor`

**修复**:
```python
# 启动时创建有界进程池
app.state.tts_executor = ProcessPoolExecutor(max_workers=2)

# 使用时
pcm = await loop.run_in_executor(app.state.tts_executor, tts_backend.generate, ...)
```

---

### 9. 错误处理缺失 / 裸露异常捕获
**文件**: `src/openai_backend.py:170-180`

```python
except Exception as exc:
    print(f"[openai_backend] streaming parse error: {exc}")
    parse_error = exc  # 继续带着错误状态处理
```

**问题**:
- 捕获所有异常但不分类处理
- 无结构化日志、无指标、无告警
- 错误状态下继续执行可能产生脏数据

---

### 10. 前端 HTML 重复代码严重 ✅ 已修复
**文件**: `public/index.html` (原 10-90 行)

**原问题**:
- 桌面端 + 移动端**完整重复**两套 DOM 结构
- 同 ID 元素出现两次 (`viewportWrap`, `video`, `languageSelect` 等)
- `document.getElementById()` 只返回第一个，移动端布局失效

**修复方式** (2026-07-20):
- 合并为单套响应式 DOM (`.app-layout` + `.sidebar` + `.center-col`)
- 桌面端使用左侧 `.sidebar`，移动端 (`max-width: 1023px`) 隐藏 sidebar、显示底部 `.controls`
- 移动端仅复制 Language/Voice/Video 三个 `<select>` (带 `Mobile` 后缀 ID)，由 `app.js` 同步 desktop ↔ mobile 选项与值
- 移除 `$$` 的布局切换逻辑、`syncMessagesBetweenLayouts()`、`updateLatestUserMessage` 的跨布局镜像逻辑
- 所有元素 ID 现唯一，避免 `getElementById` 歧义

---

### 11. 依赖地狱 / 版本冲突风险
**文件**: `src/pyproject.toml`

```toml
dependencies = [
    "misaki[en,zh]>=0.9.4",
    "fugashi[unidic-lite]>=1.5.2",
    "mecab-python3>=1.0.12",      # 需要系统 mecab
    "pyopenjtalk-plus>=0.4.1.post8",
    "kokoro-onnx>=0.5.0",         # 可能锁定特定 onnxruntime 版本
    "torch",                       # 无版本锁定 - 极高风险
    "av>=17.1.0",                  # 需要系统 ffmpeg
]
```

**风险**:
- `torch` 无版本 → 下一个 breaking release 会崩
- `mecab-python3` 需要系统库 `mecab` + `mecab-ipadic`，README 未提及
- `fugashi[unidic-lite]` 安装时下载 ~500MB 模型
- 依赖间可能存在版本冲突

---

### 12. 无健康检查 / 就绪探针
- 无 `/health`、 `/ready` 端点
- 无法接入 k8s / 负载均衡器

---

### 13. WebSocket 重连策略过于简单
**文件**: `public/app.js:300+`

```javascript
ws.onclose = () => {
    setState('disconnected');
    setTimeout(connect, 2000);  // 固定 2s，无指数退避、无抖动、无最大重试
};
```

**问题**: 服务端重启时所有客户端同时重连 → 连接风暴。

---

### 14. TTS 端点代码重复
**文件**: `src/server.py:250-350` 与 `450-550`

- `/v1/audio/speech` (OpenAI 兼容) 和内部 WebSocket TTS **逻辑 90% 相同**
- 应提取公共 `generate_tts_audio()` 函数

---

### 15. 类型提示不完整
- 大量 `dict[str, object]`、`Any` 类型
- 缺乏 `TypedDict` / Pydantic 模型定义请求/响应结构

---

## 🟡 中等问题 (Moderate)

### 16. 基于 `print()` 的日志
全代码库使用 `print()`，无：
- 结构化日志 (JSON)
- 日志级别 (DEBUG/INFO/WARN/ERROR)
- 关联 ID (追踪请求链路)
- 无法接入观测栈

**建议**: 引入 `structlog` + JSON 输出。

---

### 17. 核心逻辑缺乏测试
- `test_sentence_splitter.py` 仅覆盖分句器
- 无测试：TTS 后端、OpenAI 后端解析、WebSocket 消息处理、语音目录校验

---

### 18. 前端单文件过大
**文件**: `public/app.js` ~1000 行

应拆分为模块：
- `ws.js` - WebSocket 连接管理
- `audio.js` - 音频播放/可视化
- `vad.js` - VAD 逻辑
- `ui.js` - DOM 操作
- `state.js` - 状态机

---

### 19. 环境变量验证缺失
**文件**: `src/openai_backend.py`

```python
def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default  # "abc" -> 默认值，静默失败
```

**问题**: 无效值静默回退，配置错误难发现。

**建议**: 使用 Pydantic `BaseSettings` 启动时校验。

---

### 20. 缺少 Docker / 部署支持
- 无 `Dockerfile`
- 无 `docker-compose.yml`
- 系统依赖 (mecab, ffmpeg) 未文档化

---

## 📋 架构改进建议

| 领域 | 现状 | 建议 |
|------|------|------|
| **配置管理** | `os.getenv()` 分散 | Pydantic Settings (`BaseSettings`) |
| **依赖注入** | 全局单例 | FastAPI `Depends()` + `lifespan` |
| **TTS 后端** | 具体 `ONNXBackend` 类 | `Protocol` + 工厂模式 |
| **LLM 后端** | 具体 `OpenAICompatibleBackend` | `Protocol` + 多实现 |
| **日志系统** | `print()` | `structlog` + JSON |
| **测试策略** | 极少 | Pytest + 单元/集成测试 |
| **前端构建** | 单 `app.js` | Vite + TypeScript + 模块化 |
| **部署方式** | `uv run` 手动 | Dockerfile + 健康检查 |

---

## 🎯 优先修复清单 (建议按顺序)

| 优先级 | 任务 | 预估工时 |
|--------|------|----------|
| **P0** | 修复 `zipja` 死代码 | 10 分钟 |
| **P0** | 移除全局单例，改用 `app.state` | 1 小时 |
| **P0** | 添加 WebSocket 消息大小限制 | 30 分钟 |
| **P0** | 修复 TTS 猴子补丁竞态条件 | 1 小时 |
| **P1** | 引入 Pydantic Settings 配置校验 | 2 小时 |
| **P1** | 添加 `/health` `/ready` 端点 | 30 分钟 |
| **P1** | 去重 TTS 端点代码 | 1 小时 |
| **P1** | 修复 HTML 重复 DOM ID | 1 小时 |
| **P2** | 拆分前端 `app.js` 模块化 | 3 小时 |
| **P2** | 添加结构化日志 | 2 小时 |
| **P2** | 编写核心逻辑单元测试 | 4 小时 |
| **P3** | 编写 Dockerfile + docker-compose | 2 小时 |
| **P3** | 添加指数退避重连逻辑 | 1 小时 |

---

## 📦 依赖风险清单

| 包 | 风险 | 缓解措施 |
|-----|------|----------|
| `torch` | 无版本锁，下一个大版本破坏性变更 | `torch==2.4.*` 或明确版本 |
| `mecab-python3` | 需系统 `mecab` + `mecab-ipadic` | Dockerfile 安装系统包，README 文档化 |
| `fugashi[unidic-lite]` | 安装下载 500MB+ | 预构建镜像包含模型 |
| `av` (PyAV) | 需系统 `ffmpeg` | Dockerfile `apt-get install ffmpeg` |
| `kokoro-onnx` | 锁定特定 onnxruntime | 验证版本兼容性矩阵 |

---

## 📝 后续行动建议

1. **立即**: 创建 GitHub Issues 对应每个 P0/P1 项
2. **本周**: 完成 P0 修复，建立 CI 流水线
3. **下周**: 引入配置管理、结构化日志、健康检查
4. **本月**: 前端模块化、Docker 化、测试覆盖率 > 60%

---

*报告生成于 2026-07-20 | 如有疑问请提 Issue 讨论*