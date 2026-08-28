# pdf-to-obsidian-ollama

**一个独立的命令行工具(不是 Obsidian 插件)**:用本地 Ollama vision 模型把 PDF 转成 Markdown,直接落到 Obsidian vault,绕开 obsidian-llm-wiki 插件对 Ollama PDF 不支持的限制。

整个仓库只有一个脚本 `pdf_to_obsidian.py`,不到 400 行,没有构建步骤,不需要装进 Obsidian,`python pdf_to_obsidian.py paper.pdf` 就能跑。它和插件是互补关系而不是替代关系,见文末"与 Karpathy LLM Wiki 插件的关系"。

## 仓库内容

| 文件 | 作用 |
| --- | --- |
| `pdf_to_obsidian.py` | 主工具,CLI 入口 |
| `bench_loop_detection.py` | `_is_loop_start` 循环检测微基准,不调 Ollama,秒级跑完防回归 |
| `smoke-test-report-v4.md` | v4 真实回归报告:15 页论文 7m 10s,页 14 循环 187s → 23.8s |

## 它解决什么问题

`obsidian-llm-wiki`(Andrej Karpathy 的 LLM Wiki 插件)对 Ollama provider 调用 PDF 路径时会抛 `UnsupportedProviderError`:插件底层假设 PDF 走的是云端 OpenAI / Anthropic / Gemini 的多模态接口,没有为本地 Ollama 实现图片 + 文档的分支。本脚本独立完成以下事:

- 用 pymupdf 把 PDF 每页渲染成 PNG
- 把 PNG 编码成 base64,塞进 Ollama `/api/generate` 的 `images[]` 字段
- 把模型返回的 Markdown 写到 Obsidian vault 目录里

等于把"插件 → 云端 PDF OCR"的链路换成"CLI → 本地 Ollama → vault"。整个过程数据不出本机,不需要 token,不需要把论文上传到任何云端。

## 快速开始

先准备环境:

```bash
# 1. 安装 Python 依赖
pip install pymupdf requests

# 2. 确认 Ollama 在跑(本机端口 11434)
curl http://localhost:11434/api/tags

# 3. 拉 vision 模型(任选其一)
ollama pull bonsai-27b          # 本仓库默认
ollama pull llava:13b
ollama pull minicpm-v

# 4. 跑一次健康检查
python pdf_to_obsidian.py --check

# 5. 处理 PDF
python pdf_to_obsidian.py paper.pdf
```

要求:Python 3.8+(用到 `from __future__ import annotations` 和 `set[Path]` 类型注解),Ollama 0.1.20+(支持 `images[]` 字段的 `/api/generate`)。

## 命令详解

### `python pdf_to_obsidian.py paper.pdf`

把单个 PDF 转成同名的 `.md`,输出到 `--out` 目录(默认 `vault/pdf-imports/`)。可以同时传多个文件,脚本会按顺序串行处理。

```bash
python pdf_to_obsidian.py attention-is-all-you-need.pdf resnet.pdf
```

输出文件名规则:`<pdf_stem>.md`,例如 `paper.pdf` → `paper.md`,同名旧文件会被覆盖。

### `--dir D:\papers`

一次性处理目录下所有 `.pdf`(不递归子目录,实现是 `Path.glob("*.pdf")`)。已经存在的同名 `.md` 会被覆盖,不会做 diff。

```bash
python pdf_to_obsidian.py --dir "D:\papers"
```

注意:如果路径含空格或中文,Windows 下需要加双引号。Posix 路径用正斜杠也行。

### `--watch D:\inbox`

进入轮询模式:每 10 秒扫一遍目录,发现新出现的 `.pdf` 就处理。对每个文件先做一次 2 秒的 size 稳定检测(读 size → 等 2s → 再读 size),避免读到正在被复制中的半成品文件。Ctrl+C 中断。

```bash
python pdf_to_obsidian.py --watch "D:\inbox"
```

`seen` 集合只在内存里,重启脚本会重置。失败的 PDF 会从 `seen` 里 `discard` 掉,下轮重试。

### `--model <name>`

指定 Ollama 模型名。默认 `bonsai-27b`。任意多模态 / vision GGUF 都行,例如 `llava:13b`、`minicpm-v`、`qwen2-vl`。模型未在 `/api/tags` 列表里时只 warn 不退出,会照常尝试。

```bash
python pdf_to_obsidian.py paper.pdf --model llava:13b
python pdf_to_obsidian.py paper.pdf --model minicpm-v:Q4_K_M
```

### `--out <dir>`

指定 Markdown 输出目录。默认是脚本顶部 `DEFAULT_OUT` 常量。目录不存在会自动 `mkdir(parents=True, exist_ok=True)`。

```bash
python pdf_to_obsidian.py paper.pdf --out D:\vault\00-inbox
python pdf_to_obsidian.py paper.pdf --out /home/me/obsidian/pdf-imports
```

路径分隔符 Windows 用 `\`,Posix 用 `/`,`pathlib.Path` 都接受。

### `--check`

只做健康检查:GET Ollama `/api/tags`,列出已加载模型数,确认目标 `--model` 是否在列表里(不在也只警告不退出,因为可能模型还没 pull 但别名能匹配)。处理完后直接退出,不读任何 PDF、不创建输出目录。适合放进 cron / CI 做探活。

```bash
python pdf_to_obsidian.py --check --model bonsai-27b
```

## 配置说明

所有可调常量都在脚本顶部 `============== 配置 ==============` 区块,改完保存即可生效,无需命令行参数:

| 常量 | 默认值 | 调什么场景 |
| --- | --- | --- |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama 不在默认端口、跑在远端机器、或前置 nginx 反代时改这里 |
| `DEFAULT_MODEL` | `bonsai-27b` | 想换默认 vision 模型(例如显存不够换更小的 llava:7b) |
| `DPI` | `150` | 渲染分辨率,数字越大越清晰但 base64 payload 越大,详见"性能与质量调优" |
| `NUM_CTX` | `8192` | Ollama 上下文 token 数,长表格 / 长公式页可能要提到 16384,否则模型会截断 |
| `TIMEOUT_S` | `900`(15 分钟) | 单页推理超时,慢机器 / 大模型卡住时给足等待时间 |
| `RETRY` | `2` | 单页推理失败时的重试次数(实际行为是 1 跑 + 2 重试,共 3 次机会) |
| `DEFAULT_OUT` | `vault/pdf-imports/` | 输出目录,改完源码就能换默认 vault 路径,不需要每次都传 `--out` |
| `SYSTEM_PROMPT` | 见源码 | OCR + 摘要指令模板,改它就改了模型行为,例如要求"保留原文语言"或"输出双语" |

`SYSTEM_PROMPT` 当前内容(中英混排方便阅读):

```text
你是文档 OCR + 摘要助手。严格遵循:
- 输出纯 Markdown,不用代码块包裹
- 保留原文标点、换行、段落
- 表格用 Markdown 表格语法
- 数学公式用 LaTeX($..$ 或 $$..$$)
- 不要"以下是..."、"这张图片展示了..."之类开场白
- 只输出本页内容,不跨页总结,不写"接下页..."之类过渡
```

## 工作原理

```
+------------------+      +----------------+      +---------------------+
| PDF file on disk | ---> | pymupdf render | ---> | PNG bytes per page  |
+------------------+      +----------------+      +---------------------+
                                                         |
                                                         v
                                              +---------------------+
                                              | base64.b64encode()  |
                                              +---------------------+
                                                         |
                                                         v
+---------------------+   POST /api/generate   +---------------------------+
| Ollama              | <--------------------- | payload {                 |
|   bonsai-27b / llava|                         |   model, system, prompt,  |
|   / qwen2-vl / ...  | ---------------------> |   images: [b64str],       |
+---------------------+   response              |   think: False,           |
                                |               |   options: {num_ctx, ...}|
                                v               | }                        |
                       +---------------------+  +---------------------------+
                       | Markdown per page   |
                       +---------------------+
                                |
                                v
                       +---------------------+
                       | join chunks +       |
                       | write .md to vault  |
                       +---------------------+
```

payload 结构(JSON):

```json
{
  "model": "bonsai-27b",
  "system": "你是文档 OCR + 摘要助手。严格遵循: ...",
  "prompt": "这是文档第 3/12 页。请转录/摘要为 Markdown。",
  "images": ["iVBORw0KGgoAAAANSUhEUgAA..."],
  "stream": false,
  "options": {
    "num_ctx": 8192,
    "temperature": 0.2
  },
  "think": false
}
```

关键设计:

- 调用的是 Ollama 原生 `/api/generate`,不是 OpenAI 兼容的 `/v1/chat/completions`,这样 `images[]` 字段是 Ollama 一等公民,不会被 OpenAI SDK 当成多模态消息格式拒掉。
- `stream: false` 让响应一次性返回完整 JSON,`r.json()["response"]` 直接拿到模型输出。流式需要处理 `ndjson` chunk,代码复杂度不值。
- payload 强制 `think: False`。`bonsai-27b` 的 Modelfile 默认开启了 chain-of-thought,如果不关,模型会把 `<think>...</think>` 当成 Markdown 内容吐出来,污染正文。
- 每页独立 prompt,模型不会跨页总结,符合 `SYSTEM_PROMPT` 里的"只输出本页内容"约束。
- 失败重试:网络异常 / JSON 解析失败 / 字段缺失都触发,等 5s/10s 后重试,共 3 次机会(1 跑 + 2 重试)。
- 输出 .md 结构:标题 + 元信息(源文件 / 页数 / 模型 / 时间) + 每个页 `## 第 N 页` 章节 + 页间 `---` 分隔。

## 与 Obsidian 集成

`DEFAULT_OUT` 故意指向 `vault/pdf-imports/`,而不是 `vault/wiki/`:

- 插件 wiki 文件夹是 obsidian-llm-wiki 自己管理的目录,有 frontmatter、有索引文件,手动写进去会和插件冲突。
- `pdf-imports/` 放在 vault 根目录下,Obsidian 自动索引,你可以在 vault 里自己加 dataview / tag 查询。
- 建议结构:
  ```
  obsidian-vault/
    pdf-imports/         # 本脚本输出
      attention-is-all-you-need.md
      resnet.md
    wiki/                # LLM Wiki 插件管理,别动
    notes/               # 你自己的笔记
    inbox/               # 如果用 --watch 模式,这里放新 PDF
  ```
- watch 模式适合当 inbox:把 `D:\inbox` 设成下载 / 截图软件的输出目录,新 PDF 自动转,人工定期 review `pdf-imports/` 里的 .md。
- 输出的 .md 头部已经包含源文件名、页数、模型、时间四个 frontmatter-ish 字段(用列表而不是 YAML frontmatter,因为 LLM Wiki 插件只认自己写的 frontmatter 格式):

  ```markdown
  # attention-is-all-you-need

  - 源文件: `attention-is-all-you-need.pdf`
  - 页数: 12
  - 模型: `bonsai-27b`
  - 生成时间: 2026-08-28 14:32:11

  ---

  ## 第 1 页

  ...模型输出...

  ---

  ## 第 2 页
  ...
  ```

- 想在 Obsidian 里搜所有 PDF 导入的笔记,可以加 tag 模板或者用 dataview:

  ````markdown
  ```dataview
  LIST FROM "pdf-imports"
  SORT file.mtime DESC
  ```
  ````

## 性能与质量调优

| 参数 | 取舍 |
| --- | --- |
| `DPI` | 150 是甜点。提到 200 文本清晰度提升有限,但 base64 payload 涨 ~80%,显存不够的模型会 OOM;降到 100 小字体会糊,中文尤其明显 |
| `NUM_CTX` | 8192 够一页论文用。如果你的页面里有跨页表格 / 大量公式被截断,改 16384(显存占用 +50%) |
| `think: False` | 必须开。`bonsai-27b` 默认 thinking 模型,不关会把推理过程写进 Markdown,正文会被 `<think>...</think>` 块污染 |
| `RETRY` | 网络抖动场景可以提到 3,代价是单页失败要等多 30s 才放弃 |
| `TIMEOUT_S` | 900s(15 分钟)是给慢机器兜底。如果显卡强、模型小,可以压到 120 让卡死的请求更快 fail |
| temperature | 硬编码 0.2,在源码 `query_ollama()` 里。要更确定的输出改 0.0,要 OCR 更有创造性改 0.4 |
| 模型选择 | 显存 8GB → llava:7b / minicpm-v;16GB → bonsai-27b Q4_K_M / llava:13b;24GB+ → qwen2-vl:72b |
| 量化精度 | Q4_K_M 是质量 / 体积甜点。Q1_0 省显存但中文 OCR 退化严重,生产别用 |

## 已知限制

- **Q1_0 1-bit 量化中文 OCR 准确率约 30–40%**。`bonsai-27b` 的 Q1_0 量化版对中文字符识别很弱,大量形近字混淆。**解决**:换 Q4_K_M 以上的量化(`bonsai-27b:Q4_K_M`),或换专门 OCR 模型(`PaddleOCR`、`MinerU`),或用云端多模态(OpenAI GPT-4o / Gemini Flash)。
- **单进程串行处理,不并发**。12 页论文约 1 分钟(每页 4–8s,取决于 GPU)。改成并发需要每个页独立请求 + 写时加锁,目前没做。要并发可以用 xargs 或者自己包一层 multiprocessing。
- **仅支持 PDF**。DOCX / EPUB / 图片集不支持。`pymupdf` 支持的图片型 PDF(扫描件)能跑,但要走 OCR 强的模型。
- **不支持加密 PDF**。`fitz.open()` 遇到带密码的 PDF 会直接报错,没做解密提示。先用 qpdf / pdftk 解密再喂进来。
- **没有增量**。重新跑同一个 PDF 会覆盖同名 `.md`,不会做 diff。处理过 30 页的长论文需要小心,建议先备份。
- **watch 模式不做去重持久化**。重启脚本后 `seen` 集合清空,会重新处理上次没处理完的文件。配 `seen.discard(pdf)` 已经在失败时重置,但进程级状态不跨重启。要持久化可以加一个 `.seen` 文件存 hash。
- **没做并发写锁**。多进程同时写同一个输出目录是 race condition,但当前 CLI 不暴露这种用法,只在 `--watch` 里串行处理。

## 故障排查

| 错误现象 | 原因 | 解决 |
| --- | --- | --- |
| `[fatal] cannot reach Ollama at http://localhost:11434` | Ollama 服务没起 | `ollama serve` 或在系统托盘启动 Ollama Desktop |
| `[warn] model 'X' not found` | 模型没 pull | `ollama pull X`,或换 `--model` 改成已下载的模型名 |
| 输出 .md 是 `<think>...</think>...` 思维链 | 模型启用了 thinking | 在 Modelfile 里加 `PARAMETER num_think 0`,或确认本脚本 payload 已带 `think: False`(本仓库已强制) |
| 中文输出是乱码 / 形近字 | Q1_0 量化精度不够 | 换 Q4_K_M 以上量化,或换专门 OCR 模型,详见"已知限制" |
| 巨 PDF 跑到一半 OOM / 系统卡死 | 单页 PNG base64 太大撑爆显存 | 降 `DPI` 到 100;或拆 PDF;或用 MinerU / Marker 这类流式工具 |
| HTTP 403 或持续超时 | Ollama 进程卡死 | `pkill ollama && ollama serve &`,或重启 Ollama Desktop |
| `--watch` 不触发 | 文件还在被复制 | 脚本本身有 2s size 稳定检测,确认复制完成的 PDF 才会被识别;若仍不触发,检查 glob 路径权限 |
| 输出全是英文 / 翻译后的内容而非原文 | 模型默认走"翻译"行为 | 在 `SYSTEM_PROMPT` 里加 `- 保留原文语言,不翻译` |
| `[FAIL] Ollama failed for page N: timeout` | 单页推理超过 900s | 检查 `nvidia-smi` 显存占用,可能需要 `OLLAMA_NUM_PARALLEL=1` 降并发,或换小模型 |
| `--dir` 不递归子目录 | 设计就是只 glob 当前层 | 自己写 `find ... -name "*.pdf"` 然后 xargs 喂进来 |
| 输出 .md 没有 YAML frontmatter | 故意只用 Markdown 列表维护元信息 | 想要 frontmatter 改 `pdf_to_markdown()` 里 `chunks` 拼装那几行 |

## 与 Karpathy LLM Wiki 插件的关系

插件路径:https://github.com/karpathy/obsidian-llm-wiki

插件的设计假设是:LLM provider 都支持 OpenAI 兼容的 chat/completions + image url。但 Ollama 的 OpenAI 兼容端点对 PDF / base64 图片支持是逐步加的,直到最近的版本才稳定,在此之前 PDF 路径会直接抛 `UnsupportedProviderError`。

本脚本的存在只是补这个 gap,**不是插件的替代品**:

- 插件负责 wiki 索引、链接、frontmatter 维护
- 本脚本只负责把 PDF 灌进 vault
- 两者配合:本脚本输出原始 Markdown,再交给插件做索引 / 摘要 / 链接

未来插件如果原生支持 Ollama PDF(参考其 issue tracker),你可以选:

- **保留本脚本**:继续走本地推理,数据不出本机,适合敏感文档
- **切回插件**:省去 CLI 调用,但走云端 API,token 成本 + 数据外泄风险

按你的隐私 / 成本权衡选。本脚本的代码量不到 200 行,扔掉成本很低。

## License

MIT,见 [LICENSE](LICENSE)。
