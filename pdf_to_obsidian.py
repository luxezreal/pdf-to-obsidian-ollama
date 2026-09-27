"""
pdf_to_obsidian_v2_minimax.py — PDF → Markdown via Ollama **或** MiniMax-M3。

基于原版 pdf_to_obsidian.py (luxezreal/pdf-to-obsidian-ollama, MIT)
新增能力:
  1. --provider ollama|minimax 双通道,OpenAI 兼容协议直连 M3 视觉端点
  2. --frontmatter 输出 YAML 前置块,适配 LLM Wiki 模式
  3. --vault-root 自动写 wiki/sources/{stem}.md,反链 raw/papers/

向后兼容:不传 --provider / --frontmatter 时行为与原版完全一致。

用法:
    # 原版 Ollama(不变)
    python pdf_to_obsidian_v2_minimax.py paper.pdf
    python pdf_to_obsidian_v2_minimax.py --dir D:\\papers --model bonsai-27b

    # 新版 M3(多模态)
    set M3_API_KEY=sk-xxxxxx
    python pdf_to_obsidian_v2_minimax.py paper.pdf --provider minimax

    # 适配本 vault 的 LLM Wiki 工作流
    python pdf_to_obsidian_v2_minimax.py paper.pdf ^
        --provider minimax --vault-root D:\\科研\\黄晨弘的知识库 --frontmatter

依赖:
    pip install pymupdf requests
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import fitz  # pymupdf
import requests


# ============== Ollama 配置(原版) ==============
OLLAMA_URL      = "http://localhost:11434"
DEFAULT_MODEL   = "bonsai-27b"
DPI             = 150
NUM_CTX         = 8192
TIMEOUT_S       = 900
RETRY           = 2

# ============== M3 配置(新增) ==============
# 端点/API Key/模型全部走环境变量,避免明文落盘。
# 在 ~/.bashrc / PowerShell $PROFILE 里 export 后即可。
DEFAULT_PROVIDER = "ollama"                       # ollama | minimax
M3_ENDPOINT      = os.getenv(
    "M3_ENDPOINT",
    "https://api.minimax.chat/v1/chat/completions",
)
M3_API_KEY       = os.getenv("M3_API_KEY", "")
M3_MODEL         = os.getenv("M3_MODEL", "MiniMax-M3")
M3_TIMEOUT_S     = 180
# M3 多模态单次请求图片上限;超长 PDF 自动分批
M3_MAX_IMAGES    = 8


# ============== 通用 Prompt ==============
SYSTEM_PROMPT = """你是文档 OCR + 摘要助手。严格遵循:
- 输出纯 Markdown,不用代码块包裹
- 保留原文标点、换行、段落
- 表格用 Markdown 表格语法
- 数学公式用 LaTeX($..$ 或 $$..$$)
- 不要"以下是..."、"这张图片展示了..."之类开场白
- 只输出本页内容,不跨页总结,不写"接下页..."之类过渡
"""


# Q1_0 量化缺陷的兜底:附录纯图页模型会进入 <EOS><pad> 循环,塞这些
# stop token 给 Ollama,见 smoke-test-report.md (页 13-15 案例)。
STOP_TOKENS = ["<EOS>", "<pad>", "", ""]

# 流式循环检测:每收到 N 个 token chunk 检查一次 buffer 末尾是否开始
# 循环,命中立刻关闭连接。
STREAM_CHECK_INTERVAL = 16


# ============================================================
# 文本清理与渲染(原版保留)
# ============================================================
def dedupe_repeat_runs(text: str, max_run: int = 4, min_line_len: int = 8) -> str:
    """截掉连续重复行(防止模型死循环占满单页时间)。"""
    lines = text.splitlines()
    out: list[str] = []
    prev: str | None = None
    run = 0
    cut = False
    for line in lines:
        norm = line.strip()
        if not cut and len(norm) >= min_line_len:
            if norm == prev:
                run += 1
                if run >= max_run:
                    cut = True
                    continue
            else:
                prev = norm
                run = 1
            out.append(line)
        elif cut:
            continue
        else:
            out.append(line)
    return "\n".join(out)


def render_page(page: fitz.Page, dpi: int = DPI) -> bytes:
    """PDF 页面 → PNG 字节。"""
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    return pix.tobytes("png")


def render_pdf_pages(pdf: Path) -> list[bytes]:
    """整本 PDF → 每页 PNG 字节列表。"""
    doc = fitz.open(pdf)
    pages = [render_page(p) for p in doc]
    doc.close()
    return pages


# ============================================================
# Provider 抽象(新增)
# ============================================================
def dispatch_query(
    provider: str,
    model: str,
    image_b64: str,
    user_prompt: str,
    page_num: int,
    total: int,
) -> str:
    """根据 provider 路由到对应后端。"""
    if provider == "ollama":
        return query_ollama(model, image_b64, user_prompt, page_num, total)
    elif provider == "minimax":
        return query_minimax(model, image_b64, user_prompt, page_num, total)
    else:
        raise ValueError(f"unknown provider: {provider!r}")


def health_check(provider: str, model: str) -> None:
    """对应 provider 的连通性 + 模型存在性检查。失败 sys.exit(2)。"""
    if provider == "ollama":
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            r.raise_for_status()
            tags = {m["name"] for m in r.json().get("models", [])}
            print(f"[ollama] {len(tags)} models available", flush=True)
            if model not in tags and not any(t.startswith(model) for t in tags):
                print(f"[warn] model '{model}' not found in ollama list, "
                      f"but will try anyway", flush=True)
        except Exception as e:
            print(f"[fatal] cannot reach Ollama at {OLLAMA_URL}: {e}",
                  file=sys.stderr)
            sys.exit(2)

    elif provider == "minimax":
        if not M3_API_KEY:
            print("[fatal] M3_API_KEY env var not set", file=sys.stderr)
            sys.exit(2)
        try:
            # 极小 payload 探活:1×1 透明 PNG + 文本问候
            tiny_png_b64 = (
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42m"
                "Nk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            )
            r = requests.post(
                M3_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {M3_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "ping"},
                            {"type": "image_url",
                             "image_url": {"url":
                                 f"data:image/png;base64,{tiny_png_b64}"}},
                        ],
                    }],
                    "max_tokens": 4,
                },
                timeout=15,
            )
            r.raise_for_status()
            print(f"[minimax] endpoint reachable, model={model}", flush=True)
        except Exception as e:
            print(f"[fatal] cannot reach M3 at {M3_ENDPOINT}: {e}",
                  file=sys.stderr)
            sys.exit(2)


# ============================================================
# Ollama 通道(原版,完全保留)
# ============================================================
def query_ollama(model: str, image_b64: str, user_prompt: str,
                 page_num: int, total: int) -> str:
    """调 Ollama /api/generate (流式 + 实时循环检测),带图片。失败自动重试。"""
    payload = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": user_prompt,
        "images": [image_b64],
        "stream": True,
        "options": {"num_ctx": NUM_CTX, "temperature": 0.2},
        "think": False,
        "stop": STOP_TOKENS,
    }
    last_err = None
    for attempt in range(1, RETRY + 2):
        try:
            return _stream_with_loop_break(payload, page_num)
        except (requests.RequestException, ValueError, KeyError) as e:
            last_err = e
            if attempt <= RETRY:
                wait = 5 * attempt
                print(f"  retry {attempt}/{RETRY} after {wait}s: {e}",
                      flush=True)
                time.sleep(wait)
    raise RuntimeError(f"Ollama failed for page {page_num}: {last_err}")


def _stream_with_loop_break(payload: dict, page_num: int) -> str:
    """流式调用 Ollama,实时检测并打断死循环。"""
    text_parts: list[str] = []
    chunks_since_check = 0
    aborted = False

    with requests.post(f"{OLLAMA_URL}/api/generate", json=payload,
                       stream=True, timeout=TIMEOUT_S) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk = data.get("response", "")
            if chunk:
                text_parts.append(chunk)
                chunks_since_check += 1
                if chunks_since_check >= STREAM_CHECK_INTERVAL:
                    chunks_since_check = 0
                    buffer = "".join(text_parts)
                    if _is_loop_start(buffer):
                        print(f"  [page {page_num}] 循环检测触发 "
                              f"({len(buffer)} chars),关闭连接", flush=True)
                        aborted = True
                        r.close()
                        break
            if data.get("done"):
                break

    raw = "".join(text_parts)
    if aborted:
        return dedupe_repeat_runs(raw)
    return raw


def _is_loop_start(buffer: str, max_run: int = 4, min_line_len: int = 8) -> bool:
    """检查 buffer 末尾是否开始循环。"""
    non_empty = [ln for ln in buffer.splitlines() if ln.strip()]
    if len(non_empty) < max_run:
        return False
    last = non_empty[-max_run:]
    norm = [ln.strip() for ln in last]
    if any(len(n) < min_line_len for n in norm):
        return False
    first = norm[0]
    return all(n == first for n in norm)


# ============================================================
# M3 通道(新增)
# ============================================================
def query_minimax(model: str, image_b64: str, user_prompt: str,
                  page_num: int, total: int) -> str:
    """调 M3 /v1/chat/completions,OpenAI 兼容协议,带图片。失败自动重试。

    与 Ollama 路径差异:
    - 不流式(M3 多模态流式 + 图片通常不稳,同步拿整段更简单可靠)
    - 协议是 messages[].content[] 而非顶层 prompt + images[]
    - 不需要 dedupe 兜底(消费级 M3 没有 Q1_0 量化死循环)
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ]},
        ],
        "temperature": 0.2,
        "max_tokens": 4096,
    }
    last_err = None
    for attempt in range(1, RETRY + 2):
        try:
            r = requests.post(
                M3_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {M3_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=M3_TIMEOUT_S,
            )
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except (requests.RequestException, ValueError, KeyError) as e:
            last_err = e
            if attempt <= RETRY:
                wait = 5 * attempt
                print(f"  retry {attempt}/{RETRY} after {wait}s: {e}",
                      flush=True)
                time.sleep(wait)
    raise RuntimeError(f"M3 failed for page {page_num}: {last_err}")


# ============================================================
# Frontmatter 生成(新增)
# ============================================================
def build_frontmatter(
    pdf: Path,
    model: str,
    provider: str,
    vault_root: Path | None,
) -> str:
    """生成 YAML frontmatter,适配 LLM Wiki 模式。

    frontmatter 形如:
        ---
        title: "attention-is-all-you-need"
        type: source
        created: 2026-09-27
        updated: 2026-09-27
        tags:
          - pdf
          - source
        sources:
          - "[[raw/papers/attention-is-all-you-need.pdf]]"
        ---
    """
    today = time.strftime("%Y-%m-%d")
    title = pdf.stem.replace("-", " ").replace("_", " ")
    # 推断源文件路径(vault 模式下假定在 raw/papers/)
    source_link = f'"[[raw/papers/{pdf.name}]]"'
    fm = (
        "---\n"
        f'title: "{title}"\n'
        "type: source\n"
        f"created: {today}\n"
        f"updated: {today}\n"
        "tags:\n"
        "  - pdf\n"
        "  - source\n"
        f"provider: {provider}\n"
        f"model: \"{model}\"\n"
        "sources:\n"
        f"  - {source_link}\n"
        "---\n\n"
    )
    return fm


# ============================================================
# 主流程
# ============================================================
def pdf_to_markdown(
    pdf: Path,
    model: str,
    out_dir: Path,
    provider: str = DEFAULT_PROVIDER,
    vault_root: Path | None = None,
    frontmatter: bool = False,
) -> Path:
    """单个 PDF → 一个 .md 文件。返回输出路径。"""
    pages_png = render_pdf_pages(pdf)
    total = len(pages_png)
    print(f"[{pdf.name}] {total} pages, provider={provider}, model={model}",
          flush=True)

    today = time.strftime("%Y-%m-%d %H:%M:%S")
    chunks: list[str] = [
        f"# {pdf.stem}\n\n",
        f"- 源文件: `{pdf.name}`\n",
        f"- 页数: {total}\n",
        f"- Provider: `{provider}`\n",
        f"- 模型: `{model}`\n",
        f"- 生成时间: {today}\n\n",
    ]

    for i, png in enumerate(pages_png, 1):
        t0 = time.time()
        b64 = base64.b64encode(png).decode("ascii")
        prompt = f"这是文档第 {i}/{total} 页。请转录/摘要为 Markdown。"
        text = dispatch_query(provider, model, b64, prompt, i, total)
        if provider == "ollama":
            # Ollama 路径仍走死循环兜底(向后兼容)
            text = dedupe_repeat_runs(text)
        dt = time.time() - t0
        chunks.append(f"\n---\n\n## 第 {i} 页\n\n{text}\n")
        print(f"  [{pdf.name}] page {i}/{total}  {dt:5.1f}s  "
              f"{len(text)} chars", flush=True)

    # 决定输出路径
    if vault_root and frontmatter:
        out_dir = vault_root / "wiki" / "sources"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (pdf.stem + ".md")

    body = "\n".join(chunks)
    if frontmatter:
        body = build_frontmatter(pdf, model, provider, vault_root) + body
    out.write_text(body, encoding="utf-8")
    print(f"[OK] {out}", flush=True)
    return out


def process_files(
    paths: list[Path],
    model: str,
    out_dir: Path,
    provider: str = DEFAULT_PROVIDER,
    vault_root: Path | None = None,
    frontmatter: bool = False,
) -> list[Path]:
    """处理多个文件,返回生成的 .md 列表。"""
    outputs: list[Path] = []
    for p in paths:
        if not p.exists():
            print(f"[skip] not found: {p}", flush=True)
            continue
        if p.suffix.lower() != ".pdf":
            print(f"[skip] not PDF: {p}", flush=True)
            continue
        try:
            outputs.append(pdf_to_markdown(
                p, model, out_dir, provider, vault_root, frontmatter,
            ))
        except Exception as e:
            print(f"[FAIL] {p.name}: {e}", flush=True)
    return outputs


def watch_loop(
    dir_: Path,
    model: str,
    out_dir: Path,
    provider: str = DEFAULT_PROVIDER,
    vault_root: Path | None = None,
    frontmatter: bool = False,
    interval: int = 10,
):
    """简易 watch: 轮询目录,新出现的 PDF 自动处理。"""
    print(f"[watch] {dir_} → {out_dir}, provider={provider}, "
          f"model={model}", flush=True)
    seen: set[Path] = set()
    while True:
        for pdf in dir_.glob("*.pdf"):
            if pdf in seen:
                continue
            try:
                size1 = pdf.stat().st_size
                time.sleep(2)
                size2 = pdf.stat().st_size
                if size1 != size2:
                    continue
            except OSError:
                continue
            seen.add(pdf)
            try:
                pdf_to_markdown(
                    pdf, model, out_dir, provider, vault_root, frontmatter,
                )
            except Exception as e:
                print(f"[FAIL] {pdf.name}: {e}", flush=True)
                seen.discard(pdf)
        time.sleep(interval)


# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description=(
            "PDF → Markdown via Ollama 或 MiniMax-M3 视觉通道 "
            "(支持任意多模态模型)"
        ),
    )
    ap.add_argument("pdfs", nargs="*", help="PDF files to process")
    ap.add_argument("--provider", default=DEFAULT_PROVIDER,
                    choices=["ollama", "minimax"],
                    help=f"后端选择 (default: {DEFAULT_PROVIDER})")
    ap.add_argument("--model", default=None,
                    help=(
                        f"模型名;Ollama 默认 {DEFAULT_MODEL},"
                        f"M3 默认 {M3_MODEL}"
                    ))
    ap.add_argument("--out", type=Path, default=None,
                    help="输出目录 (vault_root 模式下被忽略)")
    ap.add_argument("--dir", type=Path,
                    help="处理该目录下所有 PDF (一次性)")
    ap.add_argument("--watch", type=Path,
                    help="监听该目录,自动处理新增 PDF")
    ap.add_argument("--vault-root", type=Path,
                    help=(
                        "Obsidian vault 根路径;开启后输出强制写到 "
                        "{vault_root}/wiki/sources/,配合 --frontmatter 使用"
                    ))
    ap.add_argument("--frontmatter", action="store_true",
                    help="输出 YAML frontmatter,适配 LLM Wiki 模式")
    ap.add_argument("--check", action="store_true",
                    help="只检查 provider + model 可达性,退出")
    args = ap.parse_args()

    # 默认模型按 provider 选
    if args.model is None:
        args.model = M3_MODEL if args.provider == "minimax" else DEFAULT_MODEL

    # 默认输出目录
    if args.out is None:
        if args.vault_root and args.frontmatter:
            args.out = args.vault_root / "wiki" / "sources"
        else:
            args.out = Path(
                r"C:\Users\30402\Documents\obsidian-vault\pdf-imports"
            )

    # 健康检查
    health_check(args.provider, args.model)

    if args.check:
        return

    args.out.mkdir(parents=True, exist_ok=True)

    if args.watch:
        watch_loop(
            args.watch, args.model, args.out,
            args.provider, args.vault_root, args.frontmatter,
        )
        return

    paths: list[Path] = [Path(p) for p in args.pdfs]
    if args.dir:
        paths.extend(args.dir.glob("*.pdf"))
    if not paths:
        ap.error("give PDF files or --dir/--watch")

    outputs = process_files(
        paths, args.model, args.out,
        args.provider, args.vault_root, args.frontmatter,
    )
    print(f"\n[done] {len(outputs)} files written to {args.out}", flush=True)


if __name__ == "__main__":
    main()