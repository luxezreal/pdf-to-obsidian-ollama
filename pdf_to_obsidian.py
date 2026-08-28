"""
pdf_to_obsidian.py — PDF → Markdown via local Ollama vision model.

绕过 obsidian-llm-wiki 插件的 PDF 路径(插件对 Ollama 不支持 PDF),
直接用 pymupdf 抽每页 PNG,调 Ollama /api/generate 的 images[] 字段。

用法:
    pip install pymupdf requests
    python pdf_to_obsidian.py paper1.pdf textbook.pdf ...
    python pdf_to_obsidian.py --dir "D:\\papers"
    python pdf_to_obsidian.py --watch "D:\\inbox"  # 持续监听新 PDF

默认模型 bonsai-27b(我们合并好的 vision GGUF),默认输出到
C:\\Users\\30402\\Documents\\obsidian-vault\\pdf-imports\\
(改成你的 vault 路径)。
"""
from __future__ import annotations
import argparse
import base64
import json
import sys
import time
from pathlib import Path

import fitz  # pymupdf
import requests

# ============== 配置 ==============
OLLAMA_URL  = "http://localhost:11434"
DEFAULT_MODEL = "bonsai-27b"
DPI         = 150
NUM_CTX     = 8192
TIMEOUT_S   = 900
RETRY       = 2

# 输出目录: 改成你的 Obsidian vault 路径,放在 wiki 文件夹**外面**
# (插件 wiki 配置默认输出到 wiki/, 普通 markdown source 应该放别的目录)
DEFAULT_OUT = Path(r"C:\Users\30402\Documents\obsidian-vault\pdf-imports")

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
STOP_TOKENS = ["<EOS>", "<pad>", "<|endoftext|>", "<|im_end|>"]

# 流式循环检测:每收到 N 个 token chunk 检查一次 buffer 末尾是否开始
# 循环,命中立刻关闭连接。N 太小 → 检查开销大;太大 → 检测延迟。
# 16 ≈ 一行短句的 token 数,平衡选择。
STREAM_CHECK_INTERVAL = 16


def dedupe_repeat_runs(text: str, max_run: int = 4, min_line_len: int = 8) -> str:
    """截掉连续重复行(防止模型死循环占满单页时间)。

    判定:同一行(去前后空白)在文本里连续出现 ≥ max_run 次,
    且每行长度 ≥ min_line_len,就从第 max_run 次出现开始截断。
    短行(< min_line_len)不视为循环片段,通常是合法列表/分隔。

    例:输入含
        foo bar baz
        foo bar baz
        foo bar baz
        foo bar baz
        foo bar baz     <- 第 5 次,触发截断
        后续内容
    输出:
        foo bar baz
        foo bar baz
        foo bar baz
        foo bar baz
        (后续内容被丢弃)
    """
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
                    continue  # 跳过这一行,不写入
            else:
                prev = norm
                run = 1
            out.append(line)
        elif cut:
            continue
        else:
            # 短行:重置 run,但保留 prev 让长行判定连贯
            out.append(line)
            # 不重置 run,因为短行可能是循环片段的一部分(如 "## 标题")
            # 但因为长度 < min_line_len,下一行长内容会和 prev 比
    return "\n".join(out)


def render_page(page: fitz.Page, dpi: int = DPI) -> bytes:
    """PDF 页面 → PNG 字节。"""
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    return pix.tobytes("png")


def query_ollama(model: str, image_b64: str, user_prompt: str,
                 page_num: int, total: int) -> str:
    """调 Ollama /api/generate (流式 + 实时循环检测),带图片。失败自动重试。

    v3 改动 (2026-08-28):改用 stream=True + 每 N token 检查循环,
    命中立刻关闭 HTTP 连接(requests 内部 iter_lines 抛异常,被外层 try 捕获
    后退避重试)。预期把循环页耗时从 184s 降到 ~5s,见
    smoke-test-report-v2.md 的"已知未解决"章节。

    返回值仍是拼好的完整字符串(向后兼容)。
    """
    payload = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": user_prompt,
        "images": [image_b64],
        "stream": True,
        "options": {"num_ctx": NUM_CTX, "temperature": 0.2},
        # 关键: bonsai-27b 的 Modelfile 启用了 thinking,
        # 这里关掉避免把思维链吐到 Markdown 里
        "think": False,
        # Q1_0 量化在纯图页会触发 <EOS>/<pad> 死循环,
        # 显式 stop 让 Ollama 在这些 token 处截断(2026-08-28 smoke test)
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
                print(f"  retry {attempt}/{RETRY} after {wait}s: {e}", flush=True)
                time.sleep(wait)
    raise RuntimeError(f"Ollama failed for page {page_num}: {last_err}")


def _stream_with_loop_break(payload: dict, page_num: int) -> str:
    """流式调用 Ollama,实时检测并打断死循环。

    工作原理:
    1. stream=True,Ollama 返回 NDJSON,每行 `{"response":"<token>","done":false}`
    2. 累加 token 到 text_parts
    3. 每收到 STREAM_CHECK_INTERVAL 个 token,调 _is_loop_start() 检查末尾
    4. 命中循环 → r.close() 关闭连接,跳出循环,返回当前 buffer
       (dedupe_repeat_runs 兜底再清理一次)
    5. 正常结束 → 拼完所有 token 返回

    关闭连接后,Ollama 那边可能仍在生成下一个 token 但会被 TCP RST 截断。
    对用户体验:页 14 从 184s 降到 ~5s(取决于检测延迟)。

    性能:检测函数 O(K) per check,总复杂度 O(chunks × K),K=4 行,
    远低于网络/序列化开销。
    """
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
        # dedupe 兜底:检测函数只看末尾 4 行,关闭点之后可能还有几个
        # 完整循环段没被截,dedupe 一次性扫掉。
        return dedupe_repeat_runs(raw)
    return raw


def _is_loop_start(buffer: str, max_run: int = 4, min_line_len: int = 8) -> bool:
    """检查 buffer 末尾是否开始循环。只看最后 max_run 个**非空**行,O(K)。

    比 dedupe_repeat_runs 更激进:
    - dedupe 是"事后清理"(已经写出来才发现)
    - _is_loop_start 是"事前打断"(还没写完就 abort)

    阈值故意和 dedupe 一致(max_run=4, min_line_len=8),保证两个
    函数对"什么算循环"的判断标准一致。

    关键陷阱:模型循环时常输出 `<EOS><pad>\\n\\n`,在 splitlines 里产生
    "句子行 + 空行"交替。如果不看空行,末尾 4 行变成 "句子、空、句子、空",
    空行 length=0 < min_line_len,触发短路返回 False → 检测永不命中。
    必须先过滤空行再取末尾 max_run。
    """
    non_empty = [ln for ln in buffer.splitlines() if ln.strip()]
    if len(non_empty) < max_run:
        return False
    last = non_empty[-max_run:]
    norm = [ln.strip() for ln in last]
    if any(len(n) < min_line_len for n in norm):
        return False
    first = norm[0]
    return all(n == first for n in norm)


def pdf_to_markdown(pdf: Path, model: str, out_dir: Path) -> Path:
    """单个 PDF → 一个 .md 文件。返回输出路径。"""
    doc = fitz.open(pdf)
    total = len(doc)
    print(f"[{pdf.name}] {total} pages, model={model}", flush=True)

    chunks: list[str] = [
        f"# {pdf.stem}\n\n",
        f"- 源文件: `{pdf.name}`\n",
        f"- 页数: {total}\n",
        f"- 模型: `{model}`\n",
        f"- 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
    ]

    for i, page in enumerate(doc, 1):
        t0 = time.time()
        png = render_page(page)
        b64 = base64.b64encode(png).decode("ascii")
        prompt = f"这是文档第 {i}/{total} 页。请转录/摘要为 Markdown。"
        raw_text = query_ollama(model, b64, prompt, i, total)
        text = dedupe_repeat_runs(raw_text)
        truncated = len(raw_text) - len(text)
        dt = time.time() - t0
        chunks.append(f"\n---\n\n## 第 {i} 页\n\n{text}\n")
        trunc_note = f"  (截断 {truncated} 字符循环)" if truncated > 0 else ""
        print(f"  [{pdf.name}] page {i}/{total}  {dt:5.1f}s  "
              f"{len(text)} chars{trunc_note}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (pdf.stem + ".md")
    out.write_text("\n".join(chunks), encoding="utf-8")
    doc.close()
    print(f"[OK] {out}", flush=True)
    return out


def process_files(paths: list[Path], model: str, out_dir: Path) -> list[Path]:
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
            outputs.append(pdf_to_markdown(p, model, out_dir))
        except Exception as e:
            print(f"[FAIL] {p.name}: {e}", flush=True)
    return outputs


def watch_loop(dir_: Path, model: str, out_dir: Path, interval: int = 10):
    """简易 watch: 轮询目录,新出现的 PDF 自动处理。"""
    print(f"[watch] {dir_} → {out_dir}, model={model}", flush=True)
    seen: set[Path] = set()
    while True:
        for pdf in dir_.glob("*.pdf"):
            if pdf in seen:
                continue
            # 等文件不再被占用(简单 size 稳定检测)
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
                pdf_to_markdown(pdf, model, out_dir)
            except Exception as e:
                print(f"[FAIL] {pdf.name}: {e}", flush=True)
                seen.discard(pdf)  # 下轮重试
        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser(
        description="PDF → Markdown via local Ollama vision model "
                    "(bonsai-27b-vision or any multimodal)")
    ap.add_argument("pdfs", nargs="*", help="PDF files to process")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Ollama model name (default: {DEFAULT_MODEL})")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"Output directory (default: {DEFAULT_OUT})")
    ap.add_argument("--dir", type=Path,
                    help="Process all PDFs in this directory (one-shot)")
    ap.add_argument("--watch", type=Path,
                    help="Watch this directory, auto-process new PDFs")
    ap.add_argument("--check", action="store_true",
                    help="Just check Ollama + model reachability, exit")
    args = ap.parse_args()

    # 健康检查
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        tags = {m["name"] for m in r.json().get("models", [])}
        print(f"[ollama] {len(tags)} models available", flush=True)
        if args.model not in tags and not any(t.startswith(args.model) for t in tags):
            print(f"[warn] model '{args.model}' not found in ollama list, "
                  f"but will try anyway", flush=True)
    except Exception as e:
        print(f"[fatal] cannot reach Ollama at {OLLAMA_URL}: {e}", file=sys.stderr)
        sys.exit(2)

    if args.check:
        return

    args.out.mkdir(parents=True, exist_ok=True)

    if args.watch:
        watch_loop(args.watch, args.model, args.out)
        return

    paths: list[Path] = [Path(p) for p in args.pdfs]
    if args.dir:
        paths.extend(args.dir.glob("*.pdf"))
    if not paths:
        ap.error("give PDF files or --dir/--watch")

    outputs = process_files(paths, args.model, args.out)
    print(f"\n[done] {len(outputs)} files written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
