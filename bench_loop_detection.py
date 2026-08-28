"""bench_loop_detection.py — 独立验证 _is_loop_start 在真实分块模式下的行为。

不调 Ollama,直接喂模拟 chunk 流,看检测是否触发以及多快触发。
"""
import sys
import time
from pathlib import Path

# 从同目录的 pdf_to_obsidian 导入
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdf_to_obsidian import (
    STREAM_CHECK_INTERVAL,
    _is_loop_start,
    _stream_with_loop_break,
    dedupe_repeat_runs,
)


def case_single_loop_sentence_realistic_chunks():
    """模拟真实场景:模型输出 1 句有意义内容,然后进入循环。
    每个 chunk 是一个 token 级字符串(可能含空格、换行、<EOS>等)。"""
    SENTENCE = "The Law will never be perfect, but its application should be just this is what we are missing, in my opinion."
    # 把句子切成接近真实 token 的 chunk(平均 4 字符,含前导空格)
    import re
    # 简单切:每 3-5 字符一片
    chunks_prefix: list[str] = []
    words = SENTENCE.split(" ")
    for w in words:
        chunks_prefix.append(" " + w)
    chunks_prefix.append(" <EOS>")
    chunks_prefix.append(" <pad>")
    chunks_prefix.append("\n")
    chunks_prefix.append("\n")  # 句间空行

    print(f"  句子切分: {len(chunks_prefix)} chunks/重复")

    # 模拟:第 1 句正常输出,然后重复 N 次进入循环
    N_REPEATS = 50
    full_chunks = chunks_prefix * (N_REPEATS + 1)  # 1 次正常 + N 次循环

    # 模拟流式检测
    text_parts: list[str] = []
    chunks_since_check = 0
    detected_at_iter: int | None = None
    detected_at_chars: int | None = None

    for i, chunk in enumerate(full_chunks):
        text_parts.append(chunk)
        chunks_since_check += 1
        if chunks_since_check >= STREAM_CHECK_INTERVAL:
            chunks_since_check = 0
            buffer = "".join(text_parts)
            if _is_loop_start(buffer):
                detected_at_iter = i
                detected_at_chars = len(buffer)
                break

    if detected_at_iter is None:
        print(f"  ❌ 检测失败:跑完 {len(full_chunks)} chunks 都没触发")
        print(f"     最终 buffer 前 200 字: {''.join(full_chunks)[:200]!r}")
        print(f"     最终 buffer 末 200 字: {''.join(full_chunks)[-200:]!r}")
    else:
        print(f"  ✓ 第 {detected_at_iter}/{len(full_chunks)} chunk 时触发 "
              f"({detected_at_chars} chars)")
        print(f"  ✓ 相对节省: {1 - detected_at_iter/len(full_chunks):.1%} chunks")


def case_realistic_ollama_chunking():
    """更激进:模拟 Ollama 真实 chunk 模式 —— 一些是单 token,一些是多 token batched。"""
    SENTENCE = "The Law will never be perfect, but its application should be just this is what we are missing, in my opinion."
    # 每 1-4 个 token 一批(模拟 Ollama 偶尔 batch)
    import random
    random.seed(42)

    words = SENTENCE.split(" ") + ["<EOS>", "<pad>", "\n", "\n"]
    # 每批:1 个、2 个、3 个 word 随机
    batches: list[str] = []
    i = 0
    while i < len(words):
        size = random.choice([1, 1, 2, 3])  # 多数 1,偶尔 batch
        batch = "".join(" " + w for w in words[i:i+size])
        batches.append(batch)
        i += size
    print(f"  句子切分: {len(batches)} batches/重复 (平均 {sum(len(b) for b in batches)/len(batches):.1f} 字符/batch)")

    N_REPEATS = 80
    full = batches * (N_REPEATS + 1)

    text_parts: list[str] = []
    chunks_since_check = 0
    detected_at = None
    detected_chars = None

    for i, chunk in enumerate(full):
        text_parts.append(chunk)
        chunks_since_check += 1
        if chunks_since_check >= STREAM_CHECK_INTERVAL:
            chunks_since_check = 0
            buffer = "".join(text_parts)
            if _is_loop_start(buffer):
                detected_at = i
                detected_chars = len(buffer)
                break

    if detected_at is None:
        print(f"  ❌ {len(full)} batches 全跑完都没触发")
    else:
        print(f"  ✓ 第 {detected_at}/{len(full)} batch 触发 ({detected_chars} chars)")
        print(f"  ✓ 节省: {1 - detected_at/len(full):.1%} batches")


def case_dedupe_post_check():
    """流式检测触发后,看 dedupe 能否把截下来的 buffer 彻底清干净。"""
    SENTENCE = "The Law will never be perfect, but its application should be just this is what we are missing, in my opinion."
    line = SENTENCE + " <EOS> <pad>"
    # 模拟检测触发时 buffer = 1 句正常 + 4 句循环
    buf = "正常开头内容\n\n" + (line + "\n\n") * 4
    out = dedupe_repeat_runs(buf)
    print(f"  输入: {len(buf)} chars,输出: {len(out)} chars,循环被截: {'是' if len(out) < len(buf) else '否'}")
    print(f"  输出末 100 字: {out[-100:]!r}")


if __name__ == "__main__":
    print("Case 1: 单句循环,均匀 token chunk")
    case_single_loop_sentence_realistic_chunks()
    print()
    print("Case 2: 单句循环,模拟 Ollama batched chunk")
    case_realistic_ollama_chunking()
    print()
    print("Case 3: dedupe 兜底")
    case_dedupe_post_check()