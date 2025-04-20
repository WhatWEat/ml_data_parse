#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build headline_test_clean.jsonl from AdaptLLM/finance-tasks Headline test split
规则：
- 用两个及以上换行分段
- 最后一段是待回答的问答 => Question
- 前面所有段直接保存为 Example QA
"""
import json
import re
import time
from pathlib import Path
from typing import List

from datasets import load_dataset
from tqdm import tqdm


def split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r'\n{2,}', text.strip()) if p.strip()]


def extract_question(paragraph: str) -> str:

    prefix_re = re.compile(
        r'^(?:Now answer this question:|Question:)\s*',
        flags=re.IGNORECASE,
    )
    parts = prefix_re.split(paragraph, maxsplit=1)
    if len(parts) == 2:
        return parts[1].strip()
    return paragraph.strip()


def main() -> None:
    start = time.time()

    ds = load_dataset("AdaptLLM/finance-tasks", name="Headline", split="test")
    out_file = Path("headline_test.jsonl").expanduser()

    total_pairs = 0
    with out_file.open("w", encoding="utf-8") as fout:
        for sample in tqdm(ds, desc="processing"):
            paragraphs = split_paragraphs(sample["input"])
            if len(paragraphs) < 2:
                example_qas = []
                last_para = paragraphs[0]
            else:
                example_qas = paragraphs[:-1]
                last_para = paragraphs[-1]

            qa_obj = {
                "id": sample["id"],
                "Example QA": example_qas,
                "Question": extract_question(last_para),
                "Answer": sample["options"][sample["gold_index"]],
                "Class_id": sample["class_id"],
                "Options": sample["options"],
                "labels": sample["gold_index"]
            }
            json.dump(qa_obj, fout, ensure_ascii=False)
            fout.write("\n")

            total_pairs += len(example_qas) + 1

    end = time.time()

    print("\n========= 统计 =========")
    print(f"样本总数              : {len(ds):,}")
    print(f"问答对总数            : {total_pairs:,}")
    print(f"脚本耗时 (秒)         : {end - start:.2f}")
    print(f"输出文件              : {out_file.resolve()}")


if __name__ == "__main__":
    main()
