"""Базовый EDA по датасету прототипа (results.csv) для ДЗ-4.

Считает фактические метрики, на которые опирается hw_4.md:
  - размер выборки, пропуски по колонкам
  - распределение возраста (+ детект нечисловых артефактов)
  - частоты городов и доля доминирующего
  - длины открытых ответов, доля ответов-заглушек
  - распределение форматов суммаризации (валидный список / отказ / текст)
  - эвристические флаги галлюцинаций и сдвига колонок

Запуск:
    python -m eval.eda
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

CSV = "DHA_hackathon/data/Примеры того, что получается при прогонки/results.csv"

# Логический порядок колонок (в файле он перемешан — дефект целостности, см. hw_4.md §1.2)
ORDER = [
    "user_ans_1", "summed_ans_1",
    "user_ans_2", "summed_ans_2",
    "user_ans_3", "summed_ans_3",
    "user_ans_4", "summed_ans_4",
]

STUBS = {
    "все", "всё", "ничего", "нет", "да", "ок", "хорошо",
    "все нравится", "всё нравится", "все устраивает",
}


def _to_age(x):
    if pd.isna(x):
        return None
    m = re.search(r"\d+", str(x))
    return int(m.group()) if m else None


def _classify_sum(x):
    """Грубая классификация формата суммаризации открытого ответа."""
    if pd.isna(x):
        return "NaN"
    s = str(x).strip()
    if s == "[]":
        return "empty_list"
    low = s.lower()
    refusal_markers = [
        "к сожалению", "не предоставили", "не указал", "уточните",
        "отлично!", "если у вас", "не совсем понимаю", "спасибо за доверие",
    ]
    if any(k in low for k in refusal_markers):
        return "refusal/apology"
    if re.match(r"^\[.*\]$", s, re.S):
        return "python_list"
    if "?" in s and len(s.split()) < 15:
        return "question(echo)"
    return "free_text/numbered"


def main() -> None:
    df = pd.read_csv(CSV)
    print("=== SHAPE ===")
    print("rows:", len(df), "cols:", len(df.columns))
    print("file column order:", list(df.columns))
    df = df[ORDER]

    print("\n=== MISSING (NaN) ===")
    miss = df.isna().sum()
    for c in ORDER:
        print(f"{c:16s} NaN={miss[c]:2d} ({miss[c] / len(df) * 100:.0f}%)")
    ans_cols = ["user_ans_1", "user_ans_2", "user_ans_3", "user_ans_4"]
    print("fully-empty rows:", int(df[ans_cols].isna().all(axis=1).sum()))

    print("\n=== AGE (user_ans_1) ===")
    ages = [_to_age(x) for x in df["user_ans_1"]]
    non_numeric = [str(x) for x, a in zip(df["user_ans_1"], ages) if a is None and not pd.isna(x)]
    valid = [a for a in ages if a is not None]
    print("non-numeric cells:", non_numeric)
    print(f"n={len(valid)} min={min(valid)} max={max(valid)} "
          f"median={np.median(valid):.0f} mean={np.mean(valid):.1f}")

    print("\n=== CITY (summed_ans_2) ===")
    cnt = Counter(df["summed_ans_2"].dropna().tolist())
    for c, n in cnt.most_common():
        print(f"  {n:2d}  {c}")
    spb = sum(n for c, n in cnt.items() if c in {"Санкт-Петербург", "Спб", "Санкт Петербург"})
    total = sum(cnt.values())
    print(f"SPb share (normalized): {spb}/{total} = {spb / total * 100:.0f}%")

    print("\n=== OPEN ANSWER LENGTHS (words) ===")
    for col in ["user_ans_3", "user_ans_4"]:
        wl = df[col].dropna().astype(str).apply(lambda s: len(s.split()))
        print(f"{col}: n={len(wl)} min={wl.min()} median={wl.median():.0f} "
              f"max={wl.max()} mean={wl.mean():.1f}")

    print("\n=== STUB ANSWERS ===")
    for col in ["user_ans_3", "user_ans_4"]:
        vals = df[col].dropna().astype(str)
        s = vals[vals.str.strip().str.lower().isin(STUBS)]
        print(f"{col}: stub-like = {len(s)} -> {list(s)}")

    print("\n=== SUMMARIZATION FORMAT ===")
    for col in ["summed_ans_3", "summed_ans_4"]:
        print(col, dict(Counter(_classify_sum(x) for x in df[col])))

    print("\n=== HALLUCINATION / MISALIGN FLAGS (heuristic) ===")
    py3 = df["summed_ans_3"].astype(str).str.contains("Python|библиотек|синтаксис", case=False, na=False).sum()
    py4 = df["summed_ans_4"].astype(str).str.contains("Python", case=False, na=False).sum()
    tr = df["summed_ans_4"].astype(str).str.contains("переводчик|на разных языках|медицинских", na=False).sum()
    bye = (df["user_ans_1"].astype(str).str.contains("свидан", na=False) |
           df["user_ans_2"].astype(str).str.contains("свидан", na=False)).sum()
    print("Python-hallucination summed_ans_3:", int(py3))
    print("Python in summed_ans_4:", int(py4))
    print("translation-hallucination summed_ans_4:", int(tr))
    print("non-numeric age (column shift):", sum(1 for a in ages if a is None))
    print("'До свидания' captured as answer:", int(bye))


if __name__ == "__main__":
    main()
