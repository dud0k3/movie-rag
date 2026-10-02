"""Word-based chunks with overlap; sentence boundaries stay intact where possible."""

import re


def chunks(text: str, target_words: int = 480, overlap_words: int = 80) -> list[str]:
    if target_words <= overlap_words or overlap_words < 0:
        raise ValueError("target_words must be greater than overlap_words")
    words = re.findall(r"\S+", re.sub(r"\s+", " ", text).strip())
    if not words:
        return []
    output = []
    start = 0
    while start < len(words):
        end = min(start + target_words, len(words))
        if end < len(words):
            lower = max(start + target_words // 2, end - 35)
            for index in range(end, lower, -1):
                if re.search(r"[.!?][\"')\]]?$", words[index - 1]):
                    end = index
                    break
        output.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(start + 1, end - overlap_words)
    return output
