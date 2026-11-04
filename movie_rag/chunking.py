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


def context_window(previous: str, center: str, following: str, margin: int = 460) -> str:
    """Add nearby text once, accounting for the overlap already in the index."""
    def overlap(left: list[str], right: list[str]) -> int:
        for count in range(min(len(left), len(right), 120), 0, -1):
            if left[-count:] == right[:count]:
                return count
        return 0

    words = center.split()
    before, after = previous.split(), following.split()
    left_overlap, right_overlap = overlap(before, words), overlap(words, after)
    head = " ".join(before[:len(before) - left_overlap])
    tail = " ".join(after[right_overlap:])
    if len(head) > margin:
        head = head[-margin:].partition(" ")[2]
        start = re.search(r"[.!?]\s+", head)
        if start:
            head = head[start.end():]
    if len(tail) > margin:
        tail = tail[:margin].rsplit(" ", 1)[0]
        boundaries = list(re.finditer(r"[.!?](?:[»\x22])?(?=\s|$)", tail))
        if boundaries:
            tail = tail[:boundaries[-1].end()]
    return " ".join(part for part in (head, center, tail) if part)
