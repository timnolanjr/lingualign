# lingualign/merge.py
from typing import List, Tuple

def merge_labels(
    spans: List[Tuple[float, float, str, float]],
    min_flip_s: float = 0.3
) -> List[Tuple[float, float, str]]:
    """
    Collapse adjacent spans with same lang, absorb flips < min_flip_s
    only if they’re uncertain or very low‐conf (extend logic as needed).
    """
    if not spans: return []
    merged, (s0,e0,l0,_) = [], spans[0]
    for s1,e1,l1,conf in spans[1:]:
        if l1 == l0:
            e0 = e1
        elif (e0-s0) < min_flip_s and l0 == "un":
            # absorb uncertain short spans
            # (requires you label windows 'un' when conf< threshold)
            # extend previous by merging
            if merged:
                ps,pe,pl = merged.pop()
                merged.append((ps, e1, pl))
            else:
                merged.append((s0, e1, l1))
            s0,e0,l0 = s1,e1,l1
        else:
            merged.append((s0,e0,l0))
            s0,e0,l0 = s1,e1,l1
    merged.append((s0,e0,l0))
    return merged
