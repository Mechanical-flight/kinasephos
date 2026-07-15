import random
from collections import defaultdict


def group_split(
    rows: list[dict],
    group_key: str = "cluster_id",
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> list[dict]:
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to one")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        group = str(row.get(group_key) or row.get("protein_id") or "")
        if not group:
            raise ValueError("Every sample needs a cluster_id or protein_id")
        groups[group].append(row)
    items = list(groups.items())
    random.Random(seed).shuffle(items)
    targets = [len(rows) * ratio for ratio in ratios]
    counts = [0, 0, 0]
    names = ("train", "val", "test")
    output: list[dict] = []
    for _, members in sorted(items, key=lambda item: len(item[1]), reverse=True):
        slot = min(range(3), key=lambda idx: counts[idx] / max(targets[idx], 1.0))
        for member in members:
            copied = dict(member)
            copied["split"] = names[slot]
            output.append(copied)
        counts[slot] += len(members)
    return output


def assert_no_leakage(rows: list[dict]) -> None:
    for key in ("protein_id", "cluster_id", "sample_id"):
        memberships: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            memberships[str(row[key])].add(str(row["split"]))
        leaking = [item for item, splits in memberships.items() if len(splits) > 1]
        if leaking:
            raise ValueError(f"{key} leakage detected: {leaking[:5]}")
