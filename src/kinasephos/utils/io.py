import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def write_json(path: str | Path, data: object) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: str | Path, rows: Iterable[Mapping[str, object]], fields: list[str]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
