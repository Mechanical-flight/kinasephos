import mimetypes
import re
import stat
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from kinasephos.utils.hashing import sha256_file
from kinasephos.utils.io import ensure_dir, write_csv, write_json

INVENTORY_FIELDS = [
    "top_level_source",
    "relative_path",
    "filename",
    "extension",
    "mime_type",
    "size_bytes",
    "modified_time_utc",
    "sha256",
    "is_archive",
    "is_readable",
    "content_type",
    "suspected_duplicate",
    "suspected_data",
    "suspected_document",
    "suspected_model_weight",
    "suspected_sensitive",
    "recommendation",
]
ARCHIVE_SUFFIXES = {".zip", ".gz", ".tgz", ".tar", ".7z"}
DATA_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".fasta", ".fa"}
DOCUMENT_SUFFIXES = {".md", ".pdf", ".doc", ".docx", ".ppt", ".pptx"}
MODEL_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".bin"}
SENSITIVE_PATTERN = re.compile(
    r"(?i)(token|password|secret|cookie|credential|private[_-]?key|\.env)"
)


def _classify(path: Path) -> dict[str, object]:
    suffix = path.suffix.lower()
    name = path.name
    is_archive = suffix in ARCHIVE_SUFFIXES
    is_data = suffix in DATA_SUFFIXES
    is_document = suffix in DOCUMENT_SUFFIXES
    is_model = suffix in MODEL_SUFFIXES
    is_sensitive = bool(SENSITIVE_PATTERN.search(name))
    if is_model:
        content_type = "model_weight"
    elif is_archive:
        content_type = "archive"
    elif suffix in {".fasta", ".fa"}:
        content_type = "fasta"
    elif is_data:
        content_type = "tabular_data"
    elif is_document:
        content_type = "document"
    else:
        content_type = "other"
    recommendation = "exclude_from_git" if is_sensitive or is_model or is_archive else "review"
    return {
        "is_archive": is_archive,
        "content_type": content_type,
        "suspected_data": is_data,
        "suspected_document": is_document,
        "suspected_model_weight": is_model,
        "suspected_sensitive": is_sensitive,
        "recommendation": recommendation,
    }


def inventory_assets(root: str | Path) -> list[dict[str, object]]:
    base = Path(root).resolve()
    rows: list[dict[str, object]] = []
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        relative = path.relative_to(base)
        parts = relative.parts
        metadata = _classify(path)
        readable = bool(path.stat().st_mode & (stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH))
        rows.append(
            {
                "top_level_source": parts[0] if len(parts) > 1 else base.name,
                "relative_path": str(relative),
                "filename": path.name,
                "extension": path.suffix.lower(),
                "mime_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                "size_bytes": path.stat().st_size,
                "modified_time_utc": datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
                "sha256": sha256_file(path) if readable else "",
                "is_readable": readable,
                **metadata,
            }
        )
    hashes: dict[str, int] = defaultdict(int)
    for row in rows:
        hashes[str(row["sha256"])] += 1
    for row in rows:
        row["suspected_duplicate"] = bool(row["sha256"] and hashes[str(row["sha256"])] > 1)
    return rows


def inspect_zip(path: str | Path, max_ratio: float = 100.0) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            normalized = Path(member.filename)
            unsafe = normalized.is_absolute() or ".." in normalized.parts
            ratio = member.file_size / max(member.compress_size, 1)
            rows.append(
                {
                    "archive": str(path),
                    "member": member.filename,
                    "compressed_size": member.compress_size,
                    "uncompressed_size": member.file_size,
                    "compression_ratio": round(ratio, 3),
                    "unsafe_path": unsafe,
                    "possible_zip_bomb": ratio > max_ratio,
                }
            )
    return rows


def write_inventory_report(root: str | Path, output: str | Path) -> dict[str, object]:
    destination = ensure_dir(output)
    rows = inventory_assets(root)
    archives: list[dict[str, object]] = []
    for row in rows:
        if row["extension"] == ".zip":
            archives.extend(inspect_zip(Path(root) / str(row["relative_path"])))
    write_csv(destination / "file_inventory.csv", rows, INVENTORY_FIELDS)
    write_json(destination / "file_inventory.json", rows)
    archive_fields = [
        "archive",
        "member",
        "compressed_size",
        "uncompressed_size",
        "compression_ratio",
        "unsafe_path",
        "possible_zip_bomb",
    ]
    write_csv(destination / "archive_inventory.csv", archives, archive_fields)
    duplicates = [row for row in rows if row["suspected_duplicate"]]
    write_csv(destination / "duplicate_files.csv", duplicates, INVENTORY_FIELDS)
    unreadable = [row for row in rows if not row["is_readable"]]
    write_csv(destination / "unreadable_files.csv", unreadable, INVENTORY_FIELDS)
    sensitive = [row for row in rows if row["suspected_sensitive"]]
    write_csv(destination / "suspected_sensitive_files.csv", sensitive, INVENTORY_FIELDS)
    summary = {
        "root": str(Path(root).resolve()),
        "file_count": len(rows),
        "total_bytes": sum(int(row["size_bytes"]) for row in rows),
        "archive_members": len(archives),
        "duplicate_files": len(duplicates),
        "unreadable_files": len(unreadable),
        "suspected_sensitive_files": len(sensitive),
    }
    write_json(destination / "summary.json", summary)
    return summary
