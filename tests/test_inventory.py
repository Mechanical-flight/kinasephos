import zipfile

from kinasephos.data.inventory import inspect_zip, write_inventory_report


def test_inventory_and_safe_zip(tmp_path):
    (tmp_path / "data.tsv").write_text("a\tb\n1\t2\n", encoding="utf-8")
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("nested/example.txt", "safe")
        handle.writestr("../escape.txt", "unsafe")
    members = inspect_zip(archive)
    assert members[0]["unsafe_path"] is False
    assert members[1]["unsafe_path"] is True
    summary = write_inventory_report(tmp_path, tmp_path / "report")
    assert summary["file_count"] >= 2
    assert (tmp_path / "report/file_inventory.csv").exists()
