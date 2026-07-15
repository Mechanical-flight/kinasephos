from kinasephos.data.windowing import build_window


def test_window_center_and_padding():
    result = build_window("MSTY", 2, "S")
    assert result.status == "valid"
    assert len(result.window) == 61
    assert result.window[30] == "S"
    assert result.window.startswith("X" * 29)


def test_window_rejects_mismatch():
    assert build_window("MSTY", 2, "T").status == "residue_mismatch"
    assert build_window("MAAA", 2).status == "center_not_sty"
