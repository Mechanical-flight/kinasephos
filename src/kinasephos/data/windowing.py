from dataclasses import dataclass

from kinasephos.constants import CENTER_INDEX, PHOSPHO_RESIDUES, WINDOW_LENGTH


@dataclass(frozen=True)
class WindowResult:
    window: str | None
    status: str


def build_window(
    sequence: str,
    position_1based: int,
    expected_residue: str | None = None,
    padding: str = "X",
) -> WindowResult:
    if len(padding) != 1:
        raise ValueError("Padding token must be one character")
    index = position_1based - 1
    if index < 0 or index >= len(sequence):
        return WindowResult(None, "position_out_of_bounds")
    residue = sequence[index].upper()
    if residue not in PHOSPHO_RESIDUES:
        return WindowResult(None, "center_not_sty")
    if expected_residue and residue != expected_residue.upper():
        return WindowResult(None, "residue_mismatch")
    left_start = index - CENTER_INDEX
    right_end = index + (WINDOW_LENGTH - CENTER_INDEX)
    left_padding = padding * max(0, -left_start)
    right_padding = padding * max(0, right_end - len(sequence))
    core = sequence[max(0, left_start) : min(len(sequence), right_end)].upper()
    window = left_padding + core + right_padding
    if len(window) != WINDOW_LENGTH or window[CENTER_INDEX] != residue:
        raise AssertionError("Window construction invariant failed")
    return WindowResult(window, "valid")
