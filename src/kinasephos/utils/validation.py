from kinasephos.constants import CENTER_INDEX, PHOSPHO_RESIDUES, WINDOW_LENGTH


def validate_window(window: str) -> None:
    if len(window) != WINDOW_LENGTH:
        raise ValueError(f"Expected {WINDOW_LENGTH} residues, got {len(window)}")
    if window[CENTER_INDEX] not in PHOSPHO_RESIDUES:
        raise ValueError("Window center must be S, T, or Y")
