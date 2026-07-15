from kinasephos.data.dataset import SiteDataset, encode_window


def test_dataset_encoding():
    window = "A" * 30 + "S" + "A" * 30
    tokens = encode_window(window)
    assert tokens.shape == (61,)
    dataset = SiteDataset(
        [{"sample_id": "s1", "window_61": window, "binary_label": "1", "family_label": "0"}],
        synthetic_teacher=True,
    )
    assert dataset[0]["teacher_embedding"].shape == (1280,)


def test_padding_and_unknown_residue_have_distinct_ids():
    window = "-" * 29 + "X" + "S" + "A" * 30
    tokens = encode_window(window)
    assert tokens[0].item() == 0
    assert tokens[29].item() == 1
