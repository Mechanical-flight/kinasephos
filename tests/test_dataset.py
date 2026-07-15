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
