from kinasephos.data.splitting import assert_no_leakage, group_split


def test_group_split_is_deterministic_and_leakage_free():
    rows = [
        {"sample_id": f"s{i}", "protein_id": f"p{i // 2}", "cluster_id": f"c{i // 4}"}
        for i in range(20)
    ]
    first = group_split(rows, seed=7)
    second = group_split(rows, seed=7)
    assert first == second
    assert_no_leakage(first)
