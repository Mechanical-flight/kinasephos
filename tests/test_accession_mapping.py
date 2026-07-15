from kinasephos.data.accession_mapping import canonicalize_uniprot


def test_accession_mapping_is_explicit():
    assert canonicalize_uniprot(" P12345-2 ") == "P12345-2"
    assert canonicalize_uniprot("OLD", {"OLD": "P12345"}) == "P12345"
