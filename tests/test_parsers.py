from kinasephos.data.parsers import parse_fasta, parse_site_table


def test_fasta_and_site_table(tmp_path):
    fasta = tmp_path / "example.fa"
    fasta.write_text(">EP1|P12345|GENE example\nMS*\n", encoding="utf-8")
    assert list(parse_fasta(fasta)) == [("P12345", "MS", "EP1|P12345|GENE example")]
    table = tmp_path / "sites.tsv"
    table.write_text(
        "EPSD ID\tUniProt\tPosition\tKinase\tSource\nE1\tP12345\tS2\tK1\tExperimental\n",
        encoding="utf-8",
    )
    site = next(
        parse_site_table(
            table,
            source_dataset="fixture",
            accession_column="UniProt",
            position_column="Position",
        )
    )
    assert site.site_key == "P12345:S2"
