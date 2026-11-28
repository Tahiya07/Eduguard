from types import SimpleNamespace as C

from sectioned_summary import combine, pack_sections, sections_from_chunks, sections_from_text, summarize_sections


def chunk(i, page, topic, text):
    return C(chunk_id=i, page=page, source="lec.pdf", topic=topic, text=text)


def test_chunks_under_one_heading_form_one_section_in_page_order():
    chunks = [chunk(2, 2, "Consensus", "Consensus text."), chunk(0, 1, "Network", "Network one."),
              chunk(1, 1, "Network", "Network two.")]
    secs = sections_from_chunks(chunks)
    assert [s.heading for s in secs] == ["Network", "Consensus"]
    assert "Network one." in secs[0].text and "Network two." in secs[0].text


def test_overlap_between_chunks_is_removed():
    a = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    b = "eta theta iota kappa lambda mu"  # starts with the last 4 words of a: below the 8-word minimum, kept
    c = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"
    secs = sections_from_chunks([chunk(0, 1, None, a), chunk(1, 1, None, c)])
    assert secs[0].text.count("alpha") == 1  # 10-word overlap dropped
    assert sections_from_chunks([chunk(0, 1, None, a), chunk(1, 1, None, b)])[0].text.count("eta theta") == 2


def test_small_sections_merge_and_long_documents_are_bounded():
    secs = sections_from_text("\n\n".join(f"Paragraph {i}. " + "word " * 300 for i in range(40)))
    groups, truncated = pack_sections(secs)
    assert 1 < len(groups) <= 12
    assert all(len(g.text) <= 8000 + 1200 for g in groups)
    assert not truncated


def test_large_sections_are_summarized_separately_in_order_and_labelled():
    groups, _ = pack_sections(sections_from_chunks([chunk(0, 1, "Network", "N " * 3600), chunk(1, 2, "Wallet", "W " * 3600)]))
    parts = summarize_sections(groups, lambda t: "About wallets." if t.strip().startswith("W") else "About networks.")
    out = combine(parts)
    assert out.splitlines() == ["Network:", "- About networks.", "", "Wallet:", "- About wallets."]


def test_default_layout_is_titles_with_one_bullet_per_point():
    parts = [("Sharding in simple words / Sharding: one blockchain", "Sharding splits the chain. Each shard has validators."),
             ("Sharding in simple words / Sharding: one blockchain", "Shards run in parallel."),
             ("Side chain", "A side chain is a parallel chain.")]
    out = combine([(h.split(" / ")[0], s) for h, s in parts])
    assert out.splitlines() == ["Sharding in simple words:", "- Sharding splits the chain.", "- Each shard has validators.",
                                "- Shards run in parallel.", "", "Side chain:", "- A side chain is a parallel chain."]


def test_short_neighbouring_sections_share_one_call():
    groups, _ = pack_sections(sections_from_chunks([chunk(0, 1, "Network", "N " * 400), chunk(1, 2, "Wallet", "W " * 400)]))
    assert len(groups) == 1 and groups[0].heading == "Network / Wallet"


def test_documents_beyond_twelve_full_calls_are_flagged():
    _, truncated = pack_sections(sections_from_text("\n\n".join("x " * 4000 for _ in range(16))))
    assert truncated


def test_formats_follow_section_boundaries():
    from sectioned_summary import format_sections, trim_to_last_sentence
    parts = [("Network", "Eclipse isolates nodes. Sybil fakes identities. BGP hijacks routes."),
             ("Consensus", "A 51% attack rewrites history. Selfish mining withholds blocks."),
             ("Wallet", "Phishing steals seeds.")]
    five = format_sections(parts, "bullets", 5).splitlines()
    assert len(five) == 5 and five[0].startswith("- Network:") and five[3].startswith("- Consensus:")
    two = format_sections(parts, "bullets", 2).splitlines()
    assert two == ["- Network / Consensus: Eclipse isolates nodes. A 51% attack rewrites history.",
                   "- Wallet: Phishing steals seeds."]
    assert format_sections(parts, "sentences", 3) == "Eclipse isolates nodes. A 51% attack rewrites history. Phishing steals seeds."
    assert trim_to_last_sentence("One done. Two not") == "One done."


def test_section_count_grows_with_document_length():
    sizes = []
    for n in (2, 6, 12):
        groups, _ = pack_sections(sections_from_text("\n\n".join("w " * 600 for _ in range(n))))  # ~1,200 chars each
        sizes.append(len(groups))
    assert sizes[0] < sizes[1] < sizes[2]
