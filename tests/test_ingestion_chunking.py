from ingestion import DocumentIngestor


def _ing():
    return DocumentIngestor(chunk_size=120, chunk_overlap=25)


def test_no_duplicate_tail_chunk():
    text = " ".join(f"Sentence number {i} is here." for i in range(20))  # 100 words
    assert len(_ing().chunk_text(text)) == 1


def test_chunks_end_on_sentence_boundaries_and_overlap_whole_sentences():
    text = " ".join(f"Fact {i} says that item {i} matters a lot." for i in range(60))
    chunks = _ing().chunk_text(text)
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.endswith("."), c.text[-40:]
        assert len(c.text.split()) <= 120
    first_last = chunks[0].text.rsplit(". ", 1)[-1]
    assert first_last.rstrip(".") in chunks[1].text  # last sentence repeated as overlap


def test_heading_is_kept_with_following_text():
    text = "Network-level Threats\nEclipse Attack: isolating a node.\nSmart Contract Threats\nReentrancy: calls back."
    chunks = DocumentIngestor(chunk_size=6, chunk_overlap=0).chunk_text(text)
    for c in chunks:
        assert not c.text.rstrip().endswith("Threats"), c.text  # never a dangling heading
    assert any(c.topic == "Smart Contract Threats" and "Reentrancy" in c.text for c in chunks)


def test_overlong_sentence_is_split():
    assert [len(c.text.split()) for c in _ing().chunk_text(" ".join(["word"] * 300))] == [120, 120, 60]


def test_running_headers_and_page_numbers_are_removed():
    pages = [f"Course Title\nBody text of page {i} explains topic {i}.\n{i}" for i in range(1, 6)]
    cleaned = DocumentIngestor._strip_running_lines(pages)
    assert all("Course Title" not in p for p in cleaned)
    assert all(f"topic {i}" in p for i, p in enumerate(cleaned, 1))
    assert all(not p.strip().endswith(str(i)) for i, p in enumerate(cleaned, 1))


def test_numbered_headings_survive_but_page_footers_do_not():
    pages = [f"Lecture section {i}\nBody text about topic {i} in some detail here.\nCourse Notes - page {i}" for i in range(1, 7)]
    cleaned = DocumentIngestor._strip_running_lines(pages)
    assert all(f"Lecture section {i}" in p for i, p in enumerate(cleaned, 1))
    assert all("Course Notes - page" not in p for p in cleaned)
