from rag_utils import is_enumerative_question, keep_supported_sentences, notes_contain_terms

NOTES = ("Threats to blockchain systems are grouped into four types: network-level threats, consensus-level "
         "threats, smart contract threats, and wallet and user-level threats. Reentrancy attack: a contract calls back "
         "into the victim before its balance is updated, as in the DAO hack.")


def test_list_questions_are_enumerative():
    assert is_enumerative_question("Mention all type of threats")
    assert is_enumerative_question("what are the types of threats")
    assert not is_enumerative_question("what is a reentrancy attack")


def test_key_terms_must_all_be_in_notes():
    assert notes_contain_terms("Mention all type of threats", NOTES)
    assert not notes_contain_terms("what is photosynthesis", NOTES)
    assert not notes_contain_terms("explain the threats of photosynthesis", NOTES)


def test_paraphrase_is_kept_but_outside_knowledge_is_dropped():
    paraphrase = ("A reentrancy attack allows an attacker to execute a contract's code before its balance is "
                  "updated, often through a call that returns before the balance is set.")
    assert keep_supported_sentences(paraphrase, [NOTES]) == paraphrase
    assert "Paris" not in keep_supported_sentences("The capital of France is Paris.", [NOTES])


def test_short_numeric_answer_is_not_refused():
    assert keep_supported_sentences("96%", ["Carbon dioxide accounts for 96% of the atmosphere."]) == "96%"


def test_buried_fact_is_found_by_its_sentence():
    import numpy as np
    from hashing_vectorizer import HashingVectorizer
    from rag_utils import sentence_rescore
    from retriever import RetrievalResult

    class Enc:  # tiny stand-in for BGE: normalized hashing vectors
        hv = HashingVectorizer(512)
        def _v(self, t):
            m = self.hv.transform(t); return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-9, None)
        encode_queries = encode_passages = _v

    filler = "Implicit differentiation is useful when a relationship is given by an equation. " * 6
    docs = [filler + "The value for section 3 is 21. " + filler,
            "The value for section 33 is 231.", "The value for section 39 is 273."]
    results = [RetrievalResult(rank=i, doc_id=i, text=t, cosine=c) for i, (t, c) in enumerate(zip(docs, [0.5, 0.74, 0.74]))]
    top = sentence_rescore("What is the value for section 3?", results, Enc(), {}, keep=3)
    assert "section 3 is 21" in top[0].text
    assert top[0].cosine > 0.5  # carries its best sentence's relevance forward
