import pytest

from input_validation import question_problem, require_question

GIBBERISH = ["hdoiwdnc", "hdoiwdnc sjdkf", "asdf asdf asdf", "qwerty uiop", "jdjdjd kdkdkd",
             "Explain hdoiwdnc sjdkfh qpwoeir", "12345", "???", ""]
QUESTIONS = ["Define recursion.", "What is a stack?", "Explain how Kubernetes schedules pods.",
             "Compare TCP and UDP.", "Design a stablecoin peg mechanism.",
             "Analyze Dijkstra's algorithm complexity in O(n log n).", "List the types of IPv6 addresses.",
             "Explain the role of mitochondria in cellular respiration.",
             "Critically evaluate Porter's five forces model.", "what is 2NF"]


@pytest.mark.parametrize("text", GIBBERISH)
def test_rejects_gibberish(text):
    assert question_problem(text) is not None
    with pytest.raises(ValueError):
        require_question(text)


@pytest.mark.parametrize("text", QUESTIONS)
def test_accepts_real_questions(text):
    assert question_problem(text) is None


def test_single_real_word_asks_for_a_full_question():
    assert "single word" in question_problem("Recursion")


@pytest.mark.parametrize("text", ["John Smith", "Md Rahim Uddin", "Operating systems",
                                  "Computer networks and security", "Dhaka University", "hello there"])
def test_rejects_names_and_bare_topics(text):
    assert "exam question or task" in question_problem(text)


def test_single_letter_variables_do_not_count_as_gibberish():
    assert question_problem("Determine whether the graph with vertices V={A,B,C,D,E} is bipartite.") is None
