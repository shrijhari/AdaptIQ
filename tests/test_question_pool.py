import csv
import pickle
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import question_pool as qp

STEMS = [
    "Why does a model with very high variance perform badly on new data?",
    "What is the main effect of adding a penalty term to the loss function?",
    "How does early stopping help during neural network training?",
    "Which problem does cross validation help to detect in a model?",
    "What happens to a decision tree when it is allowed to grow without limit?",
    "Why is feature scaling important for distance based methods?",
    "What role does the learning rate play in gradient descent?",
    "How does dropout reduce co-adaptation between hidden units?",
    "What does a kernel function let a support vector machine do?",
    "Why can a large number of correlated predictors hurt a regression?",
    "What is the purpose of holding out a validation set from the data?",
    "How does bagging lower the variance of an unstable learner?",
]


def make_chunks(n_sections=6, per_section=2):
    chunks = []
    k = 0
    for s in range(n_sections):
        for p in range(per_section):
            k += 1
            body = (f"Regularization section {s} part {p}. " +
                    "A model that memorises noise in the training set fits those "
                    "points closely but predicts new observations poorly, so "
                    "penalising complexity trades a little bias for a large "
                    "reduction in variance. " * 4)
            chunks.append({
                "chunk_id": f"S2_C{k}", "skill_id": "S2",
                "content": f"## S2_C{k} --- Title\n\nSource: book.pdf, ch\n\n{body}",
                "source": "book.pdf", "chapter": "ch",
                "section": f"Chapter > Section {s}",
                "page_start": str(10 + k), "page_end": str(11 + k),
            })
    return chunks


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A temporary data folder with S2 chunks and a tiny bank."""
    monkeypatch.setattr(qp, "DATA_DIR", tmp_path)
    monkeypatch.setattr(qp, "GENERATED_DIR", tmp_path / "generated_pool")
    with open(tmp_path / "s2_chunks.pkl", "wb") as f:
        pickle.dump(make_chunks(), f)

    with open(tmp_path / "adaptive_questions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "question", "option_a", "option_b", "option_c",
                    "option_d", "correct_answer", "difficulty", "skills"])
        w.writerow(["AQ1", "Bank question one about variance?", "a1", "b1", "c1",
                    "d1", "B", "Easy", "S2"])
        w.writerow(["AQ2", "Bank question two about bias?", "a2", "b2", "c2",
                    "d2", "C", "Hard", "S2"])
    with open(tmp_path / "adaptive_q_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "S1", "S2", "S3", "S4", "S5", "S6"])
        w.writerow(["AQ1", 0, 1, 0, 0, 0, 0])
        w.writerow(["AQ2", 0, 1, 0, 0, 0, 0])
    return tmp_path


class FakeLLM:
    """Stands in for Ollama: writes a grounded question from the passage."""

    def __init__(self, bad_every=None):
        self.calls = 0
        self.bad_every = bad_every

    def __call__(self, system, user, schema):
        self.calls += 1
        passage = user.split("PASSAGE:\n", 1)[1]
        correct = passage[:70].strip()
        options = [correct, "It always lowers the training error to zero",
                   "It removes the need for any validation data",
                   "It makes every feature equally important"]
        if self.bad_every and self.calls % self.bad_every == 0:
            options[1] = options[0]          # duplicate options -> must be rejected
        return {"question": STEMS[(self.calls - 1) % len(STEMS)],
                "option_a": options[0], "option_b": options[1],
                "option_c": options[2], "option_d": options[3],
                "correct_answer": "A",
                "explanation": "The passage says penalising complexity cuts variance."}


def good_question(**changes):
    q = {"question_id": "G_S2_0001",
         "question": "Why does overfitting hurt performance on new data?",
         "option_a": "The model memorised noise in the training set",
         "option_b": "The model has too few parameters",
         "option_c": "The data were split too evenly",
         "option_d": "The loss function was not differentiable",
         "correct_answer": "A", "difficulty": "Medium", "skills": ["S2"],
         "explanation": "Memorised noise does not generalise.",
         "source": "book.pdf, pages 1-2", "origin": "generated"}
    q.update(changes)
    return q


# ---------------------------------------------------------- check_question

def test_good_question_passes():
    assert qp.check_question(good_question()) == (True, "")


@pytest.mark.parametrize("changes, expected", [
    ({"explanation": ""}, "explanation"),
    ({"option_b": "The model memorised noise in the training set"}, "different"),
    ({"option_b": "The model memorised noise in the training sets"}, "nearly identical"),
    ({"correct_answer": "E"}, "A, B, C or D"),
    ({"difficulty": "Impossible"}, "difficulty"),
    ({"question": "Which is right? A) one B) two C) three"}, "inside the question"),
    ({"question": "According to the passage, why does overfitting hurt?"}, "passage"),
    ({"option_d": "All of the above"}, "all/none"),
    ({"question": "Why does the model memorised noise in the training set happen?"}, "given away"),
    ({"question": "What does the x-axis in Figure 11.7 represent for overfitting?"}, "figure"),
    ({"question": "Which idea is introduced in Section 3.2 about overfitting?"}, "figure"),
    ({"question": "As shown in the figure, why does overfitting hurt new data?"}, "figure"),
    ({"explanation": "As shown in Table 4, memorised noise does not generalise."}, "figure"),
])
def test_bad_questions_are_rejected_with_reason(changes, expected):
    ok, reason = qp.check_question(good_question(**changes))
    assert not ok and expected in reason


def test_missing_field_is_reported():
    q = good_question()
    del q["option_c"]
    ok, reason = qp.check_question(q)
    assert not ok and "Missing field: option_c" in reason


def test_duplicate_stem_is_rejected():
    stems = ["Why does overfitting hurt performance on new data?"]
    ok, reason = qp.check_question(good_question(), accepted_stems=stems)
    assert not ok and "duplicate" in reason


def test_answer_must_be_supported_by_source():
    ok, reason = qp.check_question(
        good_question(), chunk_text="Completely unrelated text about cooking pasta.")
    assert not ok and "source" in reason
    ok, _ = qp.check_question(
        good_question(),
        chunk_text="A model memorised noise in the training set and fails later.")
    assert ok


# ------------------------------------------------------------ generate_pool

def test_generates_valid_standard_format_questions(env, monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(qp, "_call_llm", fake)
    pool = qp.generate_pool("S2", 6)

    assert len(pool) == 6
    for q in pool:
        qp.validate_question(q)
        assert q["origin"] == "generated"
        assert q["skills"] == ["S2"]
        assert q["question_id"].startswith("G_S2_")
        assert "pages" in q["source"]
    assert {q["difficulty"] for q in pool} == {"Easy", "Medium", "Hard"}


def test_questions_come_from_different_sections(env, monkeypatch):
    monkeypatch.setattr(qp, "_call_llm", FakeLLM())
    pool = qp.generate_pool("S2", 6)
    assert len({q["source_chunk_id"] for q in pool}) == 6
    sections = {int(q["source_chunk_id"][4:]) for q in pool}
    # 6 questions over 6 sections with 2 chunks each -> one chunk per section
    assert len({(c - 1) // 2 for c in sections}) == 6


def test_second_call_uses_the_cache(env, monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(qp, "_call_llm", fake)
    first = qp.generate_pool("S2", 4)
    calls = fake.calls
    second = qp.generate_pool("S2", 4)
    assert fake.calls == calls                      # no new model calls
    assert [q["question_id"] for q in first] == [q["question_id"] for q in second]
    assert (env / "generated_pool" / "S2.json").is_file()


def test_top_up_adds_without_overwriting(env, monkeypatch):
    monkeypatch.setattr(qp, "_call_llm", FakeLLM())
    first_ids = {q["question_id"] for q in qp.generate_pool("S2", 3)}
    bigger = qp.generate_pool("S2", 5)
    ids = {q["question_id"] for q in bigger}
    assert len(bigger) == 5
    assert first_ids <= ids                         # old questions kept
    assert len(ids) == 5                            # new IDs do not clash


def test_invalid_generated_questions_are_dropped(env, monkeypatch):
    monkeypatch.setattr(qp, "_call_llm", FakeLLM(bad_every=2))
    pool = qp.generate_pool("S2", 4)
    generated = [q for q in pool if q["origin"] == "generated"]
    assert generated and all(qp.check_question(q)[0] for q in generated)
    assert qp.last_run_stats["rejected"]["options must be four different answers"] > 0


def test_bank_fallback_when_no_textbook_content(env, monkeypatch):
    monkeypatch.setattr(qp, "_call_llm", FakeLLM())
    pool = qp.generate_pool("S5", 3)                # no s5_chunks.pkl
    assert pool == qp.bank_pool("S5", 3)


def test_bank_fallback_when_ollama_is_down(env, monkeypatch):
    def broken(*args, **kwargs):
        raise ConnectionError("ollama is not running")
    monkeypatch.setattr(qp, "_call_llm", broken)

    pool = qp.generate_pool("S2", 2)                # must not raise
    assert len(pool) == 2
    assert all(q["origin"] == "bank" for q in pool)


def test_bank_tops_up_when_too_few_valid_questions(env, monkeypatch):
    monkeypatch.setattr(qp, "_call_llm", FakeLLM(bad_every=1))  # everything invalid
    pool = qp.generate_pool("S2", 2)
    assert {q["origin"] for q in pool} == {"bank"}


def test_cache_is_ignored_after_reingestion(env, monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(qp, "_call_llm", fake)
    qp.generate_pool("S2", 2)
    calls = fake.calls

    with open(env / "s2_chunks.pkl", "wb") as f:    # simulate re-ingestion
        pickle.dump(make_chunks(n_sections=7), f)
    qp.generate_pool("S2", 2)
    assert fake.calls > calls                       # regenerated from new chunks


def test_unusable_chunks_are_skipped(env, monkeypatch):
    chunks = make_chunks(n_sections=2, per_section=1)
    chunks[0]["section"] = "Contents (part 1/2)"
    chunks[1]["content"] = "## x\n\nSource: b\n\n" + "`code` " * 40
    with open(env / "s2_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    fake = FakeLLM()
    monkeypatch.setattr(qp, "_call_llm", fake)
    pool = qp.generate_pool("S2", 2)
    assert fake.calls == 0                          # nothing worth asking about
    assert all(q["origin"] == "bank" for q in pool)


def test_zero_questions_requested(env):
    assert qp.generate_pool("S2", 0) == []


def test_bank_pool_mixes_difficulty(env):
    pool = qp.bank_pool("S2", 2)
    assert {q["difficulty"] for q in pool} == {"Easy", "Hard"}


def test_hash_table_is_not_mistaken_for_a_table_reference():
    q = good_question(question="Why can a hash table help with overfitting checks?")
    assert qp.check_question(q)[0]


def test_old_cache_is_cleaned_with_the_current_rules(env, monkeypatch):
    bad = good_question(question_id="G_S2_0001",
                        question="What does the x-axis in Figure 11.7 show about overfitting?")
    keep = good_question(question_id="G_S2_0002", source_chunk_id="S2_C1")
    qp._save_cache("S2", [bad, keep])

    monkeypatch.setattr(qp, "_call_llm", FakeLLM())
    pool = qp.generate_pool("S2", 1)
    assert [q["question_id"] for q in pool] == ["G_S2_0002"]

    import json
    saved = json.loads((env / "generated_pool" / "S2.json").read_text())
    assert [q["question_id"] for q in saved["questions"]] == ["G_S2_0002"]
