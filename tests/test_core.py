import numpy as np
import pytest

from cegqa.error_analysis import categorize
from cegqa.metrics import aggregate, exact_match, f1, normalize_answer, score
from cegqa.prompts import extract_answer
from cegqa.stats import bootstrap_ci, paired_bootstrap


def test_normalize():
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"
    assert normalize_answer("an apple, a pear") == "apple pear"


def test_em_f1():
    assert exact_match("the Eiffel tower", "Eiffel Tower.") == 1.0
    assert f1("Eiffel", "Eiffel Tower") == pytest.approx(2 * 1 * 0.5 / 1.5)
    assert f1("", "x") == 0.0
    assert f1("", "") == 1.0
    s = score("Paris", ["London", "paris"])
    assert s == {"em": 1.0, "f1": 1.0}
    agg = aggregate([{"em": 1.0, "f1": 1.0}, {"em": 0.0, "f1": 0.5}])
    assert agg["em"] == 50.0 and agg["f1"] == 75.0


@pytest.mark.parametrize("raw,expected", [
    ("Paris", "Paris"),
    ("Answer: Paris.", "Paris"),
    ("<think>\n\n</think>\n\nParis\nbecause ...", "Paris"),
    ('"1945"', "1945"),
    ("The answer is: George Washington", "George Washington"),
    ("Washington D.C.", "Washington D.C."),
    ("", ""),
])
def test_extract_answer(raw, expected):
    assert extract_answer(raw) == expected


def test_bootstrap():
    x = np.r_[np.ones(60), np.zeros(40)]
    ci = bootstrap_ci(x, n_boot=500)
    assert ci["lo"] < 60 < ci["hi"]
    p = paired_bootstrap(np.zeros(100), np.r_[np.ones(30), np.zeros(70)], n_boot=500)
    assert p["diff"] == pytest.approx(30.0)
    assert p["lo"] > 0 and p["p_one_sided"] < 0.01


def test_categorize():
    base = {"answers": ["Paris"], "answer_in_prompt": True, "pred_in_context": True}
    assert categorize({**base, "prediction": "Paris", "em": 1.0, "f1": 1.0}) == "correct"
    assert categorize({**base, "prediction": "in Paris France", "em": 0.0, "f1": 0.5}) == "verbose"
    assert categorize({**base, "prediction": "Lyon", "em": 0.0, "f1": 0.0}) == "wrong_span"
    assert categorize({**base, "prediction": "Lyon", "em": 0.0, "f1": 0.0, "pred_in_context": False}) == "not_in_passage"
    assert categorize({**base, "prediction": "", "em": 0.0, "f1": 0.0}) == "empty"


def test_tiny_model_forward():
    torch = pytest.importorskip("torch")
    from cegqa.tiny.model import TinyConfig, TinyGPT

    m = TinyGPT(TinyConfig(vocab_size=100, max_seq_len=32, d_model=32, n_layers=2, n_heads=2))
    out = m(torch.randint(0, 100, (3, 16)))
    assert out.shape == (3, 16, 100)
    pc = m.param_counts()
    assert pc["total"] == pc["non_embedding"] + pc["embedding"]
    # causality: changing a future token must not change earlier logits
    x = torch.randint(0, 100, (1, 10))
    y = x.clone()
    y[0, -1] = (y[0, -1] + 1) % 100
    assert torch.allclose(m(x)[0, :-1], m(y)[0, :-1], atol=1e-5)


def test_collate_masks_prompt():
    torch = pytest.importorskip("torch")
    from cegqa.train_lora import collate

    b = collate([{"input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4]},
                 {"input_ids": [5, 6], "labels": [-100, 6]}], pad_id=0)
    assert b["input_ids"].tolist() == [[1, 2, 3, 4], [5, 6, 0, 0]]
    assert b["labels"].tolist() == [[-100, -100, 3, 4], [-100, 6, -100, -100]]
    assert b["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
