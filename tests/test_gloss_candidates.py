from spoken_to_signed.gloss_to_pose.lookup.lookup import gloss_candidates


def test_original_gloss_is_tried_first():
    assert gloss_candidates("HAUS")[0] == "HAUS"


def test_candidates_are_deduplicated():
    assert gloss_candidates("haus") == ["haus"]


def test_plus_artifact_is_removed():
    assert "rezept" in gloss_candidates("Rezept+")


def test_double_dash_artifact_is_removed():
    assert "gehen" in gloss_candidates("gehen--")


def test_trailing_ix_is_removed():
    assert "berg" in gloss_candidates("BERG-ix")


def test_wrapped_integer_is_unwrapped():
    assert "500" in gloss_candidates("500.")


def test_decimal_is_left_alone():
    assert gloss_candidates("3,14") == ["3,14"]


def test_alphanumeric_token_falls_back_to_letters():
    assert gloss_candidates("A3") == ["A3", "a3", "a"]


def test_punctuation_is_stripped_to_letters():
    assert "zürich" in gloss_candidates("Zürich.")


def test_empty_candidates_are_dropped():
    assert gloss_candidates(".") == ["."]
