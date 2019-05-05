from pandas.util.testing import assert_frame_equal

from horsekickpy.horsekick import clean_horse_kicks, MissingColumnException
import pytest
import pandas as pd


def test_clean_horse_kicks_no_columns():
    input_df = pd.DataFrame({"a": ["b"]})

    with pytest.raises(MissingColumnException):
        clean_horse_kicks(input_df)


def test_clean_horse_kicks():
    input_df = pd.DataFrame({
        "year": [1600, 1800, 1900],
        "corps": ["I", "II", "XVI"],
        "dummy": list(range(3))
    })

    reference_df = pd.DataFrame({"year": [1800], "corps": ["II"]})

    result_df = clean_horse_kicks(input_df)

    assert_frame_equal(result_df.reset_index(drop=True), reference_df)