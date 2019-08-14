import datetime
import pytz
import pandas as pd
import numpy as np

from pathlib import Path
import sys

sys.path.append(str(Path.cwd().parent))
from features import build_features


def test_create_timeseries_feature():

    # trivial tests
    start = datetime.datetime(year=2019, month=1, day=1).replace(tzinfo=pytz.utc)
    end = start + datetime.timedelta(seconds=60)
    pd_daterange = pd.date_range(start, end, freq="1T")

    data = pd.DataFrame(data=[10], index=[start], columns=["A"])

    # No shifts
    expected_result_data = data.copy()
    expected_result_data.columns = [data.columns[0] + "_0"]
    received_result_data = build_features._create_timeseries_feature(data, "5m", "mean", 0, 0)
    pd.testing.assert_frame_equal(received_result_data, expected_result_data)

    # Tests: pd.dataframe. 10 min in 1 min intervals.
    start = datetime.datetime(year=2019, month=1, day=1).replace(tzinfo=pytz.utc)
    end = start + datetime.timedelta(seconds=10 * 60)
    pd_daterange = pd.date_range(start, end, freq="1T")[:-1]

    data = pd.DataFrame(data=[10, 9, -2, 2, 5, 4, 1, 2, -20, 2], index=pd_daterange, columns=["A"])

    # * no shift, no aggregation
    expected_result_data = data.copy()
    expected_result_data.columns = [data.columns[0] + "_0"]
    received_result_data = build_features._create_timeseries_feature(data, "1m", "mean", 0, 0)
    pd.testing.assert_frame_equal(received_result_data, expected_result_data)

    # * no shift, 2 min mean aggregation
    expected_result_data = pd.DataFrame(data=[(10 + 9)/2., (9 + -2)/2., (-2 + 2)/2., (2 + 5)/2., (5 + 4)/2., (4 + 1)/2., (1 + 2)/2., (2 + -20)/2., (-20 + 2)/2., 2], columns=["A_0"], index=pd_daterange)
    received_result_data = build_features._create_timeseries_feature(data, "2m", "mean", 0, 0)
    pd.testing.assert_frame_equal(received_result_data, expected_result_data)

    # shift 2 back and front with 2min mean aggregation
    expected_result_data = pd.DataFrame(data=[(10 + 9)/2., (9 + -2)/2., (-2 + 2)/2., (2 + 5)/2., (5 + 4)/2., (4 + 1)/2., (1 + 2)/2., (2 + -20)/2., (-20 + 2)/2., 2], columns=["A_0"], index=pd_daterange)
    expected_result_data["A_+1"] = expected_result_data["A_0"].shift(periods=-1)
    expected_result_data["A_+2"] = expected_result_data["A_0"].shift(periods=-2)
    expected_result_data["A_-1"] = expected_result_data["A_0"].shift(periods=1)
    expected_result_data["A_-2"] = expected_result_data["A_0"].shift(periods=2)
    received_result_data = build_features._create_timeseries_feature(data, "2m", "mean", 2, 2)
    pd.testing.assert_frame_equal(received_result_data, expected_result_data)

    # * no shift, 2 min max aggregation
    expected_result_data = pd.DataFrame(
        data=[10, 9, 2, 5, 5, 4, 2, 2, 2, 2], columns=["A_0"], index=pd_daterange)
    received_result_data = build_features._create_timeseries_feature(data, "2m", "max", 0, 0)
    pd.testing.assert_frame_equal(received_result_data, expected_result_data)

    # * no shift, 2 min min aggregation
    expected_result_data = pd.DataFrame(
        data=[9, -2, -2, 2, 4, 1, 1, -20, -20, 2], columns=["A_0"], index=pd_daterange)
    received_result_data = build_features._create_timeseries_feature(data, "2m", "min", 0, 0)
    pd.testing.assert_frame_equal(received_result_data, expected_result_data)

if __name__ == "__main__":
    test_create_timeseries_feature()
