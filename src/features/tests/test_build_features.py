import datetime
import pytz
import pandas as pd
import numpy as np

from pathlib import Path
import sys

sys.path.append(Path.cwd().parent)
from features import build_features


def test_create_timeseries_feature():

    # trivial tests
    start = datetime.datetime(year=2019, month=1, day=1).replace(tzinfo=pytz.utc)
    end = start + datetime.timedelta(seconds=60)
    pd_daterange = pd.date_range(start, end, freq="1T")

    data = pd.DataFrame(data=[10], index=[start], columns=["A"])

    print(build_features._create_timeseries_feature(data, "5m", "mean", 1, 1))


if __name__ == "__main__":
    test_create_timeseries_feature()
