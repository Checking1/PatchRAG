# -*- coding: utf-8 -*-
from ts_benchmark.evaluation.strategy.baseline_fixed_window_forecast import (
    BaselineFixedWindowForecast,
)
from ts_benchmark.evaluation.strategy.fixed_forecast import FixedForecast
from ts_benchmark.evaluation.strategy.rolling_forecast import RollingForecast

STRATEGY = {
    "fixed_forecast": FixedForecast,
    "rolling_forecast": RollingForecast,
    "baseline_fixed_window_forecast": BaselineFixedWindowForecast,
}
