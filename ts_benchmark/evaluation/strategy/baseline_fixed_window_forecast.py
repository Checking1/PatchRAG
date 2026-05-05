# -*- coding: utf-8 -*-
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ts_benchmark.evaluation.metrics import regression_metrics
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.evaluation.strategy.forecasting import ForecastingStrategy
from ts_benchmark.models import ModelFactory
from ts_benchmark.models.model_base import BatchMaker, ModelBase
from ts_benchmark.utils.data_processing import split_channel


class BaselineFixedWindowEvalBatchMaker:
    def __init__(
        self,
        series: pd.DataFrame,
        index_list: List[int],
        input_window_size: int,
        covariates: Optional[dict] = None,
    ):
        self.series = series
        self.index_list = index_list
        self.input_window_size = input_window_size
        self.current_sample_count = 0
        self.covariates = covariates

    def make_batch_predict(self, batch_size: int, win_size: int) -> dict:
        index_list = self.index_list[
            self.current_sample_count : self.current_sample_count + batch_size
        ]
        predict_batch = self._make_batch_data(
            self.series.values, np.array(index_list), win_size
        )
        covariates_batch = self._make_batch_covariates(np.array(index_list), win_size)
        self.current_sample_count += len(index_list)
        return {
            "input": predict_batch,
            "covariates": covariates_batch,
        }

    def make_batch_eval(self, horizon: int) -> dict:
        target_batch = self._make_batch_data(
            self.series.values,
            np.array(self.index_list) + self.input_window_size,
            horizon,
        )
        covariates_batch = self._make_batch_covariates(
            np.array(self.index_list) + self.input_window_size,
            horizon,
        )
        return {
            "target": target_batch,
            "covariates": covariates_batch,
        }

    def _make_batch_covariates(self, index_list: np.ndarray, win_size: int) -> Dict:
        covariates = {} if self.covariates is None else self.covariates
        covariates_batch = {}
        if covariates.get("exog") is not None:
            covariates_batch["exog"] = self._make_batch_data(
                self.covariates["exog"], index_list, win_size
            )
        return covariates_batch

    @staticmethod
    def _make_batch_data(
        data: Any, index_list: np.ndarray, win_size: int
    ) -> np.ndarray:
        windows = sliding_window_view(data, window_shape=(win_size, *data.shape[1:]))
        data_batch = windows[index_list]
        data_batch = np.squeeze(data_batch, axis=tuple(range(1, np.ndim(data))))
        return data_batch

    def has_more_batches(self) -> bool:
        return self.current_sample_count < len(self.index_list)


class BaselineFixedWindowPredictBatchMaker(BatchMaker):
    def __init__(self, batch_maker: BaselineFixedWindowEvalBatchMaker):
        self._batch_maker = batch_maker

    def make_batch(self, batch_size: int, win_size: int) -> dict:
        return self._batch_maker.make_batch_predict(batch_size, win_size)

    def has_more_batches(self) -> bool:
        return self._batch_maker.has_more_batches()


class BaselineFixedWindowForecast(ForecastingStrategy):
    """
    Baseline-aligned forecasting strategy.

    This strategy reproduces the MCLR-PCF evaluation protocol:
    - fixed 70/10/20 train/val/test split,
    - validation and test splits start with a seq_len overlap,
    - model fits once on the train+val segment,
    - evaluation traverses the fixed test split with sliding windows.
    """

    REQUIRED_CONFIGS = [
        "horizon",
        "train_ratio_in_tv",
        "save_true_pred",
        "target_channel",
    ]

    TRAIN_RATIO = 0.7
    TEST_RATIO = 0.2

    @staticmethod
    def _precompute_forecast_shortlists(
        model: ModelBase,
        query_contexts: Optional[np.ndarray],
    ) -> None:
        if not (
            hasattr(model, "precompute_query_shortlists")
            and hasattr(model, "prepare_query_context_array")
        ):
            return
        model.precompute_query_shortlists(query_contexts)

    @classmethod
    def _get_split_boundaries(
        cls,
        data_len: int,
        seq_len: int,
    ) -> Tuple[int, int, int, int, int, int, int, int]:
        num_train = int(data_len * cls.TRAIN_RATIO)
        num_test = int(data_len * cls.TEST_RATIO)
        num_val = data_len - num_train - num_test

        train_end = num_train
        val_start = num_train - seq_len
        val_end = num_train + num_val
        test_start = data_len - num_test - seq_len
        test_end = data_len

        if val_start < 0 or test_start < 0:
            raise ValueError("Series is shorter than seq_len under baseline-aligned split.")

        return num_train, num_val, num_test, train_end, val_start, val_end, test_start, test_end

    @staticmethod
    def _get_window_count(segment_len: int, seq_len: int, horizon: int) -> int:
        return segment_len - seq_len - horizon + 1

    @staticmethod
    def _get_exact_train_ratio_in_tv(num_train: int, train_valid_len: int) -> float:
        exact_ratio = num_train / train_valid_len
        return float(np.nextafter(exact_ratio, 1.0))

    def _build_query_contexts(
        self,
        model: ModelBase,
        target_test_segment: pd.DataFrame,
        exog_test_segment: Optional[pd.DataFrame],
        horizon: int,
    ) -> Optional[np.ndarray]:
        seq_len = model.config.seq_len
        num_windows = self._get_window_count(len(target_test_segment), seq_len, horizon)
        if num_windows <= 0:
            return None

        index_list = list(range(num_windows))
        batch_maker = BaselineFixedWindowEvalBatchMaker(
            target_test_segment,
            index_list,
            seq_len,
            {"exog": exog_test_segment},
        )
        window_batch = batch_maker.make_batch_predict(num_windows, seq_len)
        covariates_batch = window_batch.get("covariates", {}) or {}
        exog_batch = covariates_batch.get("exog")
        return model.prepare_query_context_array(window_batch["input"], exog_batch)

    def _execute(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        model_factory: ModelFactory,
        series_name: str,
    ) -> List:
        model = model_factory()
        if model.batch_forecast.__annotations__.get("not_implemented_batch"):
            return self._eval_sample(series, meta_info, model, series_name)
        return self._eval_batch(series, meta_info, model, series_name)

    def _prepare_data(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        model: ModelBase,
        series_name: str,
    ):
        target_channel = self._get_scalar_config_value("target_channel", series_name)
        horizon = self._get_scalar_config_value("horizon", series_name)
        configured_train_ratio_in_tv = self._get_scalar_config_value(
            "train_ratio_in_tv", series_name
        )

        data_len = int(self._get_meta_info(meta_info, "length", len(series)))
        (
            num_train,
            num_val,
            num_test,
            train_end,
            val_start,
            val_end,
            test_start,
            test_end,
        ) = self._get_split_boundaries(data_len, model.config.seq_len)

        train_valid_data = series.iloc[:val_end, :]
        test_segment = series.iloc[test_start:test_end, :]

        if len(train_valid_data) != num_train + num_val:
            raise RuntimeError("Train/validation split length mismatch.")

        exact_train_ratio_in_tv = self._get_exact_train_ratio_in_tv(
            num_train, len(train_valid_data)
        )
        if int(len(train_valid_data) * configured_train_ratio_in_tv) != num_train:
            raise ValueError(
                "Configured train_ratio_in_tv does not reproduce the baseline train boundary."
            )

        target_train_valid_data, exog_train_valid_data = split_channel(
            train_valid_data, target_channel
        )
        target_test_segment, exog_test_segment = split_channel(test_segment, target_channel)

        num_windows = self._get_window_count(
            len(target_test_segment), model.config.seq_len, horizon
        )
        if num_windows <= 0:
            raise ValueError(
                "The fixed test split is too short to form one evaluation window."
            )

        return (
            horizon,
            exact_train_ratio_in_tv,
            target_train_valid_data,
            exog_train_valid_data,
            target_test_segment,
            exog_test_segment,
            num_windows,
            num_train,
            num_val,
            num_test,
        )

    def _fit_model(
        self,
        model: ModelBase,
        target_train_valid_data: pd.DataFrame,
        exog_train_valid_data: Optional[pd.DataFrame],
        train_ratio_in_tv: float,
    ) -> Tuple[float, Any]:
        covariates_train = {"exog": exog_train_valid_data}
        start_fit_time = time.time()
        fit_method = model.forecast_fit if hasattr(model, "forecast_fit") else model.fit
        fit_method(
            target_train_valid_data,
            covariates=covariates_train,
            train_ratio_in_tv=train_ratio_in_tv,
        )
        end_fit_time = time.time()
        eval_scaler = self._get_eval_scaler(target_train_valid_data, train_ratio_in_tv)
        return end_fit_time - start_fit_time, eval_scaler

    def _eval_sample(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        model: ModelBase,
        series_name: str,
    ) -> List:
        (
            horizon,
            train_ratio_in_tv,
            target_train_valid_data,
            exog_train_valid_data,
            target_test_segment,
            exog_test_segment,
            num_windows,
            _,
            _,
            _,
        ) = self._prepare_data(series, meta_info, model, series_name)

        fit_time, eval_scaler = self._fit_model(
            model,
            target_train_valid_data,
            exog_train_valid_data,
            train_ratio_in_tv,
        )

        if hasattr(model, "prepare_query_context_array"):
            query_contexts = self._build_query_contexts(
                model,
                target_test_segment,
                exog_test_segment,
                horizon,
            )
            self._precompute_forecast_shortlists(model, query_contexts)

        total_inference_time = 0.0
        all_test_results = []
        all_rolling_actual = []
        all_rolling_predict = []
        seq_len = model.config.seq_len

        for start in range(num_windows):
            context = target_test_segment.iloc[start : start + seq_len, :]
            test = target_test_segment.iloc[
                start + seq_len : start + seq_len + horizon, :
            ]
            exog_context = None
            if exog_test_segment is not None:
                exog_context = exog_test_segment.iloc[start : start + seq_len, :]

            start_inference_time = time.time()
            predict = model.forecast(horizon, context, covariates={"exog": exog_context})
            end_inference_time = time.time()
            total_inference_time += end_inference_time - start_inference_time

            single_series_result = self.evaluator.evaluate(
                test.to_numpy(), predict, eval_scaler, target_train_valid_data.values
            )
            inference_data = pd.DataFrame(
                predict,
                columns=test.columns,
                index=test.index,
            )
            all_rolling_actual.append(test)
            all_rolling_predict.append(inference_data)
            all_test_results.append(single_series_result)

        single_series_results = np.mean(np.stack(all_test_results), axis=0).tolist()
        save_true_pred = self._get_scalar_config_value("save_true_pred", series_name)
        actual_data_encoded = (
            self._encode_data(all_rolling_actual) if save_true_pred else np.nan
        )
        inference_data_encoded = (
            self._encode_data(all_rolling_predict) if save_true_pred else np.nan
        )

        single_series_results += [
            series_name,
            fit_time,
            total_inference_time,
            actual_data_encoded,
            inference_data_encoded,
            "",
        ]
        return single_series_results

    def _eval_batch(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        model: ModelBase,
        series_name: str,
    ) -> List:
        (
            horizon,
            train_ratio_in_tv,
            target_train_valid_data,
            exog_train_valid_data,
            target_test_segment,
            exog_test_segment,
            num_windows,
            _,
            _,
            _,
        ) = self._prepare_data(series, meta_info, model, series_name)

        fit_time, eval_scaler = self._fit_model(
            model,
            target_train_valid_data,
            exog_train_valid_data,
            train_ratio_in_tv,
        )

        index_list = list(range(num_windows))
        batch_maker = BaselineFixedWindowEvalBatchMaker(
            target_test_segment,
            index_list,
            model.config.seq_len,
            {"exog": exog_test_segment},
        )

        if hasattr(model, "prepare_query_context_array"):
            window_batch = batch_maker.make_batch_predict(len(index_list), model.config.seq_len)
            covariates_batch = window_batch.get("covariates", {}) or {}
            exog_batch = covariates_batch.get("exog")
            prepared = model.prepare_query_context_array(window_batch["input"], exog_batch)
            self._precompute_forecast_shortlists(model, prepared)
            batch_maker.current_sample_count = 0

        all_predicts = []
        total_inference_time = 0.0
        predict_batch_maker = BaselineFixedWindowPredictBatchMaker(batch_maker)
        while predict_batch_maker.has_more_batches():
            start_inference_time = time.time()
            predicts = model.batch_forecast(horizon, predict_batch_maker)
            end_inference_time = time.time()
            total_inference_time += end_inference_time - start_inference_time
            all_predicts.append(predicts)

        all_predicts = np.concatenate(all_predicts, axis=0)
        targets = batch_maker.make_batch_eval(horizon)["target"]
        if len(targets) != len(all_predicts):
            raise RuntimeError("Predictions' len don't equal targets' len!")

        all_test_results = []
        for predicts, target in zip(all_predicts, targets):
            single_series_result = self.evaluator.evaluate(
                target,
                predicts,
                eval_scaler,
                target_train_valid_data.values,
            )
            all_test_results.append(single_series_result)
        single_series_results = np.mean(np.stack(all_test_results), axis=0).tolist()

        save_true_pred = self._get_scalar_config_value("save_true_pred", series_name)
        actual_data_encoded = self._encode_data(targets) if save_true_pred else np.nan
        inference_data_encoded = (
            self._encode_data(all_predicts) if save_true_pred else np.nan
        )

        single_series_results += [
            series_name,
            fit_time,
            total_inference_time,
            actual_data_encoded,
            inference_data_encoded,
            "",
        ]
        return single_series_results

    @staticmethod
    def accepted_metrics() -> List[str]:
        return regression_metrics.__all__

    @property
    def field_names(self) -> List[str]:
        return self.evaluator.metric_names + [
            FieldNames.FILE_NAME,
            FieldNames.FIT_TIME,
            FieldNames.INFERENCE_TIME,
            FieldNames.ACTUAL_DATA,
            FieldNames.INFERENCE_DATA,
            FieldNames.LOG_INFO,
        ]
