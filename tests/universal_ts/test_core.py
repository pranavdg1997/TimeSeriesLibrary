"""Unit tests for UniversalForecaster core functionality."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from universal_ts import UniversalForecaster, evaluate
from universal_ts.exceptions import (
    BackendNotFoundError,
    BackendNotInstalledError,
    FitNotCalledError,
    DataValidationError
)


class TestUniversalForecasterInit:
    """Tests for UniversalForecaster initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        with patch('universal_ts.core._load_backend') as mock_load:
            mock_backend_class = Mock()
            mock_backend = Mock()
            mock_backend_class.return_value = mock_backend
            mock_load.return_value = mock_backend_class

            model = UniversalForecaster()

            assert model.backend_name == "prophet"
            assert model.freq is None
            assert model.country_holidays is None
            assert model.is_fitted is False
            assert model.group_id_col is None
            assert model.covariates == []
            assert model.regressors == []
            assert model.seasonalities == []
            mock_load.assert_called_once_with("prophet")

    def test_custom_backend_initialization(self):
        """Test initialization with custom backend."""
        with patch('universal_ts.core._load_backend') as mock_load:
            mock_backend_class = Mock()
            mock_backend = Mock()
            mock_backend_class.return_value = mock_backend
            mock_load.return_value = mock_backend_class

            model = UniversalForecaster(
                backend="autogluon",
                freq="D",
                country_holidays=["US", "UK"]
            )

            assert model.backend_name == "autogluon"
            assert model.freq == "D"
            assert model.country_holidays == ["US", "UK"]
            mock_load.assert_called_once_with("autogluon")

    def test_backend_not_found_error(self):
        """Test error when backend is not found."""
        with patch('universal_ts.core._load_backend') as mock_load:
            mock_load.side_effect = BackendNotFoundError("Backend not found")

            with pytest.raises(BackendNotFoundError):
                UniversalForecaster(backend="nonexistent")

    def test_backend_not_installed_error(self):
        """Test error when backend dependencies are not installed."""
        with patch('universal_ts.core._load_backend') as mock_load:
            mock_load.side_effect = BackendNotInstalledError("Dependencies not installed")

            with pytest.raises(BackendNotInstalledError):
                UniversalForecaster(backend="prophet")


class TestLoadBackend:
    """Tests for _load_backend function."""

    def test_load_prophet_backend(self):
        """Test loading Prophet backend."""
        from universal_ts.core import _load_backend

        with patch.dict('sys.modules', {'universal_ts.backends.prophet_backend': Mock()}):
            with patch('universal_ts.backends.prophet_backend.ProphetBackend') as mock_class:
                backend_class = _load_backend("prophet")
                assert backend_class == mock_class

    def test_load_autogluon_backend(self):
        """Test loading AutoGluon backend."""
        from universal_ts.core import _load_backend

        with patch.dict('sys.modules', {'universal_ts.backends.autogluon_backend': Mock()}):
            with patch('universal_ts.backends.autogluon_backend.AutoGluonBackend') as mock_class:
                backend_class = _load_backend("autogluon")
                assert backend_class == mock_class

    def test_backend_not_found(self):
        """Test error for unknown backend."""
        from universal_ts.core import _load_backend

        with pytest.raises(BackendNotFoundError):
            _load_backend("unknown_backend")

    def test_backend_import_error(self):
        """Test error when backend module cannot be imported."""
        from universal_ts.core import _load_backend

        with patch('builtins.__import__', side_effect=ImportError("Module not found")):
            with pytest.raises(BackendNotInstalledError):
                _load_backend("prophet")


class TestUniversalForecasterFit:
    """Tests for UniversalForecaster.fit method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_df = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=100, freq='D'),
            'y': np.arange(100) + np.random.normal(0, 1, 100)
        })

        self.mock_backend = Mock()
        self.mock_backend.supports_panel_data.return_value = True
        self.mock_backend.models = {'dummy': Mock()}  # Add models attribute for predict tests
        self.mock_backend.models = {'dummy': Mock()}  # Add models attribute for predict tests
        self.mock_backend.models = {'dummy': Mock()}  # Add models attribute for predict tests

    def test_fit_single_series(self):
        """Test fitting single series."""
        with patch('universal_ts.core._load_backend') as mock_load:
            mock_backend_class = Mock()
            mock_backend_class.return_value = self.mock_backend
            mock_load.return_value = mock_backend_class

            model = UniversalForecaster(backend="prophet")
            model.fit(self.sample_df)

            assert model.is_fitted is True
            assert model.group_id_col is None
            self.mock_backend.fit.assert_called_once()

    def test_fit_panel_data(self):
        """Test fitting panel data."""
        panel_df = pd.DataFrame({
            'group_id': ['A'] * 50 + ['B'] * 50,
            'ds': pd.date_range('2020-01-01', periods=50, freq='D').tolist() * 2,
            'y': np.arange(100)
        })

        with patch('universal_ts.core._load_backend') as mock_load:
            mock_backend_class = Mock()
            mock_backend_class.return_value = self.mock_backend
            mock_load.return_value = mock_backend_class

            model = UniversalForecaster(backend="autogluon")
            model.fit(panel_df, group_id="group_id")

            assert model.is_fitted is True
            assert model.group_id_col == "group_id"
            self.mock_backend.fit.assert_called_once()

    def test_fit_with_holidays(self):
        """Test fitting with holiday features."""
        with patch('universal_ts.core._load_backend') as mock_load:
            mock_backend_class = Mock()
            mock_backend_class.return_value = self.mock_backend
            mock_load.return_value = mock_backend_class

            with patch('universal_ts.core.HolidayFeatureGenerator') as mock_holiday:
                mock_holiday_gen = Mock()
                mock_holiday.return_value = mock_holiday_gen
                mock_holiday_gen.add_holiday_features.return_value = self.sample_df

                model = UniversalForecaster(
                    backend="prophet",
                    country_holidays=["US"]
                )
                model.fit(self.sample_df)

                assert model.holiday_generator is not None
                mock_holiday_gen.add_holiday_features.assert_called()

    def test_fit_frequency_inference(self):
        """Test frequency inference during fit."""
        with patch('universal_ts.core._load_backend') as mock_load:
            mock_backend_class = Mock()
            mock_backend_class.return_value = self.mock_backend
            mock_load.return_value = mock_backend_class

            with patch('universal_ts.core.prepare_dataframe') as mock_prepare:
                mock_prepare.return_value = (self.sample_df, "D")

                model = UniversalForecaster(backend="prophet")
                model.fit(self.sample_df)

                assert model.freq == "D"

    def test_fit_panel_data_backend_warning(self):
        """Test warning when backend doesn't support panel data."""
        panel_df = pd.DataFrame({
            'group_id': ['A'] * 50 + ['B'] * 50,
            'ds': pd.date_range('2020-01-01', periods=50, freq='D').tolist() * 2,
            'y': np.arange(100)
        })

        self.mock_backend.supports_panel_data.return_value = False

        with patch('universal_ts.core._load_backend') as mock_load:
            mock_backend_class = Mock()
            mock_backend_class.return_value = self.mock_backend
            mock_load.return_value = mock_backend_class

            model = UniversalForecaster(backend="prophet")

            with pytest.warns(UserWarning, match="does not natively support panel data"):
                model.fit(panel_df, group_id="group_id")


class TestUniversalForecasterPredict:
    """Tests for UniversalForecaster.predict method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_df = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=100, freq='D'),
            'y': np.arange(100)
        })

        self.forecast_df = pd.DataFrame({
            'ds': pd.date_range('2020-04-10', periods=10, freq='D'),
            'yhat': np.arange(100, 110)
        })

        self.mock_backend = Mock()
        self.mock_backend.supports_panel_data.return_value = True
        self.mock_backend.models = {'dummy': Mock()}  # Add models attribute for predict tests
        self.mock_backend.models = {'dummy': Mock()}  # Add models attribute for predict tests

    def test_predict_after_fit(self):
        """Test prediction after successful fit."""
        with patch('universal_ts.core._load_backend') as mock_load:
            mock_backend_class = Mock()
            mock_backend_class.return_value = self.mock_backend
            mock_load.return_value = mock_backend_class

            self.mock_backend.predict.return_value = self.forecast_df

            model = UniversalForecaster(backend="prophet")
            model.fit(self.sample_df)
            forecast = model.predict(horizon=10)

            assert len(forecast) == 10
            assert 'ds' in forecast.columns
            assert 'yhat' in forecast.columns
            self.mock_backend.predict.assert_called_once()

    def test_predict_without_fit_error(self):
        """Test error when predict is called before fit."""
        with patch('universal_ts.core._load_backend') as mock_load:
            mock_backend_class = Mock()
            mock_backend_class.return_value = self.mock_backend
            mock_load.return_value = mock_backend_class

            model = UniversalForecaster(backend="prophet")

            with pytest.raises(FitNotCalledError, match="Model has not been fitted"):
                model.predict(horizon=10)

    def test_predict_with_future_dataframe(self):
        """Test prediction with custom future dataframe."""
        future_df = pd.DataFrame({
            'ds': pd.date_range('2020-04-10', periods=10, freq='D'),
            'temperature': np.random.normal(20, 5, 10)
        })

        with patch('universal_ts.core._load_backend') as mock_load:
            mock_backend_class = Mock()
            mock_backend_class.return_value = self.mock_backend
            mock_load.return_value = mock_backend_class

            self.mock_backend.predict.return_value = self.forecast_df

            model = UniversalForecaster(backend="prophet")
            model.fit(self.sample_df)
            forecast = model.predict(horizon=10, df_future=future_df)

            self.mock_backend.predict.assert_called_once_with(
                horizon=10,
                df_future=future_df
            )

    def test_predict_auto_generate_future_dataframe(self):
        """Test auto-generation of future dataframe with holidays."""
        with patch('universal_ts.core._load_backend') as mock_load:
            mock_backend_class = Mock()
            mock_backend_class.return_value = self.mock_backend
            mock_load.return_value = mock_backend_class

            # Mock holiday generator
            mock_holiday_gen = Mock()
            mock_holiday_gen.add_holiday_features.return_value = self.forecast_df

            model = UniversalForecaster(
                backend="prophet",
                country_holidays=["US"]
            )
            model.holiday_generator = mock_holiday_gen
            model.is_fitted = True
            model.freq = "D"

            # Create a properly mocked model with history
            mock_model = Mock()
            mock_model.history = pd.DataFrame({
                'ds': pd.date_range('2020-01-01', periods=10, freq='D'),
                'y': range(10)
            })
            self.mock_backend.models = {'test_group': mock_model}

            # Mock backend to return forecast
            self.mock_backend.predict.return_value = self.forecast_df

            forecast = model.predict(horizon=10)

            # Should have called holiday generator
            assert len(forecast) == 10


class TestUniversalForecasterMethods:
    """Tests for additional UniversalForecaster methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_df = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=100, freq='D'),
            'y': np.arange(100)
        })

        self.mock_backend = Mock()
        self.mock_backend.supports_panel_data.return_value = True
        self.mock_backend.models = {'dummy': Mock()}  # Add models attribute for predict tests

    def test_add_regressor(self):
        """Test adding regressor."""
        with patch('universal_ts.core._load_backend') as mock_load:
            mock_backend_class = Mock()
            mock_backend_class.return_value = self.mock_backend
            mock_load.return_value = mock_backend_class

            model = UniversalForecaster(backend="prophet")
            model.add_regressor("temperature", prior_scale=0.5, mode="additive")

            assert len(model.regressors) == 1
            assert model.regressors[0]["name"] == "temperature"
            assert model.regressors[0]["prior_scale"] == 0.5
            assert model.regressors[0]["mode"] == "additive"
            assert "temperature" in model.covariates

    def test_add_regressor_after_fit_warning(self):
        """Test warning when adding regressor after fit."""
        with patch('universal_ts.core._load_backend') as mock_load:
            mock_backend_class = Mock()
            mock_backend_class.return_value = self.mock_backend
            mock_load.return_value = mock_backend_class

            model = UniversalForecaster(backend="prophet")
            model.is_fitted = True

            with pytest.warns(UserWarning, match="Adding regressor after fit"):
                model.add_regressor("temperature")

    def test_add_seasonality(self):
        """Test adding custom seasonality."""
        with patch('universal_ts.core._load_backend') as mock_load:
            mock_backend_class = Mock()
            mock_backend_class.return_value = self.mock_backend
            mock_load.return_value = mock_backend_class

            model = UniversalForecaster(backend="prophet")
            model.add_seasonality("monthly", period=30.5, fourier_order=5)

            assert len(model.seasonalities) == 1
            assert model.seasonalities[0]["name"] == "monthly"
            assert model.seasonalities[0]["period"] == 30.5
            assert model.seasonalities[0]["fourier_order"] == 5

    def test_add_seasonality_non_prophet_warning(self):
        """Test warning when adding seasonality to non-Prophet backend."""
        with patch('universal_ts.core._load_backend') as mock_load:
            mock_backend_class = Mock()
            mock_backend_class.return_value = self.mock_backend
            mock_load.return_value = mock_backend_class

            model = UniversalForecaster(backend="autogluon")

            with pytest.warns(UserWarning, match="may not support custom seasonalities"):
                model.add_seasonality("monthly", period=30.5, fourier_order=5)

    def test_get_model_info(self):
        """Test getting model information."""
        with patch('universal_ts.core._load_backend') as mock_load:
            mock_backend_class = Mock()
            mock_backend_class.return_value = self.mock_backend
            mock_load.return_value = mock_backend_class

            model = UniversalForecaster(
                backend="prophet",
                freq="D",
                country_holidays=["US"]
            )
            model.add_regressor("temperature")
            model.add_seasonality("monthly", period=30.5, fourier_order=5)

            info = model.get_model_info()

            assert info["backend"] == "prophet"
            assert info["freq"] == "D"
            assert info["country_holidays"] == ["US"]
            assert info["is_fitted"] is False
            assert len(info["regressors"]) == 1
            assert len(info["seasonalities"]) == 1

    def test_get_model_info_after_fit(self):
        """Test getting model info after fit."""
        with patch('universal_ts.core._load_backend') as mock_load:
            mock_backend_class = Mock()
            mock_backend_class.return_value = self.mock_backend
            mock_load.return_value = mock_backend_class

            # Mock backend info
            self.mock_backend.get_model_info.return_value = {"model_type": "Prophet"}

            model = UniversalForecaster(backend="prophet")
            model.fit(self.sample_df)

            info = model.get_model_info()

            assert info["is_fitted"] is True
            assert "backend_info" in info
            assert info["backend_info"]["model_type"] == "Prophet"


class TestUniversalForecasterEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_dataframe_error(self):
        """Test error with empty dataframe."""
        with patch('universal_ts.core._load_backend') as mock_load:
            mock_backend_class = Mock()
            mock_backend = Mock()
            mock_backend_class.return_value = mock_backend
            mock_load.return_value = mock_backend_class

            model = UniversalForecaster(backend="prophet")

            with pytest.raises(DataValidationError):
                model.fit(pd.DataFrame())

    def test_missing_columns_error(self):
        """Test error with missing required columns."""
        with patch('universal_ts.core._load_backend') as mock_load:
            mock_backend_class = Mock()
            mock_backend = Mock()
            mock_backend_class.return_value = mock_backend
            mock_load.return_value = mock_backend_class

            model = UniversalForecaster(backend="prophet")

            # Missing 'y' column
            df_missing_y = pd.DataFrame({
                'ds': pd.date_range('2020-01-01', periods=10)
            })

            with pytest.raises(DataValidationError):
                model.fit(df_missing_y)

            # Missing 'ds' column
            df_missing_ds = pd.DataFrame({
                'y': range(10)
            })

            with pytest.raises(DataValidationError):
                model.fit(df_missing_ds)

    def test_non_numeric_target_error(self):
        """Test error with non-numeric target column."""
        with patch('universal_ts.core._load_backend') as mock_load:
            mock_backend_class = Mock()
            mock_backend = Mock()
            mock_backend_class.return_value = mock_backend
            mock_load.return_value = mock_backend_class

            model = UniversalForecaster(backend="prophet")

            df_non_numeric = pd.DataFrame({
                'ds': pd.date_range('2020-01-01', periods=10),
                'y': ['a'] * 10
            })

            with pytest.raises(DataValidationError):
                model.fit(df_non_numeric)


class TestUniversalForecasterIntegration:
    """Integration tests that don't require actual backends."""

    def test_method_chaining(self):
        """Test method chaining for fluent interface."""
        with patch('universal_ts.core._load_backend') as mock_load:
            mock_backend_class = Mock()
            mock_backend = Mock()
            mock_backend_class.return_value = mock_backend
            mock_load.return_value = mock_backend_class

            model = UniversalForecaster(backend="prophet")

            # Test method chaining
            result = (model
                     .add_regressor("temperature")
                     .add_seasonality("monthly", period=30.5, fourier_order=5))

            assert result is model
            assert len(model.regressors) == 1
            assert len(model.seasonalities) == 1

    def test_with_real_data_structure(self):
        """Test with realistic data structure."""
        with patch('universal_ts.core._load_backend') as mock_load:
            mock_backend_class = Mock()
            mock_backend = Mock()
            mock_backend_class.return_value = mock_backend
            mock_load.return_value = mock_backend_class

            # Create realistic data
            np.random.seed(42)
            dates = pd.date_range('2020-01-01', periods=365, freq='D')
            trend = np.arange(365) * 0.1
            seasonal = 10 * np.sin(2 * np.pi * np.arange(365) / 365.25)
            noise = np.random.normal(0, 2, 365)
            values = 100 + trend + seasonal + noise

            df = pd.DataFrame({
                'ds': dates,
                'y': values
            })

            model = UniversalForecaster(backend="prophet")
            model.fit(df)

            assert model.is_fitted is True
            mock_backend.fit.assert_called_once()
