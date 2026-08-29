"""Comprehensive integration tests for all backends."""

import pytest
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
import os
import sys

# Test data generators
def create_synthetic_series(n_periods=100, freq='D', trend=0.1, noise=0.1, seasonal=True):
    """Create realistic synthetic time series data."""
    dates = pd.date_range('2020-01-01', periods=n_periods, freq=freq)

    # Trend component
    trend_component = np.arange(n_periods) * trend

    # Seasonal component
    seasonal_component = np.zeros(n_periods)
    if seasonal and freq == 'D':
        # Daily seasonality (weekly pattern)
        seasonal_component += 5 * np.sin(2 * np.pi * np.arange(n_periods) / 7)
        # Yearly seasonality
        seasonal_component += 10 * np.sin(2 * np.pi * np.arange(n_periods) / 365.25)
    elif seasonal and freq == 'H':
        # Hourly seasonality (daily pattern)
        seasonal_component += 3 * np.sin(2 * np.pi * np.arange(n_periods) / 24)

    # Noise component
    np.random.seed(42)  # For reproducible tests
    noise_component = np.random.normal(0, noise, n_periods)

    # Combine components
    values = 100 + trend_component + seasonal_component + noise_component

    return pd.DataFrame({
        'ds': dates,
        'y': values
    })


def create_panel_data(n_series=3, n_periods=50, freq='D'):
    """Create panel data with multiple series."""
    all_data = []

    for i in range(n_series):
        series_data = create_synthetic_series(
            n_periods=n_periods,
            freq=freq,
            trend=0.1 * (i + 1),
            noise=0.5
        )
        series_data['group_id'] = f"series_{i}"
        all_data.append(series_data)

    return pd.concat(all_data, ignore_index=True)


def create_covariates_data(n_periods=100, freq='D'):
    """Create data with external covariates."""
    dates = pd.date_range('2020-01-01', periods=n_periods, freq=freq)

    # Main series
    main_values = 100 + np.arange(n_periods) * 0.1 + np.random.normal(0, 2, n_periods)

    # External covariates
    temperature = 20 + 10 * np.sin(2 * np.pi * np.arange(n_periods) / 365.25) + np.random.normal(0, 2, n_periods)
    promotion = np.random.choice([0, 1], n_periods, p=[0.8, 0.2])

    return pd.DataFrame({
        'ds': dates,
        'y': main_values,
        'temperature': temperature,
        'promotion': promotion
    })


class TestProphetIntegration:
    """Integration tests for Prophet backend."""

    @pytest.mark.skipif(
        not pytest.importorskip("prophet", reason="Prophet not installed"),
        reason="Prophet backend tests"
    )
    def test_prophet_single_series_basic(self):
        """Test Prophet basic single series forecasting."""
        from universal_ts import UniversalForecaster

        # Create test data
        df = create_synthetic_series(n_periods=100, freq='D')

        # Fit and predict
        model = UniversalForecaster(backend='prophet')
        model.fit(df)
        forecast = model.predict(horizon=30)

        # Assertions
        assert len(forecast) == 30
        assert 'ds' in forecast.columns
        assert 'yhat' in forecast.columns
        assert 'yhat_lower' in forecast.columns
        assert 'yhat_upper' in forecast.columns

        # Check forecast dates are after training data
        assert forecast['ds'].min() > df['ds'].max()

        # Check model info
        info = model.get_model_info()
        assert info['backend'] == 'prophet'
        assert info['is_fitted'] is True

    @pytest.mark.skipif(
        not pytest.importorskip("prophet", reason="Prophet not installed"),
        reason="Prophet backend tests"
    )
    def test_prophet_panel_data(self):
        """Test Prophet with panel data."""
        from universal_ts import UniversalForecaster

        # Create panel data
        df = create_panel_data(n_series=3, n_periods=50)

        # Fit and predict
        model = UniversalForecaster(backend='prophet')
        model.fit(df, group_id='group_id')
        forecast = model.predict(horizon=20)

        # Assertions
        assert len(forecast) == 60  # 20 forecasts for 3 series
        assert 'group_id' in forecast.columns
        assert set(forecast['group_id']) == {'series_0', 'series_1', 'series_2'}

    @pytest.mark.skipif(
        not pytest.importorskip("prophet", reason="Prophet not installed"),
        reason="Prophet backend tests"
    )
    def test_prophet_with_holidays(self):
        """Test Prophet with holiday features."""
        from universal_ts import UniversalForecaster

        # Create yearly data to capture holidays
        df = create_synthetic_series(n_periods=365, freq='D')

        # Fit with holidays
        model = UniversalForecaster(
            backend='prophet',
            country_holidays=['US']
        )
        model.fit(df)
        forecast = model.predict(horizon=30)

        # Assertions
        assert len(forecast) == 30
        assert model.country_holidays == ['US']

        # Check that holiday features were added
        assert len(model.covariates) > 0
        assert any('holiday' in cov.lower() for cov in model.covariates)

    @pytest.mark.skipif(
        not pytest.importorskip("prophet", reason="Prophet not installed"),
        reason="Prophet backend tests"
    )
    def test_prophet_with_regressors(self):
        """Test Prophet with external regressors."""
        from universal_ts import UniversalForecaster

        # Create data with covariates
        df = create_covariates_data(n_periods=100)

        # Fit with regressors
        model = UniversalForecaster(backend='prophet')
        model.add_regressor('temperature')
        model.add_regressor('promotion')
        model.fit(df)

        # Check regressors were added
        assert len(model.regressors) == 2
        assert model.regressors[0]['name'] == 'temperature'
        assert model.regressors[1]['name'] == 'promotion'

        # Create future covariates for prediction
        future_dates = pd.date_range(df['ds'].max() + timedelta(days=1), periods=30, freq='D')
        future_df = pd.DataFrame({
            'ds': future_dates,
            'temperature': 20 + 5 * np.sin(2 * np.pi * np.arange(30) / 30),
            'promotion': np.random.choice([0, 1], 30, p=[0.8, 0.2])
        })

        forecast = model.predict(horizon=30, df_future=future_df)
        assert len(forecast) == 30

    @pytest.mark.skipif(
        not pytest.importorskip("prophet", reason="Prophet not installed"),
        reason="Prophet backend tests"
    )
    def test_prophet_custom_seasonality(self):
        """Test Prophet with custom seasonality."""
        from universal_ts import UniversalForecaster

        df = create_synthetic_series(n_periods=100, freq='D')

        # Add custom seasonality
        model = UniversalForecaster(backend='prophet')
        model.add_seasonality('monthly', period=30.5, fourier_order=5)
        model.fit(df)

        # Check seasonality was added
        assert len(model.seasonalities) == 1
        assert model.seasonalities[0]['name'] == 'monthly'
        assert model.seasonalities[0]['period'] == 30.5


class TestAutoGluonIntegration:
    """Integration tests for AutoGluon backend."""

    @pytest.mark.skipif(
        not pytest.importorskip("autogluon.timeseries", reason="AutoGluon not installed"),
        reason="AutoGluon backend tests"
    )
    def test_autogluon_single_series_basic(self):
        """Test AutoGluon basic single series forecasting."""
        from universal_ts import UniversalForecaster

        # Create test data
        df = create_synthetic_series(n_periods=100, freq='D')

        # Fit and predict
        model = UniversalForecaster(
            backend='autogluon',
            prediction_length=10,
            eval_metric='MASE',
            verbosity=0
        )
        model.fit(df, time_limit=30)  # Short time limit for tests
        forecast = model.predict(horizon=10)

        # Assertions
        assert len(forecast) == 10
        assert 'ds' in forecast.columns
        assert 'yhat' in forecast.columns

        # Check model info
        info = model.get_model_info()
        assert info['backend'] == 'autogluon'
        assert info['is_fitted'] is True

    @pytest.mark.skipif(
        not pytest.importorskip("autogluon.timeseries", reason="AutoGluon not installed"),
        reason="AutoGluon backend tests"
    )
    def test_autogluon_panel_data(self):
        """Test AutoGluon with panel data."""
        from universal_ts import UniversalForecaster

        # Create panel data
        df = create_panel_data(n_series=3, n_periods=50)

        # Fit and predict
        model = UniversalForecaster(
            backend='autogluon',
            prediction_length=10,
            verbosity=0
        )
        model.fit(df, group_id='group_id', time_limit=30)
        forecast = model.predict(horizon=10)

        # Assertions
        assert len(forecast) == 30  # 10 forecasts for 3 series
        assert 'ds' in forecast.columns
        assert 'yhat' in forecast.columns

    @pytest.mark.skipif(
        not pytest.importorskip("autogluon.timeseries", reason="AutoGluon not installed"),
        reason="AutoGluon backend tests"
    )
    def test_autogluon_gpu_detection(self):
        """Test AutoGluon GPU detection and usage."""
        from universal_ts import UniversalForecaster

        # Create test data
        df = create_synthetic_series(n_periods=50, freq='D')

        # Test GPU auto-detection
        model = UniversalForecaster(
            backend='autogluon',
            prediction_length=5,
            num_gpus=None,  # Auto-detect
            verbosity=0
        )

        # Fit should not fail regardless of GPU availability
        model.fit(df, time_limit=20)
        forecast = model.predict(horizon=5)

        assert len(forecast) == 5

        # Check GPU info if available
        if hasattr(model.backend, 'num_gpus'):
            gpu_count = model.backend.num_gpus
            assert isinstance(gpu_count, int)
            assert gpu_count >= 0

    @pytest.mark.skipif(
        not pytest.importorskip("autogluon.timeseries", reason="AutoGluon not installed"),
        reason="AutoGluon backend tests"
    )
    def test_autogluon_force_cpu(self):
        """Test AutoGluon forced CPU usage."""
        from universal_ts import UniversalForecaster

        df = create_synthetic_series(n_periods=50, freq='D')

        # Force CPU usage
        model = UniversalForecaster(
            backend='autogluon',
            prediction_length=5,
            num_gpus=0,  # Force CPU
            verbosity=0
        )

        model.fit(df, time_limit=20)
        forecast = model.predict(horizon=5)

        assert len(forecast) == 5

    @pytest.mark.skipif(
        not pytest.importorskip("autogluon.timeseries", reason="AutoGluon not installed"),
        reason="AutoGluon backend tests"
    )
    def test_autogluon_with_covariates(self):
        """Test AutoGluon with covariates."""
        from universal_ts import UniversalForecaster

        # Create data with covariates
        df = create_covariates_data(n_periods=80)

        # Fit with known covariates
        model = UniversalForecaster(
            backend='autogluon',
            prediction_length=10,
            verbosity=0
        )
        model.add_regressor('temperature')
        model.add_regressor('promotion')
        model.fit(df, time_limit=30)

        # Create future covariates
        future_dates = pd.date_range(df['ds'].max() + timedelta(days=1), periods=10, freq='D')
        future_df = pd.DataFrame({
            'ds': future_dates,
            'temperature': 20 + 5 * np.sin(2 * np.pi * np.arange(10) / 10),
            'promotion': np.random.choice([0, 1], 10, p=[0.8, 0.2])
        })

        # Test prediction with future covariates
        forecast = model.predict(horizon=10, df_future=future_df)
        assert len(forecast) == 10


class TestSktimeIntegration:
    """Integration tests for sktime backend."""

    @pytest.mark.skipif(
        not pytest.importorskip("sktime", reason="sktime not installed"),
        reason="sktime backend tests"
    )
    def test_sktime_naive_model(self):
        """Test sktime with NaiveForecaster."""
        from universal_ts import UniversalForecaster

        df = create_synthetic_series(n_periods=50, freq='D')

        model = UniversalForecaster(backend='sktime', model='naive')
        model.fit(df)
        forecast = model.predict(horizon=10)

        assert len(forecast) == 10
        assert 'ds' in forecast.columns
        assert 'yhat' in forecast.columns

    @pytest.mark.skipif(
        not pytest.importorskip("sktime", reason="sktime not installed"),
        reason="sktime backend tests"
    )
    def test_sktime_auto_ets_model(self):
        """Test sktime with AutoETS."""
        from universal_ts import UniversalForecaster

        df = create_synthetic_series(n_periods=50, freq='D')

        model = UniversalForecaster(backend='sktime', model='auto_ets')
        model.fit(df)
        forecast = model.predict(horizon=10)

        assert len(forecast) == 10
        assert 'ds' in forecast.columns
        assert 'yhat' in forecast.columns

    @pytest.mark.skipif(
        not pytest.importorskip("sktime", reason="sktime not installed"),
        reason="sktime backend tests"
    )
    def test_sktime_auto_arima_model(self):
        """Test sktime with AutoARIMA."""
        from universal_ts import UniversalForecaster

        df = create_synthetic_series(n_periods=50, freq='D')

        model = UniversalForecaster(backend='sktime', model='auto_arima')
        model.fit(df)
        forecast = model.predict(horizon=10)

        assert len(forecast) == 10
        assert 'ds' in forecast.columns
        assert 'yhat' in forecast.columns

    @pytest.mark.skipif(
        not pytest.importorskip("sktime", reason="sktime not installed"),
        reason="sktime backend tests"
    )
    def test_sktime_panel_data(self):
        """Test sktime with panel data (supports via separate models)."""
        from universal_ts import UniversalForecaster

        df = create_panel_data(n_series=2, n_periods=30)

        model = UniversalForecaster(backend='sktime', model='naive')

        # sktime supports panel data, no warning expected
        model.fit(df, group_id='group_id')

        forecast = model.predict(horizon=10)
        assert len(forecast) == 20  # 10 for each series


class TestDartsIntegration:
    """Integration tests for Darts backend."""

    @pytest.mark.skipif(
        not pytest.importorskip("darts", reason="Darts not installed"),
        reason="Darts backend tests"
    )
    def test_darts_naive_seasonal_model(self):
        """Test Darts with NaiveSeasonal model."""
        from universal_ts import UniversalForecaster

        df = create_synthetic_series(n_periods=50, freq='D')

        model = UniversalForecaster(backend='darts', model='naive_seasonal')
        model.fit(df)
        forecast = model.predict(horizon=10)

        assert len(forecast) == 10
        assert 'ds' in forecast.columns
        assert 'yhat' in forecast.columns

    @pytest.mark.skipif(
        not pytest.importorskip("darts", reason="Darts not installed"),
        reason="Darts backend tests"
    )
    def test_darts_arima_model(self):
        """Test Darts with ARIMA model."""
        from universal_ts import UniversalForecaster

        df = create_synthetic_series(n_periods=50, freq='D')

        model = UniversalForecaster(backend='darts', model='arima')
        model.fit(df)
        forecast = model.predict(horizon=10)

        assert len(forecast) == 10
        assert 'ds' in forecast.columns
        assert 'yhat' in forecast.columns

    @pytest.mark.skipif(
        not pytest.importorskip("darts", reason="Darts not installed"),
        reason="Darts backend tests"
    )
    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch not installed for Darts DL models"),
        reason="Darts deep learning models require PyTorch"
    )
    def test_darts_tide_model_gpu(self):
        """Test Darts TiDE model with GPU support."""
        from universal_ts import UniversalForecaster

        df = create_synthetic_series(n_periods=100, freq='D')

        model = UniversalForecaster(
            backend='darts',
            model='tide',
            input_chunk_length=30,
            output_chunk_length=10,
            n_epochs=2,  # Small number for quick tests
            verbosity=0
        )

        model.fit(df)
        forecast = model.predict(horizon=10)

        assert len(forecast) == 10
        assert 'ds' in forecast.columns
        assert 'yhat' in forecast.columns

    @pytest.mark.skipif(
        not pytest.importorskip("darts", reason="Darts not installed"),
        reason="Darts backend tests"
    )
    def test_darts_panel_data(self):
        """Test Darts with panel data (uses separate models)."""
        from universal_ts import UniversalForecaster

        df = create_panel_data(n_series=2, n_periods=30)

        model = UniversalForecaster(backend='darts', model='naive_seasonal')

        # Darts supports panel data, no warning expected
        model.fit(df, group_id='group_id')

        forecast = model.predict(horizon=10)
        assert len(forecast) == 20  # 10 for each series


class TestCrossBackendCompatibility:
    """Tests for cross-backend compatibility and consistency."""

    def test_all_backends_interface_consistency(self):
        """Test that all available backends have consistent interface."""
        from universal_ts import UniversalForecaster

        # Create simple test data
        df = create_synthetic_series(n_periods=30, freq='D')

        # Test each available backend
        backends = []

        # Check which backends are available
        try:
            import prophet
            backends.append('prophet')
        except ImportError:
            pass

        try:
            import autogluon.timeseries
            backends.append('autogluon')
        except ImportError:
            pass

        try:
            import sktime
            backends.append('sktime')
        except ImportError:
            pass

        try:
            import darts
            backends.append('darts')
        except ImportError:
            pass

        # Test each backend
        for backend in backends:
            try:
                # Create model
                if backend == 'autogluon':
                    model = UniversalForecaster(
                        backend=backend,
                        prediction_length=5,
                        verbosity=0
                    )
                elif backend == 'sktime':
                    model = UniversalForecaster(backend=backend, model='naive')
                elif backend == 'darts':
                    model = UniversalForecaster(backend=backend, model='naive_seasonal')
                else:
                    model = UniversalForecaster(backend=backend)

                # Test fit
                model.fit(df)

                # Test predict
                forecast = model.predict(horizon=5)

                # Common assertions
                assert len(forecast) == 5
                assert 'ds' in forecast.columns
                assert 'yhat' in forecast.columns

                # Test model info
                info = model.get_model_info()
                assert info['backend'] == backend
                assert info['is_fitted'] is True

            except Exception as e:
                pytest.fail(f"Backend {backend} failed consistency test: {e}")


class TestErrorHandlingIntegration:
    """Integration tests for error handling."""

    def test_backend_not_installed_errors(self):
        """Test proper errors when backends are not installed."""
        from universal_ts import UniversalForecaster
        from universal_ts.exceptions import BackendNotInstalledError

        # Temporarily hide imports by modifying sys.modules
        original_modules = {}

        # Test each backend
        backends_to_test = ['prophet', 'autogluon', 'sktime', 'darts']

        for backend in backends_to_test:
            # Save original module state
            if backend == 'prophet':
                module_name = 'prophet'
            elif backend == 'autogluon':
                module_name = 'autogluon.timeseries'
            elif backend == 'sktime':
                module_name = 'sktime'
            elif backend == 'darts':
                module_name = 'darts'

            if module_name in sys.modules:
                original_modules[module_name] = sys.modules[module_name]
                del sys.modules[module_name]

            try:
                # Try to create model without backend installed
                with pytest.raises(BackendNotInstalledError) as exc_info:
                    UniversalForecaster(backend=backend)

                # Check error message contains installation instructions
                error_msg = str(exc_info.value)
                assert "pip install universal-ts" in error_msg
                assert backend in error_msg

            finally:
                # Restore original modules
                for module_name, module in original_modules.items():
                    sys.modules[module_name] = module
                original_modules.clear()


class TestPerformanceIntegration:
    """Integration tests for performance and scaling."""

    @pytest.mark.skipif(
        not all([pytest.importorskip("prophet", reason="Prophet not installed")]),
        reason="Performance tests require Prophet"
    )
    def test_single_series_performance(self):
        """Test performance with different data sizes."""
        from universal_ts import UniversalForecaster
        import time

        # Test different data sizes
        data_sizes = [50, 100, 200]
        backends = ['prophet']

        # Add other available backends
        try:
            import autogluon.timeseries
            backends.append('autogluon')
        except ImportError:
            pass

        performance_results = {}

        for backend in backends:
            performance_results[backend] = {}

            for size in data_sizes:
                df = create_synthetic_series(n_periods=size, freq='D')

                try:
                    start_time = time.time()

                    if backend == 'autogluon':
                        model = UniversalForecaster(
                            backend=backend,
                            prediction_length=10,
                            verbosity=0
                        )
                        model.fit(df, time_limit=30)
                    else:
                        model = UniversalForecaster(backend=backend)
                        model.fit(df)

                    forecast = model.predict(horizon=10)

                    end_time = time.time()
                    fit_time = end_time - start_time

                    performance_results[backend][size] = {
                        'fit_time': fit_time,
                        'forecast_length': len(forecast)
                    }

                    # Basic performance assertions
                    assert fit_time < 60  # Should complete within 60 seconds
                    assert len(forecast) == 10

                except Exception as e:
                    performance_results[backend][size] = {'error': str(e)}

        # Log results (for manual inspection)
        print("\nPerformance Results:")
        for backend, results in performance_results.items():
            print(f"\n{backend}:")
            for size, result in results.items():
                if 'error' in result:
                    print(f"  Size {size}: ERROR - {result['error']}")
                else:
                    print(f"  Size {size}: {result['fit_time']:.2f}s")


# Test fixtures for pytest
@pytest.fixture
def sample_single_series():
    """Fixture providing sample single series data."""
    return create_synthetic_series(n_periods=100, freq='D')


@pytest.fixture
def sample_panel_data():
    """Fixture providing sample panel data."""
    return create_panel_data(n_series=3, n_periods=50)


@pytest.fixture
def sample_covariates_data():
    """Fixture providing sample data with covariates."""
    return create_covariates_data(n_periods=100)
