"""Unit tests for backend implementations - Simplified working version."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestBaseBackendModel:
    """Tests for BaseBackendModel abstract class."""

    def test_abstract_methods(self):
        """Test that BaseBackendModel cannot be instantiated directly."""
        try:
            from universal_ts.base import BaseBackendModel
            BaseBackendModel()
            assert False, "Should not be able to instantiate abstract class"
        except TypeError:
            pass  # Expected


class TestBackendLoading:
    """Tests for backend loading functionality."""

    def test_load_prophet_backend_available(self):
        """Test loading Prophet backend when available."""
        try:
            from universal_ts.core import _load_backend
            with patch.dict('sys.modules', {
                'universal_ts.backends.prophet_backend': Mock(),
                'universal_ts.backends.prophet_backend.PROPHET_AVAILABLE': True,
                'prophet': Mock()
            }):
                backend_class = _load_backend("prophet")
                assert backend_class is not None
        except ImportError:
            pytest.skip("Prophet not available")

    def test_load_autogluon_backend_available(self):
        """Test loading AutoGluon backend when available."""
        try:
            from universal_ts.core import _load_backend
            with patch.dict('sys.modules', {
                'universal_ts.backends.autogluon_backend': Mock(),
                'universal_ts.backends.autogluon_backend.AUTOGLUON_AVAILABLE': True,
                'autogluon.timeseries': Mock()
            }):
                backend_class = _load_backend("autogluon")
                assert backend_class is not None
        except ImportError:
            pytest.skip("AutoGluon not available")

    def test_load_sktime_backend_available(self):
        """Test loading sktime backend when available."""
        try:
            from universal_ts.core import _load_backend
            with patch.dict('sys.modules', {
                'universal_ts.backends.sktime_backend': Mock(),
                'universal_ts.backends.sktime_backend.SKTIME_AVAILABLE': True,
                'sktime': Mock()
            }):
                backend_class = _load_backend("sktime")
                assert backend_class is not None
        except ImportError:
            pytest.skip("sktime not available")

    def test_load_darts_backend_available(self):
        """Test loading Darts backend when available."""
        try:
            from universal_ts.core import _load_backend
            with patch.dict('sys.modules', {
                'universal_ts.backends.darts_backend': Mock(),
                'universal_ts.backends.darts_backend.DARTS_AVAILABLE': True,
                'darts': Mock()
            }):
                backend_class = _load_backend("darts")
                assert backend_class is not None
        except ImportError:
            pytest.skip("Darts not available")

    def test_load_nonexistent_backend(self):
        """Test error when loading non-existent backend."""
        from universal_ts.core import _load_backend
        from universal_ts.exceptions import BackendNotFoundError

        with pytest.raises(BackendNotFoundError):
            _load_backend("nonexistent")

    def test_load_backend_import_error(self):
        """Test error when backend module cannot be imported."""
        from universal_ts.core import _load_backend
        from universal_ts.exceptions import BackendNotInstalledError

        # Mock importlib to raise ImportError for Prophet
        with patch('builtins.__import__', side_effect=ImportError("Module not found")):
            with pytest.raises(BackendNotInstalledError):
                _load_backend("prophet")


class TestBackendInterface:
    """Tests for backend interface consistency."""

    def test_backend_interface_consistency(self):
        """Test that all available backends implement required methods."""
        from universal_ts.core import _load_backend
        from universal_ts.exceptions import BackendNotInstalledError

        backend_names = ["prophet", "autogluon", "sktime", "darts"]
        required_methods = ['fit', 'predict', 'supports_panel_data', 'get_model_info']

        for backend_name in backend_names:
            try:
                backend_class = _load_backend(backend_name)
            except BackendNotInstalledError:
                continue

            for method in required_methods:
                assert hasattr(backend_class, method), f"{backend_name} missing {method}"
