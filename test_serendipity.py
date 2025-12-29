"""
Tests for Serendipity Finder
Minimal test suite designed to validate core functionality.

Run with: python -m pytest test_serendipity.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Ensure current directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from serendipity_finder import SerendipityFinder


class TestSerendipityFinderCore:
    """Core functionality tests."""

    def test_can_instantiate(self):
        """Test that SerendipityFinder can be instantiated."""
        finder = SerendipityFinder()
        assert finder is not None

    def test_default_parameters(self):
        """Test default parameters are set correctly."""
        finder = SerendipityFinder()
        assert hasattr(finder, 'tail_threshold')
        assert hasattr(finder, 'correlation_threshold')
        assert finder.tail_threshold == 0.15
        assert finder.correlation_threshold == 0.6

    def test_custom_parameters(self):
        """Test custom parameters are accepted."""
        finder = SerendipityFinder(tail_threshold=0.10, correlation_threshold=0.7)
        assert finder.tail_threshold == 0.10
        assert finder.correlation_threshold == 0.7


class TestDataGeneration:
    """Data generation tests."""

    def test_generate_data_returns_dataframe(self):
        """Test that generate_serendipitous_data returns a DataFrame."""
        finder = SerendipityFinder()
        data = finder.generate_serendipitous_data(n_samples=100, n_features=5)
        assert isinstance(data, pd.DataFrame)

    def test_generate_data_correct_rows(self):
        """Test that generated data has correct number of rows."""
        finder = SerendipityFinder()
        data = finder.generate_serendipitous_data(n_samples=100, n_features=5)
        assert len(data) == 100

    def test_generate_data_has_columns(self):
        """Test that generated data has at least requested features."""
        finder = SerendipityFinder()
        data = finder.generate_serendipitous_data(n_samples=100, n_features=5)
        assert data.shape[1] >= 5

    def test_generate_data_no_nulls(self):
        """Test that generated data contains no null values."""
        finder = SerendipityFinder()
        data = finder.generate_serendipitous_data(n_samples=100, n_features=5)
        assert not data.isnull().any().any()

    def test_generate_data_numeric(self):
        """Test that all generated data is numeric."""
        finder = SerendipityFinder()
        data = finder.generate_serendipitous_data(n_samples=100, n_features=5)
        for col in data.columns:
            assert np.issubdtype(data[col].dtype, np.number)


class TestCorrelationDiscovery:
    """Correlation discovery tests."""

    def test_find_correlations_returns_list(self):
        """Test that find_hidden_correlations returns a list."""
        finder = SerendipityFinder()
        finder.generate_serendipitous_data(n_samples=200, n_features=5)
        discoveries = finder.find_hidden_correlations()
        assert isinstance(discoveries, list)

    def test_find_correlations_with_custom_data(self):
        """Test correlation finding with custom DataFrame."""
        finder = SerendipityFinder()
        np.random.seed(42)
        data = pd.DataFrame({
            'a': np.random.randn(100),
            'b': np.random.randn(100),
            'c': np.random.randn(100)
        })
        discoveries = finder.find_hidden_correlations(data)
        assert isinstance(discoveries, list)


class TestReporting:
    """Report generation tests."""

    def test_generate_report_returns_string(self):
        """Test that generate_report returns a string."""
        finder = SerendipityFinder()
        finder.generate_serendipitous_data(n_samples=100, n_features=3)
        finder.find_hidden_correlations()
        report = finder.generate_report()
        assert isinstance(report, str)
        assert len(report) > 0


class TestEndToEnd:
    """End-to-end workflow tests."""

    def test_full_workflow(self):
        """Test complete workflow from initialization to report."""
        # Initialize
        finder = SerendipityFinder()
        assert finder is not None

        # Generate data
        data = finder.generate_serendipitous_data(n_samples=200, n_features=5)
        assert len(data) == 200

        # Find correlations
        discoveries = finder.find_hidden_correlations()
        assert isinstance(discoveries, list)

        # Generate report
        report = finder.generate_report()
        assert isinstance(report, str)

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same random seed."""
        np.random.seed(42)
        finder1 = SerendipityFinder()
        data1 = finder1.generate_serendipitous_data(n_samples=50, n_features=3)

        np.random.seed(42)
        finder2 = SerendipityFinder()
        data2 = finder2.generate_serendipitous_data(n_samples=50, n_features=3)

        pd.testing.assert_frame_equal(data1, data2)


# Standalone smoke test
def test_smoke():
    """Basic smoke test to verify module loads and runs."""
    finder = SerendipityFinder()
    data = finder.generate_serendipitous_data(n_samples=50, n_features=3)
    discoveries = finder.find_hidden_correlations()
    report = finder.generate_report()

    assert finder is not None
    assert data is not None
    assert discoveries is not None
    assert report is not None
    print("✓ Smoke test passed")


if __name__ == "__main__":
    # Run basic test when executed directly
    test_smoke()
    print("All direct tests passed!")
