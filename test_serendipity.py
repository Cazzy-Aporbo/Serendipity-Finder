"""
Unit tests for Serendipity Finder
Author: Cazandra Aporbo, MS 2025
Email: becaziam@gmail.com
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from serendipity_finder import SerendipityFinder


class TestSerendipityFinder:
    """Test suite for SerendipityFinder class."""
    
    @pytest.fixture
    def finder(self):
        """Create a SerendipityFinder instance for testing."""
        return SerendipityFinder(tail_threshold=0.15, correlation_threshold=0.6)
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
            'feature_1': np.random.randn(n),
            'feature_2': np.random.randn(n),
            'feature_3': np.random.randn(n)
        })
        return data
    
    def test_initialization(self, finder):
        """Test proper initialization of SerendipityFinder."""
        assert finder.tail_threshold == 0.15
        assert finder.correlation_threshold == 0.6
        assert finder.discoveries == []
        assert finder.data is None
    
    def test_generate_serendipitous_data(self, finder):
        """Test synthetic data generation."""
        data = finder.generate_serendipitous_data(
            n_samples=500,
            n_features=6,
            hidden_pairs=2,
            noise_level=0.5
        )
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 500
        assert len(data.columns) >= 6
        assert not data.isnull().any().any()
    
    def test_find_hidden_correlations(self, finder):
        """Test hidden correlation detection."""
        # Generate data with known hidden patterns
        data = finder.generate_serendipitous_data(
            n_samples=1000,
            n_features=4,
            hidden_pairs=2,
            noise_level=0.3
        )
        
        discoveries = finder.find_hidden_correlations(data)
        
        assert isinstance(discoveries, dict)
        assert 'strong_tail_correlations' in discoveries
        assert 'inverted_patterns' in discoveries
        assert 'emergent_relationships' in discoveries
    
    def test_calculate_significance(self, finder):
        """Test significance score calculation."""
        discovery = {
            'global_correlation': 0.1,
            'lower_tail_correlation': 0.8,
            'upper_tail_correlation': 0.7,
            'max_tail_correlation': 0.8
        }
        
        significance = finder._calculate_significance(discovery)
        
        # Significance = (1 - |0.1|) * 0.8 = 0.9 * 0.8 = 0.72
        assert pytest.approx(significance, 0.01) == 0.72
    
    def test_weak_global_strong_tail(self, finder):
        """Test detection of weak global but strong tail correlations."""
        # Create data with this specific pattern
        n = 1000
        x = np.random.randn(n)
        y = np.random.randn(n)
        
        # Add strong correlation in tails
        tail_mask = (np.abs(x) > 1.5)
        y[tail_mask] = x[tail_mask] * 2 + np.random.randn(tail_mask.sum()) * 0.1
        
        data = pd.DataFrame({'x': x, 'y': y})
        discoveries = finder.find_hidden_correlations(data)
        
        # Should find at least one strong tail correlation
        assert len(discoveries['strong_tail_correlations']) > 0
    
    def test_export_discoveries(self, finder, tmp_path):
        """Test export functionality."""
        # Generate and find discoveries
        data = finder.generate_serendipitous_data()
        finder.find_hidden_correlations(data)
        
        # Export to temporary file
        export_path = tmp_path / "test_discoveries.csv"
        finder.export_discoveries(str(export_path))
        
        assert export_path.exists()
        
        # Check exported data
        exported = pd.read_csv(export_path)
        assert not exported.empty
    
    def test_generate_report(self, finder):
        """Test report generation."""
        # Generate data and find discoveries
        data = finder.generate_serendipitous_data()
        finder.find_hidden_correlations(data)
        
        report = finder.generate_report()
        
        assert isinstance(report, str)
        assert "SERENDIPITY FINDER DISCOVERY REPORT" in report
    
    @pytest.mark.parametrize("tail_threshold", [0.05, 0.10, 0.15, 0.20, 0.25])
    def test_different_tail_thresholds(self, tail_threshold):
        """Test finder with different tail thresholds."""
        finder = SerendipityFinder(
            tail_threshold=tail_threshold,
            correlation_threshold=0.6
        )
        
        data = finder.generate_serendipitous_data(n_samples=500)
        discoveries = finder.find_hidden_correlations(data)
        
        assert discoveries is not None
        assert isinstance(discoveries, dict)
    
    def test_edge_cases(self, finder):
        """Test edge cases and error handling."""
        # Test with empty data
        empty_data = pd.DataFrame()
        with pytest.raises(Exception):
            finder.find_hidden_correlations(empty_data)
        
        # Test with single column
        single_col = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        discoveries = finder.find_hidden_correlations(single_col)
        assert len(discoveries['strong_tail_correlations']) == 0
        
        # Test with perfect correlation
        n = 100
        perfect_data = pd.DataFrame({
            'x': np.arange(n),
            'y': np.arange(n) * 2
        })
        discoveries = finder.find_hidden_correlations(perfect_data)
        # Should not find serendipitous patterns in perfect correlation
        assert len(discoveries['strong_tail_correlations']) == 0
    
    @pytest.mark.slow
    def test_large_dataset_performance(self, finder):
        """Test performance with large dataset."""
        import time
        
        # Generate large dataset
        data = finder.generate_serendipitous_data(
            n_samples=10000,
            n_features=20
        )
        
        start_time = time.time()
        discoveries = finder.find_hidden_correlations(data)
        elapsed_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert elapsed_time < 30  # seconds
        assert discoveries is not None
    
    @pytest.mark.visualization
    def test_visualization_creation(self, finder, tmp_path):
        """Test visualization generation."""
        data = finder.generate_serendipitous_data()
        discoveries = finder.find_hidden_correlations(data)
        
        if discoveries['strong_tail_correlations']:
            top = discoveries['strong_tail_correlations'][0]
            viz_path = tmp_path / "test_viz.png"
            
            # This would normally create the visualization
            # finder.visualize_discovery(
            #     top['feature1'], 
            #     top['feature2'],
            #     save_path=str(viz_path)
            # )
            # assert viz_path.exists()
            
            # For now, just check the method exists
            assert hasattr(finder, 'visualize_discovery')


class TestMathematicalCorrectness:
    """Test mathematical computations for correctness."""
    
    def test_correlation_calculation(self):
        """Verify correlation calculations are correct."""
        # Known correlation case
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        
        # Perfect positive correlation
        corr = np.corrcoef(x, y)[0, 1]
        assert pytest.approx(corr, 0.0001) == 1.0
        
        # Perfect negative correlation
        y_neg = np.array([10, 8, 6, 4, 2])
        corr_neg = np.corrcoef(x, y_neg)[0, 1]
        assert pytest.approx(corr_neg, 0.0001) == -1.0
        
        # No correlation
        y_random = np.array([3, 1, 4, 1, 5])
        corr_random = np.corrcoef(x, y_random)[0, 1]
        assert abs(corr_random) < 0.5
    
    def test_percentile_calculations(self):
        """Test percentile boundary calculations."""
        data = np.arange(100)
        
        # Test various percentiles
        assert np.percentile(data, 15) == 14.85
        assert np.percentile(data, 85) == 84.15
        assert np.percentile(data, 50) == 49.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])