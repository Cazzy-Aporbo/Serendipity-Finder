#!/usr/bin/env python3
"""
Serendipity Finder: Discovering Hidden Patterns in Data Tails


An approach to data analysis that looks beyond global correlations
to find insights in the extremes of distributions.

License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class SerendipityFinder:
    """
    A class for discovering hidden correlations in the tails of distributions.
    
    This approach mimics how real scientific breakthroughs happen - not in the 
    average, but in the outliers and edge cases.
    """
    
    def __init__(self, tail_threshold: float = 0.15, correlation_threshold: float = 0.6):
        """
        Initialize the Serendipity Finder.
        
        Args:
            tail_threshold: Percentile for defining tails (default: 15%)
            correlation_threshold: Minimum correlation to flag as interesting (default: 0.6)
        """
        self.tail_threshold = tail_threshold
        self.correlation_threshold = correlation_threshold
        self.discoveries = []
        self.data = None
        
    def generate_serendipitous_data(self, n_samples: int = 1000, 
                                   n_features: int = 10,
                                   hidden_pairs: int = 3,
                                   noise_level: float = 0.5) -> pd.DataFrame:
        """
        Generate synthetic data with hidden tail correlations.
        
        This creates data where global correlations are weak, but strong
        patterns emerge in the extremes - perfect for demonstrating the concept.
        
        Args:
            n_samples: Number of data points
            n_features: Number of features to generate
            hidden_pairs: Number of feature pairs with hidden tail correlations
            noise_level: Amount of noise to add (0-1)
            
        Returns:
            DataFrame with generated data
        """
        np.random.seed(42)  # For reproducibility
        
        # Start with independent random features
        data = np.random.randn(n_samples, n_features)
        
        # Add hidden correlations in the tails
        for i in range(min(hidden_pairs, n_features // 2)):
            feature1_idx = i * 2
            feature2_idx = i * 2 + 1
            
            # Identify tail indices
            tail_low = data[:, feature1_idx] < np.percentile(data[:, feature1_idx], self.tail_threshold * 100)
            tail_high = data[:, feature1_idx] > np.percentile(data[:, feature1_idx], (1 - self.tail_threshold) * 100)
            tail_indices = tail_low | tail_high
            
            # Create strong correlation in tails
            if np.sum(tail_indices) > 0:
                # In tails, feature2 strongly correlates with feature1
                data[tail_indices, feature2_idx] = (
                    data[tail_indices, feature1_idx] * (1.5 if i % 2 == 0 else -1.2) + 
                    np.random.randn(np.sum(tail_indices)) * noise_level
                )
            
            # Add noise to middle section to obscure global correlation
            middle_indices = ~tail_indices
            data[middle_indices, feature2_idx] += np.random.randn(np.sum(middle_indices)) * 2
        
        # Create DataFrame with meaningful names
        columns = [f'feature_{i:02d}' for i in range(n_features)]
        self.data = pd.DataFrame(data, columns=columns)
        
        # Add some derived features that might show interesting patterns
        if n_features >= 4:
            self.data['derived_ratio'] = self.data.iloc[:, 0] / (self.data.iloc[:, 1] + 5)
            self.data['derived_product'] = self.data.iloc[:, 2] * self.data.iloc[:, 3]
            
        return self.data
    
    def find_hidden_correlations(self, data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Find correlations that are weak globally but strong in the tails.
        
        This is where the magic happens - we look for patterns that would be
        missed by traditional correlation analysis.
        
        Args:
            data: DataFrame to analyze (uses self.data if None)
            
        Returns:
            Dictionary of discoveries
        """
        if data is not None:
            self.data = data
        elif self.data is None:
            raise ValueError("No data provided. Generate or load data first.")
        
        discoveries = {
            'strong_tail_correlations': [],
            'inverted_patterns': [],
            'emergent_relationships': []
        }
        
        columns = self.data.select_dtypes(include=[np.number]).columns
        
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                # Calculate global correlation
                global_corr = self.data[col1].corr(self.data[col2])
                
                # Calculate tail correlations
                lower_tail_mask = (
                    (self.data[col1] <= self.data[col1].quantile(self.tail_threshold)) |
                    (self.data[col2] <= self.data[col2].quantile(self.tail_threshold))
                )
                upper_tail_mask = (
                    (self.data[col1] >= self.data[col1].quantile(1 - self.tail_threshold)) |
                    (self.data[col2] >= self.data[col2].quantile(1 - self.tail_threshold))
                )
                
                if lower_tail_mask.sum() > 10:  # Need enough points
                    lower_tail_corr = self.data.loc[lower_tail_mask, col1].corr(
                        self.data.loc[lower_tail_mask, col2]
                    )
                else:
                    lower_tail_corr = np.nan
                    
                if upper_tail_mask.sum() > 10:
                    upper_tail_corr = self.data.loc[upper_tail_mask, col1].corr(
                        self.data.loc[upper_tail_mask, col2]
                    )
                else:
                    upper_tail_corr = np.nan
                
                # Check for interesting patterns
                discovery = {
                    'feature1': col1,
                    'feature2': col2,
                    'global_correlation': global_corr,
                    'lower_tail_correlation': lower_tail_corr,
                    'upper_tail_correlation': upper_tail_corr,
                    'max_tail_correlation': np.nanmax([abs(lower_tail_corr), abs(upper_tail_corr)])
                }
                
                # Flag interesting discoveries
                if abs(global_corr) < 0.3:  # Weak global correlation
                    if abs(lower_tail_corr) > self.correlation_threshold or \
                       abs(upper_tail_corr) > self.correlation_threshold:
                        discovery['discovery_type'] = 'hidden_tail_correlation'
                        discovery['significance'] = self._calculate_significance(discovery)
                        discoveries['strong_tail_correlations'].append(discovery)
                        
                # Check for inverted patterns (opposite correlations in tails)
                if not np.isnan(lower_tail_corr) and not np.isnan(upper_tail_corr):
                    if np.sign(lower_tail_corr) != np.sign(upper_tail_corr) and \
                       abs(lower_tail_corr) > 0.4 and abs(upper_tail_corr) > 0.4:
                        discovery['discovery_type'] = 'inverted_pattern'
                        discoveries['inverted_patterns'].append(discovery)
                        
        self.discoveries = discoveries
        return discoveries
    
    def _calculate_significance(self, discovery: Dict) -> float:
        """Calculate a significance score for a discovery."""
        global_weak = 1 - abs(discovery['global_correlation'])
        tail_strong = discovery['max_tail_correlation']
        return global_weak * tail_strong
    
    def visualize_discovery(self, feature1: str, feature2: str, 
                          save_path: Optional[str] = None) -> None:
        """
        Create a visualization highlighting the hidden correlation in tails.
        
        This creates a scatterplot that tells the story: boring globally,
        fascinating in the extremes.
        
        Args:
            feature1: First feature name
            feature2: Second feature name
            save_path: Optional path to save the figure
        """
        if self.data is None:
            raise ValueError("No data available for visualization")
            
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Define tail masks
        lower_tail = (
            (self.data[feature1] <= self.data[feature1].quantile(self.tail_threshold)) |
            (self.data[feature2] <= self.data[feature2].quantile(self.tail_threshold))
        )
        upper_tail = (
            (self.data[feature1] >= self.data[feature1].quantile(1 - self.tail_threshold)) |
            (self.data[feature2] >= self.data[feature2].quantile(1 - self.tail_threshold))
        )
        middle = ~(lower_tail | upper_tail)
        
        # Global view
        ax = axes[0]
        ax.scatter(self.data.loc[middle, feature1], 
                  self.data.loc[middle, feature2], 
                  alpha=0.5, s=20, c='gray', label='Middle')
        ax.scatter(self.data.loc[lower_tail, feature1], 
                  self.data.loc[lower_tail, feature2], 
                  alpha=0.7, s=30, c='blue', label='Lower tail')
        ax.scatter(self.data.loc[upper_tail, feature1], 
                  self.data.loc[upper_tail, feature2], 
                  alpha=0.7, s=30, c='red', label='Upper tail')
        
        global_corr = self.data[feature1].corr(self.data[feature2])
        ax.set_title(f'Global View\nCorrelation: {global_corr:.3f}')
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Lower tail focus
        ax = axes[1]
        if lower_tail.sum() > 0:
            ax.scatter(self.data.loc[lower_tail, feature1], 
                      self.data.loc[lower_tail, feature2], 
                      alpha=0.7, s=30, c='blue')
            lower_corr = self.data.loc[lower_tail, feature1].corr(
                self.data.loc[lower_tail, feature2]
            )
            
            # Add trend line
            z = np.polyfit(self.data.loc[lower_tail, feature1], 
                          self.data.loc[lower_tail, feature2], 1)
            p = np.poly1d(z)
            x_line = np.linspace(self.data.loc[lower_tail, feature1].min(),
                                self.data.loc[lower_tail, feature1].max(), 100)
            ax.plot(x_line, p(x_line), "b--", alpha=0.8, linewidth=2)
            
            ax.set_title(f'Lower Tail Pattern\nCorrelation: {lower_corr:.3f}')
        else:
            ax.set_title('Lower Tail Pattern\nInsufficient data')
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.grid(True, alpha=0.3)
        
        # Upper tail focus
        ax = axes[2]
        if upper_tail.sum() > 0:
            ax.scatter(self.data.loc[upper_tail, feature1], 
                      self.data.loc[upper_tail, feature2], 
                      alpha=0.7, s=30, c='red')
            upper_corr = self.data.loc[upper_tail, feature1].corr(
                self.data.loc[upper_tail, feature2]
            )
            
            # Add trend line
            z = np.polyfit(self.data.loc[upper_tail, feature1], 
                          self.data.loc[upper_tail, feature2], 1)
            p = np.poly1d(z)
            x_line = np.linspace(self.data.loc[upper_tail, feature1].min(),
                                self.data.loc[upper_tail, feature1].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
            
            ax.set_title(f'Upper Tail Pattern\nCorrelation: {upper_corr:.3f}')
        else:
            ax.set_title('Upper Tail Pattern\nInsufficient data')
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Serendipity Finder: Hidden Correlations in Distribution Tails', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self) -> str:
        """
        Generate a text report of discoveries.
        
        Returns:
            Formatted report string
        """
        if not self.discoveries:
            return "No discoveries found. Run find_hidden_correlations() first."
        
        report = []
        report.append("=" * 60)
        report.append("SERENDIPITY FINDER DISCOVERY REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Strong tail correlations
        if self.discoveries['strong_tail_correlations']:
            report.append("HIDDEN TAIL CORRELATIONS DISCOVERED:")
            report.append("-" * 40)
            for disc in sorted(self.discoveries['strong_tail_correlations'], 
                             key=lambda x: x.get('significance', 0), reverse=True)[:5]:
                report.append(f"\n  • {disc['feature1']} ↔ {disc['feature2']}")
                report.append(f"    Global correlation: {disc['global_correlation']:.3f} (weak)")
                report.append(f"    Lower tail correlation: {disc['lower_tail_correlation']:.3f}")
                report.append(f"    Upper tail correlation: {disc['upper_tail_correlation']:.3f}")
                report.append(f"    Significance score: {disc.get('significance', 0):.3f}")
        
        # Inverted patterns
        if self.discoveries['inverted_patterns']:
            report.append("\n")
            report.append("INVERTED PATTERNS DISCOVERED:")
            report.append("-" * 40)
            for disc in self.discoveries['inverted_patterns'][:3]:
                report.append(f"\n  • {disc['feature1']} ↔ {disc['feature2']}")
                report.append(f"    Lower tail: {disc['lower_tail_correlation']:.3f}")
                report.append(f"    Upper tail: {disc['upper_tail_correlation']:.3f}")
                report.append("    Pattern: Opposite correlations in tails!")
        
        report.append("\n" + "=" * 60)
        report.append("INTERPRETATION:")
        report.append("These patterns would be completely missed by traditional")
        report.append("correlation analysis. They represent potential breakthrough")
        report.append("insights hiding in the extremes of your data.")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def export_discoveries(self, filepath: str = "discoveries.csv") -> None:
        """Export discoveries to CSV for further analysis."""
        if self.discoveries and self.discoveries['strong_tail_correlations']:
            df = pd.DataFrame(self.discoveries['strong_tail_correlations'])
            df.to_csv(filepath, index=False)
            print(f"Discoveries exported to {filepath}")


def demo_serendipity_finder():
    """
    Demonstration of the Serendipity Finder in action.
    
    This shows how the tool can uncover insights that traditional
    data analysis would miss entirely.
    """
    print("Initializing Serendipity Finder...")
    print("-" * 50)
    
    # Create finder instance
    finder = SerendipityFinder(tail_threshold=0.15, correlation_threshold=0.6)
    
    # Generate data with hidden patterns
    print("Generating synthetic data with hidden tail correlations...")
    data = finder.generate_serendipitous_data(
        n_samples=1000,
        n_features=10,
        hidden_pairs=3,
        noise_level=0.5
    )
    print(f"   Created dataset with {len(data)} samples and {len(data.columns)} features")
    
    # Find hidden correlations
    print("\n Searching for serendipitous discoveries...")
    discoveries = finder.find_hidden_correlations()
    
    # Print summary
    print(f"\n DISCOVERIES FOUND:")
    print(f"   • Hidden tail correlations: {len(discoveries['strong_tail_correlations'])}")
    print(f"   • Inverted patterns: {len(discoveries['inverted_patterns'])}")
    
    # Show detailed report
    print("\n" + finder.generate_report())
    
    # Visualize top discovery
    if discoveries['strong_tail_correlations']:
        top_discovery = discoveries['strong_tail_correlations'][0]
        print(f"\n Visualizing top discovery: {top_discovery['feature1']} vs {top_discovery['feature2']}")
        finder.visualize_discovery(
            top_discovery['feature1'], 
            top_discovery['feature2'],
            save_path="serendipity_discovery.png"
        )
    
    # Export discoveries
    finder.export_discoveries("serendipity_discoveries.csv")
    
    return finder


if __name__ == "__main__":
    # Run the demonstration
    finder = demo_serendipity_finder()
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT:")
    print("This approach mimics how real scientific breakthroughs happen.")
    print("Penicillin, market crashes, medical side-effects - all found")
    print("in the outliers, not the average. That's the power of")
    print("looking where others don't think to look.")
    print("=" * 60)
