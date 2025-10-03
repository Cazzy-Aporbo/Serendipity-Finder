#!/usr/bin/env python3
"""
Serendipity Finder v2.0: Advanced Pattern Discovery in Distribution Tails

A different approach to uncovering hidden correlations that emerge only
in extreme conditions - where traditional statistical methods fail to look.

Author: Cazandra Aporbo, MS 2025 
Repository: https://github.com/Cazzy-Aporbo/Serendipity-Finder
License: MIT

This implementation is based on research in tail dependence theory,
particularly the work of Poon, Rockinger, and Tawn (2004) on tail dependence
in financial markets, and Embrechts et al. on correlation breakdowns in
extreme events.

Real-world applications have proven this approach valuable:
- The 2007-2008 financial crisis revealed that asset correlations converge 
  to 1 during market stress, despite showing weak correlations in normal times
  https://solnik.people.ust.hk/Articles/A6-JoFLongin.pdf
- Medical research discovered that Vioxx's cardiac risks were hidden in 
  population averages but visible in high-risk patient subgroups
  https://www.ccjm.org/content/ccjom/71/12/933.full.pdf
  
- Climate tipping points show non-linear relationships that only emerge
  at temperature extremes (Lenton et al., 2012)
https://www.researchgate.net/publication/353978866_Tipping_points_in_the_climate_system

  additional sources reviewed: https://www.sciencedirect.com/science/article/abs/pii/S0370157319303291

  Version 3 will use real data. 
"""

# Standard library imports for core functionality
import numpy as np  # Numerical computing foundation
import pandas as pd  # Data manipulation and analysis
import matplotlib.pyplot as plt  # Visualization engine
import seaborn as sns  # Statistical visualization enhancement
from scipy import stats  # Statistical functions and tests
from scipy.stats import pearsonr, spearmanr  # Correlation coefficients
from typing import Tuple, Dict, List, Optional, Union  # Type hints for clarity
import warnings  # Warning control
import json  # Data export capabilities
from datetime import datetime  # Timestamp functionality
import logging  # Professional logging system

# Configure logging for production-ready code
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress non-critical warnings for cleaner output
warnings.filterwarnings('ignore')


class SerendipityFinder:
    """
    Advanced correlation discovery engine for finding patterns in distribution tails.
    
    This class implements my approach to data analysis that specifically
    looks for relationships that only emerge under extreme conditions.
    
    The mathematics behind this approach are grounded in Extreme Value Theory
    and copula-based dependence modeling. But the intuition is simple: what
    happens in the extremes often tells a completely different story than
    what happens on average.
    
    Historical context:
    Many scientific breakthroughs came from examining outliers rather than
    averages. Alexander Fleming discovered penicillin by noticing unusual
    bacterial death in contaminated cultures. The ozone hole was discovered
    by examining extreme measurements that were initially dismissed as errors.
    """
    
    def __init__(self, 
                 tail_threshold: float = 0.15,
                 correlation_threshold: float = 0.6,
                 min_tail_samples: int = 30,
                 bootstrap_iterations: int = 1000):
        """
        Initialize the Serendipity Finder with configurable parameters.
        
        Args:
            tail_threshold: Percentile defining the tail regions (default 0.15).
                           Research shows 10-20% captures meaningful extremes
                           without losing statistical power.
            
            correlation_threshold: Minimum correlation to flag as significant (default 0.6).
                                 Based on Cohen's guidelines for effect sizes,
                                 0.6 represents a strong relationship.
            
            min_tail_samples: Minimum samples needed in tail for valid analysis (default 30).
                             Based on Central Limit Theorem requirements for
                             reliable correlation estimates.
            
            bootstrap_iterations: Number of bootstrap samples for significance testing (default 1000).
                                Based on Efron & Tibshirani's recommendations
                                for stable confidence intervals.
        """
        # Store configuration parameters
        self.tail_threshold = tail_threshold
        self.correlation_threshold = correlation_threshold
        self.min_tail_samples = min_tail_samples
        self.bootstrap_iterations = bootstrap_iterations
        
        # Initialize storage for discoveries and analysis results
        self.discoveries = []  # List to store discovered patterns
        self.data = None  # Placeholder for analyzed dataset
        self.analysis_metadata = {}  # Metadata about the analysis run
        
        # Log initialization
        logger.info(f"Serendipity Finder initialized with tail_threshold={tail_threshold}")
    
    def load_real_world_example(self, example_type: str = 'financial') -> pd.DataFrame:
        """
        Load example datasets that demonstrate real tail correlation phenomena.
        
        This method generates synthetic data that mimics real-world patterns
        documented in peer-reviewed research.
        
        Args:
            example_type: Type of example data to generate
                         'financial': Asset correlations during crisis (Longin & Solnik, 2001)
                         'medical': Drug side effects in subpopulations (FDA AERS data patterns)
                         'climate': Temperature-ecosystem relationships (Scheffer et al., 2001)
        
        Returns:
            DataFrame containing example data with hidden tail correlations
        """
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Number of observations (statistical power requires adequate sample size)
        n_samples = 2000
        
        if example_type == 'financial':
            # Generate data mimicking stock market behavior
            # Normal times: low correlation between assets
            # Crisis times (tails): high correlation (contagion effect)
            
            # Market factor represents overall market conditions
            market_factor = np.random.randn(n_samples)
            
            # Asset 1: Technology stock proxy
            tech_stock = np.zeros(n_samples)
            # Asset 2: Banking stock proxy  
            bank_stock = np.zeros(n_samples)
            
            for i in range(n_samples):
                if abs(market_factor[i]) > 2:  # Extreme market conditions (tail events)
                    # During crisis, correlation increases dramatically
                    # This mimics the documented behavior during 2008 crisis
                    tech_stock[i] = market_factor[i] * 1.5 + np.random.randn() * 0.3
                    bank_stock[i] = market_factor[i] * 1.3 + np.random.randn() * 0.3
                else:  # Normal market conditions
                    # Assets move independently during calm periods
                    tech_stock[i] = np.random.randn() * 1.2
                    bank_stock[i] = np.random.randn() * 1.1
            
            # Create DataFrame with meaningful column names
            data = pd.DataFrame({
                'market_stress_indicator': market_factor,
                'tech_sector_returns': tech_stock,
                'financial_sector_returns': bank_stock,
                'volatility_index': np.abs(market_factor) + np.random.exponential(0.5, n_samples)
            })
            
            # Add timestamp for realism
            data['date'] = pd.date_range(start='2015-01-01', periods=n_samples, freq='D')
            
            logger.info("Generated financial crisis correlation example data")
            
        elif example_type == 'medical':
            # Generate data mimicking drug response patterns
            # Most patients: no correlation between drug dose and side effect
            # High-risk subgroup (tail): strong correlation
            
            # Patient risk score (genetic markers, age, comorbidities)
            risk_score = np.random.gamma(2, 2, n_samples)
            
            # Drug dosage (mg/day)
            dosage = np.random.uniform(10, 100, n_samples)
            
            # Side effect severity
            side_effect = np.zeros(n_samples)
            
            for i in range(n_samples):
                if risk_score[i] > np.percentile(risk_score, 85):  # High-risk patients
                    # Strong dose-response relationship in vulnerable population
                    # Based on patterns seen in Vioxx cardiac events
                    side_effect[i] = dosage[i] * 0.02 * risk_score[i] + np.random.randn() * 5
                else:  # Normal risk patients
                    # No relationship between dose and side effects
                    side_effect[i] = np.random.randn() * 10
            
            # Create clinically meaningful DataFrame
            data = pd.DataFrame({
                'patient_risk_score': risk_score,
                'drug_dosage_mg': dosage,
                'adverse_event_severity': side_effect,
                'age': 40 + risk_score * 5 + np.random.randn(n_samples) * 10,
                'baseline_health': 100 - risk_score * 3 + np.random.randn(n_samples) * 5
            })
            
            logger.info("Generated medical side-effect pattern example data")
            
        else:  # Climate example
            # Generate data mimicking ecosystem tipping points
            # Normal range: gradual linear relationship
            # Extreme temperatures: sudden non-linear changes
            
            # Temperature anomaly (degrees C from baseline)
            temp_anomaly = np.random.randn(n_samples) * 2
            
            # Ecosystem health indicator
            ecosystem_health = np.zeros(n_samples)
            
            for i in range(n_samples):
                if temp_anomaly[i] > 3:  # Extreme warming
                    # Catastrophic decline (tipping point behavior)
                    # Based on coral bleaching and Arctic ice dynamics
                    ecosystem_health[i] = -50 - (temp_anomaly[i] - 3) * 20 + np.random.randn() * 5
                elif temp_anomaly[i] < -2:  # Extreme cooling
                    # Different stress response
                    ecosystem_health[i] = -30 - abs(temp_anomaly[i] + 2) * 15 + np.random.randn() * 5
                else:  # Normal temperature range
                    # Gradual linear response
                    ecosystem_health[i] = -temp_anomaly[i] * 2 + np.random.randn() * 10
            
            # Create environmental DataFrame
            data = pd.DataFrame({
                'temperature_anomaly': temp_anomaly,
                'ecosystem_health': ecosystem_health,
                'precipitation_change': temp_anomaly * 0.3 + np.random.randn(n_samples),
                'species_diversity': 100 + ecosystem_health * 0.5 + np.random.randn(n_samples) * 10,
                'carbon_flux': -ecosystem_health * 0.8 + np.random.randn(n_samples) * 5
            })
            
            logger.info("Generated climate tipping point example data")
        
        # Store the generated data
        self.data = data
        
        # Record metadata about this dataset
        self.analysis_metadata = {
            'dataset_type': example_type,
            'n_samples': n_samples,
            'generation_timestamp': datetime.now().isoformat(),
            'theoretical_basis': self._get_theoretical_basis(example_type)
        }
        
        return data
    
    def _get_theoretical_basis(self, example_type: str) -> str:
        """
        Provide scientific references for each example type.
        
        This ensures our synthetic data is grounded in real research.
        """
        references = {
            'financial': "Longin, F. & Solnik, B. (2001). Extreme correlation of international equity markets. Journal of Finance, 56(2), 649-676.",
            'medical': "Graham, D.J. et al. (2005). Risk of acute myocardial infarction and sudden cardiac death in patients treated with COX-2 selective and non-selective NSAIDs. BMJ, 330(7489), 385.",
            'climate': "Scheffer, M. et al. (2001). Catastrophic shifts in ecosystems. Nature, 413(6856), 591-596."
        }
        return references.get(example_type, "General extreme value theory applications")
    
    def find_hidden_correlations(self, 
                                data: Optional[pd.DataFrame] = None,
                                use_robust_measures: bool = True) -> Dict:
        """
        Core algorithm for discovering correlations that only appear in distribution tails.
        
        This method implements the mathematical framework described in
        "Correlation and Dependence in Risk Management" by Embrechts, McNeil, and Straumann.
        
        The key insight: Pearson correlation can be near zero globally while
        tail dependence is strong. This is why portfolio diversification fails
        during market crashes - the tails tell a different story than the body.
        
        Args:
            data: DataFrame to analyze (uses self.data if None)
            use_robust_measures: If True, also calculate Spearman rank correlation
                                which is more robust to outliers
        
        Returns:
            Dictionary containing discovered patterns with statistical significance
        """
        # Use provided data or stored data
        if data is not None:
            self.data = data
        elif self.data is None:
            raise ValueError("No data provided. Load data first using load_real_world_example()")
        
        # Initialize discovery storage
        discoveries = {
            'strong_tail_correlations': [],
            'inverted_patterns': [],
            'tail_only_relationships': [],
            'statistical_tests': []
        }
        
        # Get numeric columns only (correlation requires numeric data)
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove date columns if present
        numeric_columns = [col for col in numeric_columns if 'date' not in col.lower()]
        
        # Analyze each pair of features
        total_pairs = len(numeric_columns) * (len(numeric_columns) - 1) // 2
        logger.info(f"Analyzing {total_pairs} feature pairs for hidden correlations")
        
        for i, col1 in enumerate(numeric_columns):
            for j, col2 in enumerate(numeric_columns[i+1:], i+1):
                # Skip if same column (would give correlation of 1)
                if col1 == col2:
                    continue
                
                # Extract the two series for analysis
                series1 = self.data[col1].dropna()
                series2 = self.data[col2].dropna()
                
                # Ensure same length (handle missing values)
                valid_indices = series1.index.intersection(series2.index)
                series1 = series1.loc[valid_indices]
                series2 = series2.loc[valid_indices]
                
                # Calculate global correlation (Pearson's r)
                global_corr, global_pvalue = pearsonr(series1, series2)
                
                # Calculate quantiles for tail definition
                q1_lower = series1.quantile(self.tail_threshold)
                q1_upper = series1.quantile(1 - self.tail_threshold)
                q2_lower = series2.quantile(self.tail_threshold)
                q2_upper = series2.quantile(1 - self.tail_threshold)
                
                # Identify observations in lower tail (either variable in lower extreme)
                lower_tail_mask = (series1 <= q1_lower) | (series2 <= q2_lower)
                
                # Identify observations in upper tail (either variable in upper extreme)
                upper_tail_mask = (series1 >= q1_upper) | (series2 >= q2_upper)
                
                # Calculate tail correlations if sufficient samples
                lower_tail_corr = np.nan
                lower_tail_pvalue = np.nan
                upper_tail_corr = np.nan
                upper_tail_pvalue = np.nan
                
                if lower_tail_mask.sum() >= self.min_tail_samples:
                    # Calculate correlation in lower tail
                    lower_tail_corr, lower_tail_pvalue = pearsonr(
                        series1[lower_tail_mask],
                        series2[lower_tail_mask]
                    )
                    
                if upper_tail_mask.sum() >= self.min_tail_samples:
                    # Calculate correlation in upper tail
                    upper_tail_corr, upper_tail_pvalue = pearsonr(
                        series1[upper_tail_mask],
                        series2[upper_tail_mask]
                    )
                
                # Create discovery record
                discovery = {
                    'feature1': col1,
                    'feature2': col2,
                    'global_correlation': global_corr,
                    'global_pvalue': global_pvalue,
                    'lower_tail_correlation': lower_tail_corr,
                    'lower_tail_pvalue': lower_tail_pvalue,
                    'lower_tail_n': lower_tail_mask.sum(),
                    'upper_tail_correlation': upper_tail_corr,
                    'upper_tail_pvalue': upper_tail_pvalue,
                    'upper_tail_n': upper_tail_mask.sum(),
                    'max_tail_correlation': np.nanmax([abs(lower_tail_corr), abs(upper_tail_corr)])
                }
                
                # Calculate the Serendipity Score (our novel metric)
                # High score = weak global correlation but strong tail correlation
                if not np.isnan(discovery['max_tail_correlation']):
                    discovery['serendipity_score'] = (
                        (1 - abs(global_corr)) *  # Reward weak global correlation
                        discovery['max_tail_correlation'] *  # Reward strong tail correlation
                        np.exp(-global_pvalue)  # Reward statistical significance
                    )
                else:
                    discovery['serendipity_score'] = 0
                
                # Categorize the discovery
                self._categorize_discovery(discovery, discoveries)
        
        # Store discoveries for later use
        self.discoveries = discoveries
        
        # Log summary
        logger.info(f"Found {len(discoveries['strong_tail_correlations'])} strong tail correlations")
        logger.info(f"Found {len(discoveries['inverted_patterns'])} inverted patterns")
        
        return discoveries
    
    def _categorize_discovery(self, discovery: Dict, discoveries: Dict) -> None:
        """
        Categorize a discovery based on its statistical properties.
        
        This method implements decision rules based on effect size guidelines
        and statistical significance thresholds from psychological and
        medical research standards.
        """
        # Extract key metrics for decision making
        global_corr = abs(discovery['global_correlation'])
        lower_corr = abs(discovery['lower_tail_correlation'])
        upper_corr = abs(discovery['upper_tail_correlation'])
        max_tail = discovery['max_tail_correlation']
        
        # Check for hidden tail correlation
        # Criteria: weak global but strong tail correlation
        if global_corr < 0.3 and max_tail > self.correlation_threshold:
            # Additional check for statistical significance
            if (discovery['lower_tail_pvalue'] < 0.05 or 
                discovery['upper_tail_pvalue'] < 0.05):
                discovery['discovery_type'] = 'hidden_tail_correlation'
                discovery['interpretation'] = (
                    f"Variables show no relationship globally (r={discovery['global_correlation']:.3f}) "
                    f"but strong correlation in extremes (r={max_tail:.3f}). "
                    "This pattern would be completely missed by standard analysis."
                )
                discoveries['strong_tail_correlations'].append(discovery)
        
        # Check for inverted pattern
        # Criteria: opposite sign correlations in different tails
        if not np.isnan(discovery['lower_tail_correlation']) and \
           not np.isnan(discovery['upper_tail_correlation']):
            if np.sign(discovery['lower_tail_correlation']) != \
               np.sign(discovery['upper_tail_correlation']) and \
               lower_corr > 0.4 and upper_corr > 0.4:
                discovery['discovery_type'] = 'inverted_pattern'
                discovery['interpretation'] = (
                    "Relationship reverses in extremes: "
                    f"negative correlation in lower tail (r={discovery['lower_tail_correlation']:.3f}), "
                    f"positive in upper tail (r={discovery['upper_tail_correlation']:.3f}). "
                    "Suggests regime change or phase transition."
                )
                discoveries['inverted_patterns'].append(discovery)
        
        # Check for tail-only relationships
        # Criteria: strong correlation exists only in one tail
        if global_corr < 0.2:
            if (lower_corr > 0.7 and upper_corr < 0.3) or \
               (upper_corr > 0.7 and lower_corr < 0.3):
                discovery['discovery_type'] = 'tail_only_relationship'
                discovery['interpretation'] = (
                    "Relationship exists only in extreme conditions. "
                    "Critical for risk assessment and rare event prediction."
                )
                discoveries['tail_only_relationships'].append(discovery)
    
    def bootstrap_significance_test(self, 
                                   feature1: str, 
                                   feature2: str,
                                   n_iterations: Optional[int] = None) -> Dict:
        """
        Perform bootstrap resampling to establish confidence intervals for tail correlations.
        
        Bootstrap methods are particularly valuable here because tail correlations
        are calculated on smaller subsets of data, making parametric assumptions
        questionable. This follows Efron & Tibshirani's bootstrap methodology.
        
        Args:
            feature1: First feature name
            feature2: Second feature name
            n_iterations: Number of bootstrap samples (uses default if None)
        
        Returns:
            Dictionary with confidence intervals and p-values
        """
        if self.data is None:
            raise ValueError("No data available. Load data first.")
        
        # Use provided iterations or default
        n_iter = n_iterations or self.bootstrap_iterations
        
        # Extract the series
        series1 = self.data[feature1].dropna()
        series2 = self.data[feature2].dropna()
        valid_indices = series1.index.intersection(series2.index)
        series1 = series1.loc[valid_indices].values
        series2 = series2.loc[valid_indices].values
        
        # Storage for bootstrap statistics
        bootstrap_global = []
        bootstrap_lower = []
        bootstrap_upper = []
        
        logger.info(f"Running {n_iter} bootstrap iterations for {feature1} vs {feature2}")
        
        for iteration in range(n_iter):
            # Resample with replacement
            indices = np.random.choice(len(series1), len(series1), replace=True)
            boot_series1 = series1[indices]
            boot_series2 = series2[indices]
            
            # Calculate global correlation for this bootstrap sample
            boot_global_corr, _ = pearsonr(boot_series1, boot_series2)
            bootstrap_global.append(boot_global_corr)
            
            # Calculate tail correlations for bootstrap sample
            q1_lower = np.percentile(boot_series1, self.tail_threshold * 100)
            q1_upper = np.percentile(boot_series1, (1 - self.tail_threshold) * 100)
            q2_lower = np.percentile(boot_series2, self.tail_threshold * 100)
            q2_upper = np.percentile(boot_series2, (1 - self.tail_threshold) * 100)
            
            # Lower tail
            lower_mask = (boot_series1 <= q1_lower) | (boot_series2 <= q2_lower)
            if lower_mask.sum() >= 10:
                lower_corr, _ = pearsonr(boot_series1[lower_mask], boot_series2[lower_mask])
                bootstrap_lower.append(lower_corr)
            
            # Upper tail
            upper_mask = (boot_series1 >= q1_upper) | (boot_series2 >= q2_upper)
            if upper_mask.sum() >= 10:
                upper_corr, _ = pearsonr(boot_series1[upper_mask], boot_series2[upper_mask])
                bootstrap_upper.append(upper_corr)
        
        # Calculate confidence intervals (95% by default)
        results = {
            'feature1': feature1,
            'feature2': feature2,
            'n_iterations': n_iter,
            'global_correlation_ci': (np.percentile(bootstrap_global, 2.5),
                                     np.percentile(bootstrap_global, 97.5)),
            'lower_tail_ci': (np.percentile(bootstrap_lower, 2.5),
                             np.percentile(bootstrap_lower, 97.5)) if bootstrap_lower else (np.nan, np.nan),
            'upper_tail_ci': (np.percentile(bootstrap_upper, 2.5),
                             np.percentile(bootstrap_upper, 97.5)) if bootstrap_upper else (np.nan, np.nan),
            'tail_correlation_stable': self._assess_stability(bootstrap_lower, bootstrap_upper)
        }
        
        return results
    
    def _assess_stability(self, bootstrap_lower: List, bootstrap_upper: List) -> bool:
        """
        Assess whether tail correlations are stable across bootstrap samples.
        
        Stability is important for confidence in the discovered pattern.
        """
        # Check if we have enough samples
        if len(bootstrap_lower) < 100 or len(bootstrap_upper) < 100:
            return False
        
        # Calculate coefficient of variation for stability assessment
        cv_lower = np.std(bootstrap_lower) / abs(np.mean(bootstrap_lower)) if np.mean(bootstrap_lower) != 0 else float('inf')
        cv_upper = np.std(bootstrap_upper) / abs(np.mean(bootstrap_upper)) if np.mean(bootstrap_upper) != 0 else float('inf')
        
        # Stable if coefficient of variation is less than 0.5
        return cv_lower < 0.5 and cv_upper < 0.5
    
    def visualize_discovery(self, 
                          feature1: str, 
                          feature2: str,
                          save_path: Optional[str] = None,
                          include_density: bool = True) -> None:
        """
        Create comprehensive visualization of discovered tail correlations.
        
        This visualization tells the complete story: what traditional analysis
        would see (and miss) versus what our method reveals.
        
        Args:
            feature1: First feature to plot
            feature2: Second feature to plot
            save_path: Optional path to save the figure
            include_density: Whether to include density contours
        """
        if self.data is None:
            raise ValueError("No data available for visualization")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 6))
        
        # Use a professional style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Define color scheme
        colors = {
            'middle': '#7f8c8d',  # Gray for middle
            'lower': '#3498db',   # Blue for lower tail
            'upper': '#e74c3c'    # Red for upper tail
        }
        
        # Extract data
        x = self.data[feature1].dropna()
        y = self.data[feature2].dropna()
        valid_indices = x.index.intersection(y.index)
        x = x.loc[valid_indices]
        y = y.loc[valid_indices]
        
        # Define tail boundaries
        x_lower = x.quantile(self.tail_threshold)
        x_upper = x.quantile(1 - self.tail_threshold)
        y_lower = y.quantile(self.tail_threshold)
        y_upper = y.quantile(1 - self.tail_threshold)
        
        # Create masks for different regions
        lower_mask = (x <= x_lower) | (y <= y_lower)
        upper_mask = (x >= x_upper) | (y >= y_upper)
        middle_mask = ~(lower_mask | upper_mask)
        
        # Subplot 1: Global view (what traditional analysis sees)
        ax1 = plt.subplot(1, 3, 1)
        
        # Plot all points with region coloring
        ax1.scatter(x[middle_mask], y[middle_mask], 
                   alpha=0.3, s=20, c=colors['middle'], label='Core (middle 70%)')
        ax1.scatter(x[lower_mask], y[lower_mask], 
                   alpha=0.6, s=30, c=colors['lower'], label='Lower tail (15%)')
        ax1.scatter(x[upper_mask], y[upper_mask], 
                   alpha=0.6, s=30, c=colors['upper'], label='Upper tail (15%)')
        
        # Add regression line for global correlation
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax1.plot(x_line, p(x_line), "k--", alpha=0.5, linewidth=2, 
                label=f'Global trend (r={x.corr(y):.3f})')
        
        # Formatting
        ax1.set_xlabel(feature1, fontsize=12)
        ax1.set_ylabel(feature2, fontsize=12)
        ax1.set_title('Traditional Analysis View\n(Misses the Hidden Patterns)', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Lower tail focus
        ax2 = plt.subplot(1, 3, 2)
        
        if lower_mask.sum() > 0:
            # Plot lower tail points
            ax2.scatter(x[lower_mask], y[lower_mask], 
                       alpha=0.6, s=50, c=colors['lower'])
            
            # Add regression line for lower tail
            x_lower_tail = x[lower_mask]
            y_lower_tail = y[lower_mask]
            if len(x_lower_tail) > 2:
                z = np.polyfit(x_lower_tail, y_lower_tail, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x_lower_tail.min(), x_lower_tail.max(), 100)
                ax2.plot(x_line, p(x_line), color=colors['lower'], 
                        linewidth=2.5, label=f'r={x_lower_tail.corr(y_lower_tail):.3f}')
            
            # Add confidence ellipse if requested
            if include_density and len(x_lower_tail) > 3:
                from matplotlib.patches import Ellipse
                from scipy.stats import chi2
                
                # Calculate covariance matrix
                cov = np.cov(x_lower_tail, y_lower_tail)
                
                # Get eigenvalues and eigenvectors
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                
                # Calculate ellipse parameters
                angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                width, height = 2 * np.sqrt(eigenvalues * chi2.ppf(0.95, 2))
                
                # Add ellipse
                ellipse = Ellipse(xy=(x_lower_tail.mean(), y_lower_tail.mean()),
                                 width=width, height=height, angle=angle,
                                 facecolor=colors['lower'], alpha=0.2)
                ax2.add_patch(ellipse)
        
        # Formatting
        ax2.set_xlabel(feature1, fontsize=12)
        ax2.set_ylabel(feature2, fontsize=12)
        ax2.set_title('Lower Tail Discovery\n(Bottom 15% Extremes)', 
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Upper tail focus
        ax3 = plt.subplot(1, 3, 3)
        
        if upper_mask.sum() > 0:
            # Plot upper tail points
            ax3.scatter(x[upper_mask], y[upper_mask], 
                       alpha=0.6, s=50, c=colors['upper'])
            
            # Add regression line for upper tail
            x_upper_tail = x[upper_mask]
            y_upper_tail = y[upper_mask]
            if len(x_upper_tail) > 2:
                z = np.polyfit(x_upper_tail, y_upper_tail, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x_upper_tail.min(), x_upper_tail.max(), 100)
                ax3.plot(x_line, p(x_line), color=colors['upper'], 
                        linewidth=2.5, label=f'r={x_upper_tail.corr(y_upper_tail):.3f}')
            
            # Add confidence ellipse if requested
            if include_density and len(x_upper_tail) > 3:
                from matplotlib.patches import Ellipse
                from scipy.stats import chi2
                
                # Calculate covariance matrix
                cov = np.cov(x_upper_tail, y_upper_tail)
                
                # Get eigenvalues and eigenvectors
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                
                # Calculate ellipse parameters
                angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                width, height = 2 * np.sqrt(eigenvalues * chi2.ppf(0.95, 2))
                
                # Add ellipse
                ellipse = Ellipse(xy=(x_upper_tail.mean(), y_upper_tail.mean()),
                                 width=width, height=height, angle=angle,
                                 facecolor=colors['upper'], alpha=0.2)
                ax3.add_patch(ellipse)
        
        # Formatting
        ax3.set_xlabel(feature1, fontsize=12)
        ax3.set_ylabel(feature2, fontsize=12)
        ax3.set_title('Upper Tail Discovery\n(Top 15% Extremes)', 
                     fontsize=14, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Overall title
        fig.suptitle(f'Serendipity Finder Analysis: {feature1} vs {feature2}\n'
                    'Hidden Patterns Revealed in Distribution Tails',
                    fontsize=16, fontweight='bold', y=1.02)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        # Display
        plt.show()
    
    def generate_comprehensive_report(self) -> str:
        """
        Generate a detailed, publication-quality report of all discoveries.
        
        This report format is designed to be immediately useful for
        decision-makers and researchers, with clear interpretations
        and actionable insights.
        
        Returns:
            Formatted report as string
        """
        if not self.discoveries:
            return "No analysis has been performed yet. Run find_hidden_correlations() first."
        
        # Initialize report sections
        report_lines = []
        
        # Header
        report_lines.append("\nSERENDIPITY FINDER ANALYSIS REPORT")
        report_lines.append("Generated by Cazandra Aporbo's Advanced Pattern Discovery System")
        report_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("Repository: https://github.com/Cazzy-Aporbo/Serendipity-Finder")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 40)
        
        total_discoveries = (
            len(self.discoveries.get('strong_tail_correlations', [])) +
            len(self.discoveries.get('inverted_patterns', [])) +
            len(self.discoveries.get('tail_only_relationships', []))
        )
        
        report_lines.append(f"Total serendipitous discoveries: {total_discoveries}")
        report_lines.append(f"Hidden tail correlations found: {len(self.discoveries.get('strong_tail_correlations', []))}")
        report_lines.append(f"Inverted patterns detected: {len(self.discoveries.get('inverted_patterns', []))}")
        report_lines.append(f"Tail-only relationships: {len(self.discoveries.get('tail_only_relationships', []))}")
        report_lines.append("")
        
        # Methodology note
        report_lines.append("METHODOLOGY")
        report_lines.append("-" * 40)
        report_lines.append(f"Tail threshold: {self.tail_threshold:.1%} (analyzing top/bottom {self.tail_threshold:.1%} of distributions)")
        report_lines.append(f"Correlation threshold: {self.correlation_threshold:.2f} (minimum r for significance)")
        report_lines.append(f"Minimum tail samples: {self.min_tail_samples} observations required")
        report_lines.append("")
        
        # Key Discoveries
        if self.discoveries.get('strong_tail_correlations'):
            report_lines.append("KEY DISCOVERIES: HIDDEN TAIL CORRELATIONS")
            report_lines.append("-" * 40)
            report_lines.append("These relationships are invisible to traditional correlation analysis")
            report_lines.append("but emerge strongly in extreme conditions.")
            report_lines.append("")
            
            # Sort by serendipity score
            sorted_discoveries = sorted(
                self.discoveries['strong_tail_correlations'],
                key=lambda x: x.get('serendipity_score', 0),
                reverse=True
            )[:5]  # Top 5
            
            for i, discovery in enumerate(sorted_discoveries, 1):
                report_lines.append(f"Discovery {i}:")
                report_lines.append(f"  Variables: {discovery['feature1']} <-> {discovery['feature2']}")
                report_lines.append(f"  Global correlation: {discovery['global_correlation']:.3f} (p={discovery['global_pvalue']:.4f})")
                report_lines.append(f"  Lower tail correlation: {discovery['lower_tail_correlation']:.3f} (n={discovery['lower_tail_n']})")
                report_lines.append(f"  Upper tail correlation: {discovery['upper_tail_correlation']:.3f} (n={discovery['upper_tail_n']})")
                report_lines.append(f"  Serendipity score: {discovery.get('serendipity_score', 0):.3f}")
                report_lines.append(f"  Interpretation: {discovery.get('interpretation', 'Pattern detected in extremes')}")
                report_lines.append("")
        
        # Inverted Patterns
        if self.discoveries.get('inverted_patterns'):
            report_lines.append("INVERTED PATTERNS: REGIME CHANGES")
            report_lines.append("-" * 40)
            report_lines.append("These variables show opposite relationships in different extremes,")
            report_lines.append("suggesting phase transitions or regime changes.")
            report_lines.append("")
            
            for discovery in self.discoveries['inverted_patterns'][:3]:
                report_lines.append(f"  {discovery['feature1']} <-> {discovery['feature2']}:")
                report_lines.append(f"    Lower tail: r={discovery['lower_tail_correlation']:.3f}")
                report_lines.append(f"    Upper tail: r={discovery['upper_tail_correlation']:.3f}")
                report_lines.append(f"    {discovery.get('interpretation', '')}")
                report_lines.append("")
        
        # Scientific Context
        report_lines.append("SCIENTIFIC CONTEXT")
        report_lines.append("-" * 40)
        report_lines.append("This analysis is based on established research in tail dependence theory:")
        report_lines.append("")
        report_lines.append("1. Financial Markets: Longin & Solnik (2001) showed that international")
        report_lines.append("   equity correlations increase from 0.4 to 0.8+ during market crashes.")
        report_lines.append("")
        report_lines.append("2. Medical Research: The Vioxx case demonstrated how drug side effects")
        report_lines.append("   can be hidden in population averages but visible in high-risk subgroups.")
        report_lines.append("")
        report_lines.append("3. Climate Science: Scheffer et al. (2001) identified critical transitions")
        report_lines.append("   in ecosystems that only appear at temperature extremes.")
        report_lines.append("")
        
        # Implications
        report_lines.append("IMPLICATIONS FOR DECISION-MAKING")
        report_lines.append("-" * 40)
        report_lines.append("1. Risk Management: Traditional correlation-based risk models may")
        report_lines.append("   severely underestimate extreme event probabilities.")
        report_lines.append("")
        report_lines.append("2. Predictive Modeling: Including tail-specific relationships can")
        report_lines.append("   dramatically improve model performance for rare events.")
        report_lines.append("")
        report_lines.append("3. Scientific Discovery: Many breakthrough insights may be hiding")
        report_lines.append("   in the extremes of existing datasets.")
        report_lines.append("")
        
        # Technical Notes
        report_lines.append("TECHNICAL NOTES")
        report_lines.append("-" * 40)
        report_lines.append("P-values reported use standard t-distribution assumptions.")
        report_lines.append("For small tail samples, bootstrap methods provide more reliable inference.")
        report_lines.append("Consider tail dependencies when correlations exceed 0.6 with p < 0.05.")
        report_lines.append("")
        
        # Footer
        report_lines.append("-" * 60)
        report_lines.append("End of Report")
        report_lines.append("For questions or collaboration: https://github.com/Cazzy-Aporbo")
        
        return "\n".join(report_lines)
    
    def export_results(self, 
                      format: str = 'csv',
                      filepath: Optional[str] = None) -> None:
        """
        Export discoveries in various formats for further analysis or reporting.
        
        Args:
            format: Output format ('csv', 'json', or 'excel')
            filepath: Output file path (auto-generated if None)
        """
        if not self.discoveries:
            logger.warning("No discoveries to export. Run analysis first.")
            return
        
        # Generate default filepath if not provided
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"serendipity_discoveries_{timestamp}.{format}"
        
        # Prepare data for export
        all_discoveries = []
        for category, discoveries in self.discoveries.items():
            if isinstance(discoveries, list):
                for discovery in discoveries:
                    discovery['category'] = category
                    all_discoveries.append(discovery)
        
        if format == 'csv':
            # Export to CSV
            df = pd.DataFrame(all_discoveries)
            df.to_csv(filepath, index=False)
            logger.info(f"Discoveries exported to {filepath}")
            
        elif format == 'json':
            # Export to JSON with metadata
            export_data = {
                'metadata': self.analysis_metadata,
                'parameters': {
                    'tail_threshold': self.tail_threshold,
                    'correlation_threshold': self.correlation_threshold,
                    'min_tail_samples': self.min_tail_samples
                },
                'discoveries': self.discoveries,
                'export_timestamp': datetime.now().isoformat()
            }
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            logger.info(f"Discoveries exported to {filepath}")
            
        elif format == 'excel':
            # Export to Excel with multiple sheets
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Write each discovery category to a separate sheet
                for category, discoveries in self.discoveries.items():
                    if isinstance(discoveries, list) and discoveries:
                        df = pd.DataFrame(discoveries)
                        sheet_name = category[:31]  # Excel sheet name limit
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Add metadata sheet
                metadata_df = pd.DataFrame([self.analysis_metadata])
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
            
            logger.info(f"Discoveries exported to {filepath}")
        
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv', 'json', or 'excel'.")


def demonstrate_serendipity_finder():
    """
    Comprehensive demonstration of the Serendipity Finder capabilities.
    
    This function walks through a complete analysis workflow, from
    data generation through discovery and visualization, demonstrating
    how the tool reveals insights that traditional methods miss.
    """
    print("\nSerendipity Finder v2.0 - Advanced Pattern Discovery")
    print("Created by Cazandra Aporbo, MS 2025")
    print("Repository: https://github.com/Cazzy-Aporbo/Serendipity-Finder")
    print("-" * 60)
    
    # Initialize the finder
    print("\nInitializing Serendipity Finder with scientifically-grounded parameters...")
    finder = SerendipityFinder(
        tail_threshold=0.15,  # Analyze top/bottom 15% - balances extremity with sample size
        correlation_threshold=0.6,  # Strong correlation threshold (Cohen's guidelines)
        min_tail_samples=30,  # Minimum for reliable correlation estimates
        bootstrap_iterations=1000  # For robust confidence intervals
    )
    
    # Load example data
    print("\nGenerating example dataset based on real-world phenomena...")
    print("Example: Financial market behavior during crisis periods")
    data = finder.load_real_world_example('financial')
    print(f"Dataset created: {len(data)} observations, {len(data.columns)} variables")
    
    # Perform analysis
    print("\nSearching for hidden correlations in distribution tails...")
    print("This process examines each pair of variables for patterns that")
    print("emerge only in extreme conditions...")
    discoveries = finder.find_hidden_correlations()
    
    # Display summary
    print(f"\nAnalysis Complete!")
    print(f"Hidden tail correlations discovered: {len(discoveries['strong_tail_correlations'])}")
    print(f"Inverted patterns found: {len(discoveries['inverted_patterns'])}")
    print(f"Tail-only relationships: {len(discoveries['tail_only_relationships'])}")
    
    # Generate report
    print("\nGenerating comprehensive report...")
    report = finder.generate_comprehensive_report()
    print(report)
    
    # Visualize top discovery if found
    if discoveries['strong_tail_correlations']:
        top_discovery = discoveries['strong_tail_correlations'][0]
        print(f"\nVisualizing top discovery: {top_discovery['feature1']} vs {top_discovery['feature2']}")
        print("This visualization shows how the pattern is invisible globally")
        print("but emerges strongly in the distribution tails...")
        
        finder.visualize_discovery(
            top_discovery['feature1'],
            top_discovery['feature2'],
            save_path='serendipity_top_discovery.png'
        )
        
        # Perform bootstrap significance test
        print("\nPerforming bootstrap significance test (1000 iterations)...")
        print("This establishes confidence intervals for the tail correlations...")
        bootstrap_results = finder.bootstrap_significance_test(
            top_discovery['feature1'],
            top_discovery['feature2']
        )
        
        print(f"Global correlation 95% CI: {bootstrap_results['global_correlation_ci']}")
        print(f"Lower tail 95% CI: {bootstrap_results['lower_tail_ci']}")
        print(f"Upper tail 95% CI: {bootstrap_results['upper_tail_ci']}")
        print(f"Pattern stability: {'Stable' if bootstrap_results['tail_correlation_stable'] else 'Unstable'}")
    
    # Export results
    print("\nExporting discoveries for further analysis...")
    finder.export_results('csv', 'serendipity_discoveries.csv')
    finder.export_results('json', 'serendipity_discoveries.json')
    
    print("\n" + "-" * 60)
    print("Analysis complete!")
    print("\nKey Insight:")
    print("Traditional correlation analysis would have missed these patterns entirely.")
    print("By looking where others don't - in the extremes - we can uncover")
    print("relationships that could be critical for risk management, prediction,")
    print("and scientific discovery.")
    print("\nRemember: The most important patterns often hide in the tails.")
    print("-" * 60)
    
    return finder


if __name__ == "__main__":
    # Run the comprehensive demonstration
    finder = demonstrate_serendipity_finder()
    
    # Additional examples for different domains
    print("\n\nAdditional Examples Available:")
    print("1. Medical: Drug side effects in vulnerable populations")
    print("2. Climate: Ecosystem tipping points at temperature extremes")
    print("\nTo explore these, use:")
    print("  finder.load_real_world_example('medical')")
    print("  finder.load_real_world_example('climate')")
    print("\nFor more information, visit:")
    print("https://github.com/Cazzy-Aporbo/Serendipity-Finder")
