# Serendipity Finder

**Pattern discovery in distribution extremes**

[![Python](https://img.shields.io/badge/Python-3.8+-7B8DFF?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-FF9F68?style=flat-square)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Demo-Live-050510?style=flat-square)](https://cazzy-aporbo.github.io/Serendipity-Finder/)

---

## Overview

Serendipity Finder detects correlations that exist only in distribution tails—relationships invisible to standard regression. Where global correlation shows r=0.06, tail analysis may reveal r=0.85.

Traditional correlation analysis examines relationships across entire datasets. This approach systematically misses patterns that emerge under extreme conditions: market crashes, adverse drug events, clinical deterioration, system failures.

This tool looks where others don't.

---

## How It Works

```
Global correlation:     r = 0.06  →  "No relationship"
Lower tail (≤15%):      r = 0.85  →  Strong negative correlation
Upper tail (≥85%):      r = 0.82  →  Strong positive correlation
```

The algorithm:
1. Separates data into core and tail regions
2. Computes correlations within each region independently
3. Identifies cases where tail correlations diverge from global correlations
4. Quantifies discovery significance

**Serendipity Score:**
```
S = (1 - |ρ_global|) × max(|ρ_lower|, |ρ_upper|)
```

---

## Installation

```bash
git clone https://github.com/Cazzy-Aporbo/Serendipity-Finder.git
cd Serendipity-Finder
pip install -r requirements.txt
python serendipity_finder.py
```

**Requirements:** Python 3.8+, NumPy, Pandas, Matplotlib, SciPy, Seaborn

---

## Usage

```python
from serendipity_finder import SerendipityFinder

# Initialize
finder = SerendipityFinder(tail_threshold=0.15, correlation_threshold=0.6)

# Generate synthetic data with hidden tail patterns
data = finder.generate_serendipitous_data(n_samples=1000, n_features=10)

# Discover hidden correlations
discoveries = finder.find_hidden_correlations()

# Report findings
print(finder.generate_report())

# Visualize
finder.visualize_discovery('feature_00', 'feature_01')
```

**With your own data:**
```python
import pandas as pd

df = pd.read_csv('your_data.csv')
finder = SerendipityFinder()
discoveries = finder.find_hidden_correlations(df)
finder.export_discoveries('discoveries.csv')
```

---

## Example Output

| Feature Pair | Global r | Lower Tail | Upper Tail | Significance |
|:-------------|:--------:|:----------:|:----------:|:------------:|
| feature_02 ↔ derived_product | -0.057 | -0.315 | -0.850 | 0.801 |
| feature_02 ↔ feature_07 | -0.031 | -0.556 | -0.639 | 0.619 |
| feature_00 ↔ feature_07 | -0.026 | -0.553 | -0.635 | 0.618 |

Global correlations near zero. Tail correlations exceeding 0.6. Traditional analysis would conclude "no relationship." Serendipity Finder surfaces what's actually there.

---

## Applications

| Domain | Use Case | Example |
|:-------|:---------|:--------|
| **Finance** | Tail risk | Correlations that emerge during market stress |
| **Healthcare** | Adverse events | Drug interactions in specific patient subgroups |
| **Climate** | Tipping points | Threshold behaviors in complex systems |
| **Quality Control** | Failure modes | Defects under extreme operating conditions |

---

## Why This Matters

Distribution tails are not noise. They contain:
- Physiologically distinct patient populations
- Rare but consequential system failures  
- Conditions where standard models break down
- Signals that precede catastrophic events

Ignoring outliers is a choice. This tool provides an alternative.

→ **[Read: The Cost of Ignoring Outliers](ETHICAL_IMPACT.md)**

---

## Project Status

**Current version: v3**

| Version | Status | Description |
|:--------|:-------|:------------|
| v1 | Complete | AI-assisted baseline |
| v2 | Complete | Human-structured refactor |
| v3 | Current | Dataset trials, validation workflows |
| v4 | Planned | Real-world benchmarks, reproducible reports |

---

## Files

| File | Purpose |
|:-----|:--------|
| `serendipity_finder.py` | Core algorithm |
| `serendipity_finder_v2.py` | Refactored implementation |
| `test_serendipity.py` | Test suite |
| `serendipity.html` | Interactive visualization |
| `ETHICAL_IMPACT.md` | Why tail analysis matters for health equity |

---

## Citation

```bibtex
@software{serendipity_finder_2025,
  title = {Serendipity Finder: Pattern Discovery in Distribution Extremes},
  author = {Aporbo, Cazandra},
  year = {2025},
  url = {https://github.com/Cazzy-Aporbo/Serendipity-Finder}
}
```

---

## Contact

**Cazandra Aporbo**  
Chief Data & AI Officer, [Loopchii](https://loopchii.com)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-7B8DFF?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/cazandra-aporbo)
[![Email](https://img.shields.io/badge/Email-FF9F68?style=flat-square&logo=gmail&logoColor=white)](mailto:loopchii.tech@gmail.com)
[![Loopchii](https://img.shields.io/badge/loopchii.com-050510?style=flat-square)](https://loopchii.com)

---

## License

MIT License. See [LICENSE](LICENSE).

---

<div align="center">

![](https://img.shields.io/badge/━━━━━━━━━━-050510?style=flat-square)
![](https://img.shields.io/badge/━━━━━━━━━━-7B8DFF?style=flat-square)
![](https://img.shields.io/badge/━━━━━━━━━━-FF9F68?style=flat-square)
![](https://img.shields.io/badge/━━━━━━━━━━-7B8DFF?style=flat-square)
![](https://img.shields.io/badge/━━━━━━━━━━-050510?style=flat-square)

*A [Loopchii](https://loopchii.com) research project*

</div>



