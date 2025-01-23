# Rocks Evaluation

A Python package combining deep learning and geological expertise for mineral and rock type classification.

## Core Components
- Deep Learning: CNN models for mineral classification
- Expert System: Geologist-advised rules for rock type classification based on mineral assemblages
- Uncertainty Quantification (optional): Monte Carlo dropout for prediction confidence

## Installation
```bash
git clone https://github.com/iyeszin/rocks_evaluation.git
cd rocks-evaluation
pip install -e .
```

## Project Structure
```
rocks_evaluation/
├── models/
│   ├── mineral_classifier.py     # CNN architectures
│   └── integrated_classifier.py  # Expert system + CNN integration
├── utils/
│   ├── enums.py                 # Rock type definitions
│   ├── data_classes.py          # Mineral groupings (Γ notation)
│   └── visualization.py         # Analysis plotting
└── config.py                    # System parameters
```

## Expert System Rules
The rock classification follows geologist-defined rules:
1. Mineral Assemblage Analysis
   - Granite: Requires feldspars, quartz, and micas
   - Limestone: Requires calcite with minimal silicate impurities
   - Sandstone: Requires quartz with feldspars and accessory minerals
2. Accuracy Requirements
   - Standard: ≥60% mineral identification accuracy
   - Uncertainty-aware: Handles "unknown" predictions with confidence thresholds

## Usage
```python
from rocks_evaluation import IntegratedRockClassifier, UncertaintyIntegratedRockClassifier

# Standard classification
classifier = IntegratedRockClassifier(model, label_encoder, device)

# With uncertainty quantification
uncertainty_classifier = UncertaintyIntegratedRockClassifier(
    uncertainty_model, 
    label_encoder, 
    device
)
```

## Dependencies
- PyTorch
- NumPy
- Matplotlib
- scikit-learn