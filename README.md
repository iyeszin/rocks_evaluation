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

### Development
- Run `pip install -e .` initially to install the package
- Run `pip install -e .` again after modifying any imports in `__init__.py` files
- No reinstall needed for changes to function implementations
- Editable mode (-e flag) creates a link to source code, allowing changes to take effect immediately

## Project Structure
```
rocks_evaluation/
├── rocks_evaluation/          
│   ├── models/
│   │   ├── mineral_classifier.py
│   │   └── integrated_classifier.py
│   └── utils/
│       ├── enums.py
│       ├── data_classes.py
│       └── visualization.py
├── notebooks/
│   ├── generate-validate-data-expert.ipynb           # to generate synthetic rocks as test samples for validation
│   ├── integrated-mineral-rock-system.ipynb          # to run hybrid system - baseline + expert system
│   ├── integrated-mineral-rock-system-w-unknown.ipynb   # to run hybrid system - uncertainty + expert system
│   └── evaluation_metrics.ipynb                      # plots to put at report
├── validation_rocks/          # Validation data
├── weights/                   # Model weights
├── mineral_label_encoder.joblib
├── setup.py
└── README.md
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