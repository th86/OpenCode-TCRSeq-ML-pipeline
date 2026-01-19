# TCR-seq Data Analysis & Machine Learning Workflow

A comprehensive pipeline for analyzing TCR-seq data and classifying samples into control vs diseased conditions using clone identification, motif discovery, and machine learning.

This codebase was generated, tested, and debugged by [OpenCode](https://opencode.ai/). 

## Features

- **Data Preprocessing**: Handles multiple TCR-seq file formats with quality control
- **Clone Analysis**: Identifies TCR clones and extracts diversity metrics
- **Motif Discovery**: Discovers sequence motifs using multiple approaches (k-mer, position-specific, topic modeling)
- **Machine Learning**: Trains and evaluates multiple classification models
- **Visualization**: Creates comprehensive plots and interactive dashboards
- **Cross-sample Analysis**: Identifies public/private clones and differential motifs

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd tcr_seq_workflow
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create sample configuration:
```bash
python -c "from src.utils import create_sample_config; create_sample_config()"
```

## Usage

### Basic Usage

```bash
python main.py --data_dir /path/to/tcr/data --output_dir results
```

### Advanced Usage

```bash
python main.py \
    --data_dir /path/to/tcr/data \
    --config config/config.yaml \
    --output_dir results \
    --mode full \
    --verbose
```

### Different Modes

- `full`: Complete analysis (preprocessing + ML)
- `preprocess`: Only preprocessing and feature extraction
- `ml`: Only machine learning (requires preprocessed data)

## Input Data Format

The workflow expects TCR-seq data in CSV/TSV format with the following columns:

### Required Columns
- `cdr3_amino_acid` or `cdr3`: CDR3 amino acid sequence
- `v_gene` or `V_gene`: V gene identifier
- `read_count` or `count`: Read count for each clone

### Optional Columns
- `j_gene` or `J_gene`: J gene identifier
- `frequency`: Clone frequency (calculated if not provided)

### Sample Naming
Sample files should be named to indicate condition:
- Control samples: `control_01.csv`, `healthy_02.tsv`, etc.
- Diseased samples: `disease_01.csv`, `patient_02.tsv`, etc.

## Configuration

Edit `config/config.yaml` to customize analysis parameters:

```yaml
preprocessing:
  min_read_count: 1
  max_cdr3_length: 30
  min_cdr3_length: 5

clone_analysis:
  definition: exact  # or 'similar'
  min_clone_size: 2
  similarity_threshold: 0.8

motif_discovery:
  min_motif_length: 3
  max_motif_length: 8
  min_motif_frequency: 5
  n_topics: 10

ml:
  test_size: 0.2
  cv_folds: 5
  random_state: 42
  scaler: standard
  feature_selector: none
  remove_correlated: true
```

## Output Structure

```
results/
├── combined_data.csv              # Preprocessed data
├── clone_features.csv             # Clone-based features
├── motif_features.csv             # Motif-based features
├── ml_results_summary.csv         # Model performance summary
├── best_model.joblib               # Trained model
├── feature_importance.csv         # Feature importance rankings
├── plots/                          # Visualizations
│   ├── data_overview.png
│   ├── clone_analysis.png
│   ├── motif_analysis.png
│   ├── ml_model_comparison.png
│   ├── best_model_results.png
│   ├── feature_importance.png
│   └── interactive_dashboard.html
└── analysis_summary.txt           # Text summary
```

## Machine Learning Models

The workflow trains and evaluates the following models:

1. Random Forest
2. Gradient Boosting
3. Support Vector Machine (SVM)
4. Logistic Regression
5. K-Nearest Neighbors (KNN)
6. Decision Tree
7. Naive Bayes
8. AdaBoost
9. Ridge Classifier
10. Multi-layer Perceptron (MLP)

## Feature Types

### Clone-based Features
- Number of clones
- Clone size distribution
- Shannon/Simpson diversity
- Clonality metrics
- V/J gene usage patterns
- CDR3 length statistics

### Motif-based Features
- K-mer motif counts
- Position-specific motifs
- Physico-chemical patterns
- Topic modeling motifs
- Differential motifs

## Visualization

The pipeline generates:
- Data overview plots
- Clone analysis visualizations
- Motif discovery plots
- Model performance comparisons
- Feature importance charts
- Interactive dashboards

## Example Workflow

1. **Prepare Data**: Place TCR-seq files in a directory
2. **Run Analysis**: Execute the main pipeline
3. **Review Results**: Check plots and summary reports
4. **Examine Model**: Inspect best model performance
5. **Feature Analysis**: Review important features

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Install all requirements with `pip install -r requirements.txt`
2. **File Format Errors**: Ensure CSV/TSV files have required columns
3. **Memory Issues**: Reduce dataset size or adjust parameters
4. **Poor Model Performance**: Try different preprocessing parameters or feature selection

### Logging

The workflow generates detailed logs in `tcr_workflow.log`. Use `--verbose` for more detailed output.

## Advanced Features

### Hyperparameter Tuning

```python
from src.ml_pipeline import MLPipeline

pipeline = MLPipeline(config)
results = pipeline.hyperparameter_tuning(feature_matrix, 'random_forest')
```

### Custom Analysis

```python
from src.preprocessing import TCRPreprocessor
from src.clone_analysis import CloneAnalyzer
from src.motif_discovery import MotifDiscoverer

# Load and preprocess data
preprocessor = TCRPreprocessor(config)
data = preprocessor.process_directory('data/')

# Analyze clones
clone_analyzer = CloneAnalyzer(config)
clone_features = clone_analyzer.analyze_clones(data)

# Discover motifs
motif_discoverer = MotifDiscoverer(config)
motif_features = motif_discoverer.discover_motifs(data)
```

## Citation

If you use this workflow in your research, please cite:

```
TCR-seq Analysis & Machine Learning Workflow
[Your citation information]
```

## License

[Specify your license]

## Contributing

[Contributing guidelines]

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the log files
3. Create an issue with detailed information
