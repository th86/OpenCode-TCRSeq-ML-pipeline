"""
Utility functions for TCR-seq workflow
"""

import logging
import yaml
import json
from pathlib import Path
from typing import Dict, Any


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("tcr_workflow.log")],
    )


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config if config is not None else {}
    except FileNotFoundError:
        logging.warning(f"Config file {config_path} not found, using defaults")
        return get_default_config()
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config file: {e}")
        return get_default_config()


def get_default_config() -> Dict:
    """Get default configuration"""
    return {
        "preprocessing": {
            "min_read_count": 1,
            "max_cdr3_length": 30,
            "min_cdr3_length": 5,
        },
        "clone_analysis": {
            "definition": "exact",
            "min_clone_size": 2,
            "similarity_threshold": 0.8,
        },
        "motif_discovery": {
            "min_motif_length": 3,
            "max_motif_length": 8,
            "min_motif_frequency": 5,
            "n_topics": 10,
        },
        "ml": {
            "test_size": 0.2,
            "cv_folds": 5,
            "random_state": 42,
            "scaler": "standard",
            "feature_selector": "none",
            "remove_correlated": True,
        },
        "visualization": {"figsize": [10, 6], "dpi": 300, "color_palette": "Set2"},
    }


def save_config(config: Dict, config_path: str):
    """Save configuration to YAML file"""
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def validate_config(config: Dict) -> bool:
    """Validate configuration parameters"""
    required_sections = [
        "preprocessing",
        "clone_analysis",
        "motif_discovery",
        "ml",
        "visualization",
    ]

    for section in required_sections:
        if section not in config:
            logging.warning(f"Missing config section: {section}")
            return False

    # Validate specific parameters
    if (
        config["preprocessing"]["min_cdr3_length"]
        >= config["preprocessing"]["max_cdr3_length"]
    ):
        logging.error("min_cdr3_length must be less than max_cdr3_length")
        return False

    if config["ml"]["test_size"] <= 0 or config["ml"]["test_size"] >= 1:
        logging.error("test_size must be between 0 and 1")
        return False

    return True


def create_sample_config():
    """Create a sample configuration file"""
    config = get_default_config()
    config_path = Path("config/config.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    save_config(config, str(config_path))
    print(f"Sample configuration created at {config_path}")


def format_results_summary(results: Dict) -> str:
    """Format results for display"""
    summary = []

    if "best_model" in results:
        best = results["best_model"]
        summary.append(f"Best Model: {best.get('model_name', 'Unknown')}")
        summary.append(f"Test Accuracy: {best.get('test_accuracy', 0):.3f}")
        summary.append(f"Test F1 Score: {best.get('test_f1', 0):.3f}")
        summary.append(f"Test AUC: {best.get('test_auc', 0):.3f}")

    return "\n".join(summary)


def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        "pandas",
        "numpy",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "seaborn",
        "plotly",
        "pyyaml",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("All dependencies are installed")
        return True
