#!/usr/bin/env python3
"""
TCR-seq Data Analysis & Machine Learning Workflow
Main pipeline for analyzing TCR-seq data and classifying control vs diseased samples
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.preprocessing import TCRPreprocessor
from src.clone_analysis import CloneAnalyzer
from src.motif_discovery import MotifDiscoverer
from src.ml_pipeline import MLPipeline
from src.visualization import TCRVisualizer
from src.utils import setup_logging, load_config


def main():
    parser = argparse.ArgumentParser(description="TCR-seq Analysis & ML Workflow")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing TCR-seq data files",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Configuration file path",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory for results"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "preprocess", "ml"],
        default="full",
        help="Analysis mode",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load configuration
    config = load_config(args.config)

    logger.info("Starting TCR-seq analysis workflow")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    try:
        # Initialize components
        preprocessor = TCRPreprocessor(config)
        clone_analyzer = CloneAnalyzer(config)
        motif_discoverer = MotifDiscoverer(config)
        ml_pipeline = MLPipeline(config)
        visualizer = TCRVisualizer(config)

        # Step 1: Data preprocessing
        if args.mode in ["full", "preprocess"]:
            logger.info("Step 1: Data preprocessing")
            processed_data = preprocessor.process_directory(args.data_dir)
            preprocessor.save_processed_data(processed_data, args.output_dir)

        # Step 2: Clone analysis
        if args.mode in ["full", "preprocess"]:
            logger.info("Step 2: Clone analysis")
            clone_features = clone_analyzer.analyze_clones(processed_data)
            clone_analyzer.save_features(clone_features, args.output_dir)

        # Step 3: Motif discovery
        if args.mode in ["full", "preprocess"]:
            logger.info("Step 3: Motif discovery")
            motif_features = motif_discoverer.discover_motifs(processed_data)
            motif_discoverer.save_motifs(motif_features, args.output_dir)

        # Step 4: Machine learning pipeline
        if args.mode in ["full", "ml"]:
            logger.info("Step 4: Machine learning pipeline")

            # Load processed data if not in memory
            if args.mode == "ml":
                processed_data = preprocessor.load_processed_data(args.output_dir)
                clone_features = clone_analyzer.load_features(args.output_dir)
                motif_features = motif_discoverer.load_motifs(args.output_dir)

            # Combine features
            combined_features = ml_pipeline.combine_features(
                clone_features, motif_features, processed_data
            )

            # Train and evaluate models
            results = ml_pipeline.train_models(combined_features)
            ml_pipeline.save_results(results, args.output_dir)

            # Generate visualizations
            visualizer.generate_all_visualizations(
                combined_features, results, args.output_dir
            )

        logger.info("TCR-seq analysis workflow completed successfully")

    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
