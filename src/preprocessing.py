"""
TCR-seq Data Preprocessing Module
Handles loading, cleaning, and initial processing of TCR-seq data
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
from collections import defaultdict


class TCRPreprocessor:
    """Preprocesses TCR-seq data from multiple samples"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # TCR sequence patterns
        self.v_gene_pattern = re.compile(r"^TR[AB][VDJ][0-9]+")
        self.cdr3_pattern = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")

        # Quality thresholds
        self.min_read_count = config.get("preprocessing", {}).get("min_read_count", 1)
        self.max_cdr3_length = config.get("preprocessing", {}).get(
            "max_cdr3_length", 30
        )
        self.min_cdr3_length = config.get("preprocessing", {}).get("min_cdr3_length", 5)

    def process_directory(self, data_dir: str) -> Dict:
        """Process all TCR-seq files in a directory"""
        data_path = Path(data_dir)
        processed_data = {}

        # Find all TCR-seq files
        tcr_files = []
        for ext in ["*.csv", "*.tsv", "*.txt"]:
            tcr_files.extend(data_path.glob(ext))

        if not tcr_files:
            raise ValueError(f"No TCR-seq files found in {data_dir}")

        self.logger.info(f"Found {len(tcr_files)} TCR-seq files")

        for file_path in tcr_files:
            sample_name = self._extract_sample_name(file_path)
            condition = self._determine_condition(sample_name)

            self.logger.info(f"Processing sample: {sample_name} ({condition})")

            try:
                sample_data = self._process_sample_file(
                    file_path, sample_name, condition
                )
                processed_data[sample_name] = sample_data

            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {str(e)}")
                continue

        if not processed_data:
            raise ValueError("No samples were successfully processed")

        # Create combined dataset
        combined_data = self._create_combined_dataset(processed_data)
        processed_data["combined"] = combined_data

        self.logger.info(f"Successfully processed {len(processed_data) - 1} samples")
        return processed_data

    def _extract_sample_name(self, file_path: Path) -> str:
        """Extract sample name from filename"""
        return file_path.stem

    def _determine_condition(self, sample_name: str) -> str:
        """Determine if sample is control or diseased"""
        control_keywords = ["control", "ctrl", "healthy", "normal", "cn"]
        diseased_keywords = ["disease", "patient", "diseased", "case", "dz"]

        sample_lower = sample_name.lower()

        for keyword in control_keywords:
            if keyword in sample_lower:
                return "control"

        for keyword in diseased_keywords:
            if keyword in sample_lower:
                return "diseased"

        # Default to control if unclear
        self.logger.warning(
            f"Could not determine condition for {sample_name}, defaulting to control"
        )
        return "control"

    def _process_sample_file(
        self, file_path: Path, sample_name: str, condition: str
    ) -> Dict:
        """Process a single TCR-seq sample file"""
        # Load data
        if file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
        else:  # tsv or txt
            df = pd.read_csv(file_path, sep="\t")

        # Standardize column names
        df = self._standardize_columns(df)

        # Quality control
        df = self._quality_control(df)

        # Add sample metadata
        df["sample_name"] = sample_name
        df["condition"] = condition

        # Calculate additional metrics
        df = self._calculate_metrics(df)

        # Calculate sample-level metrics
        df = self._calculate_sample_level_metrics(df)

        return {
            "data": df,
            "sample_name": sample_name,
            "condition": condition,
            "n_clones": len(df),
            "total_reads": df["read_count"].sum(),
            "file_path": str(file_path),
        }

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across different file formats"""
        column_mapping = {
            # Common variations for CDR3
            "cdr3": "cdr3_amino_acid",
            "CDR3": "cdr3_amino_acid",
            "cdr3aa": "cdr3_amino_acid",
            "amino_acid": "cdr3_amino_acid",
            # Common variations for V gene
            "v_gene": "v_gene",
            "V_gene": "v_gene",
            "v": "v_gene",
            "TRBV": "v_gene",
            # Common variations for J gene
            "j_gene": "j_gene",
            "J_gene": "j_gene",
            "j": "j_gene",
            "TRBJ": "j_gene",
            # Common variations for read count
            "count": "read_count",
            "reads": "read_count",
            "frequency": "read_count",
            "clone_count": "read_count",
            # Common variations for frequency
            "freq": "frequency",
            "clone_frequency": "frequency",
            "proportion": "frequency",
        }

        # Apply mapping
        df = df.rename(columns=column_mapping)

        # Ensure required columns exist
        required_cols = ["cdr3_amino_acid", "v_gene", "read_count"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(
                    f"Required column '{col}' not found. Available columns: {list(df.columns)}"
                )

        # Add missing columns with default values
        if "j_gene" not in df.columns:
            df["j_gene"] = "Unknown"
        if "frequency" not in df.columns:
            df["frequency"] = df["read_count"] / df["read_count"].sum()

        return df

    def _quality_control(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality control filters"""
        initial_count = len(df)

        # Filter by read count
        df = df[df["read_count"] >= self.min_read_count]

        # Filter CDR3 sequences
        df = df[
            df["cdr3_amino_acid"]
            .str.len()
            .between(self.min_cdr3_length, self.max_cdr3_length)
        ]

        # Filter valid amino acid sequences
        df = df[df["cdr3_amino_acid"].str.match(self.cdr3_pattern)]

        # Filter valid V genes
        df = df[df["v_gene"].str.match(self.v_gene_pattern, na=False)]

        final_count = len(df)

        if final_count < initial_count:
            self.logger.info(
                f"QC: Removed {initial_count - final_count} low-quality clones "
                f"({initial_count} -> {final_count})"
            )

        return df.reset_index(drop=True)

    def _calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional metrics for each clone"""
        # Calculate relative frequency
        total_reads = df["read_count"].sum()
        df["frequency"] = df["read_count"] / total_reads

        # Calculate CDR3 length
        df["cdr3_length"] = df["cdr3_amino_acid"].str.len()

        # Calculate V gene family
        df["v_family"] = df["v_gene"].str.extract(r"(TR[AB][VDJ])[0-9]+")

        # Calculate J gene family
        df["j_family"] = df["j_gene"].str.extract(r"(TR[AB][J])[0-9]+")

        # Add clone ID
        df["clone_id"] = range(1, len(df) + 1)

        return df

    def _create_combined_dataset(self, processed_data: Dict) -> pd.DataFrame:
        """Create a combined dataset from all samples"""
        all_data = []

        for sample_name, sample_info in processed_data.items():
            if sample_name == "combined":
                continue

            df = sample_info["data"].copy()
            all_data.append(df)

        combined_df = pd.concat(all_data, ignore_index=True)

        # Calculate combined metrics
        combined_df = self._calculate_sample_level_metrics(combined_df)

        return combined_df

    def _calculate_sample_level_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate sample-level diversity metrics"""
        sample_metrics = []

        for sample_name in df["sample_name"].unique():
            sample_df = df[df["sample_name"] == sample_name]

            # Shannon diversity index
            freq = sample_df["frequency"].values
            shannon = -np.sum(freq * np.log(freq + 1e-10))

            # Simpson diversity index
            simpson = 1 - np.sum(freq**2)

            # Clonality (1 - normalized Shannon)
            max_shannon = np.log(len(sample_df))
            clonality = 1 - (shannon / max_shannon) if max_shannon > 0 else 0

            # Top clone frequency
            top_clone_freq = sample_df["frequency"].max()

            # Number of productive clones
            n_clones = len(sample_df)

            sample_metrics.append(
                {
                    "sample_name": sample_name,
                    "shannon_diversity": shannon,
                    "simpson_diversity": simpson,
                    "clonality": clonality,
                    "top_clone_frequency": top_clone_freq,
                    "n_clones": n_clones,
                }
            )

        # Merge metrics back to main dataframe
        metrics_df = pd.DataFrame(sample_metrics)
        df = df.merge(metrics_df, on="sample_name", how="left")

        return df

    def save_processed_data(self, processed_data: Dict, output_dir: str):
        """Save processed data to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save individual sample data
        for sample_name, sample_info in processed_data.items():
            if sample_name == "combined":
                continue

            file_path = output_path / f"{sample_name}_processed.csv"
            sample_info["data"].to_csv(file_path, index=False)

        # Save combined data
        if "combined" in processed_data:
            combined_path = output_path / "combined_data.csv"
            processed_data["combined"].to_csv(combined_path, index=False)

        # Save summary statistics
        summary = self._generate_summary(processed_data)
        summary_path = output_path / "processing_summary.csv"
        summary.to_csv(summary_path, index=False)

        self.logger.info(f"Processed data saved to {output_dir}")

    def _generate_summary(self, processed_data: Dict) -> pd.DataFrame:
        """Generate summary statistics for all samples"""
        summary_data = []

        for sample_name, sample_info in processed_data.items():
            if sample_name == "combined":
                continue

            summary_data.append(
                {
                    "sample_name": sample_name,
                    "condition": sample_info["condition"],
                    "n_clones": sample_info["n_clones"],
                    "total_reads": sample_info["total_reads"],
                    "shannon_diversity": sample_info["data"]["shannon_diversity"].iloc[
                        0
                    ],
                    "clonality": sample_info["data"]["clonality"].iloc[0],
                    "top_clone_frequency": sample_info["data"][
                        "top_clone_frequency"
                    ].iloc[0],
                }
            )

        return pd.DataFrame(summary_data)

    def load_processed_data(self, input_dir: str) -> Dict:
        """Load previously processed data"""
        input_path = Path(input_dir)
        processed_data = {}

        # Load combined data
        combined_path = input_path / "combined_data.csv"
        if combined_path.exists():
            processed_data["combined"] = pd.read_csv(combined_path)

        # Load individual sample data
        for file_path in input_path.glob("*_processed.csv"):
            sample_name = file_path.stem.replace("_processed", "")
            df = pd.read_csv(file_path)

            processed_data[sample_name] = {
                "data": df,
                "sample_name": sample_name,
                "condition": df["condition"].iloc[0],
                "n_clones": len(df),
                "total_reads": df["read_count"].sum(),
            }

        return processed_data
