"""
TCR Clone Analysis Module
Identifies and analyzes TCR clones for feature extraction
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
import Levenshtein


class CloneAnalyzer:
    """Analyzes TCR clones and extracts features for machine learning"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Clone definition parameters
        self.clone_definition = config.get("clone_analysis", {}).get(
            "definition", "exact"
        )
        self.min_clone_size = config.get("clone_analysis", {}).get("min_clone_size", 2)
        self.similarity_threshold = config.get("clone_analysis", {}).get(
            "similarity_threshold", 0.8
        )

    def analyze_clones(self, processed_data: Dict) -> Dict:
        """Analyze clones across all samples"""
        self.logger.info("Starting clone analysis")

        clone_features = {}

        # Analyze each sample individually
        for sample_name, sample_info in processed_data.items():
            if sample_name == "combined":
                continue

            self.logger.info(f"Analyzing clones in sample: {sample_name}")
            sample_features = self._analyze_sample_clones(sample_info["data"])
            clone_features[sample_name] = sample_features

        # Analyze cross-sample clone sharing
        cross_sample_features = self._analyze_cross_sample_clones(processed_data)
        clone_features["cross_sample"] = cross_sample_features

        # Create feature matrix for ML
        feature_matrix = self._create_feature_matrix(clone_features, processed_data)
        clone_features["feature_matrix"] = feature_matrix

        self.logger.info("Clone analysis completed")
        return clone_features

    def _analyze_sample_clones(self, df: pd.DataFrame) -> Dict:
        """Analyze clones within a single sample"""
        features = {}

        # Define clones based on configuration
        if self.clone_definition == "exact":
            clones = self._define_exact_clones(df)
        elif self.clone_definition == "similar":
            clones = self._define_similar_clones(df)
        else:
            clones = self._define_exact_clones(df)  # default

        # Basic clone statistics
        features["clone_stats"] = self._calculate_clone_stats(clones, df)

        # Clone size distribution
        features["size_distribution"] = self._analyze_size_distribution(clones)

        # V/J gene usage patterns
        features["vj_usage"] = self._analyze_vj_usage(df)

        # CDR3 length distribution
        features["cdr3_length_dist"] = self._analyze_cdr3_length_distribution(df)

        # Clonality metrics
        features["clonality_metrics"] = self._calculate_clonality_metrics(df)

        # Public vs private clone analysis
        features["public_private"] = self._analyze_public_private_clones(df)

        return features

    def _define_exact_clones(self, df: pd.DataFrame) -> Dict:
        """Define clones based on exact CDR3 sequence matches"""
        clones = {}

        # Group by exact CDR3 sequence
        for cdr3, group in df.groupby("cdr3_amino_acid"):
            clone_id = f"clone_{len(clones) + 1}"
            clones[clone_id] = {
                "cdr3_sequence": cdr3,
                "size": len(group),
                "read_count": group["read_count"].sum(),
                "frequency": group["frequency"].sum(),
                "v_genes": group["v_gene"].unique(),
                "j_genes": group["j_gene"].unique(),
                "members": group.index.tolist(),
            }

        # Filter by minimum clone size
        clones = {k: v for k, v in clones.items() if v["size"] >= self.min_clone_size}

        return clones

    def _define_similar_clones(self, df: pd.DataFrame) -> Dict:
        """Define clones based on similar CDR3 sequences"""
        clones = {}
        used_indices = set()

        # Get unique CDR3 sequences
        unique_cdr3s = df["cdr3_amino_acid"].unique()

        # Calculate similarity matrix
        similarity_matrix = self._calculate_cdr3_similarity(unique_cdr3s)

        # Cluster similar sequences
        distance_matrix = 1 - similarity_matrix
        condensed_distances = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_distances, method="average")

        # Form clusters based on threshold
        clusters = fcluster(
            linkage_matrix, 1 - self.similarity_threshold, criterion="distance"
        )

        # Create clones from clusters
        for cluster_id in np.unique(clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            cluster_cdr3s = unique_cdr3s[cluster_indices]

            # Get all rows with these CDR3 sequences
            cluster_rows = df[df["cdr3_amino_acid"].isin(cluster_cdr3s)]

            if len(cluster_rows) >= self.min_clone_size:
                clone_id = f"clone_{len(clones) + 1}"
                clones[clone_id] = {
                    "cdr3_sequences": cluster_cdr3s.tolist(),
                    "size": len(cluster_rows),
                    "read_count": cluster_rows["read_count"].sum(),
                    "frequency": cluster_rows["frequency"].sum(),
                    "v_genes": cluster_rows["v_gene"].unique(),
                    "j_genes": cluster_rows["j_gene"].unique(),
                    "members": cluster_rows.index.tolist(),
                }

        return clones

    def _calculate_cdr3_similarity(self, cdr3_sequences: np.ndarray) -> np.ndarray:
        """Calculate similarity matrix between CDR3 sequences"""
        n_sequences = len(cdr3_sequences)
        similarity_matrix = np.zeros((n_sequences, n_sequences))

        for i in range(n_sequences):
            for j in range(i, n_sequences):
                if i == j:
                    similarity = 1.0
                else:
                    # Calculate normalized Levenshtein distance
                    seq1, seq2 = cdr3_sequences[i], cdr3_sequences[j]
                    max_len = max(len(seq1), len(seq2))
                    distance = Levenshtein.distance(seq1, seq2)
                    similarity = 1 - (distance / max_len)

                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

        return similarity_matrix

    def _calculate_clone_stats(self, clones: Dict, df: pd.DataFrame) -> Dict:
        """Calculate basic clone statistics"""
        stats = {
            "n_clones": len(clones),
            "n_unique_cdr3": df["cdr3_amino_acid"].nunique(),
            "largest_clone_size": max([c["size"] for c in clones.values()])
            if clones
            else 0,
            "mean_clone_size": np.mean([c["size"] for c in clones.values()])
            if clones
            else 0,
            "clone_coverage": sum([c["frequency"] for c in clones.values()]),
            "singleton_rate": (df["cdr3_amino_acid"].value_counts() == 1).mean(),
        }

        return stats

    def _analyze_size_distribution(self, clones: Dict) -> Dict:
        """Analyze clone size distribution"""
        if not clones:
            return {}

        sizes = [c["size"] for c in clones.values()]
        frequencies = [c["frequency"] for c in clones.values()]

        distribution = {
            "size_mean": np.mean(sizes),
            "size_median": np.median(sizes),
            "size_std": np.std(sizes),
            "size_max": max(sizes),
            "freq_mean": np.mean(frequencies),
            "freq_median": np.median(frequencies),
            "freq_std": np.std(frequencies),
            "gini_coefficient": self._calculate_gini_coefficient(sizes),
        }

        return distribution

    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for inequality measurement"""
        if not values:
            return 0

        values = sorted(values)
        n = len(values)
        cumsum = np.cumsum(values)

        gini = (n + 1 - 2 * sum((n + 1 - i) * v for i, v in enumerate(values))) / (
            n * sum(values)
        )

        return gini

    def _analyze_vj_usage(self, df: pd.DataFrame) -> Dict:
        """Analyze V and J gene usage patterns"""
        vj_usage = {}

        # V gene usage
        v_counts = df["v_gene"].value_counts()
        vj_usage["v_gene_counts"] = v_counts.to_dict()
        vj_usage["v_gene_diversity"] = len(v_counts)
        vj_usage["v_gene_entropy"] = self._calculate_entropy(v_counts.values)

        # J gene usage
        j_counts = df["j_gene"].value_counts()
        vj_usage["j_gene_counts"] = j_counts.to_dict()
        vj_usage["j_gene_diversity"] = len(j_counts)
        vj_usage["j_gene_entropy"] = self._calculate_entropy(j_counts.values)

        # V-J pairing
        vj_pairs = df.groupby(["v_gene", "j_gene"]).size()
        # Convert tuple keys to strings for JSON serialization
        vj_usage["vj_pair_counts"] = {
            f"{k[0]}_{k[1]}": v for k, v in vj_pairs.to_dict().items()
        }
        vj_usage["vj_pair_diversity"] = len(vj_pairs)

        # V family usage
        v_family_counts = df["v_family"].value_counts()
        vj_usage["v_family_counts"] = v_family_counts.to_dict()

        return vj_usage

    def _calculate_entropy(self, counts: np.ndarray) -> float:
        """Calculate Shannon entropy"""
        if len(counts) == 0:
            return 0

        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))

        return entropy

    def _analyze_cdr3_length_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyze CDR3 length distribution"""
        lengths = df["cdr3_length"].values

        length_dist = {
            "length_mean": np.mean(lengths),
            "length_median": np.median(lengths),
            "length_std": np.std(lengths),
            "length_min": min(lengths),
            "length_max": max(lengths),
            "length_counts": df["cdr3_length"].value_counts().to_dict(),
        }

        return length_dist

    def _calculate_clonality_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate various clonality metrics"""
        frequencies = df["frequency"].values

        # Calculate Shannon diversity first
        shannon_diversity = -np.sum(frequencies * np.log(frequencies + 1e-10))

        metrics = {
            "shannon_diversity": shannon_diversity,
            "simpson_diversity": 1 - np.sum(frequencies**2),
            "clonality": 1 - (shannon_diversity / np.log(len(df)))
            if len(df) > 1
            else 0,
            "dominance": max(frequencies),
            "evenness": shannon_diversity / np.log(len(df)) if len(df) > 1 else 0,
        }

        return metrics

    def _analyze_public_private_clones(self, df: pd.DataFrame) -> Dict:
        """Analyze public (shared) vs private clones"""
        # This would require cross-sample data
        # For now, return placeholder
        return {
            "public_clone_rate": 0.0,  # To be calculated with cross-sample data
            "private_clone_rate": 1.0,
            "public_clones": [],
            "private_clones": df["cdr3_amino_acid"].unique().tolist(),
        }

    def _analyze_cross_sample_clones(self, processed_data: Dict) -> Dict:
        """Analyze clone sharing across samples"""
        cross_sample = {}

        # Collect all CDR3 sequences from all samples
        all_sequences = {}
        sample_sequences = {}

        for sample_name, sample_info in processed_data.items():
            if sample_name == "combined":
                continue

            sequences = set(sample_info["data"]["cdr3_amino_acid"].unique())
            sample_sequences[sample_name] = sequences

            for seq in sequences:
                if seq not in all_sequences:
                    all_sequences[seq] = []
                all_sequences[seq].append(sample_name)

        # Identify public and private clones
        public_clones = {
            seq: samples for seq, samples in all_sequences.items() if len(samples) > 1
        }
        private_clones = {
            seq: samples for seq, samples in all_sequences.items() if len(samples) == 1
        }

        cross_sample["public_clones"] = public_clones
        cross_sample["private_clones"] = private_clones
        cross_sample["public_clone_rate"] = (
            len(public_clones) / len(all_sequences) if all_sequences else 0
        )
        cross_sample["private_clone_rate"] = (
            len(private_clones) / len(all_sequences) if all_sequences else 0
        )

        # Calculate sharing matrix
        sharing_matrix = self._calculate_sharing_matrix(sample_sequences)
        cross_sample["sharing_matrix"] = sharing_matrix

        return cross_sample

    def _calculate_sharing_matrix(
        self, sample_sequences: Dict[str, set]
    ) -> pd.DataFrame:
        """Calculate clone sharing matrix between samples"""
        sample_names = list(sample_sequences.keys())
        n_samples = len(sample_names)

        sharing_matrix = pd.DataFrame(0, index=sample_names, columns=sample_names)

        for i, sample1 in enumerate(sample_names):
            for j, sample2 in enumerate(sample_names):
                if i <= j:
                    shared = len(sample_sequences[sample1] & sample_sequences[sample2])
                    sharing_matrix.loc[sample1, sample2] = shared
                    sharing_matrix.loc[sample2, sample1] = shared

        return sharing_matrix

    def _create_feature_matrix(
        self, clone_features: Dict, processed_data: Dict
    ) -> pd.DataFrame:
        """Create feature matrix for machine learning"""
        features_list = []

        for sample_name, sample_info in processed_data.items():
            if sample_name == "combined":
                continue

            # Get sample features
            if sample_name in clone_features:
                sample_features = clone_features[sample_name]
            else:
                sample_features = {}

            # Create feature row
            feature_row = {
                "sample_name": sample_name,
                "condition": sample_info["condition"],
            }

            # Add clone statistics
            if "clone_stats" in sample_features:
                stats = sample_features["clone_stats"]
                for key, value in stats.items():
                    feature_row[f"clone_{key}"] = value

            # Add size distribution features
            if "size_distribution" in sample_features:
                dist = sample_features["size_distribution"]
                for key, value in dist.items():
                    feature_row[f"size_{key}"] = value

            # Add VJ usage features
            if "vj_usage" in sample_features:
                vj = sample_features["vj_usage"]
                feature_row["v_diversity"] = vj.get("v_gene_diversity", 0)
                feature_row["j_diversity"] = vj.get("j_gene_diversity", 0)
                feature_row["vj_pair_diversity"] = vj.get("vj_pair_diversity", 0)
                feature_row["v_entropy"] = vj.get("v_gene_entropy", 0)
                feature_row["j_entropy"] = vj.get("j_gene_entropy", 0)

            # Add CDR3 length features
            if "cdr3_length_dist" in sample_features:
                length = sample_features["cdr3_length_dist"]
                for key, value in length.items():
                    if isinstance(value, (int, float)):
                        feature_row[f"length_{key}"] = value

            # Add clonality metrics
            if "clonality_metrics" in sample_features:
                clonality = sample_features["clonality_metrics"]
                for key, value in clonality.items():
                    feature_row[f"clonality_{key}"] = value

            # Add cross-sample features
            if "cross_sample" in clone_features:
                cross = clone_features["cross_sample"]
                feature_row["public_clone_rate"] = cross.get("public_clone_rate", 0)
                feature_row["private_clone_rate"] = cross.get("private_clone_rate", 0)

            features_list.append(feature_row)

        feature_matrix = pd.DataFrame(features_list)
        feature_matrix = feature_matrix.fillna(0)  # Handle missing values

        return feature_matrix

    def save_features(self, clone_features: Dict, output_dir: str):
        """Save clone features to files"""
        import json
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save feature matrix
        if "feature_matrix" in clone_features:
            matrix_path = output_path / "clone_features.csv"
            clone_features["feature_matrix"].to_csv(matrix_path, index=False)

        # Save detailed features as JSON
        features_to_save = {
            k: v for k, v in clone_features.items() if k != "feature_matrix"
        }

        # Convert numpy arrays to lists for JSON serialization
        self._convert_numpy_to_list(features_to_save)

        json_path = output_path / "clone_features_detailed.json"
        with open(json_path, "w") as f:
            json.dump(features_to_save, f, indent=2)

        self.logger.info(f"Clone features saved to {output_dir}")

    def _convert_numpy_to_list(self, obj):
        """Recursively convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                obj[key] = self._convert_numpy_to_list(value)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                obj[i] = self._convert_numpy_to_list(item)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        else:
            return obj

        return obj

    def load_features(self, input_dir: str) -> Dict:
        """Load previously saved clone features"""
        import json
        from pathlib import Path

        input_path = Path(input_dir)
        clone_features = {}

        # Load feature matrix
        matrix_path = input_path / "clone_features.csv"
        if matrix_path.exists():
            clone_features["feature_matrix"] = pd.read_csv(matrix_path)

        # Load detailed features
        json_path = input_path / "clone_features_detailed.json"
        if json_path.exists():
            with open(json_path, "r") as f:
                detailed_features = json.load(f)
            clone_features.update(detailed_features)

        return clone_features
