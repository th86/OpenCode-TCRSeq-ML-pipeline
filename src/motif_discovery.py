"""
TCR Motif Discovery Module
Discovers sequence motifs in TCR CDR3 regions for classification
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from Bio import motifs
from Bio.Seq import Seq


class MotifDiscoverer:
    """Discovers and analyzes TCR sequence motifs"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Motif discovery parameters
        self.min_motif_length = config.get("motif_discovery", {}).get(
            "min_motif_length", 3
        )
        self.max_motif_length = config.get("motif_discovery", {}).get(
            "max_motif_length", 8
        )
        self.min_motif_frequency = config.get("motif_discovery", {}).get(
            "min_motif_frequency", 5
        )
        self.n_topics = config.get("motif_discovery", {}).get("n_topics", 10)

        # Amino acid groups for physico-chemical properties
        self.aa_groups = {
            "hydrophobic": ["A", "V", "I", "L", "M", "F", "W", "Y"],
            "polar": ["S", "T", "N", "Q", "C", "G"],
            "positive": ["K", "R", "H"],
            "negative": ["D", "E"],
            "aromatic": ["F", "W", "Y"],
            "aliphatic": ["A", "V", "I", "L", "M"],
        }

    def discover_motifs(self, processed_data: Dict) -> Dict:
        """Discover motifs in TCR sequences"""
        self.logger.info("Starting motif discovery")

        motif_features = {}

        # Discover motifs in each sample
        for sample_name, sample_info in processed_data.items():
            if sample_name == "combined":
                continue

            self.logger.info(f"Discovering motifs in sample: {sample_name}")
            sample_motifs = self._discover_sample_motifs(sample_info["data"])
            motif_features[sample_name] = sample_motifs

        # Discover cross-sample motifs
        cross_sample_motifs = self._discover_cross_sample_motifs(processed_data)
        motif_features["cross_sample"] = cross_sample_motifs

        # Create motif feature matrix
        motif_matrix = self._create_motif_feature_matrix(motif_features, processed_data)
        motif_features["motif_matrix"] = motif_matrix

        self.logger.info("Motif discovery completed")
        return motif_features

    def _discover_sample_motifs(self, df: pd.DataFrame) -> Dict:
        """Discover motifs in a single sample"""
        motifs = {}

        # Extract CDR3 sequences
        cdr3_sequences = df["cdr3_amino_acid"].tolist()

        # K-mer based motif discovery
        motifs["kmer_motifs"] = self._discover_kmer_motifs(cdr3_sequences)

        # Position-specific motif discovery
        motifs["position_motifs"] = self._discover_position_motifs(cdr3_sequences)

        # Physico-chemical motif discovery
        motifs["physicochemical_motifs"] = self._discover_physicochemical_motifs(
            cdr3_sequences
        )

        # Topic modeling based motifs
        motifs["topic_motifs"] = self._discover_topic_motifs(cdr3_sequences)

        # Consensus sequence motifs
        motifs["consensus_motifs"] = self._discover_consensus_motifs(cdr3_sequences)

        return motifs

    def _discover_kmer_motifs(self, sequences: List[str], top_n: int = 50) -> Dict:
        """Discover k-mer based motifs"""
        kmer_counts = defaultdict(int)

        # Count all k-mers
        for seq in sequences:
            for k in range(self.min_motif_length, self.max_motif_length + 1):
                for i in range(len(seq) - k + 1):
                    kmer = seq[i : i + k]
                    kmer_counts[kmer] += 1

        # Filter by minimum frequency
        filtered_kmers = {
            kmer: count
            for kmer, count in kmer_counts.items()
            if count >= self.min_motif_frequency
        }

        # Get top k-mers
        top_kmers = sorted(filtered_kmers.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]

        return {
            "motifs": [{"sequence": kmer, "count": count} for kmer, count in top_kmers],
            "total_kmers": len(kmer_counts),
            "unique_kmers": len(filtered_kmers),
        }

    def _discover_position_motifs(self, sequences: List[str]) -> Dict:
        """Discover position-specific motifs"""
        if not sequences:
            return {}

        # Find maximum sequence length
        max_len = max(len(seq) for seq in sequences)

        # Create position frequency matrix
        position_matrix = defaultdict(lambda: defaultdict(int))

        for seq in sequences:
            for pos, aa in enumerate(seq):
                position_matrix[pos][aa] += 1

        # Calculate position-specific motifs
        position_motifs = []
        for pos in range(max_len):
            if pos in position_matrix:
                aa_counts = position_matrix[pos]
                total = sum(aa_counts.values())

                # Find most common amino acid at each position
                most_common = max(aa_counts.items(), key=lambda x: x[1])
                frequency = most_common[1] / total

                if frequency > 0.3:  # Threshold for significance
                    position_motifs.append(
                        {
                            "position": pos,
                            "amino_acid": most_common[0],
                            "frequency": frequency,
                            "count": most_common[1],
                        }
                    )

        return {
            "motifs": position_motifs,
            "max_length": max_len,
            "positions_analyzed": len(position_matrix),
        }

    def _discover_physicochemical_motifs(self, sequences: List[str]) -> Dict:
        """Discover motifs based on physico-chemical properties"""
        property_patterns = defaultdict(int)

        for seq in sequences:
            for k in range(self.min_motif_length, self.max_motif_length + 1):
                for i in range(len(seq) - k + 1):
                    segment = seq[i : i + k]
                    pattern = self._get_physicochemical_pattern(segment)
                    property_patterns[pattern] += 1

        # Filter and get top patterns
        filtered_patterns = {
            pattern: count
            for pattern, count in property_patterns.items()
            if count >= self.min_motif_frequency
        }

        top_patterns = sorted(
            filtered_patterns.items(), key=lambda x: x[1], reverse=True
        )[:30]

        return {
            "motifs": [
                {"pattern": pattern, "count": count} for pattern, count in top_patterns
            ],
            "unique_patterns": len(filtered_patterns),
        }

    def _get_physicochemical_pattern(self, sequence: str) -> str:
        """Convert amino acid sequence to physico-chemical pattern"""
        pattern = []

        for aa in sequence:
            for group, members in self.aa_groups.items():
                if aa in members:
                    pattern.append(group[0].upper())  # Use first letter
                    break
            else:
                pattern.append("X")  # Unknown

        return "".join(pattern)

    def _discover_topic_motifs(self, sequences: List[str]) -> Dict:
        """Discover motifs using topic modeling (LDA)"""
        if len(sequences) < self.n_topics:
            return {"motifs": [], "method": "lda"}

        # Create document-term matrix using k-mers
        vectorizer = CountVectorizer(
            analyzer="char",
            ngram_range=(self.min_motif_length, self.max_motif_length),
            min_df=self.min_motif_frequency,
            max_features=1000,
        )

        try:
            dtm = vectorizer.fit_transform(sequences)

            # Apply LDA
            lda = LatentDirichletAllocation(
                n_components=self.n_topics, random_state=42, max_iter=100
            )
            lda.fit(dtm)

            # Extract topics as motifs
            feature_names = vectorizer.get_feature_names_out()
            topic_motifs = []

            for topic_idx, topic in enumerate(lda.components_):
                # Get top words for this topic
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_weights = [topic[i] for i in top_words_idx]

                topic_motifs.append(
                    {
                        "topic_id": topic_idx,
                        "motifs": top_words,
                        "weights": top_weights.tolist(),
                        "coherence": np.std(top_weights),
                    }
                )

            return {
                "motifs": topic_motifs,
                "method": "lda",
                "n_topics": self.n_topics,
                "vocabulary_size": len(feature_names),
            }

        except Exception as e:
            self.logger.warning(f"LDA topic modeling failed: {str(e)}")
            return {"motifs": [], "method": "lda", "error": str(e)}

    def _discover_consensus_motifs(self, sequences: List[str]) -> Dict:
        """Discover consensus sequence motifs using clustering"""
        if len(sequences) < 10:
            return {"motifs": [], "method": "consensus"}

        try:
            # Create k-mer frequency vectors for clustering
            vectorizer = CountVectorizer(
                analyzer="char", ngram_range=(3, 5), min_df=2, max_features=500
            )

            kmer_vectors = vectorizer.fit_transform(sequences)

            # Apply K-means clustering
            n_clusters = min(10, len(sequences) // 5)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(kmer_vectors)

            # Create consensus motifs for each cluster
            consensus_motifs = []

            for cluster_id in range(n_clusters):
                cluster_sequences = [
                    seq
                    for i, seq in enumerate(sequences)
                    if cluster_labels[i] == cluster_id
                ]

                if len(cluster_sequences) >= 3:
                    consensus = self._create_consensus_sequence(cluster_sequences)
                    consensus_motifs.append(
                        {
                            "cluster_id": cluster_id,
                            "consensus": consensus,
                            "size": len(cluster_sequences),
                            "sequences": cluster_sequences[
                                :5
                            ],  # Store first 5 as examples
                        }
                    )

            return {
                "motifs": consensus_motifs,
                "method": "consensus",
                "n_clusters": n_clusters,
            }

        except Exception as e:
            self.logger.warning(f"Consensus motif discovery failed: {str(e)}")
            return {"motifs": [], "method": "consensus", "error": str(e)}

    def _create_consensus_sequence(self, sequences: List[str]) -> str:
        """Create consensus sequence from multiple sequences"""
        if not sequences:
            return ""

        # Find maximum length
        max_len = max(len(seq) for seq in sequences)

        consensus = []

        for pos in range(max_len):
            # Count amino acids at this position
            aa_counts = defaultdict(int)
            total = 0

            for seq in sequences:
                if pos < len(seq):
                    aa = seq[pos]
                    aa_counts[aa] += 1
                    total += 1

            if total > 0:
                # Find most common amino acid
                most_common = max(aa_counts.items(), key=lambda x: x[1])
                frequency = most_common[1] / total

                # Add to consensus if frequency is high enough
                if frequency > 0.5:
                    consensus.append(most_common[0])
                else:
                    consensus.append("X")  # Ambiguous position
            else:
                consensus.append("-")  # Gap

        return "".join(consensus)

    def _discover_cross_sample_motifs(self, processed_data: Dict) -> Dict:
        """Discover motifs across all samples"""
        all_sequences = []
        sample_labels = []

        # Collect all sequences
        for sample_name, sample_info in processed_data.items():
            if sample_name == "combined":
                continue

            sequences = sample_info["data"]["cdr3_amino_acid"].tolist()
            all_sequences.extend(sequences)
            sample_labels.extend([sample_name] * len(sequences))

        cross_sample_motifs = {}

        # Discover motifs in combined dataset
        cross_sample_motifs["combined_kmer"] = self._discover_kmer_motifs(
            all_sequences, top_n=100
        )
        cross_sample_motifs["combined_position"] = self._discover_position_motifs(
            all_sequences
        )
        cross_sample_motifs["combined_physicochemical"] = (
            self._discover_physicochemical_motifs(all_sequences)
        )

        # Discover condition-specific motifs
        control_sequences = []
        diseased_sequences = []

        for sample_name, sample_info in processed_data.items():
            if sample_name == "combined":
                continue

            condition = sample_info["condition"]
            sequences = sample_info["data"]["cdr3_amino_acid"].tolist()

            if condition == "control":
                control_sequences.extend(sequences)
            else:
                diseased_sequences.extend(sequences)

        if control_sequences:
            cross_sample_motifs["control_kmer"] = self._discover_kmer_motifs(
                control_sequences, top_n=50
            )

        if diseased_sequences:
            cross_sample_motifs["diseased_kmer"] = self._discover_kmer_motifs(
                diseased_sequences, top_n=50
            )

        # Find differential motifs
        cross_sample_motifs["differential_motifs"] = self._find_differential_motifs(
            control_sequences, diseased_sequences
        )

        return cross_sample_motifs

    def _find_differential_motifs(
        self, control_seqs: List[str], diseased_seqs: List[str]
    ) -> Dict:
        """Find motifs that differentiate control from diseased samples"""
        if not control_seqs or not diseased_seqs:
            return {"motifs": []}

        # Count k-mers in each group
        control_kmers = defaultdict(int)
        diseased_kmers = defaultdict(int)

        for seq in control_seqs:
            for k in range(3, 6):  # Focus on shorter motifs for differential analysis
                for i in range(len(seq) - k + 1):
                    kmer = seq[i : i + k]
                    control_kmers[kmer] += 1

        for seq in diseased_seqs:
            for k in range(3, 6):
                for i in range(len(seq) - k + 1):
                    kmer = seq[i : i + k]
                    diseased_kmers[kmer] += 1

        # Calculate enrichment
        differential_motifs = []

        all_kmers = set(control_kmers.keys()) | set(diseased_kmers.keys())

        for kmer in all_kmers:
            control_count = control_kmers.get(kmer, 0)
            diseased_count = diseased_kmers.get(kmer, 0)

            # Calculate fold change
            if control_count > 0:
                fold_change = (diseased_count + 1) / (control_count + 1)
            else:
                fold_change = diseased_count + 1

            # Only include motifs with significant difference
            if abs(np.log2(fold_change)) > 1 and (control_count + diseased_count) >= 5:
                differential_motifs.append(
                    {
                        "motif": kmer,
                        "control_count": control_count,
                        "diseased_count": diseased_count,
                        "fold_change": fold_change,
                        "log2_fold_change": np.log2(fold_change),
                        "direction": "diseased"
                        if diseased_count > control_count
                        else "control",
                    }
                )

        # Sort by absolute fold change
        differential_motifs.sort(key=lambda x: abs(x["log2_fold_change"]), reverse=True)

        return {
            "motifs": differential_motifs[:50],  # Top 50
            "total_motifs_tested": len(all_kmers),
        }

    def _create_motif_feature_matrix(
        self, motif_features: Dict, processed_data: Dict
    ) -> pd.DataFrame:
        """Create feature matrix based on motif analysis"""
        features_list = []

        for sample_name, sample_info in processed_data.items():
            if sample_name == "combined":
                continue

            # Get sample motifs
            if sample_name in motif_features:
                sample_motifs = motif_features[sample_name]
            else:
                sample_motifs = {}

            # Create feature row
            feature_row = {
                "sample_name": sample_name,
                "condition": sample_info["condition"],
            }

            # K-mer motif features
            if "kmer_motifs" in sample_motifs:
                kmer_data = sample_motifs["kmer_motifs"]
                feature_row["n_kmer_motifs"] = len(kmer_data.get("motifs", []))
                feature_row["unique_kmer_ratio"] = kmer_data.get(
                    "unique_kmers", 0
                ) / max(kmer_data.get("total_kmers", 1), 1)

                # Add top k-mer frequencies
                top_kmers = kmer_data.get("motifs", [])[:5]
                for i, kmer_info in enumerate(top_kmers):
                    feature_row[f"top_kmer_{i + 1}_freq"] = kmer_info["count"]

            # Position motif features
            if "position_motifs" in sample_motifs:
                pos_data = sample_motifs["position_motifs"]
                feature_row["n_position_motifs"] = len(pos_data.get("motifs", []))
                feature_row["max_cdr3_length"] = pos_data.get("max_length", 0)

                # Add position diversity
                pos_motifs = pos_data.get("motifs", [])
                if pos_motifs:
                    positions = [m["position"] for m in pos_motifs]
                    feature_row["position_diversity"] = len(set(positions)) / max(
                        len(positions), 1
                    )

            # Physico-chemical motif features
            if "physicochemical_motifs" in sample_motifs:
                phys_data = sample_motifs["physicochemical_motifs"]
                feature_row["n_physicochemical_motifs"] = len(
                    phys_data.get("motifs", [])
                )
                feature_row["unique_physicochemical_patterns"] = phys_data.get(
                    "unique_patterns", 0
                )

            # Topic modeling features
            if "topic_motifs" in sample_motifs:
                topic_data = sample_motifs["topic_motifs"]
                feature_row["n_topic_motifs"] = len(topic_data.get("motifs", []))
                feature_row["vocabulary_size"] = topic_data.get("vocabulary_size", 0)

                # Add average coherence
                motifs = topic_data.get("motifs", [])
                if motifs:
                    coherences = [m.get("coherence", 0) for m in motifs]
                    feature_row["avg_topic_coherence"] = np.mean(coherences)

            # Consensus motif features
            if "consensus_motifs" in sample_motifs:
                consensus_data = sample_motifs["consensus_motifs"]
                feature_row["n_consensus_motifs"] = len(
                    consensus_data.get("motifs", [])
                )
                feature_row["n_clusters"] = consensus_data.get("n_clusters", 0)

            # Cross-sample motif features
            if "cross_sample" in motif_features:
                cross_data = motif_features["cross_sample"]

                # Differential motif features
                if "differential_motifs" in cross_data:
                    diff_motifs = cross_data["differential_motifs"]["motifs"]

                    # Count sample-specific differential motifs
                    sample_diff_motifs = [
                        m
                        for m in diff_motifs
                        if self._motif_present_in_sample(
                            m["motif"], sample_info["data"]
                        )
                    ]

                    feature_row["n_differential_motifs"] = len(sample_diff_motifs)
                    feature_row["diseased_enriched_motifs"] = len(
                        [m for m in sample_diff_motifs if m["direction"] == "diseased"]
                    )
                    feature_row["control_enriched_motifs"] = len(
                        [m for m in sample_diff_motifs if m["direction"] == "control"]
                    )

            features_list.append(feature_row)

        # Create DataFrame
        motif_matrix = pd.DataFrame(features_list)
        motif_matrix = motif_matrix.fillna(0)  # Handle missing values

        return motif_matrix

    def _motif_present_in_sample(self, motif: str, sample_df: pd.DataFrame) -> bool:
        """Check if a motif is present in any CDR3 sequence of the sample"""
        for cdr3 in sample_df["cdr3_amino_acid"]:
            if motif in cdr3:
                return True
        return False

    def save_motifs(self, motif_features: Dict, output_dir: str):
        """Save motif features to files"""
        import json
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save motif matrix
        if "motif_matrix" in motif_features:
            matrix_path = output_path / "motif_features.csv"
            motif_features["motif_matrix"].to_csv(matrix_path, index=False)

        # Save detailed motifs as JSON
        motifs_to_save = {
            k: v for k, v in motif_features.items() if k != "motif_matrix"
        }

        # Convert numpy arrays to lists for JSON serialization
        self._convert_numpy_to_list(motifs_to_save)

        json_path = output_path / "motif_features_detailed.json"
        with open(json_path, "w") as f:
            json.dump(motifs_to_save, f, indent=2)

        # Save differential motifs separately if available
        if (
            "cross_sample" in motif_features
            and "differential_motifs" in motif_features["cross_sample"]
        ):
            diff_path = output_path / "differential_motifs.csv"
            diff_motifs = pd.DataFrame(
                motif_features["cross_sample"]["differential_motifs"]["motifs"]
            )
            diff_motifs.to_csv(diff_path, index=False)

        self.logger.info(f"Motif features saved to {output_dir}")

    def _convert_numpy_to_list(self, obj):
        """Recursively convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                self._convert_numpy_to_list(value)
        elif isinstance(obj, list):
            for item in obj:
                self._convert_numpy_to_list(item)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()

    def load_motifs(self, input_dir: str) -> Dict:
        """Load previously saved motif features"""
        import json
        from pathlib import Path

        input_path = Path(input_dir)
        motif_features = {}

        # Load motif matrix
        matrix_path = input_path / "motif_features.csv"
        if matrix_path.exists():
            motif_features["motif_matrix"] = pd.read_csv(matrix_path)

        # Load detailed motifs
        json_path = input_path / "motif_features_detailed.json"
        if json_path.exists():
            with open(json_path, "r") as f:
                detailed_motifs = json.load(f)
            motif_features.update(detailed_motifs)

        return motif_features
