"""
TCR-seq Visualization Module
Creates comprehensive visualizations for TCR-seq analysis results
"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Set style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class TCRVisualizer:
    """Creates visualizations for TCR-seq analysis"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Visualization settings
        self.figsize = config.get("visualization", {}).get("figsize", (10, 6))
        self.dpi = config.get("visualization", {}).get("dpi", 300)
        self.color_palette = config.get("visualization", {}).get(
            "color_palette", "Set2"
        )

        # Create output directories
        self.plot_dir = Path(config.get("output_dir", "results")) / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_visualizations(
        self, combined_features: pd.DataFrame, ml_results: Dict, output_dir: str
    ):
        """Generate all visualizations"""
        self.logger.info("Generating visualizations")

        # Update plot directory
        self.plot_dir = Path(output_dir) / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        # Data overview visualizations
        self._plot_data_overview(combined_features)

        # Clone analysis visualizations
        self._plot_clone_analysis(combined_features)

        # Motif analysis visualizations
        self._plot_motif_analysis(combined_features)

        # Machine learning visualizations
        self._plot_ml_results(ml_results)

        # Feature importance visualizations
        if "feature_importance" in ml_results:
            self._plot_feature_importance(ml_results["feature_importance"])

        # Interactive dashboards
        self._create_interactive_dashboard(combined_features, ml_results)

        self.logger.info(f"Visualizations saved to {self.plot_dir}")

    def _plot_data_overview(self, df: pd.DataFrame):
        """Create data overview plots"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("TCR-seq Data Overview", fontsize=16, fontweight="bold")

        # Sample distribution by condition
        condition_counts = df["condition"].value_counts()
        axes[0, 0].pie(
            condition_counts.values, labels=condition_counts.index, autopct="%1.1f%%"
        )
        axes[0, 0].set_title("Sample Distribution")

        # Clone count distribution
        if "clone_n_clones" in df.columns:
            sns.histplot(data=df, x="clone_n_clones", hue="condition", ax=axes[0, 1])
            axes[0, 1].set_title("Clone Count Distribution")

        # Shannon diversity
        if "shannon_diversity" in df.columns:
            sns.boxplot(data=df, x="condition", y="shannon_diversity", ax=axes[0, 2])
            axes[0, 2].set_title("Shannon Diversity by Condition")

        # Clonality
        if "clonality_clonality" in df.columns:
            sns.boxplot(data=df, x="condition", y="clonality_clonality", ax=axes[1, 0])
            axes[1, 0].set_title("Clonality by Condition")

        # V gene diversity
        if "v_diversity" in df.columns:
            sns.boxplot(data=df, x="condition", y="v_diversity", ax=axes[1, 1])
            axes[1, 1].set_title("V Gene Diversity by Condition")

        # CDR3 length distribution
        if "mean_cdr3_length" in df.columns:
            sns.boxplot(data=df, x="condition", y="mean_cdr3_length", ax=axes[1, 2])
            axes[1, 2].set_title("Mean CDR3 Length by Condition")

        plt.tight_layout()
        plt.savefig(
            self.plot_dir / "data_overview.png", dpi=self.dpi, bbox_inches="tight"
        )
        plt.close()

    def _plot_clone_analysis(self, df: pd.DataFrame):
        """Create clone analysis visualizations"""
        # Clone statistics
        clone_cols = [
            col for col in df.columns if "clone_" in col and col != "condition"
        ]

        if clone_cols:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle("Clone Analysis", fontsize=16, fontweight="bold")

            # Clone size distribution
            if "clone_size_mean" in df.columns:
                sns.boxplot(data=df, x="condition", y="clone_size_mean", ax=axes[0, 0])
                axes[0, 0].set_title("Mean Clone Size")

            # Clone coverage
            if "clone_clone_coverage" in df.columns:
                sns.boxplot(
                    data=df, x="condition", y="clone_clone_coverage", ax=axes[0, 1]
                )
                axes[0, 1].set_title("Clone Coverage")

            # Gini coefficient
            if "size_gini_coefficient" in df.columns:
                sns.boxplot(
                    data=df, x="condition", y="size_gini_coefficient", ax=axes[1, 0]
                )
                axes[1, 0].set_title("Clone Size Gini Coefficient")

            # Singleton rate
            if "clone_singleton_rate" in df.columns:
                sns.boxplot(
                    data=df, x="condition", y="clone_singleton_rate", ax=axes[1, 1]
                )
                axes[1, 1].set_title("Singleton Rate")

            plt.tight_layout()
            plt.savefig(
                self.plot_dir / "clone_analysis.png", dpi=self.dpi, bbox_inches="tight"
            )
            plt.close()

    def _plot_motif_analysis(self, df: pd.DataFrame):
        """Create motif analysis visualizations"""
        # Motif statistics
        motif_cols = [
            col
            for col in df.columns
            if any(x in col for x in ["motif", "kmer", "position"])
        ]

        if motif_cols:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle("Motif Analysis", fontsize=16, fontweight="bold")

            # Number of k-mer motifs
            if "n_kmer_motifs" in df.columns:
                sns.boxplot(data=df, x="condition", y="n_kmer_motifs", ax=axes[0, 0])
                axes[0, 0].set_title("Number of K-mer Motifs")

            # Unique k-mer ratio
            if "unique_kmer_ratio" in df.columns:
                sns.boxplot(
                    data=df, x="condition", y="unique_kmer_ratio", ax=axes[0, 1]
                )
                axes[0, 1].set_title("Unique K-mer Ratio")

            # Position motifs
            if "n_position_motifs" in df.columns:
                sns.boxplot(
                    data=df, x="condition", y="n_position_motifs", ax=axes[1, 0]
                )
                axes[1, 0].set_title("Number of Position Motifs")

            # Differential motifs
            if "n_differential_motifs" in df.columns:
                sns.boxplot(
                    data=df, x="condition", y="n_differential_motifs", ax=axes[1, 1]
                )
                axes[1, 1].set_title("Number of Differential Motifs")

            plt.tight_layout()
            plt.savefig(
                self.plot_dir / "motif_analysis.png", dpi=self.dpi, bbox_inches="tight"
            )
            plt.close()

    def _plot_ml_results(self, ml_results: Dict):
        """Create machine learning results visualizations"""
        # Model comparison
        model_names = []
        test_accuracies = []
        test_f1_scores = []
        cv_means = []

        for model_name, results in ml_results.items():
            if model_name in ["best_model", "feature_importance"] or "error" in results:
                continue

            model_names.append(model_name)
            test_accuracies.append(results.get("test_accuracy", 0))
            test_f1_scores.append(results.get("test_f1", 0))
            cv_means.append(results.get("cv_mean", 0))

        if model_names:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(
                "Machine Learning Model Comparison", fontsize=16, fontweight="bold"
            )

            # Test accuracy (filter out None values)
            valid_indices = [
                i for i, acc in enumerate(test_accuracies) if acc is not None
            ]
            if valid_indices:
                valid_model_names = [model_names[i] for i in valid_indices]
                valid_accuracies = [test_accuracies[i] for i in valid_indices]
                axes[0].bar(valid_model_names, valid_accuracies)
            axes[0].set_title("Test Accuracy")
            axes[0].set_ylabel("Accuracy")
            axes[0].tick_params(axis="x", rotation=45)

            # Test F1 score (filter out None values)
            valid_f1_indices = [
                i for i, f1 in enumerate(test_f1_scores) if f1 is not None
            ]
            if valid_f1_indices:
                valid_f1_model_names = [model_names[i] for i in valid_f1_indices]
                valid_f1_scores = [test_f1_scores[i] for i in valid_f1_indices]
                axes[1].bar(valid_f1_model_names, valid_f1_scores)
            axes[1].set_title("Test F1 Score")
            axes[1].set_ylabel("F1 Score")
            axes[1].tick_params(axis="x", rotation=45)

            # Cross-validation mean (filter out None values)
            valid_cv_indices = [i for i, cv in enumerate(cv_means) if cv is not None]
            if valid_cv_indices:
                valid_cv_model_names = [model_names[i] for i in valid_cv_indices]
                valid_cv_means = [cv_means[i] for i in valid_cv_indices]
                axes[2].bar(valid_cv_model_names, valid_cv_means)
            axes[2].set_title("Cross-Validation Mean")
            axes[2].set_ylabel("CV Score")
            axes[2].tick_params(axis="x", rotation=45)

            plt.tight_layout()
            plt.savefig(
                self.plot_dir / "ml_model_comparison.png",
                dpi=self.dpi,
                bbox_inches="tight",
            )
            plt.close()

        # Best model detailed results
        if "best_model" in ml_results:
            self._plot_best_model_results(ml_results["best_model"])

    def _plot_best_model_results(self, best_model_results: Dict):
        """Create detailed plots for the best model"""
        if best_model_results is None:
            self.logger.warning("No best model results to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Best Model Performance", fontsize=16, fontweight="bold")

        # Confusion matrix
        if "confusion_matrix" in best_model_results:
            cm = np.array(best_model_results["confusion_matrix"])
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                ax=axes[0, 0],
                xticklabels=["Control", "Diseased"],
                yticklabels=["Control", "Diseased"],
            )
            axes[0, 0].set_title("Confusion Matrix")
            axes[0, 0].set_xlabel("Predicted")
            axes[0, 0].set_ylabel("Actual")

        # ROC curve
        if "roc_curve" in best_model_results:
            roc_data = best_model_results["roc_curve"]
            axes[0, 1].plot(
                roc_data["fpr"],
                roc_data["tpr"],
                label=f"AUC = {best_model_results.get('test_auc', 0):.3f}",
            )
            axes[0, 1].plot([0, 1], [0, 1], "k--")
            axes[0, 1].set_xlabel("False Positive Rate")
            axes[0, 1].set_ylabel("True Positive Rate")
            axes[0, 1].set_title("ROC Curve")
            axes[0, 1].legend()

        # Cross-validation scores
        if "cv_scores" in best_model_results:
            cv_scores = best_model_results["cv_scores"]
            axes[1, 0].boxplot(cv_scores)
            axes[1, 0].set_title("Cross-Validation Scores")
            axes[1, 0].set_ylabel("F1 Score")
            axes[1, 0].axhline(
                y=np.mean(cv_scores),
                color="r",
                linestyle="--",
                label=f"Mean: {np.mean(cv_scores):.3f}",
            )
            axes[1, 0].legend()

        # Performance metrics
        metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1"]
        metric_values = [best_model_results.get(m, 0) for m in metrics]

        axes[1, 1].bar(metrics, metric_values)
        axes[1, 1].set_title("Performance Metrics")
        axes[1, 1].set_ylabel("Score")
        axes[1, 1].tick_params(axis="x", rotation=45)
        axes[1, 1].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(
            self.plot_dir / "best_model_results.png", dpi=self.dpi, bbox_inches="tight"
        )
        plt.close()

    def _plot_feature_importance(self, feature_importance: Dict):
        """Create feature importance visualizations"""
        if "importances" not in feature_importance:
            return

        # Get top features
        top_features = feature_importance["top_features"][:20]
        features = [f[0] for f in top_features]
        importances = [f[1] for f in top_features]

        # Create horizontal bar plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel("Importance")
        plt.title("Top 20 Feature Importance")
        plt.gca().invert_yaxis()

        plt.tight_layout()
        plt.savefig(
            self.plot_dir / "feature_importance.png", dpi=self.dpi, bbox_inches="tight"
        )
        plt.close()

        # Create feature importance heatmap if multiple models
        if len(feature_importance.get("model_importances", {})) > 1:
            self._plot_multi_model_feature_importance(feature_importance)

    def _plot_multi_model_feature_importance(self, feature_importance: Dict):
        """Create feature importance comparison across models"""
        model_importances = feature_importance.get("model_importances", {})

        if not model_importances:
            return

        # Create DataFrame for heatmap
        all_features = set()
        for model_data in model_importances.values():
            all_features.update([f[0] for f in model_data])

        # Create matrix
        feature_matrix = pd.DataFrame(
            0, index=list(all_features), columns=model_importances.keys()
        )

        for model_name, model_data in model_importances.items():
            for feature, importance in model_data:
                feature_matrix.loc[feature, model_name] = importance

        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(feature_matrix, cmap="viridis", annot=False)
        plt.title("Feature Importance Across Models")
        plt.tight_layout()
        plt.savefig(
            self.plot_dir / "multi_model_feature_importance.png",
            dpi=self.dpi,
            bbox_inches="tight",
        )
        plt.close()

    def _create_interactive_dashboard(self, df: pd.DataFrame, ml_results: Dict):
        """Create interactive dashboard using Plotly"""
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Sample Distribution",
                "Diversity Metrics",
                "Model Performance",
                "Feature Overview",
            ),
            specs=[
                [{"type": "pie"}, {"type": "box"}],
                [{"type": "bar"}, {"type": "scatter"}],
            ],
        )

        # Sample distribution pie chart
        condition_counts = df["condition"].value_counts()
        fig.add_trace(
            go.Pie(
                labels=condition_counts.index,
                values=condition_counts.values,
                name="Sample Distribution",
            ),
            row=1,
            col=1,
        )

        # Diversity metrics box plot
        if "shannon_diversity" in df.columns:
            fig.add_trace(
                go.Box(
                    x=df["condition"],
                    y=df["shannon_diversity"],
                    name="Shannon Diversity",
                ),
                row=1,
                col=2,
            )

        # Model performance bar chart
        model_names = []
        test_accuracies = []

        for model_name, results in ml_results.items():
            if (
                model_name not in ["best_model", "feature_importance"]
                and "error" not in results
            ):
                model_names.append(model_name)
                test_accuracies.append(results.get("test_accuracy", 0))

        if model_names:
            fig.add_trace(
                go.Bar(x=model_names, y=test_accuracies, name="Test Accuracy"),
                row=2,
                col=1,
            )

        # Feature overview scatter plot
        if "clone_n_clones" in df.columns and "shannon_diversity" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["clone_n_clones"],
                    y=df["shannon_diversity"],
                    mode="markers",
                    text=df["condition"],
                    name="Clones vs Diversity",
                ),
                row=2,
                col=2,
            )

        # Update layout
        fig.update_layout(
            title_text="TCR-seq Analysis Interactive Dashboard",
            showlegend=True,
            height=800,
        )

        # Save interactive plot
        pyo.plot(
            fig,
            filename=str(self.plot_dir / "interactive_dashboard.html"),
            auto_open=False,
        )

    def create_correlation_heatmap(self, df: pd.DataFrame):
        """Create correlation heatmap of features"""
        # Select only numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ["condition"]]

        if len(numeric_cols) > 1:
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr()

            # Create heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0)
            plt.title("Feature Correlation Heatmap")
            plt.tight_layout()
            plt.savefig(
                self.plot_dir / "correlation_heatmap.png",
                dpi=self.dpi,
                bbox_inches="tight",
            )
            plt.close()

    def create_pca_plot(self, df: pd.DataFrame):
        """Create PCA visualization of feature space"""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # Select numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ["condition"]]

        if len(numeric_cols) > 2:
            # Prepare data
            X = df[numeric_cols].fillna(0)
            X_scaled = StandardScaler().fit_transform(X)

            # Apply PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            # Create plot
            plt.figure(figsize=(10, 8))
            for condition in df["condition"].unique():
                mask = df["condition"] == condition
                plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=condition, alpha=0.7)

            plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
            plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
            plt.title("PCA of TCR-seq Features")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                self.plot_dir / "pca_plot.png", dpi=self.dpi, bbox_inches="tight"
            )
            plt.close()

    def save_summary_report(self, df: pd.DataFrame, ml_results: Dict, output_dir: str):
        """Save a text summary report"""
        report_path = Path(output_dir) / "analysis_summary.txt"

        with open(report_path, "w") as f:
            f.write("TCR-seq Analysis Summary Report\n")
            f.write("=" * 50 + "\n\n")

            # Data overview
            f.write("DATA OVERVIEW\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total samples: {len(df)}\n")
            f.write(f"Control samples: {len(df[df['condition'] == 'control'])}\n")
            f.write(f"Diseased samples: {len(df[df['condition'] == 'diseased'])}\n")
            f.write(f"Features extracted: {len(df.columns) - 2}\n\n")

            # ML results
            f.write("MACHINE LEARNING RESULTS\n")
            f.write("-" * 30 + "\n")

            if "best_model" in ml_results:
                best = ml_results["best_model"]
                f.write(f"Best model: {best.get('model_name', 'Unknown')}\n")
                f.write(f"Best score: {best.get('best_score', 0):.3f}\n")
                f.write(f"Test accuracy: {best.get('test_accuracy', 0):.3f}\n")
                f.write(f"Test F1 score: {best.get('test_f1', 0):.3f}\n")
                f.write(f"Test AUC: {best.get('test_auc', 0):.3f}\n\n")

            # Feature importance
            if "feature_importance" in ml_results:
                fi = ml_results["feature_importance"]
                if "top_features" in fi:
                    f.write("TOP FEATURES\n")
                    f.write("-" * 15 + "\n")
                    for i, (feature, importance) in enumerate(fi["top_features"][:10]):
                        f.write(f"{i + 1}. {feature}: {importance:.3f}\n")

        self.logger.info(f"Summary report saved to {report_path}")
