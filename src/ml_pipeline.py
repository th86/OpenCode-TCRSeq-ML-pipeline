"""
Machine Learning Pipeline for TCR-seq Classification
Trains and evaluates models to classify control vs diseased samples
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib
import json
from pathlib import Path


class MLPipeline:
    """Machine learning pipeline for TCR-seq classification"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # ML configuration
        self.test_size = config.get("ml", {}).get("test_size", 0.2)
        self.cv_folds = config.get("ml", {}).get("cv_folds", 5)
        self.random_state = config.get("ml", {}).get("random_state", 42)

        # Models to train
        self.models = {
            "random_forest": RandomForestClassifier(random_state=self.random_state),
            "gradient_boosting": GradientBoostingClassifier(
                random_state=self.random_state
            ),
            "svm": SVC(random_state=self.random_state, probability=True),
            "logistic_regression": LogisticRegression(
                random_state=self.random_state, max_iter=1000
            ),
            "knn": KNeighborsClassifier(),
            "decision_tree": DecisionTreeClassifier(random_state=self.random_state),
            "naive_bayes": GaussianNB(),
            "ada_boost": AdaBoostClassifier(random_state=self.random_state),
            "ridge": RidgeClassifier(random_state=self.random_state),
            "mlp": MLPClassifier(random_state=self.random_state, max_iter=1000),
        }

        # Feature scaling options
        self.scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
            "none": None,
        }

        # Feature selection options
        self.feature_selectors = {
            "kbest_f": SelectKBest(f_classif),
            "kbest_mi": SelectKBest(mutual_info_classif),
            "rfe": RFE(
                estimator=RandomForestClassifier(random_state=self.random_state)
            ),
            "none": None,
        }

    def combine_features(
        self, clone_features: Dict, motif_features: Dict, processed_data: Dict
    ) -> pd.DataFrame:
        """Combine clone and motif features into a single feature matrix"""
        self.logger.info("Combining features for machine learning")

        # Get feature matrices
        clone_matrix = clone_features.get("feature_matrix", pd.DataFrame())
        motif_matrix = motif_features.get("motif_matrix", pd.DataFrame())

        # Combine matrices
        if not clone_matrix.empty and not motif_matrix.empty:
            # Merge on sample_name and condition
            combined = clone_matrix.merge(
                motif_matrix,
                on=["sample_name", "condition"],
                how="outer",
                suffixes=("_clone", "_motif"),
            )
        elif not clone_matrix.empty:
            combined = clone_matrix
        elif not motif_matrix.empty:
            combined = motif_matrix
        else:
            raise ValueError("No feature matrices available")

        # Add additional features from processed data
        additional_features = self._extract_additional_features(processed_data)

        if not additional_features.empty:
            combined = combined.merge(
                additional_features, on=["sample_name", "condition"], how="left"
            )

        # Clean and prepare features
        combined = self._prepare_features(combined)

        self.logger.info(f"Combined feature matrix shape: {combined.shape}")
        return combined

    def _extract_additional_features(self, processed_data: Dict) -> pd.DataFrame:
        """Extract additional features from processed data"""
        features_list = []

        for sample_name, sample_info in processed_data.items():
            if sample_name == "combined":
                continue

            df = sample_info["data"]

            feature_row = {
                "sample_name": sample_name,
                "condition": sample_info["condition"],
            }

            # Basic statistics
            feature_row["total_reads"] = sample_info["total_reads"]
            feature_row["n_clones"] = sample_info["n_clones"]
            feature_row["reads_per_clone"] = sample_info["total_reads"] / max(
                sample_info["n_clones"], 1
            )

            # Diversity metrics (already calculated in preprocessing)
            if not df.empty:
                feature_row["shannon_diversity"] = df["shannon_diversity"].iloc[0]
                feature_row["clonality"] = df["clonality"].iloc[0]
                feature_row["top_clone_frequency"] = df["top_clone_frequency"].iloc[0]

            # V/J gene diversity
            feature_row["v_gene_diversity"] = df["v_gene"].nunique()
            feature_row["j_gene_diversity"] = df["j_gene"].nunique()
            feature_row["vj_pair_diversity"] = len(df.groupby(["v_gene", "j_gene"]))

            # CDR3 length statistics
            cdr3_lengths = df["cdr3_length"]
            feature_row["mean_cdr3_length"] = cdr3_lengths.mean()
            feature_row["std_cdr3_length"] = cdr3_lengths.std()
            feature_row["min_cdr3_length"] = cdr3_lengths.min()
            feature_row["max_cdr3_length"] = cdr3_lengths.max()

            features_list.append(feature_row)

        return pd.DataFrame(features_list)

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare features for ML"""
        # Remove non-feature columns
        feature_cols = [
            col for col in df.columns if col not in ["sample_name", "condition"]
        ]

        # Handle missing values
        df[feature_cols] = df[feature_cols].fillna(0)

        # Handle infinite values
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0)

        # Remove constant features
        constant_features = []
        for col in feature_cols:
            if df[col].nunique() <= 1:
                constant_features.append(col)

        if constant_features:
            self.logger.info(f"Removing {len(constant_features)} constant features")
            df = df.drop(columns=constant_features)

        # Remove highly correlated features
        if self.config.get("ml", {}).get("remove_correlated", True):
            df = self._remove_correlated_features(df)

        return df

    def _remove_correlated_features(
        self, df: pd.DataFrame, threshold: float = 0.95
    ) -> pd.DataFrame:
        """Remove highly correlated features"""
        feature_cols = [
            col for col in df.columns if col not in ["sample_name", "condition"]
        ]

        if len(feature_cols) < 2:
            return df

        # Calculate correlation matrix
        corr_matrix = df[feature_cols].corr().abs()

        # Find highly correlated features
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [
            column
            for column in upper_triangle.columns
            if any(upper_triangle[column] > threshold)
        ]

        if to_drop:
            self.logger.info(f"Removing {len(to_drop)} highly correlated features")
            df = df.drop(columns=to_drop)

        return df

    def train_models(self, feature_matrix: pd.DataFrame) -> Dict:
        """Train and evaluate multiple models"""
        self.logger.info("Starting machine learning pipeline")

        # Prepare data
        X, y = self._prepare_data(feature_matrix)

        # Split data - handle case with too few samples for stratification
        if len(np.unique(y)) < 2 or len(y) < 4:
            # Not enough samples for proper train/test split
            self.logger.warning(
                "Not enough samples for train/test split, using all data for training"
            )
            X_train, X_test, y_train, y_test = X, X[:0], y, y[:0]  # Empty test set
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y,
            )

        results = {}

        # Train each model
        for model_name, model in self.models.items():
            self.logger.info(f"Training {model_name}")

            try:
                # Create pipeline
                pipeline = self._create_pipeline(model)

                # Train model
                pipeline.fit(X_train, y_train)

                # Evaluate model
                model_results = self._evaluate_model(
                    pipeline, X_train, X_test, y_train, y_test, model_name
                )

                results[model_name] = model_results

            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {str(e)}")
                results[model_name] = {"error": str(e)}

        # Find best model
        best_model = self._find_best_model(results)
        results["best_model"] = best_model

        # Feature importance analysis
        if best_model and "model" in best_model:
            feature_importance = self._analyze_feature_importance(
                best_model["model"], X_train.columns
            )
            results["feature_importance"] = feature_importance

        self.logger.info("Machine learning pipeline completed")
        return results

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for ML"""
        # Separate features and target
        feature_cols = [
            col for col in df.columns if col not in ["sample_name", "condition"]
        ]

        X = df[feature_cols]
        y = df["condition"].map({"control": 0, "diseased": 1})

        return X, y

    def _create_pipeline(self, model) -> Pipeline:
        """Create ML pipeline with preprocessing"""
        steps = []

        # Feature scaling
        scaler_name = self.config.get("ml", {}).get("scaler", "standard")
        if scaler_name in self.scalers and self.scalers[scaler_name] is not None:
            steps.append(("scaler", self.scalers[scaler_name]))

        # Feature selection
        selector_name = self.config.get("ml", {}).get("feature_selector", "none")
        if (
            selector_name in self.feature_selectors
            and self.feature_selectors[selector_name] is not None
        ):
            steps.append(("selector", self.feature_selectors[selector_name]))

        # Model
        steps.append(("model", model))

        return Pipeline(steps)

    def _evaluate_model(
        self,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        model_name: str,
    ) -> Dict:
        """Evaluate model performance"""
        # Predictions
        y_train_pred = pipeline.predict(X_train)

        # Handle empty test set case
        if len(y_test) == 0:
            self.logger.warning("Empty test set - skipping test metrics")
            y_test_pred = []
            y_test_proba = None
        else:
            y_test_pred = pipeline.predict(X_test)
            y_test_proba = (
                pipeline.predict_proba(X_test)[:, 1]
                if hasattr(pipeline, "predict_proba")
                else None
            )

        # Metrics
        metrics = {
            "model_name": model_name,
            "model": pipeline,
            # Training metrics
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "train_f1": f1_score(y_train, y_train_pred)
            if len(np.unique(y_train_pred)) > 1
            else 0.0,
            # Test metrics (only if test set is not empty)
            "test_accuracy": accuracy_score(y_test, y_test_pred)
            if len(y_test) > 0
            else None,
            "test_precision": precision_score(y_test, y_test_pred)
            if len(y_test) > 0
            else None,
            "test_recall": recall_score(y_test, y_test_pred)
            if len(y_test) > 0
            else None,
            "test_f1": f1_score(y_test, y_test_pred) if len(y_test) > 0 else None,
            "test_auc": roc_auc_score(y_test, y_test_proba)
            if y_test_proba is not None and len(y_test) > 0
            else None,
        }

        # Cross-validation (only if enough samples)
        if len(y_train) >= self.cv_folds:
            try:
                cv_scores = cross_val_score(
                    pipeline, X_train, y_train, cv=self.cv_folds
                ).tolist()
                metrics["cv_scores"] = cv_scores
                metrics["cv_mean"] = np.mean(cv_scores)
                metrics["cv_std"] = np.std(cv_scores)
            except Exception as e:
                self.logger.warning(f"Cross-validation failed: {str(e)}")
                metrics["cv_scores"] = []
                metrics["cv_mean"] = None
                metrics["cv_std"] = None
        else:
            self.logger.warning("Not enough samples for cross-validation")
            metrics["cv_scores"] = []
            metrics["cv_mean"] = None
            metrics["cv_std"] = None

        # CV statistics
        if "cv_scores" in metrics:
            metrics["cv_mean"] = np.mean(metrics["cv_scores"])
            metrics["cv_std"] = np.std(metrics["cv_scores"])

        # Classification report and confusion matrix only if test set is not empty and valid
        if len(y_test) > 0 and len(y_test_pred) > 0:
            try:
                metrics["classification_report"] = classification_report(
                    y_test, y_test_pred, output_dict=True
                )

                # Confusion matrix
                metrics["confusion_matrix"] = confusion_matrix(
                    y_test, y_test_pred
                ).tolist()

                # ROC curve data
                if y_test_proba is not None:
                    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
                    metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
            except Exception as e:
                self.logger.warning(
                    f"Could not generate classification report: {str(e)}"
                )
                metrics["classification_report"] = {}
                metrics["confusion_matrix"] = []
        else:
            metrics["classification_report"] = {}
            metrics["confusion_matrix"] = []

        # Feature names
        metrics["feature_names"] = X_train.columns.tolist()

        return metrics

    def _find_best_model(self, results: Dict) -> Optional[Dict]:
        """Find the best performing model"""
        best_model = None
        best_score = -1

        for model_name, model_results in results.items():
            if "error" in model_results:
                continue

            # Use cross-validation mean as primary metric
            score = model_results.get("cv_mean", model_results.get("test_f1", -1))

            if score > best_score:
                best_score = score
                best_model = model_results
                best_model["best_score"] = best_score
                best_model["best_metric"] = (
                    "cv_mean" if "cv_mean" in model_results else "test_f1"
                )

        return best_model

    def _analyze_feature_importance(
        self, pipeline: Pipeline, feature_names: List[str]
    ) -> Dict:
        """Analyze feature importance from the best model"""
        importance_data = {}

        # Get the final model from pipeline
        if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
            model = pipeline.named_steps["model"]
        else:
            model = pipeline

        # Different methods for different models
        if hasattr(model, "feature_importances_"):
            # Tree-based models
            importances = model.feature_importances_
            importance_data["method"] = "feature_importances_"

        elif hasattr(model, "coef_"):
            # Linear models
            importances = np.abs(model.coef_[0])
            importance_data["method"] = "coef_"

        else:
            self.logger.warning(
                "Cannot extract feature importance from this model type"
            )
            return {"method": "none", "importances": []}

        # Create feature importance ranking
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        importance_data["importances"] = feature_importance
        importance_data["top_features"] = feature_importance[:20]  # Top 20 features

        return importance_data

    def hyperparameter_tuning(
        self, feature_matrix: pd.DataFrame, model_name: str = "random_forest"
    ) -> Dict:
        """Perform hyperparameter tuning for a specific model"""
        self.logger.info(f"Performing hyperparameter tuning for {model_name}")

        # Prepare data
        X, y = self._prepare_data(feature_matrix)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Define parameter grids
        param_grids = {
            "random_forest": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
            "gradient_boosting": {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.01, 0.1, 0.2],
                "model__max_depth": [3, 5, 7],
            },
            "svm": {
                "model__C": [0.1, 1, 10],
                "model__kernel": ["rbf", "linear"],
                "model__gamma": ["scale", "auto"],
            },
            "logistic_regression": {
                "model__C": [0.1, 1, 10],
                "model__penalty": ["l1", "l2"],
                "model__solver": ["liblinear", "saga"],
            },
        }

        if model_name not in param_grids:
            raise ValueError(f"No parameter grid defined for {model_name}")

        # Create pipeline
        model = self.models[model_name]
        pipeline = self._create_pipeline(model)

        # Perform grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grids[model_name],
            cv=self.cv_folds,
            scoring="f1",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        # Evaluate best model
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = (
            best_model.predict_proba(X_test)[:, 1]
            if hasattr(best_model, "predict_proba")
            else None
        )

        results = {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "best_model": best_model,
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred),
            "test_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        return results

    def save_results(self, results: Dict, output_dir: str):
        """Save ML results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save best model
        if (
            "best_model" in results
            and results["best_model"] is not None
            and "model" in results["best_model"]
        ):
            model_path = output_path / "best_model.joblib"
            joblib.dump(results["best_model"]["model"], model_path)
            self.logger.info(f"Best model saved to {model_path}")

        # Save results summary
        summary_data = {}
        for model_name, model_results in results.items():
            if model_name in ["best_model", "feature_importance"]:
                continue

            if "error" not in model_results:
                summary_data[model_name] = {
                    "test_accuracy": model_results.get("test_accuracy"),
                    "test_f1": model_results.get("test_f1"),
                    "test_auc": model_results.get("test_auc"),
                    "cv_mean": model_results.get("cv_mean"),
                    "cv_std": model_results.get("cv_std"),
                }

        summary_path = output_path / "ml_results_summary.csv"
        summary_df = pd.DataFrame(summary_data).T
        summary_df.to_csv(summary_path)

        # Save detailed results
        detailed_results = {}
        for key, value in results.items():
            if key == "best_model" and value is not None and "model" in value:
                # Don't save the model object in JSON
                detailed_results[key] = {k: v for k, v in value.items() if k != "model"}
            else:
                detailed_results[key] = value

        # Convert numpy arrays for JSON serialization
        self._convert_numpy_to_list(detailed_results)

        json_path = output_path / "ml_results_detailed.json"
        with open(json_path, "w") as f:
            json.dump(detailed_results, f, indent=2)

        # Save feature importance
        if "feature_importance" in results:
            importance_path = output_path / "feature_importance.csv"
            importance_df = pd.DataFrame(
                results["feature_importance"]["importances"],
                columns=["feature", "importance"],
            )
            importance_df.to_csv(importance_path, index=False)

        self.logger.info(f"ML results saved to {output_dir}")

    def _convert_numpy_to_list(self, obj):
        """Convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                obj[key] = self._convert_numpy_to_list(value)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                obj[i] = self._convert_numpy_to_list(item)
        elif hasattr(obj, "__class__") and "Pipeline" in str(type(obj)):
            return str(obj)  # Convert Pipeline to string representation
        elif hasattr(obj, "__class__") and "sklearn" in str(type(obj)):
            return str(obj)  # Convert sklearn objects to string representation
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        else:
            return obj

        return obj

    def load_model(self, model_path: str) -> Any:
        """Load a trained model"""
        return joblib.load(model_path)

    def predict_sample(self, model: Any, sample_features: pd.DataFrame) -> Dict:
        """Make predictions on new samples"""
        predictions = model.predict(sample_features)
        probabilities = (
            model.predict_proba(sample_features)
            if hasattr(model, "predict_proba")
            else None
        )

        results = {
            "predictions": predictions.tolist(),
            "predicted_conditions": [
                "diseased" if p == 1 else "control" for p in predictions
            ],
            "probabilities": probabilities.tolist()
            if probabilities is not None
            else None,
        }

        return results
