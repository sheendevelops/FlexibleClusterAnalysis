import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Tuple, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MLAnalytics:
    """
    Advanced Machine Learning Analytics for AISapien
    Provides PCA, EDA, Clustering, Dimensionality Reduction, and Model Optimization
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None
        self.kmeans = None
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
    def exploratory_data_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive Exploratory Data Analysis
        """
        try:
            eda_results = {
                "basic_stats": {},
                "data_quality": {},
                "correlations": {},
                "distributions": {},
                "insights": []
            }
            
            # Basic statistics
            eda_results["basic_stats"] = {
                "shape": data.shape,
                "columns": list(data.columns),
                "dtypes": data.dtypes.to_dict(),
                "memory_usage": data.memory_usage().sum(),
                "null_counts": data.isnull().sum().to_dict(),
                "unique_counts": data.nunique().to_dict()
            }
            
            # Data quality assessment
            total_cells = data.shape[0] * data.shape[1]
            null_percentage = (data.isnull().sum().sum() / total_cells) * 100
            
            eda_results["data_quality"] = {
                "completeness_score": 100 - null_percentage,
                "duplicate_rows": data.duplicated().sum(),
                "data_types_consistency": self._check_data_consistency(data),
                "outlier_detection": self._detect_outliers(data)
            }
            
            # Correlations for numeric data
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                correlation_matrix = data[numeric_cols].corr()
                eda_results["correlations"] = {
                    "correlation_matrix": correlation_matrix.to_dict(),
                    "high_correlations": self._find_high_correlations(correlation_matrix),
                    "correlation_insights": self._analyze_correlations(correlation_matrix)
                }
            
            # Distribution analysis
            eda_results["distributions"] = self._analyze_distributions(data)
            
            # Generate insights
            eda_results["insights"] = self._generate_eda_insights(data, eda_results)
            
            return eda_results
            
        except Exception as e:
            return {"error": f"EDA failed: {str(e)}"}
    
    def principal_component_analysis(self, data: pd.DataFrame, n_components: Optional[int] = None) -> Dict[str, Any]:
        """
        Principal Component Analysis with optimal component selection
        """
        try:
            # Prepare numeric data
            numeric_data = data.select_dtypes(include=[np.number]).dropna()
            
            if numeric_data.empty:
                return {"error": "No numeric data available for PCA"}
            
            # Standardize the data
            scaled_data = self.scaler.fit_transform(numeric_data)
            
            # Determine optimal number of components if not specified
            if n_components is None:
                n_components = min(len(numeric_data.columns), len(numeric_data))
            
            # Perform PCA
            self.pca = PCA(n_components=n_components)
            pca_result = self.pca.fit_transform(scaled_data)
            
            # Calculate cumulative explained variance
            cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
            
            # Find optimal number of components (90% variance explained)
            optimal_components = np.argmax(cumulative_variance >= 0.90) + 1
            
            pca_analysis = {
                "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
                "cumulative_variance": cumulative_variance.tolist(),
                "optimal_components": int(optimal_components),
                "total_variance_explained": float(cumulative_variance[-1]),
                "component_loadings": self.pca.components_.tolist(),
                "feature_importance": self._calculate_feature_importance(),
                "dimensionality_reduction": {
                    "original_dimensions": len(numeric_data.columns),
                    "reduced_dimensions": optimal_components,
                    "reduction_ratio": optimal_components / len(numeric_data.columns)
                }
            }
            
            return pca_analysis
            
        except Exception as e:
            return {"error": f"PCA failed: {str(e)}"}
    
    def clustering_analysis(self, data: pd.DataFrame, max_clusters: int = 10) -> Dict[str, Any]:
        """
        Comprehensive clustering analysis with optimal cluster selection
        """
        try:
            # Prepare data
            numeric_data = data.select_dtypes(include=[np.number]).dropna()
            
            if len(numeric_data) < 2:
                return {"error": "Insufficient data for clustering"}
            
            # Standardize data
            scaled_data = self.scaler.fit_transform(numeric_data)
            
            # Elbow method
            elbow_results = self.elbow_method(scaled_data, max_clusters)
            
            # Silhouette analysis
            silhouette_results = self.silhouette_analysis(scaled_data, max_clusters)
            
            # Gap statistic
            gap_results = self.gap_statistic(scaled_data, max_clusters)
            
            # Determine optimal clusters
            optimal_k = self._determine_optimal_clusters(elbow_results, silhouette_results, gap_results)
            
            # Perform final clustering
            self.kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = self.kmeans.fit_predict(scaled_data)
            
            # Cluster analysis
            cluster_analysis = {
                "optimal_clusters": optimal_k,
                "cluster_labels": cluster_labels.tolist(),
                "cluster_centers": self.kmeans.cluster_centers_.tolist(),
                "inertia": float(self.kmeans.inertia_),
                "silhouette_score": float(silhouette_score(scaled_data, cluster_labels)),
                "cluster_sizes": np.bincount(cluster_labels).tolist(),
                "cluster_profiles": self._analyze_cluster_profiles(numeric_data, cluster_labels),
                "elbow_method": elbow_results,
                "silhouette_analysis": silhouette_results,
                "gap_statistic": gap_results
            }
            
            return cluster_analysis
            
        except Exception as e:
            return {"error": f"Clustering analysis failed: {str(e)}"}
    
    def elbow_method(self, data: np.ndarray, max_clusters: int = 10) -> Dict[str, Any]:
        """
        Elbow method for optimal cluster selection
        """
        try:
            k_range = range(1, min(max_clusters + 1, len(data)))
            inertias = []
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(data)
                inertias.append(kmeans.inertia_)
            
            # Calculate elbow point
            elbow_point = self._find_elbow_point(list(k_range), inertias)
            
            return {
                "k_values": list(k_range),
                "inertias": inertias,
                "elbow_point": elbow_point,
                "optimal_k_elbow": elbow_point
            }
            
        except Exception as e:
            return {"error": f"Elbow method failed: {str(e)}"}
    
    def silhouette_analysis(self, data: np.ndarray, max_clusters: int = 10) -> Dict[str, Any]:
        """
        Silhouette analysis for cluster validation
        """
        try:
            k_range = range(2, min(max_clusters + 1, len(data)))
            silhouette_scores = []
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(data)
                score = silhouette_score(data, cluster_labels)
                silhouette_scores.append(score)
            
            # Find optimal k
            optimal_k = k_range[np.argmax(silhouette_scores)]
            
            return {
                "k_values": list(k_range),
                "silhouette_scores": silhouette_scores,
                "optimal_k_silhouette": optimal_k,
                "max_silhouette_score": max(silhouette_scores)
            }
            
        except Exception as e:
            return {"error": f"Silhouette analysis failed: {str(e)}"}
    
    def gap_statistic(self, data: np.ndarray, max_clusters: int = 10, n_refs: int = 10) -> Dict[str, Any]:
        """
        Gap statistic for optimal cluster number determination
        """
        try:
            k_range = range(1, min(max_clusters + 1, len(data)))
            gaps = []
            
            # Calculate gap statistic
            for k in k_range:
                # Cluster original data
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(data)
                original_inertia = kmeans.inertia_
                
                # Generate reference datasets and cluster them
                ref_inertias = []
                for _ in range(n_refs):
                    ref_data = np.random.uniform(
                        low=data.min(axis=0),
                        high=data.max(axis=0),
                        size=data.shape
                    )
                    ref_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    ref_kmeans.fit(ref_data)
                    ref_inertias.append(ref_kmeans.inertia_)
                
                # Calculate gap
                gap = np.log(np.mean(ref_inertias)) - np.log(original_inertia)
                gaps.append(gap)
            
            # Find optimal k using gap statistic
            optimal_k = k_range[np.argmax(gaps)]
            
            return {
                "k_values": list(k_range),
                "gap_values": gaps,
                "optimal_k_gap": optimal_k,
                "max_gap": max(gaps)
            }
            
        except Exception as e:
            return {"error": f"Gap statistic failed: {str(e)}"}
    
    def dimensionality_reduction(self, data: pd.DataFrame, method: str = "pca", n_components: int = 2) -> Dict[str, Any]:
        """
        Multiple dimensionality reduction techniques
        """
        try:
            numeric_data = data.select_dtypes(include=[np.number]).dropna()
            
            if numeric_data.empty:
                return {"error": "No numeric data for dimensionality reduction"}
            
            scaled_data = self.scaler.fit_transform(numeric_data)
            
            if method.lower() == "pca":
                reducer = PCA(n_components=n_components)
                reduced_data = reducer.fit_transform(scaled_data)
                explained_variance = reducer.explained_variance_ratio_
                
            elif method.lower() == "tsne":
                reducer = TSNE(n_components=n_components, random_state=42)
                reduced_data = reducer.fit_transform(scaled_data)
                explained_variance = None
                
            else:
                return {"error": f"Unsupported method: {method}"}
            
            return {
                "method": method,
                "reduced_data": reduced_data.tolist(),
                "original_dimensions": scaled_data.shape[1],
                "reduced_dimensions": n_components,
                "explained_variance": explained_variance.tolist() if explained_variance is not None else None,
                "reduction_quality": self._assess_reduction_quality(scaled_data, reduced_data)
            }
            
        except Exception as e:
            return {"error": f"Dimensionality reduction failed: {str(e)}"}
    
    def grid_search_optimization(self, data: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
        """
        Grid search for hyperparameter optimization
        """
        try:
            numeric_data = data.select_dtypes(include=[np.number]).dropna()
            
            if numeric_data.empty:
                return {"error": "No numeric data for optimization"}
            
            # For clustering optimization
            param_grid = {
                'n_clusters': range(2, min(11, len(numeric_data))),
                'init': ['k-means++', 'random'],
                'max_iter': [100, 300, 500],
                'n_init': [10, 20]
            }
            
            # Custom scoring function for clustering
            def clustering_score(estimator, X):
                labels = estimator.fit_predict(X)
                if len(np.unique(labels)) > 1:
                    return silhouette_score(X, labels)
                return -1
            
            scaled_data = self.scaler.fit_transform(numeric_data)
            
            # Perform grid search
            kmeans = KMeans(random_state=42)
            grid_search = GridSearchCV(
                kmeans,
                param_grid,
                scoring=clustering_score,
                cv=3,
                n_jobs=-1
            )
            
            grid_search.fit(scaled_data)
            
            return {
                "best_parameters": grid_search.best_params_,
                "best_score": float(grid_search.best_score_),
                "all_results": [
                    {
                        "params": result.parameters,
                        "score": float(result.mean_test_score)
                    }
                    for result in grid_search.cv_results_['params']
                ],
                "optimization_summary": {
                    "total_combinations": len(grid_search.cv_results_['params']),
                    "best_improvement": float(grid_search.best_score_),
                    "parameter_importance": self._analyze_parameter_importance(grid_search)
                }
            }
            
        except Exception as e:
            return {"error": f"Grid search optimization failed: {str(e)}"}
    
    def analyze_user_patterns(self, user_data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze user interaction patterns using ML techniques
        """
        try:
            if not user_data:
                return {"error": "No user data available"}
            
            # Convert to DataFrame
            df = pd.DataFrame(user_data)
            
            # Text analysis for messages
            text_features = None
            if 'message' in df.columns:
                text_features = self._analyze_text_patterns(df['message'].tolist())
            
            # Temporal analysis
            temporal_features = None
            if 'timestamp' in df.columns:
                temporal_features = self._analyze_temporal_patterns(df['timestamp'].tolist())
            
            # Interaction analysis
            interaction_features = self._analyze_interaction_patterns(df)
            
            # Perform clustering on user patterns
            feature_matrix = self._create_feature_matrix(df)
            if feature_matrix is not None:
                clustering_results = self.clustering_analysis(pd.DataFrame(feature_matrix))
            else:
                clustering_results = {"error": "Could not create feature matrix"}
            
            return {
                "text_patterns": text_features,
                "temporal_patterns": temporal_features,
                "interaction_patterns": interaction_features,
                "user_clustering": clustering_results,
                "insights": self._generate_user_insights(df, text_features, temporal_features)
            }
            
        except Exception as e:
            return {"error": f"User pattern analysis failed: {str(e)}"}
    
    def _check_data_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data type consistency and quality"""
        consistency_score = 0
        issues = []
        
        for col in data.columns:
            if data[col].dtype == 'object':
                # Check for mixed types in object columns
                try:
                    pd.to_numeric(data[col].dropna())
                    issues.append(f"Column {col} contains numeric data stored as object")
                except:
                    pass
            
        consistency_score = max(0, 100 - len(issues) * 10)
        return {"score": consistency_score, "issues": issues}
    
    def _detect_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outliers = {}
        
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_count = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            outliers[col] = {
                "count": int(outlier_count),
                "percentage": float(outlier_count / len(data) * 100)
            }
        
        return outliers
    
    def _find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.8) -> List[Dict]:
        """Find highly correlated variable pairs"""
        high_corr = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    high_corr.append({
                        "var1": corr_matrix.columns[i],
                        "var2": corr_matrix.columns[j],
                        "correlation": float(corr_val)
                    })
        
        return high_corr
    
    def _analyze_correlations(self, corr_matrix: pd.DataFrame) -> List[str]:
        """Generate insights from correlation analysis"""
        insights = []
        
        # Find strongest positive correlation
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_vals = corr_matrix.where(mask).stack()
        
        if not corr_vals.empty:
            max_corr = corr_vals.max()
            min_corr = corr_vals.min()
            
            if max_corr > 0.7:
                max_pair = corr_vals.idxmax()
                insights.append(f"Strong positive correlation between {max_pair[0]} and {max_pair[1]} ({max_corr:.3f})")
            
            if min_corr < -0.7:
                min_pair = corr_vals.idxmin()
                insights.append(f"Strong negative correlation between {min_pair[0]} and {min_pair[1]} ({min_corr:.3f})")
        
        return insights
    
    def _analyze_distributions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data distributions"""
        distributions = {}
        
        for col in data.select_dtypes(include=[np.number]).columns:
            distributions[col] = {
                "mean": float(data[col].mean()),
                "median": float(data[col].median()),
                "std": float(data[col].std()),
                "skewness": float(data[col].skew()),
                "kurtosis": float(data[col].kurtosis())
            }
        
        return distributions
    
    def _generate_eda_insights(self, data: pd.DataFrame, eda_results: Dict) -> List[str]:
        """Generate actionable insights from EDA"""
        insights = []
        
        # Data quality insights
        if eda_results["data_quality"]["completeness_score"] < 80:
            insights.append("Data has significant missing values - consider imputation strategies")
        
        if eda_results["data_quality"]["duplicate_rows"] > 0:
            insights.append(f"Found {eda_results['data_quality']['duplicate_rows']} duplicate rows")
        
        # Distribution insights
        for col, dist in eda_results["distributions"].items():
            if abs(dist["skewness"]) > 2:
                insights.append(f"Column {col} is highly skewed - consider transformation")
        
        return insights
    
    def _calculate_feature_importance(self) -> Dict[str, float]:
        """Calculate feature importance from PCA loadings"""
        if self.pca is None:
            return {}
        
        # Calculate importance as sum of absolute loadings across components
        importance = np.sum(np.abs(self.pca.components_), axis=0)
        importance = importance / importance.sum()  # Normalize
        
        return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}
    
    def _find_elbow_point(self, k_values: List[int], inertias: List[float]) -> int:
        """Find elbow point using the rate of change"""
        if len(inertias) < 3:
            return k_values[0] if k_values else 2
        
        # Calculate rate of change
        rates = []
        for i in range(1, len(inertias)):
            rate = inertias[i-1] - inertias[i]
            rates.append(rate)
        
        # Find the point where rate of change drops significantly
        if len(rates) < 2:
            return k_values[1] if len(k_values) > 1 else 2
        
        rate_changes = []
        for i in range(1, len(rates)):
            rate_change = rates[i-1] - rates[i]
            rate_changes.append(rate_change)
        
        # Find the maximum rate change
        if rate_changes:
            elbow_idx = np.argmax(rate_changes) + 2  # +2 because we start from k=1 and skip first rate
            return k_values[min(elbow_idx, len(k_values)-1)]
        
        return k_values[1] if len(k_values) > 1 else 2
    
    def _determine_optimal_clusters(self, elbow: Dict, silhouette: Dict, gap: Dict) -> int:
        """Determine optimal number of clusters from multiple methods"""
        candidates = []
        
        if "optimal_k_elbow" in elbow:
            candidates.append(elbow["optimal_k_elbow"])
        
        if "optimal_k_silhouette" in silhouette:
            candidates.append(silhouette["optimal_k_silhouette"])
        
        if "optimal_k_gap" in gap:
            candidates.append(gap["optimal_k_gap"])
        
        if candidates:
            # Return the mode, or median if no mode
            from collections import Counter
            count = Counter(candidates)
            if count:
                return count.most_common(1)[0][0]
        
        return 3  # Default fallback
    
    def _analyze_cluster_profiles(self, data: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
        """Analyze characteristics of each cluster"""
        profiles = {}
        
        for cluster_id in np.unique(labels):
            cluster_data = data[labels == cluster_id]
            
            profile = {
                "size": len(cluster_data),
                "percentage": len(cluster_data) / len(data) * 100,
                "means": cluster_data.mean().to_dict(),
                "characteristics": []
            }
            
            # Find distinguishing characteristics
            overall_means = data.mean()
            for col in data.columns:
                cluster_mean = cluster_data[col].mean()
                overall_mean = overall_means[col]
                
                if abs(cluster_mean - overall_mean) > data[col].std():
                    direction = "higher" if cluster_mean > overall_mean else "lower"
                    profile["characteristics"].append(f"{direction} {col}")
            
            profiles[f"cluster_{cluster_id}"] = profile
        
        return profiles
    
    def _assess_reduction_quality(self, original: np.ndarray, reduced: np.ndarray) -> Dict[str, float]:
        """Assess quality of dimensionality reduction"""
        # Calculate reconstruction error for PCA-like methods
        if hasattr(self, 'pca') and self.pca is not None:
            reconstructed = self.pca.inverse_transform(reduced)
            mse = np.mean((original - reconstructed) ** 2)
            return {
                "reconstruction_error": float(mse),
                "quality_score": float(max(0, 1 - mse))
            }
        
        return {"quality_score": 0.5}  # Default for non-linear methods
    
    def _analyze_parameter_importance(self, grid_search) -> Dict[str, float]:
        """Analyze which parameters had the most impact"""
        # Simplified parameter importance based on score variance
        results = grid_search.cv_results_
        param_importance = {}
        
        for param in grid_search.param_grid.keys():
            param_scores = []
            for params, score in zip(results['params'], results['mean_test_score']):
                param_scores.append((params[param], score))
            
            # Calculate variance in scores for this parameter
            unique_values = list(set([p[0] for p in param_scores]))
            if len(unique_values) > 1:
                scores_by_value = {}
                for val in unique_values:
                    scores_by_value[val] = [s for p, s in param_scores if p == val]
                
                avg_scores = [np.mean(scores_by_value[val]) for val in unique_values]
                importance = np.var(avg_scores)
                param_importance[param] = float(importance)
        
        return param_importance
    
    def _analyze_text_patterns(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze text patterns in user messages"""
        try:
            # TF-IDF analysis
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Calculate average TF-IDF scores
            avg_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Get top terms
            top_indices = np.argsort(avg_scores)[-10:]
            top_terms = [(feature_names[i], avg_scores[i]) for i in top_indices]
            
            return {
                "vocabulary_size": len(feature_names),
                "top_terms": top_terms,
                "avg_message_length": np.mean([len(text.split()) for text in texts]),
                "text_diversity": len(set(texts)) / len(texts) if texts else 0
            }
        except:
            return {"error": "Text analysis failed"}
    
    def _analyze_temporal_patterns(self, timestamps: List[str]) -> Dict[str, Any]:
        """Analyze temporal patterns in user interactions"""
        try:
            # Convert timestamps to datetime
            dates = [datetime.fromisoformat(ts.replace('Z', '+00:00')) if 'T' in ts else datetime.now() for ts in timestamps]
            
            # Extract time features
            hours = [d.hour for d in dates]
            days = [d.weekday() for d in dates]
            
            return {
                "most_active_hour": int(max(set(hours), key=hours.count)),
                "most_active_day": int(max(set(days), key=days.count)),
                "interaction_frequency": len(timestamps),
                "time_span_days": (max(dates) - min(dates)).days if len(dates) > 1 else 0
            }
        except:
            return {"error": "Temporal analysis failed"}
    
    def _analyze_interaction_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze user interaction patterns"""
        patterns = {
            "total_interactions": len(df),
            "unique_users": df['user_id'].nunique() if 'user_id' in df.columns else 1,
            "avg_interactions_per_user": len(df) / df['user_id'].nunique() if 'user_id' in df.columns else len(df)
        }
        
        # Emotion patterns if available
        if 'emotion_detected' in df.columns:
            emotion_counts = df['emotion_detected'].value_counts().to_dict()
            patterns["emotion_distribution"] = emotion_counts
            patterns["dominant_emotion"] = df['emotion_detected'].mode().iloc[0] if not df['emotion_detected'].mode().empty else "neutral"
        
        return patterns
    
    def _create_feature_matrix(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Create feature matrix for user pattern clustering"""
        try:
            features = []
            
            # Message length features
            if 'message' in df.columns:
                msg_lengths = [len(str(msg)) for msg in df['message']]
                features.extend([np.mean(msg_lengths), np.std(msg_lengths)])
            
            # Temporal features
            if 'timestamp' in df.columns:
                temporal = self._analyze_temporal_patterns(df['timestamp'].tolist())
                if 'error' not in temporal:
                    features.extend([
                        temporal.get('most_active_hour', 12),
                        temporal.get('most_active_day', 1),
                        temporal.get('interaction_frequency', 1)
                    ])
            
            # Convert to matrix format
            if features and len(features) > 0:
                return np.array(features).reshape(1, -1)
            
            return None
        except:
            return None
    
    def _generate_user_insights(self, df: pd.DataFrame, text_features: Dict, temporal_features: Dict) -> List[str]:
        """Generate actionable insights from user analysis"""
        insights = []
        
        if text_features and 'error' not in text_features:
            if text_features.get('text_diversity', 0) < 0.3:
                insights.append("User tends to repeat similar messages - consider proactive suggestions")
            
            if text_features.get('avg_message_length', 0) < 10:
                insights.append("User prefers short messages - adapt response style accordingly")
        
        if temporal_features and 'error' not in temporal_features:
            active_hour = temporal_features.get('most_active_hour', 12)
            if active_hour < 6 or active_hour > 22:
                insights.append("User is active during unusual hours - consider time-sensitive responses")
        
        return insights

    def export_analysis_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Export analysis results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ml_analysis_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            return f"Analysis results exported to {filename}"
        except Exception as e:
            return f"Export failed: {str(e)}"