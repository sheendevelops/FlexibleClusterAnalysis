import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from utils.ml_analytics import MLAnalytics
from utils.database_helper import db
import json
from datetime import datetime, timedelta

class MLIntegration:
    """
    Integration layer between AISapien and ML Analytics
    Provides intelligent data analysis for improving decision-making
    """
    
    def __init__(self):
        self.ml_analytics = MLAnalytics()
        
    def analyze_conversation_patterns(self, user_id: str = None, days: int = 30) -> Dict[str, Any]:
        """
        Analyze conversation patterns to improve personalization
        """
        try:
            # Fetch conversation data
            if user_id:
                conversations = db.find_many(
                    'conversations', 
                    {'user_id': user_id},
                    limit=1000
                )
            else:
                conversations = db.get_recent_records('conversations', limit=1000)
            
            if not conversations:
                return {"error": "No conversation data available"}
            
            # Convert to DataFrame
            df = pd.DataFrame(conversations)
            
            # Perform comprehensive analysis
            analysis_results = {
                "eda_analysis": self.ml_analytics.exploratory_data_analysis(df),
                "user_patterns": self.ml_analytics.analyze_user_patterns(conversations),
                "conversation_clusters": self._analyze_conversation_themes(df),
                "temporal_insights": self._analyze_conversation_timing(df),
                "emotion_analysis": self._analyze_emotion_patterns(df),
                "response_effectiveness": self._analyze_response_quality(df)
            }
            
            # Generate actionable insights
            analysis_results["actionable_insights"] = self._generate_conversation_insights(analysis_results)
            
            # Store analysis results
            self._store_analysis_results("conversation_analysis", analysis_results, user_id)
            
            return analysis_results
            
        except Exception as e:
            return {"error": f"Conversation analysis failed: {str(e)}"}
    
    def optimize_model_performance(self) -> Dict[str, Any]:
        """
        Optimize the performance of the three AI models using ML insights
        """
        try:
            # Analyze decision patterns
            decisions = db.find_many('master_decisions', limit=1000)
            
            if not decisions:
                return {"error": "No decision data available for optimization"}
            
            df = pd.DataFrame(decisions)
            
            # Perform optimization analysis
            optimization_results = {
                "decision_clustering": self.ml_analytics.clustering_analysis(df),
                "parameter_optimization": self._optimize_decision_parameters(df),
                "model_effectiveness": self._analyze_model_effectiveness(df),
                "performance_metrics": self._calculate_performance_metrics(df)
            }
            
            # Generate optimization recommendations
            optimization_results["recommendations"] = self._generate_optimization_recommendations(optimization_results)
            
            return optimization_results
            
        except Exception as e:
            return {"error": f"Model optimization failed: {str(e)}"}
    
    def analyze_user_segmentation(self) -> Dict[str, Any]:
        """
        Segment users based on interaction patterns using clustering
        """
        try:
            # Fetch user profiles and interactions
            profiles = db.find_many('user_profiles', limit=1000)
            interactions = db.find_many('user_interactions', limit=5000)
            
            if not profiles:
                return {"error": "No user data available for segmentation"}
            
            # Create user feature matrix
            user_features = self._create_user_feature_matrix(profiles, interactions)
            
            if user_features is None or user_features.empty:
                return {"error": "Could not create user feature matrix"}
            
            # Perform clustering analysis
            clustering_results = self.ml_analytics.clustering_analysis(user_features)
            
            # Analyze segments
            segment_analysis = {
                "clustering_results": clustering_results,
                "segment_profiles": self._analyze_user_segments(user_features, clustering_results),
                "personalization_strategies": self._generate_personalization_strategies(clustering_results),
                "segment_insights": self._generate_segment_insights(user_features, clustering_results)
            }
            
            return segment_analysis
            
        except Exception as e:
            return {"error": f"User segmentation failed: {str(e)}"}
    
    def predict_user_needs(self, user_id: str) -> Dict[str, Any]:
        """
        Predict user needs using ML analysis of historical patterns
        """
        try:
            # Fetch user data
            user_interactions = db.find_many(
                'user_interactions',
                {'user_id': user_id},
                limit=100
            )
            
            if not user_interactions:
                return {"error": "No user interaction history available"}
            
            df = pd.DataFrame(user_interactions)
            
            # Analyze patterns
            pattern_analysis = self.ml_analytics.analyze_user_patterns(user_interactions)
            
            # Predict needs based on patterns
            predictions = {
                "predicted_topics": self._predict_topic_interests(df),
                "optimal_interaction_time": self._predict_optimal_timing(df),
                "communication_preferences": self._predict_communication_style(df),
                "likely_questions": self._predict_question_types(df),
                "engagement_level": self._predict_engagement_level(df)
            }
            
            # Generate recommendations
            predictions["recommendations"] = self._generate_user_recommendations(predictions, pattern_analysis)
            
            return predictions
            
        except Exception as e:
            return {"error": f"User need prediction failed: {str(e)}"}
    
    def analyze_learning_effectiveness(self) -> Dict[str, Any]:
        """
        Analyze how effectively the system is learning and improving
        """
        try:
            # Fetch learning data
            knowledge_data = db.find_many('learning_sources', limit=500)
            decisions = db.find_many('master_decisions', limit=1000)
            
            if not knowledge_data and not decisions:
                return {"error": "No learning data available"}
            
            # Analyze knowledge acquisition
            learning_analysis = {
                "knowledge_growth": self._analyze_knowledge_growth(knowledge_data),
                "decision_improvement": self._analyze_decision_improvement(decisions),
                "learning_efficiency": self._calculate_learning_efficiency(knowledge_data, decisions),
                "knowledge_gaps": self._identify_knowledge_gaps(knowledge_data)
            }
            
            # Generate improvement suggestions
            learning_analysis["improvement_suggestions"] = self._generate_learning_improvements(learning_analysis)
            
            return learning_analysis
            
        except Exception as e:
            return {"error": f"Learning effectiveness analysis failed: {str(e)}"}
    
    def _analyze_conversation_themes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze themes and topics in conversations using clustering"""
        try:
            if 'user_message' not in df.columns:
                return {"error": "No message data for theme analysis"}
            
            # Extract text features
            messages = df['user_message'].dropna().tolist()
            
            if len(messages) < 2:
                return {"error": "Insufficient messages for clustering"}
            
            # Use TF-IDF for text clustering
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            text_features = vectorizer.fit_transform(messages)
            
            # Find optimal clusters
            max_clusters = min(10, len(messages) // 2)
            if max_clusters < 2:
                return {"error": "Too few messages for clustering"}
            
            silhouette_scores = []
            k_range = range(2, max_clusters + 1)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(text_features)
                score = silhouette_score(text_features, labels)
                silhouette_scores.append(score)
            
            optimal_k = k_range[np.argmax(silhouette_scores)]
            
            # Final clustering
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            cluster_labels = kmeans.fit_predict(text_features)
            
            # Analyze themes
            feature_names = vectorizer.get_feature_names_out()
            themes = {}
            
            for i in range(optimal_k):
                cluster_center = kmeans.cluster_centers_[i]
                top_indices = np.argsort(cluster_center)[-5:]
                top_terms = [feature_names[idx] for idx in top_indices]
                cluster_messages = [msg for j, msg in enumerate(messages) if cluster_labels[j] == i]
                
                themes[f"theme_{i}"] = {
                    "keywords": top_terms,
                    "message_count": len(cluster_messages),
                    "sample_messages": cluster_messages[:3]
                }
            
            return {
                "optimal_clusters": optimal_k,
                "themes": themes,
                "cluster_distribution": np.bincount(cluster_labels).tolist()
            }
            
        except Exception as e:
            return {"error": f"Theme analysis failed: {str(e)}"}
    
    def _analyze_conversation_timing(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns in conversations"""
        try:
            if 'timestamp' not in df.columns:
                return {"error": "No timestamp data available"}
            
            # Convert timestamps
            df['datetime'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['date'] = df['datetime'].dt.date
            
            # Analyze patterns
            hourly_dist = df['hour'].value_counts().sort_index()
            daily_dist = df['day_of_week'].value_counts().sort_index()
            
            # Find peak times
            peak_hour = hourly_dist.idxmax()
            peak_day = daily_dist.idxmax()
            
            # Calculate conversation frequency
            daily_counts = df.groupby('date').size()
            avg_daily_conversations = daily_counts.mean()
            
            return {
                "peak_hour": int(peak_hour),
                "peak_day": int(peak_day),
                "hourly_distribution": hourly_dist.to_dict(),
                "daily_distribution": daily_dist.to_dict(),
                "avg_daily_conversations": float(avg_daily_conversations),
                "total_conversation_days": len(daily_counts),
                "conversation_consistency": float(daily_counts.std() / daily_counts.mean()) if daily_counts.mean() > 0 else 0
            }
            
        except Exception as e:
            return {"error": f"Timing analysis failed: {str(e)}"}
    
    def _analyze_emotion_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze emotional patterns in conversations"""
        try:
            if 'emotion_detected' not in df.columns:
                return {"error": "No emotion data available"}
            
            emotion_counts = df['emotion_detected'].value_counts()
            total_conversations = len(df)
            
            # Calculate emotion distribution
            emotion_distribution = (emotion_counts / total_conversations * 100).to_dict()
            
            # Analyze emotion trends over time
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'])
                df_sorted = df.sort_values('datetime')
                
                # Look for emotion patterns
                emotion_trends = {}
                for emotion in emotion_counts.index:
                    emotion_data = df_sorted[df_sorted['emotion_detected'] == emotion]
                    if len(emotion_data) > 1:
                        # Simple trend calculation
                        emotion_trends[emotion] = {
                            "count": len(emotion_data),
                            "first_occurrence": emotion_data['datetime'].min().isoformat(),
                            "last_occurrence": emotion_data['datetime'].max().isoformat()
                        }
            
            return {
                "emotion_distribution": emotion_distribution,
                "dominant_emotion": emotion_counts.index[0] if not emotion_counts.empty else "neutral",
                "emotion_variety": len(emotion_counts),
                "emotion_trends": emotion_trends if 'emotion_trends' in locals() else {}
            }
            
        except Exception as e:
            return {"error": f"Emotion analysis failed: {str(e)}"}
    
    def _analyze_response_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the quality and effectiveness of responses"""
        try:
            if 'assistant_response' not in df.columns:
                return {"error": "No response data available"}
            
            responses = df['assistant_response'].dropna()
            
            # Calculate response metrics
            response_lengths = [len(str(response)) for response in responses]
            avg_response_length = np.mean(response_lengths)
            response_length_std = np.std(response_lengths)
            
            # Analyze response patterns
            response_analysis = {
                "avg_response_length": float(avg_response_length),
                "response_length_consistency": float(response_length_std / avg_response_length) if avg_response_length > 0 else 0,
                "total_responses": len(responses),
                "response_variety": len(set(responses)) / len(responses) if len(responses) > 0 else 0
            }
            
            # Check for model insights usage
            if 'model_insights' in df.columns:
                insights_usage = df['model_insights'].apply(
                    lambda x: len(json.loads(x) if isinstance(x, str) else x) if x else 0
                )
                response_analysis["avg_insights_per_response"] = float(insights_usage.mean())
            
            return response_analysis
            
        except Exception as e:
            return {"error": f"Response quality analysis failed: {str(e)}"}
    
    def _create_user_feature_matrix(self, profiles: List[Dict], interactions: List[Dict]) -> Optional[pd.DataFrame]:
        """Create feature matrix for user clustering"""
        try:
            user_features = []
            
            # Process each user profile
            for profile in profiles:
                user_id = profile['user_id']
                
                # Basic profile features
                features = {
                    'user_id': user_id,
                    'profile_completeness': profile.get('profile_completeness', 0),
                    'engagement_level_numeric': self._encode_engagement_level(profile.get('engagement_level', 'New User')),
                    'preferences_count': len(profile.get('preferences', {})),
                    'interests_count': len(profile.get('interests', [])),
                    'needs_count': len(profile.get('needs', []))
                }
                
                # Interaction features
                user_interactions = [i for i in interactions if i.get('user_id') == user_id]
                features.update({
                    'total_interactions': len(user_interactions),
                    'avg_message_length': np.mean([len(str(i.get('message', ''))) for i in user_interactions]) if user_interactions else 0,
                    'unique_emotions': len(set(i.get('emotion_detected', 'neutral') for i in user_interactions)),
                    'recent_activity': len([i for i in user_interactions if self._is_recent(i.get('timestamp'))]) if user_interactions else 0
                })
                
                user_features.append(features)
            
            return pd.DataFrame(user_features).set_index('user_id')
            
        except Exception as e:
            return None
    
    def _encode_engagement_level(self, level: str) -> int:
        """Encode engagement level to numeric value"""
        mapping = {
            'New User': 1,
            'Casual': 2,
            'Regular': 3,
            'Active': 4,
            'Power User': 5
        }
        return mapping.get(level, 1)
    
    def _is_recent(self, timestamp: str, days: int = 7) -> bool:
        """Check if timestamp is within recent days"""
        try:
            if not timestamp:
                return False
            
            ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return (datetime.now() - ts).days <= days
        except:
            return False
    
    def _generate_conversation_insights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate actionable insights from conversation analysis"""
        insights = []
        
        # EDA insights
        eda = analysis_results.get('eda_analysis', {})
        if 'data_quality' in eda:
            completeness = eda['data_quality'].get('completeness_score', 100)
            if completeness < 80:
                insights.append("Conversation data has gaps - consider prompting for more user engagement")
        
        # User pattern insights
        patterns = analysis_results.get('user_patterns', {})
        if 'text_patterns' in patterns and patterns['text_patterns'].get('text_diversity', 1) < 0.3:
            insights.append("Users repeat similar messages - implement proactive topic suggestions")
        
        # Temporal insights
        temporal = analysis_results.get('temporal_insights', {})
        if 'peak_hour' in temporal:
            peak_hour = temporal['peak_hour']
            if peak_hour < 6 or peak_hour > 22:
                insights.append("Users are active during unusual hours - consider time-sensitive features")
        
        # Emotion insights
        emotion = analysis_results.get('emotion_analysis', {})
        if 'dominant_emotion' in emotion and emotion['dominant_emotion'] in ['frustrated', 'confused']:
            insights.append("Users show signs of frustration - review response clarity and helpfulness")
        
        return insights
    
    def _store_analysis_results(self, analysis_type: str, results: Dict[str, Any], user_id: str = None):
        """Store analysis results in database"""
        try:
            analysis_record = {
                'analysis_type': analysis_type,
                'results': json.dumps(results, default=str),
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'success': 'error' not in results
            }
            
            # Store in learning_sources table as analysis insights
            db.insert_one('learning_sources', {
                'source_type': 'ml_analysis',
                'content_summary': f"{analysis_type} analysis results",
                'insights_extracted': results,
                'knowledge_domain': 'user_analytics',
                'timestamp': datetime.now(),
                'success': 'error' not in results
            })
            
        except Exception as e:
            print(f"Failed to store analysis results: {e}")
    
    def get_ml_dashboard_data(self) -> Dict[str, Any]:
        """Get data for ML analytics dashboard"""
        try:
            # Get recent analysis results
            recent_analyses = db.find_many(
                'learning_sources',
                {'source_type': 'ml_analysis'},
                limit=10,
                order_by='timestamp DESC'
            )
            
            # Get summary statistics
            total_conversations = db.count_records('conversations')
            total_users = db.count_records('user_profiles')
            total_decisions = db.count_records('master_decisions')
            
            return {
                "summary_stats": {
                    "total_conversations": total_conversations,
                    "total_users": total_users,
                    "total_decisions": total_decisions,
                    "total_analyses": len(recent_analyses)
                },
                "recent_analyses": recent_analyses,
                "ml_capabilities": {
                    "pca": "Principal Component Analysis for dimensionality reduction",
                    "clustering": "User segmentation and pattern recognition",
                    "eda": "Exploratory Data Analysis for insights",
                    "optimization": "Grid search and hyperparameter tuning",
                    "prediction": "User behavior and need prediction"
                }
            }
            
        except Exception as e:
            return {"error": f"Dashboard data retrieval failed: {str(e)}"}