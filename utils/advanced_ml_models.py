import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from typing import Dict, List, Any, Tuple, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AssociationRuleMining:
    """
    Association Rule Mining using Apriori algorithm for discovering relationships
    between emotions, conversation patterns, and user behaviors
    """
    
    def __init__(self):
        self.transaction_encoder = TransactionEncoder()
        self.frequent_itemsets = None
        self.association_rules_df = None
        
    def prepare_transaction_data(self, conversations: List[Dict]) -> List[List[str]]:
        """
        Prepare conversation data for association rule mining
        """
        transactions = []
        
        for conv in conversations:
            transaction = []
            
            # Add emotion information
            emotion = conv.get('emotion_detected', 'neutral')
            transaction.append(f"emotion_{emotion}")
            
            # Add time-based features
            timestamp = conv.get('timestamp', '')
            if timestamp:
                try:
                    dt = pd.to_datetime(timestamp)
                    hour = dt.hour
                    
                    if 6 <= hour < 12:
                        transaction.append("time_morning")
                    elif 12 <= hour < 18:
                        transaction.append("time_afternoon")
                    elif 18 <= hour < 22:
                        transaction.append("time_evening")
                    else:
                        transaction.append("time_night")
                    
                    # Day of week
                    day = dt.strftime('%A').lower()
                    transaction.append(f"day_{day}")
                except:
                    pass
            
            # Add message characteristics
            user_message = str(conv.get('user_message', ''))
            message_length = len(user_message)
            
            if message_length < 50:
                transaction.append("message_short")
            elif message_length < 150:
                transaction.append("message_medium")
            else:
                transaction.append("message_long")
            
            # Add response characteristics
            assistant_response = str(conv.get('assistant_response', ''))
            response_length = len(assistant_response)
            
            if response_length < 100:
                transaction.append("response_short")
            elif response_length < 300:
                transaction.append("response_medium")
            else:
                transaction.append("response_long")
            
            # Add model insights presence
            if conv.get('model_insights'):
                transaction.append("has_insights")
            else:
                transaction.append("no_insights")
            
            # Add keyword-based topics
            keywords = self._extract_topics(user_message)
            for keyword in keywords:
                transaction.append(f"topic_{keyword}")
            
            transactions.append(transaction)
        
        return transactions
    
    def _extract_topics(self, text: str) -> List[str]:
        """
        Extract topic keywords from text
        """
        text_lower = text.lower()
        
        topic_keywords = {
            'help': ['help', 'assist', 'support', 'guidance'],
            'question': ['question', 'ask', 'wondering', 'curious'],
            'problem': ['problem', 'issue', 'trouble', 'difficulty'],
            'information': ['information', 'data', 'facts', 'details'],
            'advice': ['advice', 'suggestion', 'recommendation', 'opinion'],
            'explanation': ['explain', 'clarify', 'understand', 'meaning'],
            'learning': ['learn', 'study', 'education', 'knowledge'],
            'work': ['work', 'job', 'career', 'business'],
            'personal': ['personal', 'life', 'relationship', 'family'],
            'technology': ['technology', 'computer', 'software', 'ai']
        }
        
        found_topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                found_topics.append(topic)
        
        return found_topics[:3]  # Limit to top 3 topics
    
    def mine_association_rules(self, conversations: List[Dict], 
                             min_support: float = 0.1, 
                             min_confidence: float = 0.6) -> Dict[str, Any]:
        """
        Mine association rules from conversation data
        """
        try:
            # Prepare transaction data
            transactions = self.prepare_transaction_data(conversations)
            
            if not transactions or len(transactions) < 5:
                return {"error": "Insufficient data for association rule mining"}
            
            # Encode transactions
            te_ary = self.transaction_encoder.fit_transform(transactions)
            df = pd.DataFrame(te_ary, columns=self.transaction_encoder.columns_)
            
            # Find frequent itemsets
            self.frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
            
            if self.frequent_itemsets.empty:
                return {"error": "No frequent itemsets found with current support threshold"}
            
            # Generate association rules
            self.association_rules_df = association_rules(
                self.frequent_itemsets, 
                metric="confidence", 
                min_threshold=min_confidence
            )
            
            if self.association_rules_df.empty:
                return {"error": "No association rules found with current confidence threshold"}
            
            # Analyze rules
            results = self._analyze_association_rules()
            
            return {
                "frequent_itemsets_count": len(self.frequent_itemsets),
                "association_rules_count": len(self.association_rules_df),
                "top_rules": results["top_rules"],
                "emotion_patterns": results["emotion_patterns"],
                "temporal_patterns": results["temporal_patterns"],
                "interaction_patterns": results["interaction_patterns"],
                "insights": results["insights"]
            }
            
        except Exception as e:
            return {"error": f"Association rule mining failed: {str(e)}"}
    
    def _analyze_association_rules(self) -> Dict[str, Any]:
        """
        Analyze discovered association rules
        """
        if self.association_rules_df is None or self.association_rules_df.empty:
            return {"error": "No association rules to analyze"}
        
        # Sort by confidence and lift
        sorted_rules = self.association_rules_df.sort_values(['confidence', 'lift'], ascending=False)
        
        # Top rules
        top_rules = []
        for _, row in sorted_rules.head(10).iterrows():
            antecedents = list(row['antecedents'])
            consequents = list(row['consequents'])
            
            top_rules.append({
                "antecedents": antecedents,
                "consequents": consequents,
                "support": float(row['support']),
                "confidence": float(row['confidence']),
                "lift": float(row['lift']),
                "interpretation": self._interpret_rule(antecedents, consequents, row['confidence'])
            })
        
        # Emotion-specific patterns
        emotion_patterns = self._find_emotion_patterns()
        
        # Temporal patterns
        temporal_patterns = self._find_temporal_patterns()
        
        # Interaction patterns
        interaction_patterns = self._find_interaction_patterns()
        
        # Generate insights
        insights = self._generate_rule_insights(top_rules)
        
        return {
            "top_rules": top_rules,
            "emotion_patterns": emotion_patterns,
            "temporal_patterns": temporal_patterns,
            "interaction_patterns": interaction_patterns,
            "insights": insights
        }
    
    def _interpret_rule(self, antecedents: List[str], consequents: List[str], confidence: float) -> str:
        """
        Provide human-readable interpretation of association rules
        """
        ant_str = ", ".join(antecedents)
        con_str = ", ".join(consequents)
        
        return f"When {ant_str} occurs, {con_str} follows with {confidence:.1%} confidence"
    
    def _find_emotion_patterns(self) -> List[Dict]:
        """
        Find patterns specifically related to emotions
        """
        emotion_rules = []
        
        for _, row in self.association_rules_df.iterrows():
            antecedents = list(row['antecedents'])
            consequents = list(row['consequents'])
            
            has_emotion_ant = any('emotion_' in item for item in antecedents)
            has_emotion_con = any('emotion_' in item for item in consequents)
            
            if has_emotion_ant or has_emotion_con:
                emotion_rules.append({
                    "antecedents": antecedents,
                    "consequents": consequents,
                    "confidence": float(row['confidence']),
                    "lift": float(row['lift'])
                })
        
        return emotion_rules[:5]  # Top 5 emotion patterns
    
    def _find_temporal_patterns(self) -> List[Dict]:
        """
        Find patterns related to time and day
        """
        temporal_rules = []
        
        for _, row in self.association_rules_df.iterrows():
            antecedents = list(row['antecedents'])
            consequents = list(row['consequents'])
            
            has_time = any(any(t in item for t in ['time_', 'day_']) for item in antecedents + consequents)
            
            if has_time:
                temporal_rules.append({
                    "antecedents": antecedents,
                    "consequents": consequents,
                    "confidence": float(row['confidence']),
                    "lift": float(row['lift'])
                })
        
        return temporal_rules[:5]  # Top 5 temporal patterns
    
    def _find_interaction_patterns(self) -> List[Dict]:
        """
        Find patterns related to interaction characteristics
        """
        interaction_rules = []
        
        for _, row in self.association_rules_df.iterrows():
            antecedents = list(row['antecedents'])
            consequents = list(row['consequents'])
            
            has_interaction = any(any(i in item for i in ['message_', 'response_', 'topic_']) 
                                for item in antecedents + consequents)
            
            if has_interaction:
                interaction_rules.append({
                    "antecedents": antecedents,
                    "consequents": consequents,
                    "confidence": float(row['confidence']),
                    "lift": float(row['lift'])
                })
        
        return interaction_rules[:5]  # Top 5 interaction patterns
    
    def _generate_rule_insights(self, top_rules: List[Dict]) -> List[str]:
        """
        Generate actionable insights from association rules
        """
        insights = []
        
        for rule in top_rules[:5]:
            antecedents = rule['antecedents']
            consequents = rule['consequents']
            confidence = rule['confidence']
            
            # Emotion-based insights
            if any('emotion_' in item for item in antecedents):
                emotion = [item.replace('emotion_', '') for item in antecedents if 'emotion_' in item][0]
                if any('response_' in item for item in consequents):
                    response_type = [item.replace('response_', '') for item in consequents if 'response_' in item][0]
                    insights.append(f"Users with {emotion} emotion typically receive {response_type} responses")
            
            # Temporal insights
            if any('time_' in item for item in antecedents):
                time_period = [item.replace('time_', '') for item in antecedents if 'time_' in item][0]
                if any('topic_' in item for item in consequents):
                    topic = [item.replace('topic_', '') for item in consequents if 'topic_' in item][0]
                    insights.append(f"During {time_period}, users often discuss {topic}")
        
        return insights

class ProbabilisticModels:
    """
    Gaussian Mixture Models for modeling relationships between emotions, 
    conversation shapes, and consequences
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.emotion_gmm = None
        self.conversation_gmm = None
        self.consequence_gmm = None
        
    def prepare_features(self, conversations: List[Dict]) -> pd.DataFrame:
        """
        Prepare features for probabilistic modeling
        """
        features = []
        
        for conv in conversations:
            feature_dict = {}
            
            # Emotional features
            emotion = conv.get('emotion_detected', 'neutral')
            feature_dict['emotion_category'] = emotion
            
            # Conversation shape features
            user_msg = str(conv.get('user_message', ''))
            assistant_msg = str(conv.get('assistant_response', ''))
            
            feature_dict['message_length'] = len(user_msg)
            feature_dict['response_length'] = len(assistant_msg)
            feature_dict['length_ratio'] = len(assistant_msg) / max(len(user_msg), 1)
            
            # Temporal features
            timestamp = conv.get('timestamp', '')
            if timestamp:
                try:
                    dt = pd.to_datetime(timestamp)
                    feature_dict['hour'] = dt.hour
                    feature_dict['day_of_week'] = dt.dayofweek
                    feature_dict['is_weekend'] = 1 if dt.dayofweek >= 5 else 0
                except:
                    feature_dict['hour'] = 12
                    feature_dict['day_of_week'] = 1
                    feature_dict['is_weekend'] = 0
            else:
                feature_dict['hour'] = 12
                feature_dict['day_of_week'] = 1
                feature_dict['is_weekend'] = 0
            
            # Consequence features (based on follow-up patterns)
            feature_dict['has_insights'] = 1 if conv.get('model_insights') else 0
            feature_dict['response_complexity'] = self._calculate_response_complexity(assistant_msg)
            feature_dict['interaction_quality'] = self._estimate_interaction_quality(conv)
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def _calculate_response_complexity(self, response: str) -> float:
        """
        Calculate complexity score of response
        """
        if not response:
            return 0.0
        
        # Factors: length, sentence count, vocabulary diversity
        sentences = response.split('.')
        words = response.split()
        unique_words = set(words)
        
        complexity = (
            len(words) * 0.3 +  # Length component
            len(sentences) * 0.3 +  # Structure component
            len(unique_words) * 0.4  # Vocabulary component
        )
        
        return min(complexity / 100, 1.0)  # Normalize to 0-1
    
    def _estimate_interaction_quality(self, conversation: Dict) -> float:
        """
        Estimate quality of interaction based on available features
        """
        quality_score = 0.5  # Base score
        
        # Adjust based on emotion
        emotion = conversation.get('emotion_detected', 'neutral')
        emotion_scores = {
            'happy': 0.9, 'excited': 0.9,
            'neutral': 0.5, 'confused': 0.4,
            'frustrated': 0.2, 'angry': 0.1
        }
        quality_score = emotion_scores.get(emotion, 0.5)
        
        # Adjust based on response length (balanced responses are better)
        response = str(conversation.get('assistant_response', ''))
        if 50 <= len(response) <= 300:
            quality_score += 0.1
        
        # Adjust based on insights presence
        if conversation.get('model_insights'):
            quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    def fit_gaussian_mixture_models(self, conversations: List[Dict], 
                                  n_components: int = 3) -> Dict[str, Any]:
        """
        Fit Gaussian Mixture Models for different aspects of conversations
        """
        try:
            # Prepare features
            df = self.prepare_features(conversations)
            
            if df.empty or len(df) < 10:
                return {"error": "Insufficient data for probabilistic modeling"}
            
            # Encode categorical variables
            categorical_cols = ['emotion_category']
            for col in categorical_cols:
                if col in df.columns:
                    le = LabelEncoder()
                    df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
            
            # Select numerical features for GMM
            numerical_cols = [
                'message_length', 'response_length', 'length_ratio',
                'hour', 'day_of_week', 'is_weekend',
                'has_insights', 'response_complexity', 'interaction_quality'
            ]
            
            if 'emotion_category_encoded' in df.columns:
                numerical_cols.append('emotion_category_encoded')
            
            # Prepare data for GMM
            gmm_data = df[numerical_cols].fillna(0)
            scaled_data = self.scaler.fit_transform(gmm_data)
            
            # Fit emotion-focused GMM
            emotion_features = [
                'emotion_category_encoded', 'interaction_quality', 
                'response_complexity', 'hour'
            ] if 'emotion_category_encoded' in df.columns else [
                'interaction_quality', 'response_complexity', 'hour'
            ]
            
            emotion_data = self.scaler.fit_transform(df[emotion_features].fillna(0))
            self.emotion_gmm = GaussianMixture(n_components=n_components, random_state=42)
            emotion_labels = self.emotion_gmm.fit_predict(emotion_data)
            
            # Fit conversation shape GMM
            shape_features = ['message_length', 'response_length', 'length_ratio', 'response_complexity']
            shape_data = self.scaler.fit_transform(df[shape_features].fillna(0))
            self.conversation_gmm = GaussianMixture(n_components=n_components, random_state=42)
            shape_labels = self.conversation_gmm.fit_predict(shape_data)
            
            # Fit consequence GMM
            consequence_features = ['interaction_quality', 'has_insights', 'response_complexity']
            consequence_data = self.scaler.fit_transform(df[consequence_features].fillna(0))
            self.consequence_gmm = GaussianMixture(n_components=n_components, random_state=42)
            consequence_labels = self.consequence_gmm.fit_predict(consequence_data)
            
            # Analyze clusters
            results = self._analyze_gmm_clusters(df, emotion_labels, shape_labels, consequence_labels)
            
            return {
                "emotion_clusters": results["emotion_analysis"],
                "shape_clusters": results["shape_analysis"],
                "consequence_clusters": results["consequence_analysis"],
                "cluster_relationships": results["relationships"],
                "probabilistic_insights": results["insights"],
                "model_performance": {
                    "emotion_aic": self.emotion_gmm.aic(emotion_data),
                    "emotion_bic": self.emotion_gmm.bic(emotion_data),
                    "shape_aic": self.conversation_gmm.aic(shape_data),
                    "shape_bic": self.conversation_gmm.bic(shape_data),
                    "consequence_aic": self.consequence_gmm.aic(consequence_data),
                    "consequence_bic": self.consequence_gmm.bic(consequence_data)
                }
            }
            
        except Exception as e:
            return {"error": f"Probabilistic modeling failed: {str(e)}"}
    
    def _analyze_gmm_clusters(self, df: pd.DataFrame, 
                            emotion_labels: np.ndarray,
                            shape_labels: np.ndarray,
                            consequence_labels: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the clusters formed by GMM
        """
        # Add cluster labels to dataframe
        df_analysis = df.copy()
        df_analysis['emotion_cluster'] = emotion_labels
        df_analysis['shape_cluster'] = shape_labels
        df_analysis['consequence_cluster'] = consequence_labels
        
        # Analyze emotion clusters
        emotion_analysis = {}
        for cluster in np.unique(emotion_labels):
            cluster_data = df_analysis[df_analysis['emotion_cluster'] == cluster]
            emotion_analysis[f"cluster_{cluster}"] = {
                "size": len(cluster_data),
                "dominant_emotion": cluster_data['emotion_category'].mode().iloc[0] if not cluster_data['emotion_category'].mode().empty else 'unknown',
                "avg_interaction_quality": float(cluster_data['interaction_quality'].mean()),
                "avg_response_complexity": float(cluster_data['response_complexity'].mean()),
                "peak_hour": int(cluster_data['hour'].mode().iloc[0]) if not cluster_data['hour'].mode().empty else 12
            }
        
        # Analyze shape clusters
        shape_analysis = {}
        for cluster in np.unique(shape_labels):
            cluster_data = df_analysis[df_analysis['shape_cluster'] == cluster]
            shape_analysis[f"cluster_{cluster}"] = {
                "size": len(cluster_data),
                "avg_message_length": float(cluster_data['message_length'].mean()),
                "avg_response_length": float(cluster_data['response_length'].mean()),
                "avg_length_ratio": float(cluster_data['length_ratio'].mean()),
                "conversation_style": self._classify_conversation_style(cluster_data)
            }
        
        # Analyze consequence clusters
        consequence_analysis = {}
        for cluster in np.unique(consequence_labels):
            cluster_data = df_analysis[df_analysis['consequence_cluster'] == cluster]
            consequence_analysis[f"cluster_{cluster}"] = {
                "size": len(cluster_data),
                "avg_interaction_quality": float(cluster_data['interaction_quality'].mean()),
                "insight_rate": float(cluster_data['has_insights'].mean()),
                "avg_complexity": float(cluster_data['response_complexity'].mean()),
                "outcome_type": self._classify_outcome_type(cluster_data)
            }
        
        # Analyze relationships between clusters
        relationships = self._analyze_cluster_relationships(df_analysis)
        
        # Generate insights
        insights = self._generate_probabilistic_insights(emotion_analysis, shape_analysis, consequence_analysis)
        
        return {
            "emotion_analysis": emotion_analysis,
            "shape_analysis": shape_analysis,
            "consequence_analysis": consequence_analysis,
            "relationships": relationships,
            "insights": insights
        }
    
    def _classify_conversation_style(self, cluster_data: pd.DataFrame) -> str:
        """
        Classify conversation style based on cluster characteristics
        """
        avg_msg_len = cluster_data['message_length'].mean()
        avg_resp_len = cluster_data['response_length'].mean()
        avg_ratio = cluster_data['length_ratio'].mean()
        
        if avg_msg_len < 50 and avg_resp_len < 100:
            return "brief_exchanges"
        elif avg_msg_len > 150 and avg_resp_len > 300:
            return "detailed_discussions"
        elif avg_ratio > 3:
            return "explanatory_responses"
        elif avg_ratio < 1:
            return "concise_responses"
        else:
            return "balanced_conversations"
    
    def _classify_outcome_type(self, cluster_data: pd.DataFrame) -> str:
        """
        Classify outcome type based on cluster characteristics
        """
        avg_quality = cluster_data['interaction_quality'].mean()
        insight_rate = cluster_data['has_insights'].mean()
        
        if avg_quality > 0.7 and insight_rate > 0.5:
            return "high_value_interactions"
        elif avg_quality > 0.5:
            return "satisfactory_interactions"
        elif insight_rate > 0.5:
            return "informative_interactions"
        else:
            return "basic_interactions"
    
    def _analyze_cluster_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze relationships between different cluster types
        """
        relationships = {}
        
        # Emotion-Shape relationships
        emotion_shape = df.groupby(['emotion_cluster', 'shape_cluster']).size().reset_index(name='count')
        relationships['emotion_shape'] = emotion_shape.to_dict('records')
        
        # Shape-Consequence relationships
        shape_consequence = df.groupby(['shape_cluster', 'consequence_cluster']).size().reset_index(name='count')
        relationships['shape_consequence'] = shape_consequence.to_dict('records')
        
        # Emotion-Consequence relationships
        emotion_consequence = df.groupby(['emotion_cluster', 'consequence_cluster']).size().reset_index(name='count')
        relationships['emotion_consequence'] = emotion_consequence.to_dict('records')
        
        return relationships
    
    def _generate_probabilistic_insights(self, emotion_analysis: Dict, 
                                       shape_analysis: Dict, 
                                       consequence_analysis: Dict) -> List[str]:
        """
        Generate actionable insights from probabilistic analysis
        """
        insights = []
        
        # Emotion cluster insights
        emotion_clusters = list(emotion_analysis.keys())
        if len(emotion_clusters) > 1:
            highest_quality_cluster = max(emotion_clusters, 
                                        key=lambda x: emotion_analysis[x]['avg_interaction_quality'])
            dominant_emotion = emotion_analysis[highest_quality_cluster]['dominant_emotion']
            insights.append(f"Users with {dominant_emotion} emotion tend to have the highest interaction quality")
        
        # Shape cluster insights
        shape_clusters = list(shape_analysis.keys())
        if len(shape_clusters) > 1:
            most_common_style = max(shape_clusters, key=lambda x: shape_analysis[x]['size'])
            style = shape_analysis[most_common_style]['conversation_style']
            insights.append(f"Most conversations follow {style.replace('_', ' ')} pattern")
        
        # Consequence insights
        consequence_clusters = list(consequence_analysis.keys())
        if len(consequence_clusters) > 1:
            best_outcome_cluster = max(consequence_clusters, 
                                     key=lambda x: consequence_analysis[x]['avg_interaction_quality'])
            outcome_type = consequence_analysis[best_outcome_cluster]['outcome_type']
            insights.append(f"Optimal outcomes are characterized by {outcome_type.replace('_', ' ')}")
        
        return insights
    
    def predict_probabilities(self, conversation_features: Dict) -> Dict[str, Any]:
        """
        Predict probabilities for emotion, shape, and consequence clusters
        """
        if not all([self.emotion_gmm, self.conversation_gmm, self.consequence_gmm]):
            return {"error": "Models not fitted yet"}
        
        try:
            # Prepare feature vector
            feature_vector = self._prepare_single_feature_vector(conversation_features)
            
            # Predict probabilities for each model
            emotion_probs = self.emotion_gmm.predict_proba([feature_vector[:4]])[0]
            shape_probs = self.conversation_gmm.predict_proba([feature_vector[4:8]])[0]
            consequence_probs = self.consequence_gmm.predict_proba([feature_vector[8:]])[0]
            
            return {
                "emotion_probabilities": emotion_probs.tolist(),
                "shape_probabilities": shape_probs.tolist(),
                "consequence_probabilities": consequence_probs.tolist(),
                "predictions": {
                    "most_likely_emotion_cluster": int(np.argmax(emotion_probs)),
                    "most_likely_shape_cluster": int(np.argmax(shape_probs)),
                    "most_likely_consequence_cluster": int(np.argmax(consequence_probs))
                }
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def _prepare_single_feature_vector(self, features: Dict) -> np.ndarray:
        """
        Prepare a single feature vector for prediction
        """
        # This is a simplified version - in practice, you'd want to match
        # the exact feature preparation used during training
        vector = [
            features.get('interaction_quality', 0.5),
            features.get('response_complexity', 0.5),
            features.get('hour', 12),
            features.get('emotion_encoded', 0),
            features.get('message_length', 100),
            features.get('response_length', 200),
            features.get('length_ratio', 2.0),
            features.get('response_complexity', 0.5),
            features.get('interaction_quality', 0.5),
            features.get('has_insights', 0),
            features.get('response_complexity', 0.5)
        ]
        
        return np.array(vector)