import streamlit as st
import json
import os
from datetime import datetime
import time
from models.master_model import AISapienMaster
from utils.content_filter import ContentFilter
from utils.emotion_detector import EmotionDetector
from utils.web_scraper import get_website_text_content
from utils.ml_integration import MLIntegration
from config import get_config_info

# Initialize session state
if 'aisapien' not in st.session_state:
    st.session_state.aisapien = AISapienMaster()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_id' not in st.session_state:
    st.session_state.user_id = "default_user"
if 'learning_mode' not in st.session_state:
    st.session_state.learning_mode = False

def load_conversations():
    """Load conversation history from database"""
    try:
        from utils.database_helper import db
        return db.get_recent_records('conversations', limit=100)
    except Exception as e:
        print(f"Error loading conversations: {e}")
        return []

def save_conversation(user_id, user_message, assistant_response, model_insights, emotion_detected):
    """Save conversation to database"""
    try:
        from utils.database_helper import db
        db.insert_one('conversations', {
            'user_id': user_id,
            'user_message': user_message,
            'assistant_response': assistant_response,
            'model_insights': model_insights,
            'emotion_detected': emotion_detected
        })
    except Exception as e:
        print(f"Error saving conversation: {e}")

def main():
    st.set_page_config(
        page_title="AISapien - Ethical AI Assistant",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 AISapien - Ethical AI Assistant")
    st.markdown("*An AI system with conscience, logic, and personality models working together*")
    
    # Sidebar for controls and status
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # User identification
        user_id = st.text_input("User ID", value=st.session_state.user_id)
        if user_id != st.session_state.user_id:
            st.session_state.user_id = user_id
            st.session_state.aisapien.set_current_user(user_id)
        
        # Learning mode toggle
        learning_mode = st.toggle("Learning Mode", value=st.session_state.learning_mode)
        st.session_state.learning_mode = learning_mode
        
        st.divider()
        
        # Status display
        st.header("📊 System Status")
        
        # Show LLM backend info
        config_info = get_config_info()
        if config_info['llm_backend'] == 'ollama':
            st.info(f"🤖 Local AI: {config_info['ollama_model']}")
        else:
            st.info(f"☁️ OpenAI: {config_info['openai_model']}")
        
        # Get system status
        status = st.session_state.aisapien.get_system_status()
        
        st.metric("Knowledge Areas", status['knowledge_areas'])
        st.metric("Skills Acquired", status['skills_count'])
        st.metric("Users Known", status['users_count'])
        
        # Display current time awareness
        current_time = datetime.now()
        time_of_day = st.session_state.aisapien.get_time_of_day()
        st.info(f"Time Awareness: {time_of_day}")
        
        # Display acquired skills
        if status['recent_skills']:
            st.subheader("🎯 Recent Skills")
            for skill in status['recent_skills'][-5:]:
                st.text(f"• {skill}")
        
        st.divider()
        
        # Learning tools
        st.header("📚 Learning Tools")
        
        # URL learning
        url_input = st.text_input("Learn from URL")
        if st.button("Learn from URL") and url_input:
            with st.spinner("Learning from URL..."):
                try:
                    content = get_website_text_content(url_input)
                    if content:
                        result = st.session_state.aisapien.learn_from_text(content, source=f"URL: {url_input}")
                        if result['success']:
                            st.success(f"Learned: {result['message']}")
                        else:
                            st.error(result['message'])
                    else:
                        st.error("Could not extract content from URL")
                except Exception as e:
                    st.error(f"Error learning from URL: {str(e)}")
        
        # File upload learning
        uploaded_file = st.file_uploader("Upload document to learn from", type=['txt', 'pdf'])
        if uploaded_file and st.button("Learn from File"):
            with st.spinner("Learning from file..."):
                try:
                    if uploaded_file.type == "text/plain":
                        content = str(uploaded_file.read(), "utf-8")
                        result = st.session_state.aisapien.learn_from_text(content, source=f"File: {uploaded_file.name}")
                        if result['success']:
                            st.success(f"Learned: {result['message']}")
                        else:
                            st.error(result['message'])
                    else:
                        st.warning("Currently only text files are supported")
                except Exception as e:
                    st.error(f"Error learning from file: {str(e)}")
        
        st.divider()
        
        # ML Analytics Section
        st.header("🧠 ML Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Analyze Patterns", help="Analyze conversation patterns using ML"):
                with st.spinner("Running pattern analysis..."):
                    analysis = st.session_state.ml_integration.analyze_conversation_patterns(st.session_state.user_id)
                    st.session_state.latest_analysis = analysis
                    if 'error' not in analysis:
                        st.success("Pattern analysis complete!")
                        if 'actionable_insights' in analysis:
                            for insight in analysis['actionable_insights'][:2]:
                                st.info(insight)
                    else:
                        st.error(f"Analysis failed: {analysis['error']}")
            
            if st.button("User Segmentation", help="Cluster users by behavior patterns"):
                with st.spinner("Performing user segmentation..."):
                    segmentation = st.session_state.ml_integration.analyze_user_segmentation()
                    st.session_state.latest_segmentation = segmentation
                    if 'error' not in segmentation:
                        st.success("User segmentation complete!")
                    else:
                        st.error(f"Segmentation failed: {segmentation['error']}")
        
        with col2:
            if st.button("Predict Needs", help="Predict user needs using ML"):
                with st.spinner("Predicting user needs..."):
                    predictions = st.session_state.ml_integration.predict_user_needs(st.session_state.user_id)
                    st.session_state.latest_predictions = predictions
                    if 'error' not in predictions:
                        st.success("Predictions ready!")
                        if 'recommendations' in predictions:
                            for rec in predictions['recommendations'][:2]:
                                st.success(rec)
                    else:
                        st.error(f"Prediction failed: {predictions['error']}")
            
            if st.button("Optimize Models", help="Optimize AI model performance"):
                with st.spinner("Optimizing models..."):
                    optimization = st.session_state.ml_integration.optimize_model_performance()
                    st.session_state.latest_optimization = optimization
                    if 'error' not in optimization:
                        st.success("Model optimization complete!")
                    else:
                        st.error(f"Optimization failed: {optimization['error']}")
    
    # Main chat interface
    st.header("💬 Chat with AISapien")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.chat_message("user").write(message['content'])
            else:
                with st.chat_message("assistant"):
                    st.write(message['content'])
                    
                    # Show model contributions if available
                    if 'model_insights' in message:
                        with st.expander("🔍 Model Insights"):
                            insights = message['model_insights']
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.subheader("❤️ Conscience Model")
                                st.write(insights.get('conscience', 'No input'))
                            
                            with col2:
                                st.subheader("🧮 Logic Model")
                                st.write(insights.get('logic', 'No input'))
                            
                            with col3:
                                st.subheader("👤 Personality Model")
                                st.write(insights.get('personality', 'No input'))
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # Process with AISapien
        with st.spinner("AISapien is thinking..."):
            try:
                response = st.session_state.aisapien.process_chat_message(
                    user_input, 
                    st.session_state.user_id
                )
                
                # Add assistant response to history
                assistant_message = {
                    'role': 'assistant',
                    'content': response['message'],
                    'timestamp': datetime.now().isoformat()
                }
                
                if 'model_insights' in response:
                    assistant_message['model_insights'] = response['model_insights']
                
                st.session_state.chat_history.append(assistant_message)
                
                # Save conversation to database
                save_conversation(
                    st.session_state.user_id,
                    user_input,
                    response['message'],
                    response.get('model_insights', {}),
                    response.get('detected_emotion', 'neutral')
                )
                
                # Rerun to show new messages
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing message: {str(e)}")
    
    # Additional features tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🧠 Knowledge Base", "📊 ML Analytics", "📈 Advanced Tools", "⚙️ Settings"])
    
    with tab1:
        st.subheader("Knowledge Base Overview")
        knowledge = st.session_state.aisapien.get_knowledge_summary()
        
        if knowledge:
            for category, items in knowledge.items():
                st.write(f"**{category.title()}**: {len(items)} items")
                with st.expander(f"View {category}"):
                    for item in items[-10:]:  # Show last 10 items
                        st.text(f"• {item}")
        else:
            st.info("No knowledge acquired yet. Start learning by uploading documents or URLs!")
    
    with tab2:
        st.subheader("System Analytics")
        
        # Emotion analysis of recent conversations
        if st.session_state.chat_history:
            emotions = []
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    emotion = EmotionDetector.detect_emotion(message['content'])
                    emotions.append(emotion)
            
            if emotions:
                emotion_counts = {}
                for emotion in emotions:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                
                st.write("**Emotion Distribution in Conversations:**")
                for emotion, count in emotion_counts.items():
                    st.write(f"- {emotion.title()}: {count}")
        
        # Learning progress
        st.write("**Learning Progress:**")
        skills = st.session_state.aisapien.get_acquired_skills()
        if skills:
            st.write(f"Total skills acquired: {len(skills)}")
            recent_skills = skills[-5:] if len(skills) > 5 else skills
            st.write("Recent skills:")
            for skill in recent_skills:
                st.write(f"- {skill}")
    
    with tab3:
        st.subheader("System Settings")
        
        # Clear conversation history
        if st.button("Clear Chat History", type="secondary"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
            st.rerun()
        
        # Reset knowledge base
        if st.button("Reset Knowledge Base", type="secondary"):
            if st.button("Confirm Reset", type="primary"):
                st.session_state.aisapien.reset_knowledge_base()
                st.success("Knowledge base reset!")
                st.rerun()
        
        # Export data
        if st.button("Export Conversation Data"):
            conversations = load_conversations()
            st.download_button(
                label="Download conversations.json",
                data=json.dumps(conversations, indent=2),
                file_name=f"conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
