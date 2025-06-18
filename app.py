import streamlit as st
import json
import os
from datetime import datetime
import time
from models.master_model import AISapienMaster
from utils.content_filter import ContentFilter
from utils.emotion_detector import EmotionDetector
from utils.web_scraper import get_website_text_content

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
    """Load conversation history from file"""
    try:
        with open('data/conversations.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_conversations(conversations):
    """Save conversation history to file"""
    with open('data/conversations.json', 'w') as f:
        json.dump(conversations, f, indent=2)

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
                
                # Save conversation
                conversations = load_conversations()
                conversations.append({
                    'user_id': st.session_state.user_id,
                    'user_message': user_input,
                    'assistant_response': response['message'],
                    'timestamp': datetime.now().isoformat(),
                    'model_insights': response.get('model_insights', {})
                })
                save_conversations(conversations)
                
                # Rerun to show new messages
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing message: {str(e)}")
    
    # Additional features tabs
    tab1, tab2, tab3 = st.tabs(["🧠 Knowledge Base", "📈 Analytics", "⚙️ Settings"])
    
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
