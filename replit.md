# AISapien - Ethical AI Assistant

## Overview

AISapien is an ethical AI assistant built with Streamlit that implements a multi-model architecture designed to prioritize human wellbeing and ethical decision-making. The system combines three specialized AI models (Conscience, Logic, and Personality) orchestrated by a master model to provide thoughtful, personalized, and ethically-grounded responses to user interactions.

## System Architecture

The application follows a modular architecture with clear separation of concerns:

### Frontend
- **Streamlit**: Web-based interface for real-time chat interactions
- **Responsive UI**: Control panel with user management and system status
- **Real-time chat**: Interactive conversation interface with history

### Backend Models
- **Master Model (AISapienMaster)**: Central orchestrator that synthesizes inputs from all other models
- **Conscience Model**: Focuses on human ethics, empathy, and humanitarian values
- **Logic Model**: Handles rational analysis, efficiency, and practical outcomes  
- **Personality Model**: Learns and remembers individual users and their preferences

### Utilities
- **Content Filter**: Identifies potentially harmful or inappropriate content
- **Emotion Detector**: Analyzes emotional context in user messages
- **OpenAI Helper**: Manages interactions with OpenAI's GPT-4o model
- **Web Scraper**: Extracts content from websites using Trafilatura

## Key Components

### 1. Multi-Model Decision Framework
The system implements a three-layer decision-making process where each specialized model contributes its expertise:
- **Conscience Model**: Evaluates ethical implications and human impact
- **Logic Model**: Provides rational analysis and efficiency considerations
- **Personality Model**: Adds personalization based on user history

### 2. Ethical Content Filtering
Proactive content filtering system that:
- Identifies concerning keywords and patterns
- Considers context to allow educational discussions
- Maintains safety while preserving functionality

### 3. User Personalization
Persistent user profiles that track:
- Communication preferences
- Interaction history
- Individual needs and context
- Emotional patterns

### 4. Knowledge Management
JSON-based knowledge storage for:
- Ethical principles and cases
- Logical frameworks
- User profiles and interactions
- Conversation history
- Acquired skills and capabilities

## Data Flow

1. **User Input**: Message received through Streamlit interface
2. **Content Filtering**: Initial safety and appropriateness check
3. **Emotion Detection**: Analyze emotional context of the input
4. **Model Analysis**: Parallel processing by Conscience, Logic, and Personality models
5. **Master Synthesis**: AISapienMaster weighs all inputs and generates response
6. **Personalization**: Response tailored to user's profile and preferences
7. **Learning**: System updates knowledge base and user profiles
8. **Output**: Formatted response delivered through chat interface

## External Dependencies

### Core Technologies
- **Python 3.11**: Runtime environment
- **Streamlit 1.45.1+**: Web application framework
- **OpenAI API**: GPT-4o model for natural language processing
- **Trafilatura 2.0.0+**: Web content extraction
- **Requests 2.32.4+**: HTTP client for web scraping

### Data Storage
- **JSON Files**: Persistent storage for all application data
- **File-based Architecture**: No external database required, all data stored locally

### AI Services
- **Ollama Integration**: Local LLM support for offline operation (Primary)
- **OpenAI GPT-4o**: Cloud-based alternative (Secondary)
- **Configurable Backend**: Easy switching between local and cloud AI
- **Custom Prompt Engineering**: Structured prompts for ethical and logical analysis
- **JSON Response Format**: Structured data exchange with AI models

## Deployment Strategy

### Replit Configuration
- **Python 3.11 Module**: Ensures consistent runtime environment
- **Autoscale Deployment**: Handles variable load automatically
- **Port 5000**: Standard deployment configuration
- **Streamlit Server**: Direct execution with proper port binding

### Environment Setup
- **UV Lock File**: Ensures reproducible dependency installation
- **PyProject.toml**: Modern Python packaging configuration
- **Streamlit Config**: Optimized for headless deployment

### Development Workflow
- **Parallel Execution**: Run button launches the application efficiently
- **Live Reload**: Streamlit's built-in development server
- **Port Management**: Automatic port assignment and waiting

## Changelog

- June 18, 2025: Initial setup with OpenAI integration
- June 21, 2025: Added PostgreSQL database support 
- June 21, 2025: Integrated Ollama for offline AI operation

## User Preferences

Preferred communication style: Simple, everyday language.