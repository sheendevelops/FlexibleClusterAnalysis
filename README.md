# AISapien - Ethical AI Assistant

## Overview

AISapien is an advanced AI assistant built with Streamlit that implements a multi-model architecture designed to prioritize human wellbeing and ethical decision-making. The system combines three specialized AI models (Conscience, Logic, and Personality) orchestrated by a master model to provide thoughtful, personalized, and ethically-grounded responses.

## Architecture

### Core Components
- **Master Model (AISapienMaster)**: Central orchestrator that synthesizes inputs from all other models
- **Conscience Model**: Focuses on human ethics, empathy, and humanitarian values
- **Logic Model**: Handles rational analysis, efficiency, and practical outcomes  
- **Personality Model**: Learns and remembers individual users and their preferences

### Technology Stack
- **Frontend**: Streamlit web interface
- **Backend**: Python 3.11+ with OpenAI GPT-4o integration
- **Database**: PostgreSQL for persistent data storage
- **Dependencies**: See requirements.txt for full list

## Installation Instructions

### Prerequisites
- Python 3.11 or newer
- PostgreSQL database (local or remote)
- OpenAI API key

### Local Setup

1. **Clone/Download the project files**
   - Download all files to a local directory
   - Maintain the folder structure as provided

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up PostgreSQL Database**
   - Install PostgreSQL locally or use a cloud service
   - Create a new database for AISapien
   - Note the connection details (host, port, username, password, database name)

4. **Configure Environment Variables**
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   DATABASE_URL=postgresql://username:password@localhost:5432/aisapien_db
   ```

5. **Initialize the Database**
   Run the database setup script:
   ```bash
   python setup_database.py
   ```

6. **Start the Application**
   ```bash
   streamlit run app.py --server.port 5000
   ```

   The application will be available at `http://localhost:5000`

### Cloud Deployment (Replit)

1. **Fork this project on Replit**
2. **Add secrets in Replit**:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - Database credentials (if using external PostgreSQL)
3. **Run the project** - Replit will handle dependencies automatically

## Configuration

### Streamlit Configuration
The `.streamlit/config.toml` file contains server settings:
```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
```

### Database Configuration
PostgreSQL tables are automatically created on first run:
- `user_profiles` - User personality data and preferences
- `conversations` - Chat history and interactions
- `conscience_knowledge` - Ethical analysis and cases
- `logic_knowledge` - Efficiency and optimization data
- `master_decisions` - AI decision history
- `skills_acquired` - Learning progress tracking
- `ethical_principles` - Core ethical guidelines
- `logical_frameworks` - Analytical frameworks
- `learning_sources` - Document and URL learning history

## Usage Guide

### Getting Started
1. Access the web interface at the configured port
2. Set your User ID in the sidebar
3. Start chatting with AISapien
4. Use the learning tools to upload documents or URLs

### Features
- **Real-time Chat**: Interactive conversation with ethical AI
- **Multi-Model Insights**: See how each model contributes to responses
- **Learning Mode**: Enable to allow system learning from interactions
- **Document Learning**: Upload text files for knowledge expansion
- **URL Learning**: Learn from web content
- **User Personalization**: System adapts to your communication style
- **Progress Tracking**: Monitor acquired skills and knowledge areas

### Advanced Features
- **Content Filtering**: Automatic detection of concerning content
- **Emotion Detection**: Understands emotional context in conversations
- **Time Awareness**: Provides contextually appropriate responses based on time of day
- **Export Data**: Download conversation history and insights

## Development

### Project Structure
```
aisapien/
├── app.py                 # Main Streamlit application
├── models/               # AI model implementations
│   ├── master_model.py   # Master orchestrator
│   ├── conscience_model.py # Ethics and humanity focus
│   ├── logic_model.py    # Efficiency and reasoning
│   └── personality_model.py # User personalization
├── utils/                # Utility functions
│   ├── database_helper.py # PostgreSQL operations
│   ├── openai_helper.py  # OpenAI API integration
│   ├── content_filter.py # Safety filtering
│   ├── emotion_detector.py # Emotion analysis
│   └── web_scraper.py    # Content extraction
├── .streamlit/           # Streamlit configuration
├── data/                 # Legacy JSON files (for reference)
├── requirements.txt      # Python dependencies
├── setup_database.py     # Database initialization
└── README.md            # This file
```

### Adding New Features
1. **New Model Types**: Extend the base model pattern in the `models/` directory
2. **Additional Utils**: Add helper functions in the `utils/` directory
3. **Database Schema**: Modify `database_helper.py` and update `setup_database.py`
4. **UI Components**: Extend `app.py` with new Streamlit components

### Environment Variables
- `OPENAI_API_KEY`: Required for AI functionality
- `DATABASE_URL`: PostgreSQL connection string
- `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`: Individual PostgreSQL settings

## API Integration

### OpenAI Configuration
The system uses GPT-4o (latest model as of May 2024) for all AI processing:
- Text generation and analysis
- Structured JSON responses
- Sentiment analysis
- Content summarization

### External Services
- **Trafilatura**: Web content extraction
- **PostgreSQL**: Data persistence
- **Streamlit**: Web interface framework

## Security and Privacy

### Data Protection
- All user data stored locally in your PostgreSQL database
- No data sent to third parties except OpenAI for AI processing
- Content filtering prevents processing of harmful content
- User consent required for learning from potentially sensitive topics

### API Security
- OpenAI API key should be kept secure and not shared
- Database credentials should use strong passwords
- Consider using environment variables for all secrets

## Troubleshooting

### Common Issues

**Database Connection Errors**:
- Verify PostgreSQL is running
- Check DATABASE_URL format
- Ensure database exists and user has permissions

**OpenAI API Errors**:
- Verify API key is correct and active
- Check OpenAI account has sufficient credits
- Monitor rate limits for high usage

**Import Errors**:
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify Python version is 3.11+

**Streamlit Issues**:
- Clear browser cache
- Check port 5000 is not in use by another service
- Restart the application

### Performance Optimization
- Monitor database size and clean old records periodically
- Implement pagination for large conversation histories
- Consider caching frequently accessed data
- Monitor OpenAI API usage and costs

## Contributing

### Development Setup
1. Fork the repository
2. Create a virtual environment
3. Install dependencies in development mode
4. Make changes and test thoroughly
5. Submit pull requests with clear descriptions

### Code Standards
- Follow PEP 8 for Python code style
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Include error handling for external API calls

## License

This project is designed for educational and research purposes. Please ensure compliance with:
- OpenAI API terms of service
- PostgreSQL licensing
- Streamlit licensing
- Any applicable local regulations for AI systems

## Support

### Getting Help
1. Check this README for common solutions
2. Review error logs in the Streamlit interface
3. Verify all environment variables are set correctly
4. Check PostgreSQL and OpenAI service status

### Reporting Issues
When reporting issues, include:
- Error messages and stack traces
- Environment details (OS, Python version)
- Steps to reproduce the problem
- Expected vs actual behavior

## Version History

- **v1.0**: Initial release with multi-model architecture
- **v1.1**: Added PostgreSQL database integration
- **v1.2**: Enhanced learning capabilities and content filtering

---

**Note**: This system is designed to prioritize human wellbeing and ethical considerations in all interactions. It includes safeguards to prevent harmful content processing and encourages positive, constructive conversations.