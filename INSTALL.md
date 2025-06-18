# AISapien Installation Guide

## Quick Start

### Option 1: Local Installation

1. **Download and extract** all project files
2. **Install Python 3.11+** from python.org
3. **Install dependencies**:
   ```bash
   pip install streamlit openai psycopg2-binary sqlalchemy requests trafilatura
   ```
4. **Set up PostgreSQL** (local or cloud)
5. **Create environment file** (.env):
   ```
   OPENAI_API_KEY=your_key_here
   DATABASE_URL=postgresql://user:password@localhost:5432/aisapien
   ```
6. **Initialize database**:
   ```bash
   python setup_database.py
   ```
7. **Run application**:
   ```bash
   streamlit run app.py --server.port 5000
   ```

### Option 2: Replit Deployment

1. **Upload project files** to Replit
2. **Add secrets**:
   - OPENAI_API_KEY
   - DATABASE_URL (if using external PostgreSQL)
3. **Run** - dependencies install automatically

## Getting Your OpenAI API Key

1. Visit https://platform.openai.com/
2. Create account or sign in
3. Go to API Keys section
4. Create new secret key
5. Copy the key (starts with "sk-")

## PostgreSQL Setup Options

### Local PostgreSQL
```bash
# Install PostgreSQL
# Create database
createdb aisapien

# Set environment variables
export DATABASE_URL="postgresql://username:password@localhost:5432/aisapien"
```

### Cloud PostgreSQL (Recommended)
- **Neon**: Free tier available at neon.tech
- **Supabase**: Free tier at supabase.com
- **AWS RDS**: Pay-per-use
- **Google Cloud SQL**: Pay-per-use

## Troubleshooting

**Can't connect to database?**
- Check DATABASE_URL format
- Verify PostgreSQL is running
- Test connection with psql

**OpenAI errors?**
- Verify API key is correct
- Check account has credits
- Test with simple API call

**Import errors?**
- Ensure Python 3.11+
- Install all dependencies
- Check virtual environment

## File Structure

Your project should look like:
```
aisapien/
├── app.py
├── setup_database.py
├── README.md
├── models/
├── utils/
├── .streamlit/
└── .env
```

## Next Steps

After installation:
1. Access web interface at localhost:5000
2. Set your user ID
3. Start chatting with AISapien
4. Try uploading documents or URLs for learning
5. Monitor the system status in the sidebar