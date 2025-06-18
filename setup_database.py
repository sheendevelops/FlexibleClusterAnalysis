#!/usr/bin/env python3
"""
Database setup script for AISapien
Creates all necessary tables and initializes default data
"""

import os
import sys
import psycopg2
from psycopg2.extras import Json

def get_database_url():
    """Get database URL from environment variables"""
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        # Try to construct from individual components
        host = os.environ.get('PGHOST', 'localhost')
        port = os.environ.get('PGPORT', '5432')
        user = os.environ.get('PGUSER', 'postgres')
        password = os.environ.get('PGPASSWORD', '')
        database = os.environ.get('PGDATABASE', 'aisapien')
        
        if password:
            database_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        else:
            database_url = f"postgresql://{user}@{host}:{port}/{database}"
    
    return database_url

def create_tables(conn):
    """Create all necessary tables"""
    with conn.cursor() as cursor:
        # User profiles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) UNIQUE NOT NULL,
                preferences JSONB DEFAULT '{}',
                communication_style JSONB DEFAULT '{}',
                interests JSONB DEFAULT '[]',
                needs JSONB DEFAULT '[]',
                personality_traits JSONB DEFAULT '{}',
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_emotional_tone VARCHAR(50) DEFAULT 'neutral',
                profile_completeness INTEGER DEFAULT 0,
                engagement_level VARCHAR(50) DEFAULT 'New User'
            )
        """)

        # User interactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_interactions (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                message TEXT NOT NULL,
                insights JSONB DEFAULT '{}',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                emotion_detected VARCHAR(50),
                response_generated TEXT
            )
        """)

        # Conscience knowledge table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conscience_knowledge (
                id SERIAL PRIMARY KEY,
                scenario TEXT,
                context TEXT,
                analysis JSONB DEFAULT '{}',
                ethical_score FLOAT,
                humanitarian_impact TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source VARCHAR(255)
            )
        """)

        # Logic knowledge table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logic_knowledge (
                id SERIAL PRIMARY KEY,
                scenario TEXT,
                context TEXT,
                analysis JSONB DEFAULT '{}',
                efficiency_score FLOAT,
                optimization_opportunities JSONB DEFAULT '[]',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source VARCHAR(255)
            )
        """)

        # Master decisions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS master_decisions (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                input_message TEXT NOT NULL,
                model_insights JSONB DEFAULT '{}',
                final_response TEXT NOT NULL,
                decision_factors JSONB DEFAULT '[]',
                time_of_day VARCHAR(20),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Skills acquired table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skills_acquired (
                id SERIAL PRIMARY KEY,
                skill_name VARCHAR(255) NOT NULL,
                skill_category VARCHAR(100) NOT NULL,
                description TEXT,
                source VARCHAR(255),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                user_message TEXT NOT NULL,
                assistant_response TEXT NOT NULL,
                model_insights JSONB DEFAULT '{}',
                emotion_detected VARCHAR(50),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Ethical principles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ethical_principles (
                id SERIAL PRIMARY KEY,
                principle VARCHAR(255) UNIQUE NOT NULL,
                description TEXT,
                category VARCHAR(100),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Logical frameworks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logical_frameworks (
                id SERIAL PRIMARY KEY,
                framework VARCHAR(255) UNIQUE NOT NULL,
                description TEXT,
                category VARCHAR(100),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Learning sources table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_sources (
                id SERIAL PRIMARY KEY,
                source_url VARCHAR(500),
                source_type VARCHAR(50),
                content_summary TEXT,
                insights_extracted JSONB DEFAULT '{}',
                knowledge_domain VARCHAR(255),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN DEFAULT TRUE
            )
        """)

    conn.commit()
    print("✓ All tables created successfully")

def insert_default_data(conn):
    """Insert default ethical principles and logical frameworks"""
    with conn.cursor() as cursor:
        # Default ethical principles
        ethical_principles = [
            "Human dignity and respect",
            "Fairness and justice",
            "Compassion and empathy",
            "Non-maleficence (do no harm)",
            "Autonomy and freedom",
            "Truth and honesty",
            "Protection of vulnerable populations",
            "Environmental responsibility"
        ]

        for principle in ethical_principles:
            cursor.execute("""
                INSERT INTO ethical_principles (principle, category) 
                VALUES (%s, %s) 
                ON CONFLICT (principle) DO NOTHING
            """, (principle, 'core'))

        # Default logical frameworks
        logical_frameworks = [
            "Cost-benefit analysis",
            "Risk assessment",
            "Efficiency optimization",
            "Resource allocation",
            "Statistical analysis",
            "Logical reasoning",
            "Process improvement",
            "Performance metrics"
        ]

        for framework in logical_frameworks:
            cursor.execute("""
                INSERT INTO logical_frameworks (framework, category) 
                VALUES (%s, %s) 
                ON CONFLICT (framework) DO NOTHING
            """, (framework, 'core'))

    conn.commit()
    print("✓ Default data inserted successfully")

def main():
    """Main setup function"""
    try:
        # Get database URL
        database_url = get_database_url()
        print(f"Connecting to database...")

        # Connect to database
        conn = psycopg2.connect(database_url)
        print("✓ Database connection established")

        # Create tables
        print("Creating database tables...")
        create_tables(conn)

        # Insert default data
        print("Inserting default data...")
        insert_default_data(conn)

        print("\n🎉 Database setup completed successfully!")
        print("\nYour AISapien database is ready to use.")
        print("You can now run: streamlit run app.py")

    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Setup error: {e}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()