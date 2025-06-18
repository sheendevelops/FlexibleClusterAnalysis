import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class UserProfile(Base):
    __tablename__ = 'user_profiles'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(255), unique=True, nullable=False)
    preferences = Column(JSON, default={})
    communication_style = Column(JSON, default={})
    interests = Column(JSON, default=[])
    needs = Column(JSON, default=[])
    personality_traits = Column(JSON, default={})
    last_updated = Column(DateTime, default=datetime.utcnow)
    last_emotional_tone = Column(String(50), default='neutral')
    profile_completeness = Column(Integer, default=0)
    engagement_level = Column(String(50), default='New User')

class UserInteraction(Base):
    __tablename__ = 'user_interactions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    insights = Column(JSON, default={})
    timestamp = Column(DateTime, default=datetime.utcnow)
    emotion_detected = Column(String(50))
    response_generated = Column(Text)

class ConscienceKnowledge(Base):
    __tablename__ = 'conscience_knowledge'
    
    id = Column(Integer, primary_key=True)
    scenario = Column(Text)
    context = Column(Text)
    analysis = Column(JSON, default={})
    ethical_score = Column(Float)
    humanitarian_impact = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    source = Column(String(255))

class LogicKnowledge(Base):
    __tablename__ = 'logic_knowledge'
    
    id = Column(Integer, primary_key=True)
    scenario = Column(Text)
    context = Column(Text)
    analysis = Column(JSON, default={})
    efficiency_score = Column(Float)
    optimization_opportunities = Column(JSON, default=[])
    timestamp = Column(DateTime, default=datetime.utcnow)
    source = Column(String(255))

class MasterDecision(Base):
    __tablename__ = 'master_decisions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(255), nullable=False)
    input_message = Column(Text, nullable=False)
    model_insights = Column(JSON, default={})
    final_response = Column(Text, nullable=False)
    decision_factors = Column(JSON, default=[])
    time_of_day = Column(String(20))
    timestamp = Column(DateTime, default=datetime.utcnow)

class SkillsAcquired(Base):
    __tablename__ = 'skills_acquired'
    
    id = Column(Integer, primary_key=True)
    skill_name = Column(String(255), nullable=False)
    skill_category = Column(String(100), nullable=False)  # technical, knowledge_domains, etc.
    description = Column(Text)
    source = Column(String(255))
    timestamp = Column(DateTime, default=datetime.utcnow)

class Conversation(Base):
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(255), nullable=False)
    user_message = Column(Text, nullable=False)
    assistant_response = Column(Text, nullable=False)
    model_insights = Column(JSON, default={})
    emotion_detected = Column(String(50))
    timestamp = Column(DateTime, default=datetime.utcnow)

class EthicalPrinciple(Base):
    __tablename__ = 'ethical_principles'
    
    id = Column(Integer, primary_key=True)
    principle = Column(String(255), unique=True, nullable=False)
    description = Column(Text)
    category = Column(String(100))
    timestamp = Column(DateTime, default=datetime.utcnow)

class LogicalFramework(Base):
    __tablename__ = 'logical_frameworks'
    
    id = Column(Integer, primary_key=True)
    framework = Column(String(255), unique=True, nullable=False)
    description = Column(Text)
    category = Column(String(100))
    timestamp = Column(DateTime, default=datetime.utcnow)

class LearningSource(Base):
    __tablename__ = 'learning_sources'
    
    id = Column(Integer, primary_key=True)
    source_url = Column(String(500))
    source_type = Column(String(50))  # url, file, text
    content_summary = Column(Text)
    insights_extracted = Column(JSON, default={})
    knowledge_domain = Column(String(255))
    timestamp = Column(DateTime, default=datetime.utcnow)
    success = Column(Boolean, default=True)

# Database connection and session management
class DatabaseManager:
    def __init__(self):
        self.database_url = os.environ.get('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Create all tables in the database"""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get a database session"""
        return self.SessionLocal()
    
    def init_default_data(self):
        """Initialize database with default ethical principles and logical frameworks"""
        session = self.get_session()
        try:
            # Add default ethical principles
            default_principles = [
                "Human dignity and respect",
                "Fairness and justice",
                "Compassion and empathy",
                "Non-maleficence (do no harm)",
                "Autonomy and freedom",
                "Truth and honesty",
                "Protection of vulnerable populations",
                "Environmental responsibility"
            ]
            
            for principle in default_principles:
                existing = session.query(EthicalPrinciple).filter_by(principle=principle).first()
                if not existing:
                    session.add(EthicalPrinciple(principle=principle, category="core"))
            
            # Add default logical frameworks
            default_frameworks = [
                "Cost-benefit analysis",
                "Risk assessment",
                "Efficiency optimization",
                "Resource allocation",
                "Statistical analysis",
                "Logical reasoning",
                "Process improvement",
                "Performance metrics"
            ]
            
            for framework in default_frameworks:
                existing = session.query(LogicalFramework).filter_by(framework=framework).first()
                if not existing:
                    session.add(LogicalFramework(framework=framework, category="core"))
            
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

# Global database manager instance
db_manager = DatabaseManager()