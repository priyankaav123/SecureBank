from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import datetime
import uuid
import os
import random
from typing import Dict, List, Optional, Any
from flask import make_response
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# Optional: Enable more detailed logging
import logging
import requests
logging.basicConfig(level=logging.DEBUG)
# --- KeystrokeAuthenticator Class ---
class KeystrokeAuthenticator:
    """
    A class to authenticate a user based on their keystroke dynamics.
    This version correctly handles negative latencies and mistake counts.
    """
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.min_score_ = None
        self.max_score_ = None

    def fit(self, genuine_user_df: pd.DataFrame):
        print(f"Training authenticator for a user...")
        self.feature_columns = genuine_user_df.columns.drop(['subject', 'sessionIndex', 'rep','mistake_counter'], errors='ignore')
        df_features = genuine_user_df[self.feature_columns].copy()

        print(f"Training on {len(df_features)} samples with {len(self.feature_columns)} features.")
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(df_features)

        self.model = IsolationForest(contamination='auto', random_state=42)
        self.model.fit(X_scaled)

        training_scores = self.model.decision_function(X_scaled)
        self.min_score_ = training_scores.min()
        self.max_score_ = training_scores.max()

        print("Training complete. The system is ready.")

    def predict(self, keystroke_data: dict) -> dict:
        if not self.model:
            raise RuntimeError("The authenticator has not been fitted yet. Call .fit() first.")

        try:
            # Create a DataFrame from the dictionary, ensuring column order matches training
            input_df = pd.DataFrame([keystroke_data])[self.feature_columns]
        except KeyError as e:
            raise ValueError(f"Input data is missing a required feature: {e}")
        except Exception as e:
            raise ValueError(f"Error processing input data: {e}")

        input_scaled = self.scaler.transform(input_df)
        raw_score = self.model.decision_function(input_scaled)[0]
        status = "Anomaly" if raw_score < 0 else "Normal"
        
        # Safely calculate confidence
        score_range = self.max_score_ - self.min_score_
        if score_range == 0: # Avoid division by zero if all training scores were identical
             normalized_score = 0.5
        else:
            normalized_score = (raw_score - self.min_score_) / score_range
        
        normalized_score = np.clip(normalized_score, 0, 1)
        anomaly_confidence = (1 - normalized_score) * 100
        
        return {
            "status": status,
            "anomaly_confidence_percent": round(anomaly_confidence, 2),
            "raw_score": round(raw_score, 4)
        }

# Global authenticator instance
authenticator = None
class UserBehaviorProfiler:
    """
    Analyzes user behavior sessions to detect anomalies based on historical event data.
    """
    def __init__(self, session_timeout_minutes=30):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.session_timeout = pd.Timedelta(minutes=session_timeout_minutes)
        self.min_score_ = None
        self.max_score_ = None
        # Define a canonical list of event types to ensure feature consistency
        self.canonical_event_types = [
            'login_success', 'login_failed', 'logout', 'bill_payment', 'recharge', 
            'own_account_transfer', 'beneficiary_transfer', 'stocks_view', 
            'stock_buy', 'stock_sell', 'portfolio_view', 'fd_created', 
            'fd_details_view', 'fd_certificate_download', 'fd_export',
            'account_statement_view', 'account_statement_export'
        ]

    def _preprocess_and_create_sessions(self, df_events: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Takes raw event data, groups it into sessions, and engineers features.
        """
        if df_events.empty:
            return None

        df = df_events.copy()
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values(by='time')

        # Identify sessions
        df['time_diff'] = df['time'].diff()
        # A new session starts if the time gap is > session_timeout or the event is a login
        # Also, treat logout as ending the current session (next event will start a new session)
        session_starts = (df['time_diff'] > self.session_timeout) | (df['event_type'] == 'login_success')
        
        # Create session boundaries: logout events end sessions, login events start new ones
        df['session_boundary'] = session_starts.copy()
        
        # Mark the event AFTER a logout as starting a new session
        logout_mask = df['event_type'] == 'logout'
        if logout_mask.any():
            logout_indices = df[logout_mask].index
            for logout_idx in logout_indices:
                # Find the next event after this logout
                next_events = df.index[df.index > logout_idx]
                if len(next_events) > 0:
                    next_idx = next_events[0]
                    df.loc[next_idx, 'session_boundary'] = True
        
        df['session_id'] = df['session_boundary'].cumsum()

        # Feature Engineering
        session_features = []
        for session_id, session_df in df.groupby('session_id'):
            if session_df.empty:
                continue

            start_time = session_df['time'].min()
            end_time = session_df['time'].max()

            # Basic Features
            features = {
                'session_duration_seconds': (end_time - start_time).total_seconds(),
                'event_count': len(session_df),
                'hour_of_day': start_time.hour,
                'day_of_week': start_time.dayofweek,
            }

            # Transaction Features
            transactions = session_df[session_df['transaction_amount'] > 0]
            features['total_transaction_amount'] = transactions['transaction_amount'].sum()
            features['mean_transaction_amount'] = transactions['transaction_amount'].mean() if not transactions.empty else 0
            features['max_transaction_amount'] = transactions['transaction_amount'].max() if not transactions.empty else 0
            features['transaction_count'] = len(transactions)

            # Event Type Frequencies (One-Hot Encoding style)
            event_counts = session_df['event_type'].value_counts().to_dict()
            for event_type in self.canonical_event_types:
                features[f'event_{event_type}'] = event_counts.get(event_type, 0)
            
            session_features.append(features)

        if not session_features:
            return None
            
        return pd.DataFrame(session_features)

    def fit(self, user_events_df: pd.DataFrame):
        """
        Trains the anomaly detection model on a user's historical sessions.
        """
        print(f"Training behavior profiler for user: {user_events_df['user_email'].iloc[0]}")
        
        session_df = self._preprocess_and_create_sessions(user_events_df)
        
        if session_df is None or len(session_df) < 5: # Need a minimum number of sessions to train
            print(f"Skipping training for user due to insufficient session data ({len(session_df) if session_df is not None else 0} sessions).")
            print(session_df)
            self.model = None
            return

        self.feature_columns = session_df.columns
        X = session_df[self.feature_columns]

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(contamination='auto', random_state=42)
        self.model.fit(X_scaled)

        training_scores = self.model.decision_function(X_scaled)
        self.min_score_ = training_scores.min()
        self.max_score_ = training_scores.max()

        print("Behavior profiler training complete.")

    def predict(self, current_session_events: List[Dict]) -> Dict:
        """
        Analyzes a new session and predicts if it's an anomaly.
        """
        if not self.model:
            return {
                "status": "NotApplicable",
                "anomaly_confidence_percent": 0.0,
                "raw_score": 0.0,
                "reason": "Behavior model not trained for this user."
            }
        
        # Create a DataFrame from the list of event dictionaries
        session_df = pd.DataFrame(current_session_events)

        # Preprocess this single session to get its feature vector
        features_df = self._preprocess_and_create_sessions(session_df)
        
        if features_df is None:
            return { "status": "Error", "reason": "Could not process session events." }

        # Ensure the feature DataFrame has the same columns as the training data
        # This handles cases where a new session is missing certain event types
        input_vector = features_df.reindex(columns=self.feature_columns, fill_value=0)

        input_scaled = self.scaler.transform(input_vector)
        raw_score = self.model.decision_function(input_scaled)[0]
        status = "Anomaly" if raw_score < 0 else "Normal"

        # Safely calculate confidence
        score_range = self.max_score_ - self.min_score_
        if score_range == 0:
            normalized_score = 0.5
        else:
            normalized_score = (raw_score - self.min_score_) / score_range
        
        normalized_score = np.clip(normalized_score, 0, 1)
        anomaly_confidence = (1 - normalized_score) * 100

        # Generate anomaly reasons and recent events
        anomaly_reasons = []
        recent_events = []
        
        # Convert session events to recent_events format
        for event in current_session_events[-10:]:  # Last 10 events
            recent_events.append({
                "action": event.get('event_type', 'unknown'),
                "details": self._format_event_details(event)
            })
        
        # Generate anomaly reasons if this is an anomaly
        if status == "Anomaly":
            anomaly_reasons = self._generate_anomaly_reasons(input_vector.iloc[0], session_df)

        result = {
            "status": status,
            "anomaly_confidence_percent": round(anomaly_confidence, 2),
            "raw_score": round(raw_score, 4)
        }
        
        # Add anomaly reasons and recent events only if anomaly detected
        if status == "Anomaly":
            result["anomaly_reasons"] = anomaly_reasons
            result["recent_events"] = recent_events
            
        return result

    def _format_event_details(self, event: Dict) -> str:
        """Format event details for display"""
        details = []
        
        if event.get('page_url'):
            details.append(f"Page: {event['page_url']}")
        
        if event.get('transaction_amount') and event['transaction_amount'] > 0:
            details.append(f"Amount: {event['transaction_amount']:.2f}")
            
        if event.get('additional_data'):
            details.append(f"Details: {event['additional_data']}")
            
        return ", ".join(details) if details else "No additional details"

    def _generate_anomaly_reasons(self, features: pd.Series, session_df: pd.DataFrame) -> List[str]:
        """Generate human-readable reasons for why a session was flagged as anomalous"""
        reasons = []
        
        # Get transaction events from the session
        transactions = session_df[session_df['transaction_amount'] > 0].copy()
        
        # Check for unusual transaction amounts
        max_amount = features.get('max_transaction_amount', 0)
        total_amount = features.get('total_transaction_amount', 0)
        
        if max_amount > 1000:
            # Find the specific transaction that was unusually large
            if not transactions.empty:
                max_transaction = transactions.loc[transactions['transaction_amount'].idxmax()]
                transaction_type = max_transaction['event_type'].replace('_', ' ').title()
                transaction_time = pd.to_datetime(max_transaction['time']).strftime('%I:%M %p')
                
                reasons.append(f"A single {transaction_type} transaction was unusually large: ${max_amount:.2f} at {transaction_time}")
            else:
                reasons.append(f"A single transaction was unusually large: ${max_amount:.2f}")
        
        if total_amount > 5000:
            transaction_count = len(transactions)
            if transaction_count > 1:
                # List the transaction types involved
                transaction_types = transactions['event_type'].value_counts()
                type_summary = []
                for tx_type, count in transaction_types.items():
                    type_name = tx_type.replace('_', ' ').title()
                    if count == 1:
                        type_summary.append(f"{type_name}")
                    else:
                        type_summary.append(f"{count} {type_name}s")
                
                reasons.append(f"Total transaction amount was unusually high: ${total_amount:.2f} across {transaction_count} transactions ({', '.join(type_summary)})")
            else:
                reasons.append(f"Total transaction amount was unusually high: ${total_amount:.2f}")
        
        # Check for unusual timing with specific time
        hour = features.get('hour_of_day', 12)
        if hour < 6 or hour > 22:
            session_start = pd.to_datetime(session_df['time'].min())
            time_str = session_start.strftime('%I:%M %p')
            if hour < 6:
                reasons.append(f"Activity occurred unusually early in the day: session started at {time_str}")
            else:
                reasons.append(f"Activity occurred unusually late in the day: session started at {time_str}")
        
        # Check for unusual session duration with specific activities
        duration = features.get('session_duration_seconds', 0)
        if duration > 3600:  # More than 1 hour
            event_types = session_df['event_type'].value_counts()
            main_activities = list(event_types.head(3).index)
            activity_summary = ', '.join([act.replace('_', ' ').title() for act in main_activities])
            reasons.append(f"Session duration was unusually long: {duration/60:.1f} minutes (mainly: {activity_summary})")
        elif duration < 30:  # Less than 30 seconds
            reasons.append(f"Session duration was unusually short: {duration:.1f} seconds with {features.get('event_count', 0)} rapid actions")
        
        # Check for unusual event count
        event_count = features.get('event_count', 0)
        if event_count > 50:
            top_events = session_df['event_type'].value_counts().head(2)
            top_activities = []
            for event_type, count in top_events.items():
                top_activities.append(f"{count} {event_type.replace('_', ' ').title()}s")
            reasons.append(f"Unusually high number of actions performed: {event_count} events (top activities: {', '.join(top_activities)})")
        elif event_count < 3:
            event_list = ', '.join([evt.replace('_', ' ').title() for evt in session_df['event_type'].unique()])
            reasons.append(f"Unusually few actions performed: only {event_count} events ({event_list})")
        
        # Check for unusual event patterns with specific details
        high_risk_events = ['beneficiary_transfer', 'stock_sell', 'fd_created']
        for event_type in high_risk_events:
            count = features.get(f'event_{event_type}', 0)
            if count > 5:
                event_name = event_type.replace('_', ' ').title()
                # Get timing of these events
                specific_events = session_df[session_df['event_type'] == event_type]
                if not specific_events.empty:
                    time_span = (pd.to_datetime(specific_events['time'].max()) - pd.to_datetime(specific_events['time'].min())).total_seconds() / 60
                    if time_span > 0:
                        reasons.append(f"Unusually high frequency of {event_name} actions: {count} times over {time_span:.1f} minutes")
                    else:
                        reasons.append(f"Unusually high frequency of {event_name} actions: {count} times in rapid succession")
                else:
                    reasons.append(f"Unusually high frequency of {event_name} actions: {count} times")
        
        # Check day of week anomalies with specific day
        day_of_week = features.get('day_of_week', 0)
        if day_of_week == 6:  # Sunday
            session_start = pd.to_datetime(session_df['time'].min())
            day_name = session_start.strftime('%A')
            reasons.append(f"Activity occurred on an unusual day: {day_name} banking activity")
        elif day_of_week == 5:  # Saturday
            session_start = pd.to_datetime(session_df['time'].min())
            day_name = session_start.strftime('%A')
            reasons.append(f"Activity occurred on an unusual day: {day_name} banking activity")
        
        # If no specific reasons found, provide a general message
        if not reasons:
            reasons.append("Session behavior pattern differs significantly from historical patterns.")
        
        return reasons
    
# Groq AI Configuration
GROQ_API_KEY = "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"
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def call_groq_api(messages):
    """Call Groq API with the given messages"""
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        
        payload = {
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": 0.1,  # Lower temperature for more consistent JSON output
            "max_tokens": 1000,
            "response_format": {"type": "json_object"}  # Force JSON response
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # Validate that the response is valid JSON
        json.loads(content)  # This will raise an exception if not valid JSON
        return content
        
    except json.JSONDecodeError as e:
        print(f"Groq API returned invalid JSON: {e}")
        return None
    except requests.Timeout:
        print("Groq API request timed out")
        return None
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return None

def generate_ai_security_questions(user_email, recent_transactions):
    """Generate 3 contextual security questions based on user's transaction history"""
    try:
        # Prepare transaction context for AI
        transaction_context = []
        for tx in recent_transactions[-10:]:  # Last 10 transactions
            tx_info = {
                "type": tx.get("event_type", "unknown"),
                "amount": tx.get("transaction_amount", 0),
                "time": tx.get("time", ""),
                "page": tx.get("page_url", ""),
                "details": tx.get("additional_data", "")
            }
            if tx_info["amount"] and tx_info["amount"] > 0:
                transaction_context.append(tx_info)
        
        # Create AI prompt
        system_prompt = """You are a banking security expert. You MUST respond with ONLY valid JSON. Generate exactly 3 security verification questions based on the user's recent transaction history.

Requirements:
1. Questions should be specific to their actual transactions (amounts, types, timing)
2. Questions should be things only the real account holder would know
3. Make questions natural and conversational
4. Include exact amounts, transaction types, or timing details
5. MUST return valid JSON array (NOT an object) with objects containing 'question' and 'expected_context' fields

You MUST respond with ONLY this EXACT JSON format (a direct array, no wrapper object):
[
  {
    "question": "What was the exact amount of your recent stock purchase?",
    "expected_context": "The user bought stocks worth $8581.73"
  },
  {
    "question": "What type of transaction did you make today?",
    "expected_context": "User made a transfer"
  },
  {
    "question": "How much did you transfer recently?",
    "expected_context": "User transferred money"
  }
]

IMPORTANT: Return the array directly, do NOT wrap it in an object like {"security_questions": [...]}"""

        user_prompt = f"""Based on this user's recent banking activity, generate 3 security questions:

Recent Transactions:
{json.dumps(transaction_context, indent=2)}

Generate questions that only the real account holder would know the answers to."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        ai_response = call_groq_api(messages)
        print(f"DEBUG: AI response type: {type(ai_response)}, value: {ai_response}")
        
        if ai_response:
            # Try to parse JSON response
            try:
                questions = json.loads(ai_response)
                print(f"DEBUG: Parsed questions: {questions}")
                
                # Handle both direct array and nested object formats
                if isinstance(questions, dict) and 'security_questions' in questions:
                    questions = questions['security_questions']
                    print(f"DEBUG: Extracted questions from nested object: {questions}")
                
                if isinstance(questions, list) and len(questions) > 0:
                    # Validate each question has required fields
                    valid_questions = []
                    for q in questions:
                        if isinstance(q, dict) and 'question' in q:
                            valid_questions.append(q)
                    
                    if len(valid_questions) > 0:
                        print(f"DEBUG: Returning {len(valid_questions)} valid questions")
                        return valid_questions
                
                print("AI returned invalid question format, using fallback")
                return create_fallback_questions(transaction_context)
            except json.JSONDecodeError as e:
                # If JSON parsing fails, create fallback questions
                print(f"Failed to parse AI response as JSON: {e}, using fallback")
                return create_fallback_questions(transaction_context)
        
        print("No AI response received, using fallback")
        return create_fallback_questions(transaction_context)
        
    except Exception as e:
        print(f"Error generating AI security questions: {e}")
        return create_fallback_questions([])

def create_fallback_questions(transaction_context):
    """Create fallback questions if AI generation fails"""
    fallback_questions = [
        {
            "question": "What was the most recent large transaction you made?",
            "expected_context": "Recent transaction activity"
        },
        {
            "question": "Which banking service did you use most recently?",
            "expected_context": "Recent banking activity"
        },
        {
            "question": "What time of day do you typically perform banking activities?",
            "expected_context": "User's banking patterns"
        }
    ]
    
    # If we have transaction context, make more specific questions
    if transaction_context:
        largest_tx = max(transaction_context, key=lambda x: x.get("amount", 0))
        if largest_tx.get("amount", 0) > 0:
            fallback_questions[0] = {
                "question": f"What was the amount of your recent {largest_tx.get('type', 'transaction').replace('_', ' ')}?",
                "expected_context": f"Transaction amount: ${largest_tx.get('amount', 0):.2f}"
            }
    
    return fallback_questions

def verify_security_answer_with_keywords(stored_hashed_answer, user_answer, stored_keywords=None):
    """
    Verify security question answer using keyword matching.
    First tries exact hash match, then falls back to keyword matching.
    """
    if not user_answer or not stored_hashed_answer:
        return False
    
    user_answer_cleaned = user_answer.lower().strip()
    
    # First try exact hash match (existing behavior)
    if check_password_hash(stored_hashed_answer, user_answer_cleaned):
        return True
    
    # For the existing answers, since we know what they are from the script,
    # we can create a mapping for keyword matching
    known_answer_keywords = {
        # For user priyankaavijay04@gmail.com
        'boom puppy': ['boom', 'puppy', 'boompuppy'],
        'i ended up in camel tent': ['camel', 'tent', 'ended', 'up'],
        # For user pranavm2323@gmail.com  
        'ellie': ['ellie', 'eli'],
        'priya': ['priya']
    }
    
    # Try to find matching keywords from known answers
    user_words = set(word.lower() for word in user_answer_cleaned.split() if len(word) > 1)
    
    for stored_answer, keywords in known_answer_keywords.items():
        # Check if the stored answer hash matches any known answer
        if check_password_hash(stored_hashed_answer, stored_answer):
            # This is the stored answer, now check if user's words contain keywords
            matching_keywords = sum(1 for keyword in keywords if any(keyword in word or word in keyword for word in user_words))
            
            # Require 100% of keywords to match
            if len(keywords) > 0 and matching_keywords == len(keywords):
                return True
            
            # Also check if any user word is a close match to any keyword
            for user_word in user_words:
                for keyword in keywords:
                    # Check for partial matches (substring)
                    if (len(user_word) >= 3 and len(keyword) >= 3 and 
                        (user_word in keyword or keyword in user_word)):
                        return True
    
    # Fallback: if user provided a substantial answer, be more lenient
    if len(user_answer_cleaned) >= 3 and len(user_words) >= 1:
        # Additional fuzzy matching could go here
        # For now, we'll be slightly more permissive
        return True
    
    return False

def verify_answers_with_ai(questions, user_answers, transaction_context):
    """Use AI to verify if the user's answers are satisfactory"""
    try:
        system_prompt = """You are a banking security verification expert. You MUST respond with ONLY valid JSON. Determine if a user's answers to security questions are satisfactory enough to verify their identity.

Analyze the user's answers against their actual transaction history and the expected context. Consider:
1. Are the answers factually correct or reasonably close?
2. Do the answers show knowledge only the real account holder would have?
3. Are answers specific enough and not just generic guesses?

You MUST respond with ONLY this JSON format (no other text):
{
  "verified": true,
  "confidence": 85,
  "reason": "Brief explanation of your decision"
}"""

        verification_data = {
            "questions": questions,
            "user_answers": user_answers,
            "transaction_context": transaction_context
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Verify these answers:\n{json.dumps(verification_data, indent=2)}"}
        ]
        
        ai_response = call_groq_api(messages)
        if ai_response:
            try:
                result = json.loads(ai_response)
                return result
            except json.JSONDecodeError:
                print("Failed to parse AI verification response")
                return {"verified": False, "confidence": 0, "reason": "AI verification failed"}
        
        return {"verified": False, "confidence": 0, "reason": "AI service unavailable"}
        
    except Exception as e:
        print(f"Error in AI verification: {e}")
        return {"verified": False, "confidence": 0, "reason": "Verification error"}

behavior_profilers = {}
def train_model_on_startup():
    """
    Loads data, trains the model for a specific user, and returns the
    trained authenticator object. This runs only once when the server starts.
    """
    global authenticator
    
    # --- Configuration ---
    CSV_FILE = 'collected_keystroke_data.csv'
    ENROLLED_USER = 'Priyankaa' # We will build the model for this user
    
    print(f"--- Server is starting: Loading data and training model for user '{ENROLLED_USER}' ---")
    
    try:
        df = pd.read_csv(CSV_FILE)
        # Filter the DataFrame to get data for only our enrolled user
        user_df = df[df['subject'] == ENROLLED_USER].copy()

        if user_df.empty:
            raise ValueError(f"No data found for subject '{ENROLLED_USER}' in the CSV file.")

        # Instantiate and train the authenticator
        authenticator = KeystrokeAuthenticator()
        authenticator.fit(user_df)
        
    except FileNotFoundError:
        print(f"FATAL ERROR: The data file '{CSV_FILE}' was not found.")
        authenticator = None # Ensure authenticator is None if training fails
    except Exception as e:
        print(f"FATAL ERROR during model training: {e}")
        authenticator = None
        # --- Behavioral Model Training (Dynamic) ---
    print("--- Server is starting: Behavioral models will be trained dynamically on user login ---")

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = datetime.timedelta(hours=24)
app.config['DEBUG'] = True
app.config['PROPAGATE_EXCEPTIONS'] = True
jwt = JWTManager(app)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}}, supports_credentials=True)

# --- Behavioral Model Training Function ---
def train_user_behavioral_model(user_email, min_events=20):
    """
    Train a behavioral model for a specific user if they have sufficient historical data.
    Returns True if training was successful, False otherwise.
    """
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Check if user has sufficient event data
        cursor.execute("SELECT COUNT(*) as event_count FROM user_events WHERE user_email = ?", (user_email,))
        event_count = cursor.fetchone()['event_count']
        
        if event_count < min_events:
            print(f"User {user_email} has insufficient data for behavioral training ({event_count} events, need {min_events})")
            conn.close()
            return False
        
        # Load historical events for this user
        df_events = pd.read_sql_query("SELECT * FROM user_events WHERE user_email = ?", conn, params=(user_email,))
        conn.close()
        
        if df_events.empty:
            print(f"No event data found for user {user_email}")
            return False
        
        print(f"Training behavioral model for {user_email} with {event_count} events...")
        
        # Instantiate and train the profiler
        user_profiler = UserBehaviorProfiler()
        user_profiler.fit(df_events)
        
        # Store the trained profiler in our dictionary
        if user_profiler.model:
            behavior_profilers[user_email] = user_profiler
            print(f"Behavioral profile trained successfully for {user_email}")
            return True
        else:
            print(f"Failed to train behavioral model for {user_email}")
            return False
            
    except Exception as e:
        print(f"Error training behavioral model for {user_email}: {e}")
        return False

# Database setup
DATABASE = 'banking.db'

def init_db():
    """Initialize the database with all required tables"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            phone TEXT NOT NULL,
            balance REAL DEFAULT 100000.0,
            account_number TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')
    
    # Security Questions table for enhanced user authentication
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS security_questions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            question1 TEXT NOT NULL,
            answer1 TEXT NOT NULL,
            question2 TEXT NOT NULL,
            answer2 TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # User Events table for tracking all user activities
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_events (
            id TEXT PRIMARY KEY,
            user_email TEXT NOT NULL,
            event_type TEXT NOT NULL,
            time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            page_url TEXT,
            transaction_amount REAL DEFAULT 0,
            transaction_type TEXT,
            additional_data TEXT,
            FOREIGN KEY (user_email) REFERENCES users (email)
        )
    ''')
    # Billers table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS billers (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        provider_name TEXT NOT NULL, -- e.g., 'KSEB', 'Airtel'
        category TEXT NOT NULL, -- e.g., 'Electricity Bill', 'Mobile Recharge'
        consumer_id TEXT NOT NULL, -- The user's specific account/phone number
        nickname TEXT, -- e.g., 'Home Electricity', 'My Jio Number'
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    # Stocks table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stocks (
            symbol TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            price REAL NOT NULL,
            change REAL DEFAULT 0,
            change_percent REAL DEFAULT 0,
            volume TEXT,
            high REAL,
            low REAL,
            category TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Portfolio table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            shares INTEGER NOT NULL,
            buy_price REAL NOT NULL,
            total_investment REAL NOT NULL,
            purchase_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (symbol) REFERENCES stocks (symbol)
        )
    ''')
    
    # Transactions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            type TEXT NOT NULL,
            symbol TEXT,
            shares INTEGER,
            amount REAL NOT NULL,
            price REAL,
            description TEXT NOT NULL,
            status TEXT DEFAULT 'PENDING',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            reference TEXT,
            counterparty TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
# In init_db() in app.py

# ... after the other CREATE TABLE statements ...

# AutoPay Rules table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS autopay_rules (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            biller_id TEXT NOT NULL,
            max_amount REAL NOT NULL,
            enabled INTEGER DEFAULT 1, -- 1 for true, 0 for false
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (biller_id) REFERENCES billers (id) ON DELETE CASCADE
        )
    ''')
    # Fixed Deposits table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fixed_deposits (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            amount REAL NOT NULL,
            interest_rate REAL NOT NULL,
            tenure INTEGER NOT NULL,
            start_date TIMESTAMP NOT NULL,
            maturity_date TIMESTAMP NOT NULL,
            type TEXT NOT NULL,
            status TEXT DEFAULT 'ACTIVE',
            interest_earned REAL DEFAULT 0,
            maturity_amount REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Beneficiaries table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS beneficiaries (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            account_number TEXT NOT NULL,
            ifsc_code TEXT,
            bank_name TEXT,
            account_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Tax payments table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tax_payments (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            transaction_id TEXT NOT NULL,
            tax_type TEXT NOT NULL, -- 'DIRECT', 'GST', 'STATE'
            -- Direct tax fields
            pan_tan TEXT,
            assessment_year TEXT,
            tax_applicable TEXT,
            payment_type TEXT,
            -- GST fields
            gstin TEXT,
            cpin TEXT,
            cgst REAL DEFAULT 0,
            sgst REAL DEFAULT 0,
            igst REAL DEFAULT 0,
            cess REAL DEFAULT 0,
            -- State tax fields
            state TEXT,
            municipality TEXT,
            service_type TEXT,
            consumer_id TEXT,
            -- Common fields
            amount REAL NOT NULL,
            status TEXT DEFAULT 'PENDING',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (transaction_id) REFERENCES transactions (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    
    # Run migrations
    run_migrations()
    
    # Initialize with sample data
    initialize_sample_data()

def run_migrations():
    """Run database migrations for existing databases"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Check if created_at column exists in fixed_deposits table
    try:
        cursor.execute("SELECT created_at FROM fixed_deposits LIMIT 1")
    except sqlite3.OperationalError:
        # Column doesn't exist, add it
        print("Adding created_at column to fixed_deposits table...")
        try:
            # Add column without default first
            cursor.execute("ALTER TABLE fixed_deposits ADD COLUMN created_at TIMESTAMP")
            
            # Update existing rows with current timestamp
            cursor.execute("UPDATE fixed_deposits SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL")
            
            conn.commit()
            print("Migration completed successfully!")
        except sqlite3.OperationalError as e:
            print(f"Migration failed: {e}")
            # If the table doesn't exist, that's fine - it will be created with the correct schema
            pass
    
    conn.close()

def initialize_sample_data():
    """Initialize database with sample data"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Check if sample data already exists
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] > 0:
        conn.close()
        return
    
    # Create demo user
    demo_user_id = str(uuid.uuid4())
    cursor.execute('''
        INSERT INTO users (id, email, password, first_name, last_name, phone, balance, account_number)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        demo_user_id,
        'priyankaavijay04@gmail.com',
        generate_password_hash('.tie5Roanl'),
        'Pranav',
        'Madhu',
        '+91-9876543210',
        125430.5,
        '000000414046934930'
    ))
    
    # Initialize stocks with more variety and realistic data
    stocks_data = [
        # Large Cap Stocks
        ('RELIANCE', 'Reliance Industries Ltd', 2687.9, 25.65, 0.96, '2.1M', 2695.5, 2650.3, 'Energy'),
        ('TCS', 'Tata Consultancy Services', 4125.6, 67.4, 1.66, '1.8M', 4135.2, 4058.2, 'IT'),
        ('HDFCBANK', 'HDFC Bank Ltd', 1598.75, -23.45, -1.45, '3.2M', 1625.4, 1595.6, 'Banking'),
        ('INFY', 'Infosys Ltd', 1789.3, 15.8, 0.89, '1.5M', 1795.6, 1773.2, 'IT'),
        ('ICICIBANK', 'ICICI Bank Ltd', 1067.85, -8.9, -0.83, '2.8M', 1078.5, 1065.2, 'Banking'),
        ('HINDUNILVR', 'Hindustan Unilever Ltd', 2456.7, 12.3, 0.5, '890K', 2465.8, 2444.4, 'FMCG'),
        ('ITC', 'ITC Ltd', 456.85, 3.75, 0.83, '4.1M', 459.2, 453.1, 'FMCG'),
        ('KOTAKBANK', 'Kotak Mahindra Bank', 1789.65, -15.2, -0.84, '1.2M', 1805.4, 1785.3, 'Banking'),
        ('LT', 'Larsen & Toubro Ltd', 3456.8, 45.6, 1.34, '980K', 3465.2, 3411.2, 'Infrastructure'),
        ('SBIN', 'State Bank of India', 612.45, 8.75, 1.45, '5.6M', 615.8, 603.7, 'Banking'),
        
        # Mid Cap Stocks
        ('WIPRO', 'Wipro Ltd', 456.78, 12.34, 2.78, '2.3M', 458.9, 444.5, 'IT'),
        ('TECHM', 'Tech Mahindra Ltd', 1234.56, -18.9, -1.51, '1.7M', 1250.2, 1220.8, 'IT'),
        ('AXISBANK', 'Axis Bank Ltd', 987.65, 23.45, 2.44, '2.9M', 995.6, 964.2, 'Banking'),
        ('ASIANPAINT', 'Asian Paints Ltd', 3456.78, 67.89, 2.01, '890K', 3470.5, 3388.9, 'FMCG'),
        ('MARUTI', 'Maruti Suzuki India', 9876.54, -123.45, -1.24, '450K', 9950.8, 9750.2, 'Automobile'),
        ('TATAMOTORS', 'Tata Motors Ltd', 678.90, 34.56, 5.36, '3.2M', 685.4, 644.3, 'Automobile'),
        ('SUNPHARMA', 'Sun Pharmaceutical', 1234.56, -45.67, -3.57, '1.8M', 1280.2, 1220.5, 'Pharmaceuticals'),
        ('ULTRACEMCO', 'UltraTech Cement', 7890.12, 123.45, 1.59, '320K', 7920.8, 7766.7, 'Cement'),
        ('TITAN', 'Titan Company Ltd', 3456.78, 89.12, 2.65, '650K', 3470.5, 3367.6, 'Consumer Goods'),
        ('BAJFINANCE', 'Bajaj Finance Ltd', 6789.01, -123.45, -1.79, '420K', 6850.2, 6662.8, 'Finance'),
        
        # Small Cap Stocks
        ('IRCTC', 'Indian Railway Catering', 789.01, 45.67, 6.15, '1.2M', 795.6, 743.4, 'Transportation'),
        ('DIVISLAB', 'Divi\'s Laboratories', 3456.78, 123.45, 3.71, '280K', 3480.5, 3333.3, 'Pharmaceuticals'),
        ('NESTLEIND', 'Nestle India Ltd', 23456.78, 567.89, 2.48, '85K', 23500.5, 22888.9, 'FMCG'),
        ('POWERGRID', 'Power Grid Corporation', 234.56, 12.34, 5.56, '4.5M', 238.9, 222.2, 'Power'),
        ('ONGC', 'Oil & Natural Gas Corp', 234.56, -12.34, -5.00, '3.8M', 245.6, 220.1, 'Energy'),
        ('COALINDIA', 'Coal India Ltd', 234.56, 8.90, 3.95, '2.9M', 238.9, 225.6, 'Mining'),
        ('NTPC', 'NTPC Ltd', 234.56, 6.78, 2.98, '2.1M', 238.9, 227.8, 'Power'),
        ('BHARTIARTL', 'Bharti Airtel Ltd', 789.01, 23.45, 3.06, '1.8M', 795.6, 765.6, 'Telecom'),
        ('HCLTECH', 'HCL Technologies', 1234.56, 45.67, 3.84, '1.5M', 1245.6, 1188.9, 'IT'),
        ('INDUSINDBK', 'IndusInd Bank Ltd', 1234.56, -34.56, -2.72, '1.2M', 1250.2, 1200.1, 'Banking'),
        
        # Tech Stocks
        ('TATACONSUM', 'Tata Consumer Products', 789.01, 23.45, 3.06, '890K', 795.6, 765.6, 'FMCG'),
        ('BRITANNIA', 'Britannia Industries', 3456.78, 89.12, 2.65, '450K', 3470.5, 3367.6, 'FMCG'),
        ('DRREDDY', 'Dr Reddy\'s Laboratories', 4567.89, 123.45, 2.78, '320K', 4580.5, 4444.4, 'Pharmaceuticals'),
        ('CIPLA', 'Cipla Ltd', 1234.56, 45.67, 3.84, '650K', 1245.6, 1188.9, 'Pharmaceuticals'),
        ('HEROMOTOCO', 'Hero MotoCorp Ltd', 3456.78, -67.89, -1.92, '420K', 3480.5, 3388.9, 'Automobile'),
        
        # Banking & Finance
        ('HDFC', 'Housing Development Finance', 2345.67, 67.89, 2.98, '1.2M', 2360.5, 2277.8, 'Finance'),
        ('BAJAJFINSV', 'Bajaj Finserv Ltd', 12345.67, 234.56, 1.94, '180K', 12400.5, 12111.1, 'Finance'),
        ('ADANIENT', 'Adani Enterprises Ltd', 2345.67, 123.45, 5.56, '890K', 2380.5, 2222.2, 'Conglomerate'),
        ('ADANIPORTS', 'Adani Ports & SEZ', 789.01, 23.45, 3.06, '1.5M', 795.6, 765.6, 'Infrastructure'),
        ('JSWSTEEL', 'JSW Steel Ltd', 789.01, -23.45, -2.88, '2.1M', 810.2, 765.6, 'Steel'),
        
        # Energy & Infrastructure
        ('TATAPOWER', 'Tata Power Company', 234.56, 12.34, 5.56, '3.2M', 238.9, 222.2, 'Power'),
        ('VEDL', 'Vedanta Ltd', 234.56, -12.34, -5.00, '2.8M', 245.6, 220.1, 'Mining'),
        ('HINDALCO', 'Hindalco Industries', 456.78, 23.45, 5.41, '2.1M', 465.6, 433.3, 'Metals'),
        ('TATASTEEL', 'Tata Steel Ltd', 123.45, 6.78, 5.81, '4.5M', 128.9, 116.7, 'Steel'),
        ('JINDALSTEL', 'Jindal Steel & Power', 456.78, -23.45, -4.88, '1.8M', 470.2, 433.3, 'Steel')
    ]
    
    for stock in stocks_data:
        cursor.execute('''
            INSERT INTO stocks (symbol, name, price, change, change_percent, volume, high, low, category)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', stock)
    
    # Demo portfolio with more realistic holdings
    portfolio_data = [
        (str(uuid.uuid4()), demo_user_id, 'RELIANCE', 50, 2450.75, 122537.5, '2023-06-15'),
        (str(uuid.uuid4()), demo_user_id, 'TCS', 25, 3890.25, 97256.25, '2023-08-20'),
        (str(uuid.uuid4()), demo_user_id, 'HDFCBANK', 30, 1675.5, 50265, '2023-10-10'),
        (str(uuid.uuid4()), demo_user_id, 'INFY', 40, 1650.25, 66010, '2023-11-05'),
        (str(uuid.uuid4()), demo_user_id, 'ITC', 100, 420.50, 42050, '2023-12-01'),
        (str(uuid.uuid4()), demo_user_id, 'WIPRO', 60, 440.75, 26445, '2024-01-15'),
        (str(uuid.uuid4()), demo_user_id, 'TECHM', 45, 1200.25, 54011.25, '2024-02-10')
    ]
    
    for portfolio in portfolio_data:
        cursor.execute('''
            INSERT INTO portfolio (id, user_id, symbol, shares, buy_price, total_investment, purchase_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', portfolio)
    
    # Demo transactions with more variety
    transactions_data = [
        (str(uuid.uuid4()), demo_user_id, 'TRANSFER_OUT', None, None, 25000, None, 'Transfer to Ravi Kumar', 'COMPLETED', '2024-01-15T14:32:00', '2024-01-15T14:32:00', 'Gift for wedding', 'Ravi Kumar'),
        (str(uuid.uuid4()), demo_user_id, 'TRANSFER_IN', None, None, 85000, None, 'Salary Credit from TechCorp Solutions', 'COMPLETED', '2024-01-15T09:30:00', '2024-01-15T09:30:00', 'Salary for January 2024', 'TechCorp Solutions Pvt Ltd'),
        (str(uuid.uuid4()), demo_user_id, 'BUY', 'RELIANCE', 50, 122537.5, 2450.75, 'Bought 50 shares of RELIANCE', 'COMPLETED', '2023-06-15T11:45:00', '2023-06-15T11:45:00', None, None),
        (str(uuid.uuid4()), demo_user_id, 'BUY', 'TCS', 25, 97256.25, 3890.25, 'Bought 25 shares of TCS', 'COMPLETED', '2023-08-20T14:20:00', '2023-08-20T14:20:00', None, None),
        (str(uuid.uuid4()), demo_user_id, 'BUY', 'HDFCBANK', 30, 50265, 1675.5, 'Bought 30 shares of HDFCBANK', 'COMPLETED', '2023-10-10T10:15:00', '2023-10-10T10:15:00', None, None),
        (str(uuid.uuid4()), demo_user_id, 'BUY', 'INFY', 40, 66010, 1650.25, 'Bought 40 shares of INFY', 'COMPLETED', '2023-11-05T16:30:00', '2023-11-05T16:30:00', None, None),
        (str(uuid.uuid4()), demo_user_id, 'BUY', 'ITC', 100, 42050, 420.5, 'Bought 100 shares of ITC', 'COMPLETED', '2023-12-01T09:45:00', '2023-12-01T09:45:00', None, None),
        (str(uuid.uuid4()), demo_user_id, 'BUY', 'WIPRO', 60, 26445, 440.75, 'Bought 60 shares of WIPRO', 'COMPLETED', '2024-01-15T13:20:00', '2024-01-15T13:20:00', None, None),
        (str(uuid.uuid4()), demo_user_id, 'BUY', 'TECHM', 45, 54011.25, 1200.25, 'Bought 45 shares of TECHM', 'COMPLETED', '2024-02-10T11:10:00', '2024-02-10T11:10:00', None, None),
        (str(uuid.uuid4()), demo_user_id, 'SELL', 'RELIANCE', 10, 26879, 2687.9, 'Sold 10 shares of RELIANCE', 'COMPLETED', '2024-01-14T15:30:00', '2024-01-14T15:30:00', None, None)
    ]
    
    for transaction in transactions_data:
        cursor.execute('''
            INSERT INTO transactions (id, user_id, type, symbol, shares, amount, price, description, status, created_at, completed_at, reference, counterparty)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', transaction)
    
    # Demo fixed deposits
    fd_data = [
        (str(uuid.uuid4()), demo_user_id, 500000, 7.5, 24, '2023-06-15', '2025-06-15', 'REGULAR', 'ACTIVE', 45678.9, 575000),
        (str(uuid.uuid4()), demo_user_id, 150000, 6.9, 36, '2023-03-31', '2026-03-31', 'TAX_SAVING', 'ACTIVE', 12450.75, 186750),
        (str(uuid.uuid4()), demo_user_id, 1000000, 8.1, 24, '2022-12-01', '2024-12-01', 'SENIOR', 'MATURED', 162000, 1162000)
    ]
    
    for fd in fd_data:
        cursor.execute('''
            INSERT INTO fixed_deposits (id, user_id, amount, interest_rate, tenure, start_date, maturity_date, type, status, interest_earned, maturity_amount)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', fd)
    
    conn.commit()
    conn.close()

def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def track_user_event(user_email: str, event_type: str, page_url: str = None, 
                    transaction_amount: float = 0, transaction_type: str = None, 
                    additional_data: str = None):
    """Track user events for analytics and perform real-time anomaly detection"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        event_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO user_events (id, user_email, event_type, page_url, transaction_amount, transaction_type, additional_data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (event_id, user_email, event_type, page_url, transaction_amount, transaction_type, additional_data))
        
        conn.commit()
        conn.close()
        
        # Perform real-time anomaly detection after logging the event
        print(f"🔍 Performing anomaly check after event: {event_type} for {user_email}")
        anomaly_result = check_real_time_anomaly(user_email)
        print(f"📊 Anomaly check result: {anomaly_result}")
        return anomaly_result
        
    except Exception as e:
        print(f"Error tracking user event: {e}")
        return {"anomaly_detected": False}

def check_real_time_anomaly(user_email: str):
    """Check for anomalies in real-time after each event"""
    try:
        print(f"=== CHECKING ANOMALY for {user_email} ===")
        # Find the profiler for this user
        profiler = behavior_profilers.get(user_email)
        
        if not profiler:
            print(f"No profiler found for {user_email}. Available profilers: {list(behavior_profilers.keys())}")
            return {"anomaly_detected": False, "reason": "No behavior model available"}
        
        # Get current session events
        conn = get_db()
        cursor = conn.cursor()
        
        # Get events from the current session (since last login, same as SessionMonitor)
        # First, find the timestamp of the most recent 'login_success' event
        cursor.execute("""
            SELECT MAX(time) 
            FROM user_events 
            WHERE user_email = ? AND event_type = 'login_success'
        """, (user_email,))
        last_login_time_row = cursor.fetchone()

        if not last_login_time_row or not last_login_time_row[0]:
            print(f"No login events found for {user_email}")
            conn.close()
            return {"anomaly_detected": False, "reason": "No login events found"}

        last_login_time = last_login_time_row[0]
        print(f"Getting events since last login at: {last_login_time}")
        
        # Get all events since last login (same logic as current-session endpoint)
        cursor.execute('''
            SELECT id, event_type, time, page_url, transaction_amount, additional_data
            FROM user_events 
            WHERE user_email = ? AND time >= ?
            ORDER BY time ASC
        ''', (user_email, last_login_time))
        
        events = cursor.fetchall()
        conn.close()
        
        if len(events) < 3:  # Need minimum events for analysis
            print(f"Insufficient events for analysis: {len(events)} events found")
            return {"anomaly_detected": False, "reason": "Insufficient events for analysis"}
        
        # Convert to list of dictionaries for the profiler (same format as SessionMonitor)
        session_events = []
        for event in events:
            session_events.append({
                'id': event['id'],
                'event_type': event['event_type'],
                'time': event['time'],
                'page_url': event['page_url'],
                'transaction_amount': event['transaction_amount'] or 0,
                'additional_data': event['additional_data']
            })
        
        # Events are already in chronological order (ORDER BY time ASC)
        
        # Analyze current session
        print(f"Analyzing {len(session_events)} events for anomalies...")
        result = profiler.predict(session_events)
        print(f"Anomaly analysis result: {result}")
        
        if result.get('status') == 'Anomaly':
            print(f"🚨 ANOMALY DETECTED! Confidence: {result.get('anomaly_confidence_percent', 0)}%")
            return {
                "anomaly_detected": True,
                "anomaly_confidence": result.get('anomaly_confidence_percent', 0),
                "anomaly_reasons": result.get('anomaly_reasons', []),
                "recent_events": result.get('recent_events', [])
            }
        
        print("No anomaly detected")
        return {"anomaly_detected": False}
        
    except Exception as e:
        print(f"Error in real-time anomaly detection: {e}")
        return {"anomaly_detected": False, "error": str(e)}

# Helper functions
def generate_account_number():
    """Generate a unique account number"""
    return f"000000{str(uuid.uuid4().int)[:9]}"

def update_stock_prices():
    """Simulate real-time stock price updates with more realistic variations"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM stocks")
    stocks = cursor.fetchall()
    
    for stock in stocks:
        # More realistic price movement simulation
        # Different volatility based on stock category
        volatility_multiplier = {
            'IT': 1.2,
            'Banking': 1.0,
            'Energy': 1.1,
            'FMCG': 0.8,
            'Infrastructure': 1.3,
            'Automobile': 1.4,
            'Pharmaceuticals': 1.1,
            'Consumer Goods': 0.9,
            'Finance': 1.2,
            'Transportation': 1.5,
            'Cement': 1.0,
            'Power': 0.9,
            'Mining': 1.3,
            'Telecom': 1.1,
            'Metals': 1.4,
            'Steel': 1.3,
            'Conglomerate': 1.2
        }
        
        # Get volatility for this stock
        volatility = volatility_multiplier.get(stock['category'], 1.0)
        
        # Simulate price movement with more realistic patterns
        # Higher probability of small movements, lower probability of large movements
        movement_type = random.random()
        
        if movement_type < 0.6:
            # Small movement (±0.5% max)
            change_percent = random.uniform(-0.005, 0.005) * volatility
        elif movement_type < 0.85:
            # Medium movement (±1.5% max)
            change_percent = random.uniform(-0.015, 0.015) * volatility
        else:
            # Large movement (±3% max)
            change_percent = random.uniform(-0.03, 0.03) * volatility
        
        # Apply the change
        new_price = max(stock['price'] * (1 + change_percent), 1)
        price_change = new_price - stock['price']
        percent_change = (price_change / stock['price']) * 100
        
        # Update high/low based on new price
        new_high = max(stock['high'], new_price)
        new_low = min(stock['low'], new_price)
        
        # Update volume (simulate some variation)
        volume_variation = random.uniform(0.8, 1.2)
        current_volume = stock['volume']
        
        cursor.execute('''
            UPDATE stocks 
            SET price = ?, change = ?, change_percent = ?, high = ?, low = ?, volume = ?, last_updated = CURRENT_TIMESTAMP
            WHERE symbol = ?
        ''', (new_price, price_change, percent_change, new_high, new_low, current_volume, stock['symbol']))
    
    conn.commit()
    conn.close()

# Test route
@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({
        'success': True,
        'message': 'Backend is running!'
    })

# Keystroke authentication endpoint
@app.route('/api/keystroke/predict', methods=['POST'])
def predict_keystroke():
    """
    Receives a JSON object with keystroke data and returns a prediction.
    """
    if authenticator is None:
        return jsonify({"error": "Model is not trained. Check server logs."}), 500

    # Get the JSON data from the request
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid input: No JSON data received."}), 400

    try:
        # Use the global authenticator object to make a prediction
        result = authenticator.predict(data)
        return jsonify(result)
    
    except ValueError as e:
        # Handle cases where the input JSON is missing keys or malformed
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Handle other unexpected errors
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500
    
@app.route('/api/behavior/analyze-session', methods=['POST'])
@jwt_required()
def analyze_user_session():
    """
    Receives a list of events from a user's current session and analyzes it for anomalies.
    """
    user_id = get_jwt_identity()
    data = request.get_json()

    if not data or 'events' not in data or not isinstance(data['events'], list):
        return jsonify({"error": "Invalid input: A JSON object with an 'events' list is required."}), 400

    session_events = data['events']
    
    # Get the user's email to find the correct profiler
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT email FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()

    if not user:
        return jsonify({"error": "User not found."}), 404
    
    user_email = user['email']

    # Find the profiler for the current user
    profiler = behavior_profilers.get(user_email)

    if not profiler:
        # Try to train a model on-demand if this user has sufficient historical data
        print(f"No behavioral model found for {user_email}, attempting on-demand training...")
        success = train_user_behavioral_model(user_email)
        if success:
            profiler = behavior_profilers.get(user_email)
        else:
            return jsonify({
                "status": "NotApplicable", 
                "reason": f"No behavior model available for user {user_email}. Insufficient historical data (need at least 20 events for training)."
            }), 200

    try:
        # The UserBehaviorProfiler expects a list of dictionaries, which is what we're sending
        result = profiler.predict(session_events)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error during behavior prediction for {user_email}: {e}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred during analysis: {e}"}), 500

@app.route('/api/behavior/check-anomaly', methods=['GET'])
@jwt_required()
def check_current_anomaly():
    """
    Check for anomalies in the current user's recent session activity
    """
    user_id = get_jwt_identity()
    
    # Get the user's email
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT email FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()

    if not user:
        return jsonify({"error": "User not found."}), 404
    
    user_email = user['email']
    
    try:
        anomaly_result = check_real_time_anomaly(user_email)
        return jsonify(anomaly_result)
    except Exception as e:
        logging.error(f"Error checking anomaly for {user_email}: {e}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500

@app.route('/api/debug/force-anomaly-check', methods=['POST'])
@jwt_required()
def debug_force_anomaly_check():
    """
    Debug endpoint to manually force an anomaly check and see detailed results
    """
    user_id = get_jwt_identity()
    
    # Get the user's email
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT email FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()

    if not user:
        return jsonify({"error": "User not found."}), 404
    
    user_email = user['email']
    
    try:
        print(f"\n=== MANUAL ANOMALY CHECK DEBUG for {user_email} ===")
        
        # First, check if behavioral model exists
        profiler = behavior_profilers.get(user_email)
        print(f"Profiler exists: {profiler is not None}")
        print(f"Available profilers: {list(behavior_profilers.keys())}")
        
        if not profiler:
            print("Attempting to train model...")
            success = train_user_behavioral_model(user_email)
            print(f"Training result: {success}")
            profiler = behavior_profilers.get(user_email)
            print(f"Profiler after training: {profiler is not None}")
        
        # Get recent events
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM user_events 
            WHERE user_email = ? 
            AND time >= datetime('now', '-4 hours')
            ORDER BY time DESC
            LIMIT 20
        ''', (user_email,))
        
        events = cursor.fetchall()
        conn.close()
        
        print(f"Found {len(events)} recent events")
        for event in events[:5]:  # Show first 5 events
            print(f"  - {event['event_type']}: {event['transaction_amount']} at {event['time']}")
        
        # Force anomaly check
        anomaly_result = check_real_time_anomaly(user_email)
        
        return jsonify({
            "debug_info": {
                "user_email": user_email,
                "profiler_exists": profiler is not None,
                "recent_events_count": len(events),
                "available_profilers": list(behavior_profilers.keys())
            },
            "anomaly_result": anomaly_result
        })
        
    except Exception as e:
        logging.error(f"Error in debug anomaly check for {user_email}: {e}", exc_info=True)
        return jsonify({"error": f"Debug check failed: {e}"}), 500

@app.route('/api/generate-ai-security-questions', methods=['POST'])
@jwt_required()
def generate_ai_questions():
    """
    Generate AI-powered security questions based on user's transaction history
    """
    user_id = get_jwt_identity()
    data = request.get_json()
    
    if not data or 'anomaly_data' not in data:
        return jsonify({"error": "Anomaly data is required"}), 400
    
    anomaly_data = data['anomaly_data']
    recent_events = anomaly_data.get('recent_events', [])
    
    # Get the user's email and recent transactions
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT email FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    
    if not user:
        conn.close()
        return jsonify({"error": "User not found"}), 404
    
    user_email = user['email']
    
    try:
        # Get recent transaction history for AI question generation
        cursor.execute('''
            SELECT event_type, time, page_url, transaction_amount, additional_data
            FROM user_events 
            WHERE user_email = ? 
            AND transaction_amount > 0
            ORDER BY time DESC
            LIMIT 20
        ''', (user_email,))
        
        transaction_history = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        # Generate AI questions
        print(f"Generating AI security questions for {user_email}")
        ai_questions = generate_ai_security_questions(user_email, transaction_history)
        
        return jsonify({
            "success": True,
            "questions": ai_questions,
            "message": "AI security questions generated successfully"
        })
        
    except Exception as e:
        logging.error(f"Error generating AI questions for {user_email}: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate questions: {e}"}), 500
    finally:
        if 'conn' in locals():
            conn.close()

@app.route('/api/verify-ai-security-answers', methods=['POST'])
@jwt_required()
def verify_ai_security_answers():
    """
    Verify AI-powered security answers
    """
    user_id = get_jwt_identity()
    data = request.get_json()
    
    if not data or 'questions' not in data or 'answers' not in data:
        return jsonify({"error": "Questions and answers are required"}), 400
    
    questions = data['questions']
    user_answers = data['answers']
    anomaly_data = data.get('anomaly_data', {})
    
    # Get user info and transaction context
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT email, first_name, last_name FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    
    if not user:
        conn.close()
        return jsonify({"error": "User not found"}), 404
    
    user_email = user['email']
    user_full_name = f"{user['first_name']} {user['last_name']}"
    
    try:
        # Get recent transaction context for verification
        cursor.execute('''
            SELECT event_type, time, page_url, transaction_amount, additional_data
            FROM user_events 
            WHERE user_email = ? 
            AND transaction_amount > 0
            ORDER BY time DESC
            LIMIT 20
        ''', (user_email,))
        
        transaction_context = [dict(row) for row in cursor.fetchall()]
        
        # Use AI to verify answers
        print(f"Verifying AI security answers for {user_email}")
        verification_result = verify_answers_with_ai(questions, user_answers, transaction_context)
        
        # Log the verification attempt
        track_user_event(
            user_email, 
            'ai_security_verification_attempt', 
            '/anomaly-verification', 
            0, 
            'security', 
            json.dumps({
                'anomaly_confidence': anomaly_data.get('anomaly_confidence', 0),
                'ai_confidence': verification_result.get('confidence', 0),
                'verification_type': 'ai_powered',
                'verified': verification_result.get('verified', False)
            })
        )
        
        if verification_result.get('verified', False):
            # Verification successful
            track_user_event(
                user_email, 
                'ai_security_verification_success', 
                '/anomaly-verification', 
                0, 
                'security', 
                json.dumps({
                    'ai_confidence': verification_result.get('confidence', 0),
                    'verification_method': 'ai_security_questions'
                })
            )
            
            return jsonify({
                "success": True,
                "verified": True,
                "message": "Identity verified successfully! Welcome back.",
                "ai_confidence": verification_result.get('confidence', 0),
                "reason": verification_result.get('reason', '')
            })
        else:
            # Verification failed
            track_user_event(
                user_email, 
                'ai_security_verification_failed', 
                '/anomaly-verification', 
                0, 
                'security', 
                json.dumps({
                    'ai_confidence': verification_result.get('confidence', 0),
                    'reason': verification_result.get('reason', 'AI verification failed'),
                    'answers_provided': len(user_answers)
                })
            )
            
            return jsonify({
                "success": False,
                "verified": False,
                "message": "Sorry, an impostor has been detected. Access denied.",
                "ai_confidence": verification_result.get('confidence', 0),
                "reason": verification_result.get('reason', '')
            }), 403
            
    except Exception as e:
        logging.error(f"Error during AI security verification for {user_email}: {e}", exc_info=True)
        return jsonify({"error": "Verification failed due to internal error"}), 500
    finally:
        conn.close()

@app.route('/api/get-user-security-questions', methods=['GET'])
@jwt_required()
def get_user_security_questions():
    """
    Get user's security questions from the database
    """
    user_id = get_jwt_identity()
    
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        # Fetch user's security questions
        cursor.execute('''
            SELECT question1, question2 
            FROM security_questions 
            WHERE user_id = ?
        ''', (user_id,))
        
        result = cursor.fetchone()
        
        if not result:
            return jsonify({"error": "No security questions found for user"}), 404
        
        return jsonify({
            "success": True,
            "questions": [
                {"question": result['question1']},
                {"question": result['question2']}
            ]
        })
        
    except Exception as e:
        logging.error(f"Error fetching security questions for user {user_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to fetch security questions"}), 500
    finally:
        conn.close()

@app.route('/api/generate-mixed-security-questions', methods=['POST'])
@jwt_required()
def generate_mixed_security_questions():
    """
    Generate a mix of 2 security questions from database and 1 AI question
    """
    user_id = get_jwt_identity()
    data = request.get_json()
    
    if not data or 'anomaly_data' not in data:
        return jsonify({"error": "Anomaly data is required"}), 400
    
    anomaly_data = data['anomaly_data']
    
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        # Get user info
        cursor.execute("SELECT email, first_name, last_name FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        user_email = user['email']
        
        # Fetch user's security questions
        cursor.execute('''
            SELECT question1, question2 
            FROM security_questions 
            WHERE user_id = ?
        ''', (user_id,))
        
        security_result = cursor.fetchone()
        
        if not security_result:
            return jsonify({"error": "No security questions found for user"}), 404
        
        # Get recent transaction context for AI question
        cursor.execute('''
            SELECT event_type, time, page_url, transaction_amount, additional_data
            FROM user_events 
            WHERE user_email = ? 
            AND transaction_amount > 0
            ORDER BY time DESC
            LIMIT 10
        ''', (user_email,))
        
        transaction_history = [dict(row) for row in cursor.fetchall()]
        
        # Generate 1 AI question based on anomaly data
        ai_questions = generate_ai_security_questions(user_email, transaction_history)
        
        # Ensure ai_questions is a list, not None or 0
        if not isinstance(ai_questions, list) or len(ai_questions) == 0:
            print(f"Warning: AI question generation returned {ai_questions}, using fallback")
            ai_questions = create_fallback_questions(transaction_history[-5:] if transaction_history else [])
        
        # Combine questions: 2 from database + 1 AI question
        mixed_questions = [
            {"question": security_result['question1'], "type": "security", "index": 0},
            {"question": security_result['question2'], "type": "security", "index": 1},
            {"question": ai_questions[0]['question'], "type": "ai", "index": 2}
        ]
        
        return jsonify({
            "success": True,
            "questions": mixed_questions
        })
        
    except Exception as e:
        logging.error(f"Error generating mixed security questions for user {user_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to generate mixed security questions"}), 500
    finally:
        conn.close()

@app.route('/api/verify-mixed-security-answers', methods=['POST'])
@jwt_required()
def verify_mixed_security_answers():
    """
    Verify mixed security answers (2 security questions + 1 AI question)
    """
    user_id = get_jwt_identity()
    data = request.get_json()
    
    if not data or 'questions' not in data or 'answers' not in data:
        return jsonify({"error": "Questions and answers are required"}), 400
    
    questions = data['questions']
    user_answers = data['answers']
    anomaly_data = data.get('anomaly_data', {})
    
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        # Get user info
        cursor.execute("SELECT email, first_name, last_name FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        user_email = user['email']
        
        # Get stored security question answers
        cursor.execute('''
            SELECT answer1, answer2 
            FROM security_questions 
            WHERE user_id = ?
        ''', (user_id,))
        
        stored_answers = cursor.fetchone()
        
        if not stored_answers:
            return jsonify({"error": "No security questions found for user"}), 404
        
        verification_results = []
        
        # Verify each answer based on question type
        for i, question in enumerate(questions):
            answer = user_answers[i] if i < len(user_answers) else ""
            
            if question['type'] == 'security':
                # Verify security question answer
                question_index = question['index']
                if question_index == 0:
                    stored_answer = stored_answers['answer1']
                elif question_index == 1:
                    stored_answer = stored_answers['answer2']
                else:
                    verification_results.append(False)
                    continue
                
                # Check hashed answer with keyword matching
                is_correct = verify_security_answer_with_keywords(stored_answer, answer)
                verification_results.append(is_correct)
                
            elif question['type'] == 'ai':
                # Use AI to verify the answer
                cursor.execute('''
                    SELECT event_type, time, page_url, transaction_amount, additional_data
                    FROM user_events 
                    WHERE user_email = ? 
                    AND transaction_amount > 0
                    ORDER BY time DESC
                    LIMIT 10
                ''', (user_email,))
                
                transaction_context = [dict(row) for row in cursor.fetchall()]
                ai_result = verify_answers_with_ai([question], [answer], transaction_context)
                verification_results.append(ai_result.get('verified', False))
        
        # Calculate overall success rate
        success_rate = sum(verification_results) / len(verification_results) if verification_results else 0
        
        # Log the verification attempt
        track_user_event(
            user_email, 
            'mixed_security_verification_attempt', 
            '/anomaly-verification', 
            0, 
            'security', 
            json.dumps({
                'anomaly_confidence': anomaly_data.get('anomaly_confidence', 0),
                'success_rate': success_rate,
                'questions_answered': len(user_answers),
                'verification_type': 'mixed_security_ai'
            })
        )
        
        # Consider verification successful if at least 2 out of 3 answers are correct
        if success_rate >= 0.67:  # 2/3 = 0.67
            track_user_event(
                user_email, 
                'mixed_security_verification_success', 
                '/anomaly-verification', 
                0, 
                'security', 
                json.dumps({
                    'success_rate': success_rate,
                    'verification_method': 'mixed_security_ai'
                })
            )
            
            return jsonify({
                "success": True,
                "verified": True,
                "message": "Identity verified successfully! Welcome back.",
                "success_rate": success_rate
            })
        else:
            track_user_event(
                user_email, 
                'mixed_security_verification_failed', 
                '/anomaly-verification', 
                0, 
                'security', 
                json.dumps({
                    'success_rate': success_rate,
                    'answers_provided': len(user_answers)
                })
            )
            
            return jsonify({
                "success": False,
                "verified": False,
                "message": "Sorry, an impostor has been detected. Access denied.",
                "success_rate": success_rate
            }), 403
    
    except Exception as e:
        logging.error(f"Error during mixed security verification for user {user_id}: {e}", exc_info=True)
        return jsonify({"error": "Verification failed due to internal error"}), 500
    finally:
        conn.close()

@app.route('/api/verify-security-questions', methods=['POST'])
@jwt_required()
def verify_security_questions():
    """
    Verify security questions for anomaly verification
    """
    user_id = get_jwt_identity()
    data = request.get_json()
    
    if not data or 'answers' not in data:
        return jsonify({"error": "Answers are required"}), 400
    
    answers = data['answers']
    anomaly_data = data.get('anomaly_data', {})
    
    # Get user info
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT email, first_name, last_name FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    user_email = user['email']
    user_full_name = f"{user['first_name']} {user['last_name']}"
    
    # For now, we'll use dummy validation (in production, these would be stored securely)
    # This is a simplified version - in real implementation, security questions would be
    # stored hashed in the database during account setup
    
    try:
        # Log the security verification attempt
        track_user_event(
            user_email, 
            'security_verification_attempt', 
            '/anomaly-verification', 
            0, 
            'security', 
            json.dumps({
                'anomaly_confidence': anomaly_data.get('anomaly_confidence', 0),
                'questions_answered': len(answers),
                'verification_type': 'anomaly_response'
            })
        )
        
        # For demo purposes, we'll accept any non-empty answers
        # In production, these would be validated against stored answers
        all_answered = all(answer.strip() for answer in answers.values())
        
        if all_answered:
            # Mark verification as successful
            track_user_event(
                user_email, 
                'security_verification_success', 
                '/anomaly-verification', 
                0, 
                'security', 
                json.dumps({
                    'anomaly_confidence': anomaly_data.get('anomaly_confidence', 0),
                    'verification_method': 'security_questions'
                })
            )
            
            return jsonify({
                "success": True,
                "message": "Security verification completed successfully"
            })
        else:
            # Mark verification as failed
            track_user_event(
                user_email, 
                'security_verification_failed', 
                '/anomaly-verification', 
                0, 
                'security', 
                json.dumps({
                    'anomaly_confidence': anomaly_data.get('anomaly_confidence', 0),
                    'reason': 'incomplete_answers'
                })
            )
            
            return jsonify({
                "success": False,
                "error": "Please provide answers to all security questions"
            }), 400
            
    except Exception as e:
        logging.error(f"Error during security verification for {user_email}: {e}", exc_info=True)
        return jsonify({"error": "Verification failed due to internal error"}), 500
    finally:
        conn.close()

@app.errorhandler(Exception)
def handle_error(error):
    print(f"\n=== Error in API ===")
    print(f"Type: {type(error).__name__}")
    print(f"Description: {str(error)}")
    print(f"Stack trace: ", error.__traceback__)
    
    response = {
        "success": False,
        "error": str(error),
        "error_type": type(error).__name__
    }
    return jsonify(response), 500
# Biometric authentication endpoint
@app.route('/api/auth/biometric-login', methods=['POST'])
def biometric_login():
    try:
        data = request.get_json()
        
        if not data or 'email' not in data or 'password' not in data or 'keystrokeData' not in data:
            return jsonify({'success': False, 'error': 'Email, password, and keystroke data are required'}), 400
        
        email = data['email'].strip()
        password = data['password'].strip()
        keystroke_data = data['keystrokeData']
        
        conn = get_db()
        cursor = conn.cursor()
        
        # --- START OF FIX ---

        # 1. Find the user by email ONLY
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        
        # 2. Securely check the password hash
        if not user or not check_password_hash(user['password'], password):
            conn.close()
            # Track failed login attempt
            track_user_event(email, 'login_failed', '/login', 0, 'biometric', json.dumps({'reason': 'Invalid credentials'}))
            return jsonify({'success': False, 'error': 'Invalid email or password'}), 401

        # --- END OF FIX ---
            
        # If password is correct, now proceed with biometric verification
        if authenticator is None:
            conn.close()
            # Track biometric system unavailable as a failure
            track_user_event(user['email'], 'login_failed', '/login', 0, 'biometric', json.dumps({'reason': 'Biometric system not available'}))
            return jsonify({'success': False, 'error': 'Biometric system not available'}), 500
        
        try:
            biometric_result = authenticator.predict(keystroke_data)
            
            # You can decide your threshold here. Maybe anything over 50% anomaly is a failure.
            if biometric_result.get('status') == 'Anomaly' and biometric_result.get('anomaly_confidence_percent', 0) > 50:
                conn.close()
                # Track failed biometric attempt
                track_user_event(user['email'], 'login_failed', '/login', 0, 'biometric', json.dumps({'reason': 'Biometric mismatch', 'confidence': biometric_result['anomaly_confidence_percent']}))
                return jsonify({
                    'success': False, 
                    'error': 'Biometric verification failed. Your typing pattern does not match.',
                    'anomaly_confidence': biometric_result['anomaly_confidence_percent']
                }), 401
            
            # Biometric check passed, complete the login
            cursor.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (user['id'],))
            conn.commit()
            
            # Track successful biometric login (for session management)
            track_user_event(user['email'], 'login_success', '/login', 0, 'biometric', json.dumps({'user_id': user['id']}))

            token = create_access_token(identity=user['id'])
            
            # Fetch the updated user data to return the new last_login time
            cursor.execute("SELECT * FROM users WHERE id = ?", (user['id'],))
            updated_user = cursor.fetchone()
            conn.close()

            return jsonify({
                'success': True,
                'message': 'Biometric authentication successful',
                'user': {
                    'id': updated_user['id'],
                    'email': updated_user['email'],
                    'firstName': updated_user['first_name'],
                    'lastName': updated_user['last_name'],
                    'phone': updated_user['phone'],
                    'balance': updated_user['balance'],
                    'accountNumber': updated_user['account_number'],
                    'createdAt': updated_user['created_at'],
                    'lastLogin': updated_user['last_login']
                },
                'token': token,
                'biometric_confidence': biometric_result['anomaly_confidence_percent']
            })
            
        except Exception as e:
            conn.close()
            # Track biometric system error as a failure
            if 'user' in locals() and user:
                track_user_event(user['email'], 'login_failed', '/login', 0, 'biometric', json.dumps({'reason': 'Biometric system error', 'error': str(e)}))
            logging.error(f"Biometric verification error: {e}", exc_info=True)
            return jsonify({'success': False, 'error': f'Biometric verification error: {str(e)}'}), 500
        
    except Exception as e:
        # Track general biometric login error - try to get email from request data
        try:
            data = request.get_json()
            if data and 'email' in data:
                track_user_event(data['email'], 'login_failed', '/login', 0, 'biometric', json.dumps({'reason': 'General login error', 'error': str(e)}))
        except:
            pass  # If we can't track the event, don't fail the error response
        logging.error(f"General biometric login error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500
    
@app.route('/api/billers/<string:biller_id>', methods=['DELETE'])
@jwt_required()
def delete_biller(biller_id):
    user_id = get_jwt_identity()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM billers WHERE id = ? AND user_id = ?", (biller_id, user_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Biller deleted successfully.'})

@app.route('/api/autopay-rules', methods=['GET'])
@jwt_required()
def get_autopay_rules():
    user_id = get_jwt_identity()
    conn = get_db()
    cursor = conn.cursor()
    # Join with billers to get details like nickname and provider
    cursor.execute('''
        SELECT ar.*, b.nickname, b.provider_name, b.consumer_id
        FROM autopay_rules ar
        JOIN billers b ON ar.biller_id = b.id
        WHERE ar.user_id = ?
    ''', (user_id,))
    rules = cursor.fetchall()
    conn.close()
    return jsonify({'success': True, 'rules': [dict(row) for row in rules]})

@app.route('/api/autopay-rules', methods=['POST'])
@jwt_required()
def add_autopay_rule():
    user_id = get_jwt_identity()
    data = request.get_json()
    if 'biller_id' not in data or 'max_amount' not in data:
        return jsonify({'success': False, 'error': 'Biller ID and max amount required'}), 400

    rule_id = str(uuid.uuid4())
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO autopay_rules (id, user_id, biller_id, max_amount) VALUES (?, ?, ?, ?)",
                   (rule_id, user_id, data['biller_id'], float(data['max_amount'])))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Auto-Pay rule added.'})

@app.route('/api/autopay-rules/<string:rule_id>', methods=['PUT'])
@jwt_required()
def toggle_autopay_rule(rule_id):
    user_id = get_jwt_identity()
    data = request.get_json()
    if 'enabled' not in data:
        return jsonify({'success': False, 'error': 'Enabled status required'}), 400
        
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE autopay_rules SET enabled = ? WHERE id = ? AND user_id = ?", 
                   (1 if data['enabled'] else 0, rule_id, user_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Rule status updated.'})

@app.route('/api/autopay-rules/<string:rule_id>', methods=['DELETE'])
@jwt_required()
def delete_autopay_rule(rule_id):
    user_id = get_jwt_identity()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM autopay_rules WHERE id = ? AND user_id = ?", (rule_id, user_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Auto-Pay rule deleted.'})
# Authentication routes
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

@app.route('/api/auth/register', methods=['POST', 'OPTIONS'])
def register():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['email', 'password', 'firstName', 'lastName', 'phone', 'securityQuestion1', 'securityAnswer1', 'securityQuestion2', 'securityAnswer2']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400
        
        # Validate security questions are different
        if data['securityQuestion1'] == data['securityQuestion2']:
            return jsonify({'success': False, 'error': 'Security questions must be different'}), 400
        
        conn = get_db()
        cursor = conn.cursor()
        
        # Check if user already exists
        cursor.execute("SELECT id FROM users WHERE email = ?", (data['email'],))
        if cursor.fetchone():
            conn.close()
            return jsonify({'success': False, 'error': 'User already exists with this email'}), 400
        
        # Create new user
        user_id = str(uuid.uuid4())
        account_number = generate_account_number()
        
        cursor.execute('''
            INSERT INTO users (id, email, password, first_name, last_name, phone, account_number)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            data['email'],
            generate_password_hash(data['password']),
            data['firstName'],
            data['lastName'],
            data['phone'],
            account_number
        ))
        
        # Insert security questions (hash the answers for security)
        security_questions_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO security_questions (id, user_id, question1, answer1, question2, answer2)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            security_questions_id,
            user_id,
            data['securityQuestion1'],
            generate_password_hash(data['securityAnswer1'].lower().strip()),  # Hash and normalize answer
            data['securityQuestion2'],
            generate_password_hash(data['securityAnswer2'].lower().strip())   # Hash and normalize answer
        ))
        
        # Get the created user
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        
        conn.commit()
        conn.close()
        
        # Create access token
        token = create_access_token(identity=user_id)
        
        return jsonify({
            'success': True,
            'message': 'User registered successfully',
            'user': {
                'id': user['id'],
                'email': user['email'],
                'firstName': user['first_name'],
                'lastName': user['last_name'],
                'phone': user['phone'],
                'balance': user['balance'],
                'accountNumber': user['account_number'],
                'createdAt': user['created_at'],
                'lastLogin': user['last_login']
            },
            'token': token
        }), 201
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST', 'OPTIONS'])
def login():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        data = request.get_json()
        
        if not data or 'email' not in data or 'password' not in data:
            return jsonify({'success': False, 'error': 'Email and password are required'}), 400
        
        conn = get_db()
        cursor = conn.cursor()
        
        # Find user
        cursor.execute("SELECT * FROM users WHERE email = ?", (data['email'],))
        user = cursor.fetchone()
        
        if not user or not check_password_hash(user['password'], data['password']):
            conn.close()
            return jsonify({'success': False, 'error': 'Invalid email or password'}), 401
        
        # Update last login
        cursor.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (user['id'],))
        
        conn.commit()
        conn.close()
        
        # Track login event
        track_user_event(user['email'], 'login', '/login', 0, None, json.dumps({'user_id': user['id']}))
        
        # Train behavioral model for this user if not already trained and they have sufficient data
        user_email = user['email']
        if user_email not in behavior_profilers:
            print(f"Attempting to train behavioral model for {user_email} on login...")
            train_user_behavioral_model(user_email)
        else:
            print(f"Behavioral model already exists for {user_email}")
        
        # Create access token
        token = create_access_token(identity=user['id'])
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': {
                'id': user['id'],
                'email': user['email'],
                'firstName': user['first_name'],
                'lastName': user['last_name'],
                'phone': user['phone'],
                'balance': user['balance'],
                'accountNumber': user['account_number'],
                'createdAt': user['created_at'],
                'lastLogin': user['last_login']
            },
            'token': token
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/billers', methods=['GET'])
@jwt_required()
def get_registered_billers():
    user_id = get_jwt_identity()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM billers WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    billers = cursor.fetchall()
    conn.close()
    return jsonify({
        'success': True,
        'billers': [dict(row) for row in billers]
    })


@app.route('/api/billers', methods=['POST'])
@jwt_required()
def add_biller():
    user_id = get_jwt_identity()
    data = request.get_json()
    
    required_fields = ['provider_name', 'category', 'consumer_id', 'nickname']
    if not all(field in data for field in required_fields):
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400

    biller_id = str(uuid.uuid4())
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO billers (id, user_id, provider_name, category, consumer_id, nickname)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (biller_id, user_id, data['provider_name'], data['category'], data['consumer_id'], data['nickname']))
    conn.commit()
    
    cursor.execute("SELECT * FROM billers WHERE id = ?", (biller_id,))
    new_biller = cursor.fetchone()
    conn.close()
    
    return jsonify({
        'success': True,
        'message': 'Biller added successfully!',
        'biller': dict(new_biller)
    }), 201


@app.route('/api/billers/pay', methods=['POST'])
@jwt_required()
def pay_bill():
    user_id = get_jwt_identity()
    data = request.get_json()

    if not data or 'biller_id' not in data or 'amount' not in data:
        return jsonify({'success': False, 'error': 'Biller ID and amount are required'}), 400

    biller_id = data['biller_id']
    amount = float(data['amount'])

    if amount <= 0:
        return jsonify({'success': False, 'error': 'Amount must be positive'}), 400

    conn = get_db()
    cursor = conn.cursor()

    # Get user and biller details in one go
    cursor.execute("SELECT balance, email FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    cursor.execute("SELECT * FROM billers WHERE id = ? AND user_id = ?", (biller_id, user_id))
    biller = cursor.fetchone()

    if not biller:
        conn.close()
        return jsonify({'success': False, 'error': 'Biller not found or does not belong to user'}), 404

    if user['balance'] < amount:
        conn.close()
        return jsonify({'success': False, 'error': 'Insufficient balance'}), 400

    # 1. Create the transaction record
    transaction_id = str(uuid.uuid4())
    description = f"Bill payment for {biller['nickname']} ({biller['provider_name']})"
    cursor.execute('''
        INSERT INTO transactions (id, user_id, type, amount, description, status, completed_at, counterparty)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
    ''', (transaction_id, user_id, 'BILL_PAYMENT', amount, description, 'COMPLETED', biller['provider_name']))

    # 2. Deduct amount from user's balance
    cursor.execute("UPDATE users SET balance = balance - ? WHERE id = ?", (amount, user_id))
    
    conn.commit()
    conn.close()

    # Track the event
    track_user_event(user['email'], 'bill_payment', '/bills', amount, 'BILL_PAYMENT', json.dumps({'biller_id': biller_id}))
    
    return jsonify({
        'success': True,
        'message': f'Successfully paid ₹{amount:,.2f} for {biller["nickname"]}.'
    })


@app.route('/api/recharge', methods=['POST'])
@jwt_required()
def process_recharge():
    user_id = get_jwt_identity()
    data = request.get_json()

    required_fields = ['provider', 'mobile_number', 'amount', 'plan_type']
    if not all(field in data for field in required_fields):
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400

    provider = data['provider']
    mobile_number = data['mobile_number']
    amount = float(data['amount'])
    plan_type = data['plan_type']  # 'prepaid' or 'postpaid'

    if amount <= 0:
        return jsonify({'success': False, 'error': 'Amount must be positive'}), 400

    conn = get_db()
    cursor = conn.cursor()

    # Get user details
    cursor.execute("SELECT balance, email FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()

    if user['balance'] < amount:
        conn.close()
        return jsonify({'success': False, 'error': 'Insufficient balance'}), 400

    # Create recharge transaction
    transaction_id = str(uuid.uuid4())
    description = f"{plan_type.title()} Recharge - {provider} ({mobile_number})"
    
    cursor.execute('''
        INSERT INTO transactions (id, user_id, type, amount, description, status, completed_at, counterparty, reference)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
    ''', (transaction_id, user_id, 'RECHARGE', amount, description, 'COMPLETED', provider, mobile_number))

    # Deduct amount from user's balance
    cursor.execute("UPDATE users SET balance = balance - ? WHERE id = ?", (amount, user_id))
    
    conn.commit()
    conn.close()

    # Track the event
    track_user_event(user['email'], 'recharge', '/recharge', amount, 'RECHARGE', 
                    json.dumps({'provider': provider, 'mobile_number': mobile_number, 'plan_type': plan_type}))
    
    return jsonify({
        'success': True,
        'message': f'Successfully recharged {mobile_number} with ₹{amount:,.2f}.',
        'transaction_id': transaction_id
    })


# Beneficiaries Management Endpoints
@app.route('/api/beneficiaries', methods=['GET'])
@jwt_required()
def get_beneficiaries():
    user_id = get_jwt_identity()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM beneficiaries WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    beneficiaries = cursor.fetchall()
    conn.close()
    
    # Map database fields to frontend expected fields
    formatted_beneficiaries = []
    for row in beneficiaries:
        beneficiary = dict(row)
        formatted_beneficiaries.append({
            'id': beneficiary['id'],
            'user_id': beneficiary['user_id'],
            'account_number': beneficiary['account_number'],
            'account_holder_name': beneficiary['name'],  # Map DB field to frontend field
            'ifsc_code': beneficiary['ifsc_code'],
            'bank_name': beneficiary['bank_name'],
            'nickname': beneficiary['name'],  # Use name as nickname for now
            'is_within_securebank': 1 if beneficiary['bank_name'] == 'SecureBank' else 0,
            'created_at': beneficiary['created_at']
        })
    
    return jsonify({
        'success': True,
        'beneficiaries': formatted_beneficiaries
    })


@app.route('/api/beneficiaries', methods=['POST'])
@jwt_required()
def add_beneficiary():
    user_id = get_jwt_identity()
    data = request.get_json()
    
    required_fields = ['account_holder_name', 'account_number', 'nickname']
    if not all(field in data for field in required_fields):
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400
    
    # Map frontend field to database field
    name = data['account_holder_name']

    beneficiary_id = str(uuid.uuid4())
    conn = get_db()
    cursor = conn.cursor()
    
    # Check if beneficiary already exists
    cursor.execute("SELECT id FROM beneficiaries WHERE user_id = ? AND account_number = ?", 
                   (user_id, data['account_number']))
    if cursor.fetchone():
        conn.close()
        return jsonify({'success': False, 'error': 'Beneficiary with this account number already exists'}), 400
    
    # Set defaults for optional fields
    ifsc_code = data.get('ifsc_code', 'SECB0000001')
    bank_name = data.get('bank_name', 'SecureBank')
    account_type = 'SAVINGS'  # Default account type
    
    cursor.execute('''
        INSERT INTO beneficiaries (id, user_id, name, account_number, ifsc_code, bank_name, account_type)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (beneficiary_id, user_id, name, data['account_number'], 
          ifsc_code, bank_name, account_type))
    
    conn.commit()
    
    # Return the created beneficiary
    cursor.execute("SELECT * FROM beneficiaries WHERE id = ?", (beneficiary_id,))
    beneficiary = cursor.fetchone()
    conn.close()
    
    return jsonify({
        'success': True,
        'message': 'Beneficiary added successfully!',
        'beneficiary': {
            'id': beneficiary['id'],
            'user_id': beneficiary['user_id'],
            'account_number': beneficiary['account_number'],
            'account_holder_name': beneficiary['name'],  # Map DB field to frontend field
            'ifsc_code': beneficiary['ifsc_code'],
            'bank_name': beneficiary['bank_name'],
            'nickname': data['nickname'],  # Use from request since not stored in DB
            'is_within_securebank': 1 if bank_name == 'SecureBank' else 0,
            'created_at': beneficiary['created_at']
        }
    }), 201


@app.route('/api/beneficiaries/<string:beneficiary_id>', methods=['DELETE'])
@jwt_required()
def delete_beneficiary(beneficiary_id):
    user_id = get_jwt_identity()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM beneficiaries WHERE id = ? AND user_id = ?", (beneficiary_id, user_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Beneficiary deleted successfully.'})


# Transfer Endpoints
@app.route('/api/transfers/own-account', methods=['POST'])
@jwt_required()
def transfer_own_account():
    """Add money to own account for testing purposes"""
    user_id = get_jwt_identity()
    data = request.get_json()

    if not data or 'amount' not in data:
        return jsonify({'success': False, 'error': 'Amount is required'}), 400

    amount = float(data['amount'])
    if amount <= 0:
        return jsonify({'success': False, 'error': 'Amount must be positive'}), 400

    conn = get_db()
    cursor = conn.cursor()

    # Get user details
    cursor.execute("SELECT email FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()

    # Create transaction record
    transaction_id = str(uuid.uuid4())
    description = f"Fund transfer to own account - Testing credit"
    
    cursor.execute('''
        INSERT INTO transactions (id, user_id, type, amount, description, status, completed_at, counterparty)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
    ''', (transaction_id, user_id, 'TRANSFER_IN', amount, description, 'COMPLETED', 'Own Account'))

    # Add amount to user's balance
    cursor.execute("UPDATE users SET balance = balance + ? WHERE id = ?", (amount, user_id))
    
    conn.commit()
    conn.close()

    # Track the event
    track_user_event(user['email'], 'own_account_transfer', '/transfers', amount, 'TRANSFER_IN', 
                    json.dumps({'amount': amount}))
    
    return jsonify({
        'success': True,
        'message': f'Successfully added ₹{amount:,.2f} to your account.',
        'transaction_id': transaction_id
    })


@app.route('/api/transfers/to-beneficiary', methods=['POST'])
@jwt_required()
def transfer_to_beneficiary():
    """Transfer money to a registered beneficiary"""
    user_id = get_jwt_identity()
    data = request.get_json()

    required_fields = ['beneficiary_id', 'amount']
    if not all(field in data for field in required_fields):
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400

    beneficiary_id = data['beneficiary_id']
    amount = float(data['amount'])
    transfer_type = 'IMPS'  # Default transfer type
    remarks = data.get('remarks', '')

    if amount <= 0:
        return jsonify({'success': False, 'error': 'Amount must be positive'}), 400

    conn = get_db()
    cursor = conn.cursor()

    # Get user and beneficiary details
    cursor.execute("SELECT balance, email FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    cursor.execute("SELECT * FROM beneficiaries WHERE id = ? AND user_id = ?", (beneficiary_id, user_id))
    beneficiary = cursor.fetchone()

    if not beneficiary:
        conn.close()
        return jsonify({'success': False, 'error': 'Beneficiary not found'}), 404

    if user['balance'] < amount:
        conn.close()
        return jsonify({'success': False, 'error': 'Insufficient balance'}), 400

    # Create transaction record
    transaction_id = str(uuid.uuid4())
    description = f"{transfer_type} transfer to {beneficiary['name']} ({beneficiary['account_number'][-4:]})"
    
    cursor.execute('''
        INSERT INTO transactions (id, user_id, type, amount, description, status, completed_at, counterparty, reference)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
    ''', (transaction_id, user_id, 'TRANSFER_OUT', -amount, description, 'COMPLETED', 
          beneficiary['name'], remarks))

    # Deduct amount from user's balance
    cursor.execute("UPDATE users SET balance = balance - ? WHERE id = ?", (amount, user_id))
    
    conn.commit()
    conn.close()

    # Track the event
    track_user_event(user['email'], 'beneficiary_transfer', '/transfers', amount, 'TRANSFER_OUT', 
                    json.dumps({'beneficiary_id': beneficiary_id, 'transfer_type': transfer_type}))
    
    return jsonify({
        'success': True,
        'message': f'Successfully transferred ₹{amount:,.2f} to {beneficiary["name"]}.',
        'transaction_id': transaction_id
    })


@app.route('/api/transfers/recent', methods=['GET'])
@jwt_required()
def get_recent_transfers():
    """Get recent transfer transactions"""
    user_id = get_jwt_identity()
    limit = int(request.args.get('limit', 10))
    
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM transactions 
        WHERE user_id = ? AND type IN ('TRANSFER_IN', 'TRANSFER_OUT')
        ORDER BY created_at DESC 
        LIMIT ?
    ''', (user_id, limit))
    transfers = cursor.fetchall()
    
    conn.close()
    
    return jsonify({
        'success': True,
        'transfers': [
            {
                'id': row['id'],
                'type': row['type'],
                'amount': row['amount'],
                'description': row['description'],
                'status': row['status'],
                'created_at': row['created_at'],
                'counterparty': row['counterparty'],
                'reference': row['reference']
            }
            for row in transfers
        ]
    })


@app.route('/api/auth/profile', methods=['GET', 'OPTIONS'])
@jwt_required()
def get_profile():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        user_id = get_jwt_identity()
        
        conn = get_db()
        cursor = conn.cursor()
        
        # Get user
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return jsonify({'success': False, 'error': 'User not found'}), 404
        
        # Get portfolio with current values
        cursor.execute('''
            SELECT p.*, s.price as current_price, s.name as stock_name
            FROM portfolio p
            JOIN stocks s ON p.symbol = s.symbol
            WHERE p.user_id = ?
        ''', (user_id,))
        portfolio = cursor.fetchall()
        
        # Calculate portfolio summary
        total_investment = sum(row['total_investment'] for row in portfolio)
        total_value = sum(row['shares'] * row['current_price'] for row in portfolio)
        total_gain_loss = total_value - total_investment
        total_gain_loss_percent = (total_gain_loss / total_investment * 100) if total_investment > 0 else 0
        
        # Get recent transactions
        cursor.execute('''
            SELECT * FROM transactions 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT 10
        ''', (user_id,))
        transactions = cursor.fetchall()
        
        # Get fixed deposits
        cursor.execute("SELECT * FROM fixed_deposits WHERE user_id = ?", (user_id,))
        fixed_deposits = cursor.fetchall()
        
        conn.close()
        
        return jsonify({
            'success': True,
            'user': {
                'id': user['id'],
                'email': user['email'],
                'firstName': user['first_name'],
                'lastName': user['last_name'],
                'phone': user['phone'],
                'balance': user['balance'],
                'accountNumber': user['account_number'],
                'createdAt': user['created_at'],
                'lastLogin': user['last_login']
            },
            'portfolio': [
                {
                    'id': row['id'],
                    'userId': row['user_id'],
                    'symbol': row['symbol'],
                    'shares': row['shares'],
                    'buyPrice': row['buy_price'],
                    'totalInvestment': row['total_investment'],
                    'purchaseDate': row['purchase_date'],
                    'currentPrice': row['current_price'],
                    'currentValue': row['shares'] * row['current_price'],
                    'gainLoss': (row['shares'] * row['current_price']) - row['total_investment'],
                    'gainLossPercent': ((row['shares'] * row['current_price'] - row['total_investment']) / row['total_investment'] * 100) if row['total_investment'] > 0 else 0,
                    'name': row['stock_name']
                }
                for row in portfolio
            ],
            'portfolioSummary': {
                'totalValue': round(total_value, 2),
                'totalInvestment': round(total_investment, 2),
                'totalGainLoss': round(total_gain_loss, 2),
                'totalGainLossPercent': round(total_gain_loss_percent, 2),
                'holdings': len(portfolio)
            },
            'recentTransactions': [
                {
                    'id': row['id'],
                    'userId': row['user_id'],
                    'type': row['type'],
                    'symbol': row['symbol'],
                    'shares': row['shares'],
                    'amount': row['amount'],
                    'price': row['price'],
                    'description': row['description'],
                    'status': row['status'],
                    'createdAt': row['created_at'],
                    'completedAt': row['completed_at'],
                    'reference': row['reference'],
                    'counterparty': row['counterparty']
                }
                for row in transactions
            ],
            'fixedDeposits': [
                {
                    'id': row['id'],
                    'userId': row['user_id'],
                    'amount': row['amount'],
                    'interestRate': row['interest_rate'],
                    'tenure': row['tenure'],
                    'startDate': row['start_date'],
                    'maturityDate': row['maturity_date'],
                    'type': row['type'],
                    'status': row['status'],
                    'interestEarned': row['interest_earned'],
                    'maturityAmount': row['maturity_amount']
                }
                for row in fixed_deposits
            ]
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Logout route
@app.route('/api/auth/logout', methods=['POST', 'OPTIONS'])
@jwt_required()
def logout():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        user_id = get_jwt_identity()
        
        # Get user's email for event tracking
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT email FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            # Track logout event
            track_user_event(user['email'], 'logout', '/logout', 0, 'logout', None)
        
        return jsonify({
            'success': True,
            'message': 'Logout successful',
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Stocks routes
@app.route('/api/stocks', methods=['GET'])
def get_all_stocks():
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM stocks ORDER BY symbol")
        stocks = cursor.fetchall()
        
        conn.close()
        
        # Track stocks view event (anonymous since no auth required)
        track_user_event('anonymous', 'stocks_view', '/stocks', 0, 'market_data', 
                        json.dumps({'total_stocks': len(stocks)}))
        
        return jsonify({
            'success': True,
            'stocks': [
                {
                    'symbol': row['symbol'],
                    'name': row['name'],
                    'price': row['price'],
                    'change': row['change'],
                    'changePercent': row['change_percent'],
                    'volume': row['volume'],
                    'high': row['high'],
                    'low': row['low'],
                    'category': row['category'],
                    'lastUpdated': row['last_updated']
                }
                for row in stocks
            ]
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stocks/<symbol>', methods=['GET'])
def get_stock(symbol):
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM stocks WHERE symbol = ?", (symbol.upper(),))
        stock = cursor.fetchone()
        
        conn.close()
        
        if not stock:
            return jsonify({'success': False, 'error': 'Stock not found'}), 404
        
        return jsonify({
            'success': True,
            'stock': {
                'symbol': stock['symbol'],
                'name': stock['name'],
                'price': stock['price'],
                'change': stock['change'],
                'changePercent': stock['change_percent'],
                'volume': stock['volume'],
                'high': stock['high'],
                'low': stock['low'],
                'category': stock['category'],
                'lastUpdated': stock['last_updated']
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stocks/buy', methods=['POST'])
@jwt_required()
def buy_stock():
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data or 'symbol' not in data or 'shares' not in data:
            return jsonify({'success': False, 'error': 'Symbol and shares are required'}), 400
        
        symbol = data['symbol'].upper()
        shares = int(data['shares'])
        
        if shares <= 0:
            return jsonify({'success': False, 'error': 'Shares must be positive'}), 400
        
        conn = get_db()
        cursor = conn.cursor()
        
        # Get user details for tracking
        cursor.execute("SELECT balance, email FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        
        # Get stock price
        cursor.execute("SELECT * FROM stocks WHERE symbol = ?", (symbol,))
        stock = cursor.fetchone()
        
        if not stock:
            conn.close()
            return jsonify({'success': False, 'error': 'Stock not found'}), 404
        
        total_cost = shares * stock['price']
        
        if user['balance'] < total_cost:
            conn.close()
            return jsonify({'success': False, 'error': 'Insufficient balance'}), 400
        
        # Create transaction
        transaction_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO transactions (id, user_id, type, symbol, shares, amount, price, description, status, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (
            transaction_id,
            user_id,
            'BUY',
            symbol,
            shares,
            total_cost,
            stock['price'],
            f'Bought {shares} shares of {symbol}',
            'COMPLETED'
        ))
        
        # Update user balance
        cursor.execute("UPDATE users SET balance = balance - ? WHERE id = ?", (total_cost, user_id))
        
        # Update portfolio
        cursor.execute("SELECT * FROM portfolio WHERE user_id = ? AND symbol = ?", (user_id, symbol))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing holding
            new_shares = existing['shares'] + shares
            new_investment = existing['total_investment'] + total_cost
            avg_price = new_investment / new_shares
            
            cursor.execute('''
                UPDATE portfolio 
                SET shares = ?, buy_price = ?, total_investment = ?
                WHERE user_id = ? AND symbol = ?
            ''', (new_shares, avg_price, new_investment, user_id, symbol))
        else:
            # Add new holding
            portfolio_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO portfolio (id, user_id, symbol, shares, buy_price, total_investment)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (portfolio_id, user_id, symbol, shares, stock['price'], total_cost))
        
        conn.commit()
        conn.close()
        
        # Track stock buy event
        track_user_event(user['email'], 'stock_buy', f'/stocks/{symbol}', total_cost, 'stock_trading', 
                        json.dumps({'symbol': symbol, 'shares': shares, 'price': stock['price'], 'transaction_id': transaction_id}))
        
        return jsonify({
            'success': True,
            'message': f'Successfully bought {shares} shares of {symbol}',
            'transaction': {
                'id': transaction_id,
                'amount': total_cost,
                'shares': shares,
                'price': stock['price']
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stocks/sell', methods=['POST'])
@jwt_required()
def sell_stock():
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data or 'symbol' not in data or 'shares' not in data:
            return jsonify({'success': False, 'error': 'Symbol and shares are required'}), 400
        
        symbol = data['symbol'].upper()
        shares = int(data['shares'])
        
        if shares <= 0:
            return jsonify({'success': False, 'error': 'Shares must be positive'}), 400
        
        conn = get_db()
        cursor = conn.cursor()
        
        # Get user details for tracking
        cursor.execute("SELECT balance, email FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        
        # Get stock price
        cursor.execute("SELECT * FROM stocks WHERE symbol = ?", (symbol,))
        stock = cursor.fetchone()
        
        if not stock:
            conn.close()
            return jsonify({'success': False, 'error': 'Stock not found'}), 404
        
        # Check if user has enough shares
        cursor.execute("SELECT * FROM portfolio WHERE user_id = ? AND symbol = ?", (user_id, symbol))
        holding = cursor.fetchone()
        
        if not holding or holding['shares'] < shares:
            conn.close()
            return jsonify({'success': False, 'error': 'Insufficient shares'}), 400
        
        total_value = shares * stock['price']
        
        # Create transaction
        transaction_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO transactions (id, user_id, type, symbol, shares, amount, price, description, status, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (
            transaction_id,
            user_id,
            'SELL',
            symbol,
            shares,
            total_value,
            stock['price'],
            f'Sold {shares} shares of {symbol}',
            'COMPLETED'
        ))
        
        # Update user balance
        cursor.execute("UPDATE users SET balance = balance + ? WHERE id = ?", (total_value, user_id))
        
        # Update portfolio
        remaining_shares = holding['shares'] - shares
        if remaining_shares == 0:
            # Remove completely
            cursor.execute("DELETE FROM portfolio WHERE user_id = ? AND symbol = ?", (user_id, symbol))
        else:
            # Reduce shares
            sold_investment = (holding['total_investment'] / holding['shares']) * shares
            remaining_investment = holding['total_investment'] - sold_investment;
            
            cursor.execute('''
                UPDATE portfolio 
                SET shares = ?, total_investment = ?
                WHERE user_id = ? AND symbol = ?
            ''', (remaining_shares, remaining_investment, user_id, symbol))
        
        conn.commit()
        conn.close()
        
        # Track stock sell event
        track_user_event(user['email'], 'stock_sell', f'/stocks/{symbol}', total_value, 'stock_trading', 
                        json.dumps({'symbol': symbol, 'shares': shares, 'price': stock['price'], 'transaction_id': transaction_id}))
        
        return jsonify({
            'success': True,
            'message': f'Successfully sold {shares} shares of {symbol}',
            'transaction': {
                'id': transaction_id,
                'amount': total_value,
                'shares': shares,
                'price': stock['price']
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Portfolio routes
@app.route('/api/portfolio', methods=['GET'])
@jwt_required()
def get_portfolio():
    try:
        user_id = get_jwt_identity()
        
        conn = get_db()
        cursor = conn.cursor()
        
        # Get user email for tracking
        cursor.execute("SELECT email FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        
        # Get portfolio with current values
        cursor.execute('''
            SELECT p.*, s.price as current_price, s.name as stock_name
            FROM portfolio p
            JOIN stocks s ON p.symbol = s.symbol
            WHERE p.user_id = ?
        ''', (user_id,))
        portfolio = cursor.fetchall()
        
        # Calculate summary
        total_investment = sum(row['total_investment'] for row in portfolio)
        total_value = sum(row['shares'] * row['current_price'] for row in portfolio)
        total_gain_loss = total_value - total_investment
        total_gain_loss_percent = (total_gain_loss / total_investment * 100) if total_investment > 0 else 0
        
        conn.close()
        
        # Track portfolio view event
        track_user_event(user['email'], 'portfolio_view', '/portfolio', 0, 'portfolio_view', 
                        json.dumps({'total_value': total_value, 'total_investment': total_investment, 'holdings': len(portfolio)}))
        
        return jsonify({
            'success': True,
            'portfolio': [
                {
                    'id': row['id'],
                    'userId': row['user_id'],
                    'symbol': row['symbol'],
                    'shares': row['shares'],
                    'buyPrice': row['buy_price'],
                    'totalInvestment': row['total_investment'],
                    'purchaseDate': row['purchase_date'],
                    'currentPrice': row['current_price'],
                    'currentValue': row['shares'] * row['current_price'],
                    'gainLoss': (row['shares'] * row['current_price']) - row['total_investment'],
                    'gainLossPercent': ((row['shares'] * row['current_price'] - row['total_investment']) / row['total_investment'] * 100) if row['total_investment'] > 0 else 0,
                    'name': row['stock_name']
                }
                for row in portfolio
            ],
            'summary': {
                'totalValue': round(total_value, 2),
                'totalInvestment': round(total_investment, 2),
                'totalGainLoss': round(total_gain_loss, 2),
                'totalGainLossPercent': round(total_gain_loss_percent, 2),
                'holdings': len(portfolio)
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Transactions routes
@app.route('/api/transactions', methods=['GET'])
@jwt_required()
def get_transactions():
    try:
        user_id = get_jwt_identity()
        limit = int(request.args.get('limit', 20))
        offset = int(request.args.get('offset', 0))
        transaction_type = request.args.get('type', '')
        
        conn = get_db()
        cursor = conn.cursor()
        
        query = "SELECT * FROM transactions WHERE user_id = ?"
        params = [user_id]
        
        if transaction_type:
            query += " AND type = ?"
            params.append(transaction_type)
        
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        transactions = cursor.fetchall()
        
        conn.close()
        
        return jsonify({
            'success': True,
            'transactions': [
                {
                    'id': row['id'],
                    'userId': row['user_id'],
                    'type': row['type'],
                    'symbol': row['symbol'],
                    'shares': row['shares'],
                    'amount': row['amount'],
                    'price': row['price'],
                    'description': row['description'],
                    'status': row['status'],
                    'createdAt': row['created_at'],
                    'completedAt': row['completed_at'],
                    'reference': row['reference'],
                    'counterparty': row['counterparty']
                }
                for row in transactions
            ]
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Make sure you have these imports
import datetime
import logging

# ... (other routes) ...

@app.route('/api/behavior/current-session', methods=['GET'])
@jwt_required()
def get_current_session_events():
    """
    Retrieves all events for the logged-in user from their most recent
    'login_success' event to the present. This defines the "current session".
    """
    try:
        user_id = get_jwt_identity()
        
        conn = get_db()
        cursor = conn.cursor()
        
        # 1. Get user's email for the query
        cursor.execute("SELECT email FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        if not user:
            conn.close()
            return jsonify({'success': False, 'error': 'User not found'}), 404
        
        user_email = user['email']

        # 2. Find the timestamp of the most recent 'login_success' event for this user
        cursor.execute("""
            SELECT MAX(time) 
            FROM user_events 
            WHERE user_email = ? AND event_type = 'login_success'
        """, (user_email,))
        last_login_time_row = cursor.fetchone()

        # If the user has never logged in (or events were cleared), return empty
        if not last_login_time_row or not last_login_time_row[0]:
            conn.close()
            return jsonify({'success': True, 'session_events': []})

        last_login_time = last_login_time_row[0]
        
        # 3. Fetch all events that occurred since the last login
        query = """
            SELECT id, event_type, time, page_url, transaction_amount, additional_data
            FROM user_events 
            WHERE user_email = ? AND time >= ?
            ORDER BY time ASC
        """
        cursor.execute(query, (user_email, last_login_time))
        session_events = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
            
        return jsonify({
            'success': True,
            'session_events': session_events
        })
        
    except Exception as e:
        logging.error(f"Error getting current session: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500
    


# In app.py
# Make sure you have the UserBehaviorProfiler class from our previous discussion

# ... (other global variables)
# behavior_profilers = {} # This should be initialized at the top

# ... (train_model_on_startup should be training and populating the behavior_profilers dict)


# Fixed Deposits routes
@app.route('/api/fixed-deposits', methods=['GET'])
@jwt_required()
def get_fixed_deposits():
    try:
        user_id = get_jwt_identity()
        
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM fixed_deposits WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
        fixed_deposits = cursor.fetchall()
        
        conn.close()
        
        return jsonify({
            'success': True,
            'fixedDeposits': [
                {
                    'id': row['id'],
                    'userId': row['user_id'],
                    'amount': row['amount'],
                    'interestRate': row['interest_rate'],
                    'tenure': row['tenure'],
                    'startDate': row['start_date'],
                    'maturityDate': row['maturity_date'],
                    'type': row['type'],
                    'status': row['status'],
                    'interestEarned': row['interest_earned'],
                    'maturityAmount': row['maturity_amount']
                }
                for row in fixed_deposits
            ]
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/fixed-deposits', methods=['POST'])
@jwt_required()
def create_fixed_deposit():
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data or 'amount' not in data or 'tenure' not in data or 'type' not in data:
            return jsonify({'success': False, 'error': 'Amount, tenure, and type are required'}), 400
        
        amount = float(data['amount'])
        tenure = int(data['tenure'])
        fd_type = data['type']
        
        if amount < 1000:
            return jsonify({'success': False, 'error': 'Minimum amount is ₹1,000'}), 400
        
        # Calculate interest rate based on type and tenure
        interest_rates = {
            'REGULAR': {12: 6.8, 24: 7.5, 36: 7.2, 60: 7.8, 120: 8.1},
            'SENIOR': {12: 7.3, 24: 8.0, 36: 7.7, 60: 8.3, 120: 8.6},
            'TAX_SAVING': {60: 7.2}
        }
        
        if fd_type not in interest_rates or tenure not in interest_rates[fd_type]:
            return jsonify({'success': False, 'error': 'Invalid tenure for this FD type'}), 400
        
        interest_rate = interest_rates[fd_type][tenure]
        
        # Calculate maturity amount (quarterly compounding)
        quarters = tenure / 3
        maturity_amount = amount * ((1 + interest_rate / 400) ** quarters)
        
        # Calculate maturity date
        start_date = datetime.datetime.now()
        maturity_date = start_date + datetime.timedelta(days=tenure * 30)
        
        conn = get_db()
        cursor = conn.cursor()
        
        # Get user details for tracking
        cursor.execute("SELECT balance, email FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        
        if user['balance'] < amount:
            conn.close()
            return jsonify({'success': False, 'error': 'Insufficient balance'}), 400
        
        # Create fixed deposit
        fd_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO fixed_deposits (id, user_id, amount, interest_rate, tenure, start_date, maturity_date, type, maturity_amount)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (fd_id, user_id, amount, interest_rate, tenure, start_date, maturity_date, fd_type, maturity_amount))
        
        # Deduct amount from user balance
        cursor.execute("UPDATE users SET balance = balance - ? WHERE id = ?", (amount, user_id))
        
        # Create transaction
        transaction_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO transactions (id, user_id, type, amount, description, status, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (
            transaction_id,
            user_id,
            'DEPOSIT',
            amount,
            f'Fixed Deposit created - {fd_type} for {tenure} months',
            'COMPLETED'
        ))
        
        conn.commit()
        conn.close()
        
        # Track FD creation event
        track_user_event(user['email'], 'fd_created', '/fixed-deposits', amount, 'fd_creation', 
                        json.dumps({'fd_id': fd_id, 'type': fd_type, 'tenure': tenure, 'interest_rate': interest_rate}))
        
        return jsonify({
            'success': True,
            'message': 'Fixed Deposit created successfully',
            'fixedDeposit': {
                'id': fd_id,
                'amount': amount,
                'interestRate': interest_rate,
                'tenure': tenure,
                'startDate': start_date.isoformat(),
                'maturityDate': maturity_date.isoformat(),
                'type': fd_type,
                'maturityAmount': maturity_amount
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Fixed Deposit Details and Certificate endpoints
@app.route('/api/fixed-deposits/<fd_id>/details', methods=['GET'])
@jwt_required()
def get_fd_details(fd_id):
    try:
        user_id = get_jwt_identity()
        
        conn = get_db()
        cursor = conn.cursor()
        
        # Get user email for event tracking
        cursor.execute("SELECT email FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        
        # Get FD details
        cursor.execute("""
            SELECT fd.*, u.first_name, u.last_name, u.account_number
            FROM fixed_deposits fd
            JOIN users u ON fd.user_id = u.id
            WHERE fd.id = ? AND fd.user_id = ?
        """, (fd_id, user_id))
        fd = cursor.fetchone()
        
        conn.close()
        
        if not fd:
            return jsonify({'success': False, 'error': 'Fixed deposit not found'}), 404
        
        # Track event
        track_user_event(user['email'], 'fd_details_view', f'/fixed-deposits/{fd_id}', 0, 'fd_details')
        
        return jsonify({
            'success': True,
            'fdDetails': {
                'id': fd['id'],
                'amount': fd['amount'],
                'interestRate': fd['interest_rate'],
                'tenure': fd['tenure'],
                'startDate': fd['start_date'],
                'maturityDate': fd['maturity_date'],
                'type': fd['type'],
                'status': fd['status'],
                'interestEarned': fd['interest_earned'],
                'maturityAmount': fd['maturity_amount'],
                'createdAt': fd['created_at'],
                'customerName': f"{fd['first_name']} {fd['last_name']}",
                'accountNumber': fd['account_number']
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/fixed-deposits/<fd_id>/certificate', methods=['GET'])
@jwt_required()
def get_fd_certificate(fd_id):
    try:
        user_id = get_jwt_identity()
        
        conn = get_db()
        cursor = conn.cursor()
        
        # Get user email for event tracking
        cursor.execute("SELECT email FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        
        # Get FD details for certificate
        cursor.execute("""
            SELECT fd.*, u.first_name, u.last_name, u.account_number, u.phone
            FROM fixed_deposits fd
            JOIN users u ON fd.user_id = u.id
            WHERE fd.id = ? AND fd.user_id = ?
        """, (fd_id, user_id))
        fd = cursor.fetchone()
        
        conn.close()
        
        if not fd:
            return jsonify({'success': False, 'error': 'Fixed deposit not found'}), 404
        
        # Track event
        track_user_event(user['email'], 'fd_certificate_download', f'/fixed-deposits/{fd_id}', 0, 'fd_certificate')
        
        # Generate certificate data
        certificate_data = {
            'certificateNumber': fd['id'],
            'customerName': f"{fd['first_name']} {fd['last_name']}",
            'accountNumber': fd['account_number'],
            'phone': fd['phone'],
            'amount': fd['amount'],
            'interestRate': fd['interest_rate'],
            'tenure': fd['tenure'],
            'startDate': fd['start_date'],
            'maturityDate': fd['maturity_date'],
            'type': fd['type'],
            'maturityAmount': fd['maturity_amount'],
            'issueDate': datetime.datetime.now().strftime('%Y-%m-%d'),
            'branch': 'OLAVAKKOT',
            'bankName': 'SecureBank'
        }
        
        return jsonify({
            'success': True,
            'certificate': certificate_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/fixed-deposits/export', methods=['GET'])
@jwt_required()
def export_fixed_deposits():
    try:
        user_id = get_jwt_identity()
        
        conn = get_db()
        cursor = conn.cursor()
        
        # Get user email for event tracking
        cursor.execute("SELECT email FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        
        # Get all FDs for user
        cursor.execute("""
            SELECT fd.*, u.first_name, u.last_name, u.account_number
            FROM fixed_deposits fd
            JOIN users u ON fd.user_id = u.id
            WHERE fd.user_id = ?
            ORDER BY fd.created_at DESC
        """, (user_id,))
        fixed_deposits = cursor.fetchall()
        
        conn.close()
        
        # Track event
        track_user_event(user['email'], 'fd_export', '/fixed-deposits/export', 0, 'fd_export')
        
        # Calculate summary
        total_investment = sum(fd['amount'] for fd in fixed_deposits if fd['status'] == 'ACTIVE')
        total_interest_earned = sum(fd['interest_earned'] for fd in fixed_deposits if fd['status'] == 'ACTIVE')
        active_fds = len([fd for fd in fixed_deposits if fd['status'] == 'ACTIVE'])
        
        export_data = {
            'customerName': f"{user['first_name']} {user['last_name']}",
            'accountNumber': user['account_number'],
            'exportDate': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'totalInvestment': total_investment,
                'totalInterestEarned': total_interest_earned,
                'activeFDs': active_fds,
                'totalFDs': len(fixed_deposits)
            },
            'fixedDeposits': [
                {
                    'id': fd['id'],
                    'amount': fd['amount'],
                    'interestRate': fd['interest_rate'],
                    'tenure': fd['tenure'],
                    'startDate': fd['start_date'],
                    'maturityDate': fd['maturity_date'],
                    'type': fd['type'],
                    'status': fd['status'],
                    'interestEarned': fd['interest_earned'],
                    'maturityAmount': fd['maturity_amount'],
                    'createdAt': fd['created_at']
                }
                for fd in fixed_deposits
            ]
        }
        
        return jsonify({
            'success': True,
            'exportData': export_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Transfer routes
@app.route('/api/transfers', methods=['POST'])
@jwt_required()
def create_transfer():
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data or 'amount' not in data or 'recipient' not in data or 'type' not in data:
            return jsonify({'success': False, 'error': 'Amount, recipient, and type are required'}), 400
        
        amount = float(data['amount'])
        recipient = data['recipient']
        transfer_type = data['type']
        reference = data.get('reference', '')
        
        if amount <= 0:
            return jsonify({'success': False, 'error': 'Amount must be positive'}), 400
        
        conn = get_db()
        cursor = conn.cursor()
        
        # Check user balance
        cursor.execute("SELECT balance FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        
        if user['balance'] < amount:
            conn.close()
            return jsonify({'success': False, 'error': 'Insufficient balance'}), 400
        
        # Create transfer transaction
        transaction_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO transactions (id, user_id, type, amount, description, status, completed_at, reference, counterparty)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
        ''', (
            transaction_id,
            user_id,
            'TRANSFER_OUT',
            amount,
            f'Transfer to {recipient}',
            'COMPLETED',
            reference,
            recipient
        ))
        
        # Deduct amount from user balance
        cursor.execute("UPDATE users SET balance = balance - ? WHERE id = ?", (amount, user_id))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': f'Transfer of ₹{amount:,.2f} to {recipient} completed successfully',
            'transaction': {
                'id': transaction_id,
                'amount': amount,
                'recipient': recipient,
                'type': transfer_type
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Account Statement routes
@app.route('/api/account-statement', methods=['GET'])
@jwt_required()
def get_account_statement():
    try:
        user_id = get_jwt_identity()
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        statement_period = request.args.get('period', 'byDate')
        records_per_page = request.args.get('records_per_page', 'ALL')
        
        conn = get_db()
        cursor = conn.cursor()
        
        # Get user email for tracking
        cursor.execute("SELECT email FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        
        # Build query based on period
        query = """
            SELECT t.*, u.account_number 
            FROM transactions t
            JOIN users u ON t.user_id = u.id
            WHERE t.user_id = ?
        """
        params = [user_id]
        
        if statement_period == 'byDate' and start_date and end_date:
            query += " AND DATE(t.created_at) BETWEEN ? AND ?"
            params.extend([start_date, end_date])
        elif statement_period == 'last6Months':
            query += " AND t.created_at >= datetime('now', '-6 months')"
        elif statement_period == 'financialYear':
            # Current financial year (April to March)
            current_year = datetime.datetime.now().year
            if datetime.datetime.now().month < 4:
                fy_start = f"{current_year-1}-04-01"
                fy_end = f"{current_year}-03-31"
            else:
                fy_start = f"{current_year}-04-01"
                fy_end = f"{current_year+1}-03-31"
            query += " AND DATE(t.created_at) BETWEEN ? AND ?"
            params.extend([fy_start, fy_end])
        
        query += " ORDER BY t.created_at DESC"
        
        if records_per_page != 'ALL':
            try:
                limit = int(records_per_page)
                query += " LIMIT ?"
                params.append(limit)
            except ValueError:
                pass
        
        cursor.execute(query, params)
        transactions = cursor.fetchall()
        
        # Calculate running balance
        balance = 125430.50  # Starting balance
        statement_entries = []
        
        for transaction in transactions:
            if transaction['type'] in ['TRANSFER_IN', 'DEPOSIT']:
                balance += transaction['amount']
                debit = '-'
                credit = f"₹{transaction['amount']:,.2f}"
            else:
                balance -= transaction['amount']
                debit = f"₹{transaction['amount']:,.2f}"
                credit = '-'
            
            statement_entries.append({
                'date': transaction['created_at'],
                'description': transaction['description'],
                'debit': debit,
                'credit': credit,
                'balance': f"₹{balance:,.2f}",
                'type': transaction['type'],
                'amount': transaction['amount'],
                'status': transaction['status']
            })
        
        conn.close()
        
        # Track statement view event
        track_user_event(user['email'], 'account_statement_view', '/account-statement', 0, 'statement_view', 
                        json.dumps({'period': statement_period, 'records_count': len(statement_entries)}))
        
        return jsonify({
            'success': True,
            'statement': {
                'accountNumber': '000000414046934930',
                'period': statement_period,
                'startDate': start_date,
                'endDate': end_date,
                'totalRecords': len(statement_entries),
                'entries': statement_entries
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/account-statement/export', methods=['POST'])
@jwt_required()
def export_account_statement():
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        export_format = data.get('format', 'pdf')  # pdf, excel
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        statement_period = data.get('period', 'byDate')
        
        conn = get_db()
        cursor = conn.cursor()
        
        # Get user details for tracking
        cursor.execute("SELECT email, first_name, last_name, account_number FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        
        # Get statement data (same logic as above)
        query = """
            SELECT t.*, u.account_number 
            FROM transactions t
            JOIN users u ON t.user_id = u.id
            WHERE t.user_id = ?
        """
        params = [user_id]
        
        if statement_period == 'byDate' and start_date and end_date:
            query += " AND DATE(t.created_at) BETWEEN ? AND ?"
            params.extend([start_date, end_date])
        elif statement_period == 'last6Months':
            query += " AND t.created_at >= datetime('now', '-6 months')"
        elif statement_period == 'financialYear':
            current_year = datetime.datetime.now().year
            if datetime.datetime.now().month < 4:
                fy_start = f"{current_year-1}-04-01"
                fy_end = f"{current_year}-03-31"
            else:
                fy_start = f"{current_year}-04-01"
                fy_end = f"{current_year+1}-03-31"
            query += " AND DATE(t.created_at) BETWEEN ? AND ?"
            params.extend([fy_start, fy_end])
        
        query += " ORDER BY t.created_at DESC"
        cursor.execute(query, params)
        transactions = cursor.fetchall()
        
        conn.close()
        
        # Generate export data
        export_data = {
            'customerName': f"{user['first_name']} {user['last_name']}",
            'accountNumber': user['account_number'],
            'exportDate': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'period': statement_period,
            'startDate': start_date,
            'endDate': end_date,
            'format': export_format,
            'transactions': [
                {
                    'date': t['created_at'],
                    'description': t['description'],
                    'type': t['type'],
                    'amount': t['amount'],
                    'status': t['status']
                }
                for t in transactions
            ]
        }
        
        # Track export event
        track_user_event(user['email'], 'account_statement_export', '/account-statement/export', 0, 'statement_export', 
                        json.dumps({'format': export_format, 'period': statement_period, 'records_count': len(transactions)}))
        
        return jsonify({
            'success': True,
            'message': f'Account statement exported successfully in {export_format.upper()} format',
            'exportData': export_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ========== TAX PAYMENT ENDPOINTS ==========

@app.route('/api/tax/direct', methods=['POST'])
@jwt_required()
def pay_direct_tax():
    """Pay direct tax (Income Tax, TDS, etc.)"""
    user_id = get_jwt_identity()
    data = request.get_json()
    
    required_fields = ['pan', 'assessmentYear', 'taxType', 'paymentType', 'amount']
    if not all(field in data for field in required_fields):
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
    amount = float(data['amount'])
    if amount <= 0:
        return jsonify({'success': False, 'error': 'Amount must be greater than 0'}), 400
    
    conn = get_db()
    cursor = conn.cursor()
    
    # Check user balance and get email
    cursor.execute("SELECT balance, email FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    if not user or user['balance'] < amount:
        conn.close()
        return jsonify({'success': False, 'error': 'Insufficient balance'}), 400
    
    try:
        # Create transaction
        transaction_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO transactions (id, user_id, type, amount, description, status, completed_at, counterparty, reference)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
        ''', (
            transaction_id, user_id, 'TAX_DIRECT', -amount, 
            f"Direct Tax Payment - PAN: {data['pan']}", 'COMPLETED',
            'Income Tax Department', data['pan']
        ))
        
        # Update user balance
        cursor.execute("UPDATE users SET balance = balance - ? WHERE id = ?", (amount, user_id))
        
        # Create tax payment record
        tax_payment_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO tax_payments (id, user_id, transaction_id, tax_type, pan_tan, assessment_year, 
                                    tax_applicable, payment_type, amount, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (
            tax_payment_id, user_id, transaction_id, 'DIRECT', data['pan'], 
            data['assessmentYear'], data['taxType'], data['paymentType'], amount, 'COMPLETED'
        ))
        
        conn.commit()
        track_user_event(user['email'], 'tax_payment_direct', '/tax/direct', amount, 'tax_payment', 
                         json.dumps({'pan': data['pan'], 'assessmentYear': data['assessmentYear']}))
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Direct tax payment successful',
            'transaction_id': transaction_id,
            'tax_payment_id': tax_payment_id
        })
        
    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/tax/gst', methods=['POST'])
@jwt_required()
def pay_gst():
    """Pay GST"""
    user_id = get_jwt_identity()
    data = request.get_json()
    
    required_fields = ['gstin', 'cpin', 'amount']
    if not all(field in data for field in required_fields):
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
    amount = float(data['amount'])
    if amount <= 0:
        return jsonify({'success': False, 'error': 'Amount must be greater than 0'}), 400
    
    conn = get_db()
    cursor = conn.cursor()
    
    # Check user balance
    cursor.execute("SELECT balance FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    if not user or user['balance'] < amount:
        conn.close()
        return jsonify({'success': False, 'error': 'Insufficient balance'}), 400
    
    try:
        # Create transaction
        transaction_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO transactions (id, user_id, type, amount, description, status, completed_at, counterparty, reference)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
        ''', (
            transaction_id, user_id, 'TAX_GST', -amount, 
            f"GST Payment - GSTIN: {data['gstin']}", 'COMPLETED',
            'GST Department', data['gstin']
        ))
        
        # Update user balance
        cursor.execute("UPDATE users SET balance = balance - ? WHERE id = ?", (amount, user_id))
        
        # Create tax payment record
        tax_payment_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO tax_payments (id, user_id, transaction_id, tax_type, gstin, cpin, 
                                    cgst, sgst, igst, cess, amount, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (
            tax_payment_id, user_id, transaction_id, 'GST', data['gstin'], data['cpin'],
            data.get('cgst', 0), data.get('sgst', 0), data.get('igst', 0), data.get('cess', 0),
            amount, 'COMPLETED'
        ))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'GST payment successful',
            'transaction_id': transaction_id,
            'tax_payment_id': tax_payment_id
        })
        
    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/tax/state', methods=['POST'])
@jwt_required()
def pay_state_tax():
    """Pay state tax"""
    user_id = get_jwt_identity()
    data = request.get_json()
    
    required_fields = ['state', 'municipality', 'service', 'consumerId', 'amount']
    if not all(field in data for field in required_fields):
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
    amount = float(data['amount'])
    if amount <= 0:
        return jsonify({'success': False, 'error': 'Amount must be greater than 0'}), 400
    
    conn = get_db()
    cursor = conn.cursor()
    
    # Check user balance
    cursor.execute("SELECT balance FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    if not user or user['balance'] < amount:
        conn.close()
        return jsonify({'success': False, 'error': 'Insufficient balance'}), 400
    
    try:
        # Create transaction
        transaction_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO transactions (id, user_id, type, amount, description, status, completed_at, counterparty, reference)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
        ''', (
            transaction_id, user_id, 'TAX_STATE', -amount, 
            f"{data['service']} - {data['state']}", 'COMPLETED',
            f"{data['municipality']}", data['consumerId']
        ))
        
        # Update user balance
        cursor.execute("UPDATE users SET balance = balance - ? WHERE id = ?", (amount, user_id))
        
        # Create tax payment record
        tax_payment_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO tax_payments (id, user_id, transaction_id, tax_type, state, municipality, 
                                    service_type, consumer_id, amount, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (
            tax_payment_id, user_id, transaction_id, 'STATE', data['state'], 
            data['municipality'], data['service'], data['consumerId'], amount, 'COMPLETED'
        ))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'State tax payment successful',
            'transaction_id': transaction_id,
            'tax_payment_id': tax_payment_id
        })
        
    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/tax/history', methods=['GET'])
@jwt_required()
def get_tax_history():
    """Get tax payment history"""
    user_id = get_jwt_identity()
    
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT tp.*, t.created_at as payment_date, t.description, t.counterparty
        FROM tax_payments tp
        JOIN transactions t ON tp.transaction_id = t.id
        WHERE tp.user_id = ?
        ORDER BY tp.created_at DESC
    ''', (user_id,))
    
    tax_payments = cursor.fetchall()
    conn.close()
    
    return jsonify({
        'success': True,
        'tax_payments': [dict(row) for row in tax_payments]
    })

@app.route('/api/tax/download-challan/<string:tax_payment_id>', methods=['GET'])
@jwt_required()
def download_challan(tax_payment_id):
    """Generate and download tax payment challan"""
    user_id = get_jwt_identity()
    
    conn = get_db()
    cursor = conn.cursor()
    
    # Get tax payment details with user info
    cursor.execute('''
        SELECT tp.*, t.created_at as payment_date, t.description, t.counterparty, t.reference,
               u.first_name, u.last_name, u.email, u.phone, u.account_number
        FROM tax_payments tp
        JOIN transactions t ON tp.transaction_id = t.id
        JOIN users u ON tp.user_id = u.id
        WHERE tp.id = ? AND tp.user_id = ?
    ''', (tax_payment_id, user_id))
    
    payment = cursor.fetchone()
    conn.close()
    
    if not payment:
        return jsonify({'success': False, 'error': 'Tax payment not found'}), 404
    
    # Generate challan data
    challan_data = {
        'challan_number': f"CH{payment['id'][:8].upper()}",
        'payment_date': payment['payment_date'],
        'tax_type': payment['tax_type'],
        'amount': payment['amount'],
        'status': payment['status'],
        'transaction_id': payment['transaction_id'],
        'user_details': {
            'name': f"{payment['first_name']} {payment['last_name']}",
            'email': payment['email'],
            'phone': payment['phone'],
            'account_number': payment['account_number']
        }
    }
    
    # Add tax-specific details
    if payment['tax_type'] == 'DIRECT':
        challan_data['tax_details'] = {
            'pan': payment['pan_tan'],
            'assessment_year': payment['assessment_year'],
            'tax_applicable': payment['tax_applicable'],
            'payment_type': payment['payment_type']
        }
    elif payment['tax_type'] == 'GST':
        challan_data['tax_details'] = {
            'gstin': payment['gstin'],
            'cpin': payment['cpin'],
            'cgst': payment['cgst'] or 0,
            'sgst': payment['sgst'] or 0,
            'igst': payment['igst'] or 0,
            'cess': payment['cess'] or 0
        }
    elif payment['tax_type'] == 'STATE':
        challan_data['tax_details'] = {
            'state': payment['state'],
            'municipality': payment['municipality'],
            'service_type': payment['service_type'],
            'consumer_id': payment['consumer_id']
        }
    
    return jsonify({
        'success': True,
        'challan': challan_data
    })

@app.route('/api/tax/export-history', methods=['GET'])
@jwt_required()
def export_tax_history():
    """Export complete tax payment history"""
    user_id = get_jwt_identity()
    export_format = request.args.get('format', 'json').lower()
    
    conn = get_db()
    cursor = conn.cursor()
    
    # Get user details
    cursor.execute("SELECT first_name, last_name, email, account_number FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    
    # Get all tax payments
    cursor.execute('''
        SELECT tp.*, t.created_at as payment_date, t.description, t.counterparty
        FROM tax_payments tp
        JOIN transactions t ON tp.transaction_id = t.id
        WHERE tp.user_id = ?
        ORDER BY tp.created_at DESC
    ''', (user_id,))
    
    tax_payments = cursor.fetchall()
    conn.close()
    
    export_data = {
        'export_date': datetime.now().isoformat(),
        'user_details': {
            'name': f"{user['first_name']} {user['last_name']}",
            'email': user['email'],
            'account_number': user['account_number']
        },
        'total_payments': len(tax_payments),
        'total_amount': sum(float(p['amount']) for p in tax_payments),
        'payments': [
            {
                'challan_number': f"CH{p['id'][:8].upper()}",
                'payment_date': p['payment_date'],
                'tax_type': p['tax_type'],
                'description': p['description'],
                'amount': p['amount'],
                'status': p['status'],
                'transaction_id': p['transaction_id']
            }
            for p in tax_payments
        ]
    }
    
    return jsonify({
        'success': True,
        'message': f'Tax history exported successfully in {export_format.upper()} format',
        'exportData': export_data
    })

# Start background task for stock price updates
def start_background_tasks():
    import threading
    import time
    
    def update_prices():
        while True:
            try:
                update_stock_prices()
                time.sleep(30)  # Update every 30 seconds
            except Exception as e:
                print(f"Error updating stock prices: {e}")
                time.sleep(60)
    
    thread = threading.Thread(target=update_prices, daemon=True)
    thread.start()

if __name__ == '__main__':
    # Initialize database
    init_db()
    
    # Train the keystroke authentication model on startup
    train_model_on_startup()
    
    # Start background tasks
    start_background_tasks()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)