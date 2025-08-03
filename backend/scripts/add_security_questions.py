#!/usr/bin/env python3
"""
Script to add security questions for existing users in the SecureBank database.
This script adds security questions for specific users who registered before 
the security questions feature was implemented.
"""

import sqlite3
import uuid
from werkzeug.security import generate_password_hash

# Database path
DATABASE = 'banking.db'

def add_security_questions_for_users():
    """Add security questions for specified users"""
    
    # User data with security questions
    users_data = [
        {
            'email': 'priyankaavijay04@gmail.com',
            'question1': 'What was the nickname given to you by your grandfather?',
            'answer1': 'Boom puppy',
            'question2': 'During DOE what happened when trying to scale the hill?',
            'answer2': 'I ended up in a camel tent'
        },
        {
            'email': 'pranavm2323@gmail.com',
            'question1': 'What is your first pets name?',
            'answer1': 'Ellie',
            'question2': 'What is your best friends name?',
            'answer2': 'Priya'
        }
    ]
    
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    try:
        for user_data in users_data:
            email = user_data['email']
            
            # First, check if user exists
            cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
            user = cursor.fetchone()
            
            if not user:
                print(f"❌ User {email} not found in database. Skipping...")
                continue
                
            user_id = user[0]
            
            # Check if security questions already exist for this user
            cursor.execute("SELECT id FROM security_questions WHERE user_id = ?", (user_id,))
            existing_questions = cursor.fetchone()
            
            if existing_questions:
                print(f"⚠️  Security questions already exist for {email}. Skipping...")
                continue
            
            # Generate unique ID for security questions record
            security_questions_id = str(uuid.uuid4())
            
            # Hash and normalize the answers (same logic as registration)
            hashed_answer1 = generate_password_hash(user_data['answer1'].lower().strip())
            hashed_answer2 = generate_password_hash(user_data['answer2'].lower().strip())
            
            # Insert security questions (created_at will be set automatically by DEFAULT CURRENT_TIMESTAMP)
            cursor.execute('''
                INSERT INTO security_questions (id, user_id, question1, answer1, question2, answer2)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                security_questions_id,
                user_id,
                user_data['question1'],
                hashed_answer1,
                user_data['question2'],
                hashed_answer2
            ))
            
            print(f"✅ Successfully added security questions for {email}")
            print(f"   Question 1: {user_data['question1']}")
            print(f"   Question 2: {user_data['question2']}")
            print()
        
        # Commit all changes
        conn.commit()
        print("🎉 All security questions have been added successfully!")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        conn.rollback()
        
    finally:
        conn.close()

def verify_security_questions():
    """Verify that security questions were added correctly"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    try:
        # Check security questions for both users
        emails = ['priyankaavijay04@gmail.com', 'pranavm2323@gmail.com']
        
        for email in emails:
            cursor.execute('''
                SELECT u.email, u.first_name, u.last_name, sq.question1, sq.question2
                FROM users u
                JOIN security_questions sq ON u.id = sq.user_id
                WHERE u.email = ?
            ''', (email,))
            
            result = cursor.fetchone()
            if result:
                print(f"✅ {result[0]} ({result[1]} {result[2]}):")
                print(f"   Q1: {result[3]}")
                print(f"   Q2: {result[4]}")
                print()
            else:
                print(f"❌ No security questions found for {email}")
                
    except Exception as e:
        print(f"❌ Error during verification: {str(e)}")
        
    finally:
        conn.close()

if __name__ == "__main__":
    print("🔐 SecureBank Security Questions Setup Script")
    print("=" * 50)
    print()
    
    print("Adding security questions for specified users...")
    add_security_questions_for_users()
    
    print("\nVerifying security questions were added correctly...")
    verify_security_questions()
    
    print("Script completed!")