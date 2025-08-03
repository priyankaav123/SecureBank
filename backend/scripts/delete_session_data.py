#!/usr/bin/env python3
"""
Script to delete the last x session data entries for a specific user email.
This is useful for testing anomaly detection by removing recent activity.
"""

import sqlite3
import sys
import os

# Add the parent directory to the path to import from backend
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

DATABASE = '../banking.db'

def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def show_recent_events(user_email, limit=10):
    """Show recent events for the user"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, event_type, time, page_url, transaction_amount, additional_data
        FROM user_events 
        WHERE user_email = ?
        ORDER BY time DESC
        LIMIT ?
    """, (user_email, limit))
    
    events = cursor.fetchall()
    conn.close()
    
    if not events:
        print(f"No events found for user: {user_email}")
        return []
    
    print(f"\nRecent {len(events)} events for {user_email}:")
    print("-" * 80)
    print(f"{'ID':<8} {'Event Type':<20} {'Time':<20} {'Amount':<10} {'Page'}")
    print("-" * 80)
    
    for event in events:
        amount = f"${event['transaction_amount']:.2f}" if event['transaction_amount'] and event['transaction_amount'] > 0 else "-"
        page = event['page_url'] or "-"
        print(f"{event['id'][:8]:<8} {event['event_type']:<20} {event['time'][:19]:<20} {amount:<10} {page}")
    
    return events

def delete_recent_events(user_email, count):
    """Delete the last x events for a user"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Get the IDs of the last x events
    cursor.execute("""
        SELECT id FROM user_events 
        WHERE user_email = ?
        ORDER BY time DESC
        LIMIT ?
    """, (user_email, count))
    
    event_ids = [row['id'] for row in cursor.fetchall()]
    
    if not event_ids:
        print(f"No events found to delete for user: {user_email}")
        conn.close()
        return 0
    
    # Delete the events
    placeholders = ','.join(['?' for _ in event_ids])
    cursor.execute(f"""
        DELETE FROM user_events 
        WHERE id IN ({placeholders})
    """, event_ids)
    
    deleted_count = cursor.rowcount
    conn.commit()
    conn.close()
    
    return deleted_count

def main():
    print("🧹 Session Data Cleanup Tool")
    print("=" * 50)
    
    # Get user email
    user_email = input("Enter user email: ").strip()
    if not user_email:
        print("❌ Email cannot be empty!")
        sys.exit(1)
    
    # Show recent events first
    recent_events = show_recent_events(user_email)
    if not recent_events:
        sys.exit(1)
    
    # Get number of events to delete
    try:
        count = int(input(f"\nHow many recent events to delete? (1-{len(recent_events)}): "))
        if count <= 0:
            print("❌ Count must be positive!")
            sys.exit(1)
        if count > len(recent_events):
            print(f"❌ Only {len(recent_events)} events available!")
            sys.exit(1)
    except ValueError:
        print("❌ Please enter a valid number!")
        sys.exit(1)
    
    # Confirm deletion
    print(f"\n⚠️  You are about to delete the last {count} events for {user_email}")
    confirm = input("Are you sure? (y/N): ").strip().lower()
    
    if confirm != 'y' and confirm != 'yes':
        print("❌ Operation cancelled!")
        sys.exit(0)
    
    # Perform deletion
    print(f"\n🗑️  Deleting {count} events...")
    deleted_count = delete_recent_events(user_email, count)
    
    if deleted_count > 0:
        print(f"✅ Successfully deleted {deleted_count} events!")
        
        # Show remaining events
        print(f"\nRemaining events for {user_email}:")
        show_recent_events(user_email, 10)
        
        print(f"\n💡 Tip: This will help reset the anomaly detection for testing purposes.")
    else:
        print("❌ No events were deleted!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user!")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)