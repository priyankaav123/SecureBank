import sqlite3

conn = sqlite3.connect('banking.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

cursor.execute("""
    SELECT user_email, COUNT(*) as event_count 
    FROM user_events 
    WHERE user_email IS NOT NULL AND user_email != '' AND user_email != 'anonymous'
    GROUP BY user_email 
    ORDER BY COUNT(*) DESC
""")

users_with_events = cursor.fetchall()
print('Users and their event counts (need >= 20 for behavioral training):')

for user_row in users_with_events:
    user_email = user_row['user_email']
    event_count = user_row['event_count']
    status = 'TRAINABLE' if event_count >= 20 else 'INSUFFICIENT'
    print(f'  - {user_email}: {event_count} events [{status}]')

conn.close()