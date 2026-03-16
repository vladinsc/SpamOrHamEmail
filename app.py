import base64
import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import time
import model as spamOrHam
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']


def authenticate_gmail():
    creds = None
    if os.path.exists('secrets/token.json'):
        creds = Credentials.from_authorized_user_file('secrets/token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('secrets/credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            with open('secrets/token.json', 'w') as token:
                token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

def get_email_body(payload):
    body = ""
    if 'parts' in payload:
        for part in payload['parts']:
            if part['mimeType'] == 'text/plain':
                data = part['body'].get('data', '')
                body += base64.urlsafe_b64decode(data).decode('utf-8')
            elif 'parts' in part:
                body += get_email_body(part)
    elif payload['mimeType'] == 'text/plain':
        data = payload['body'].get('data', '')
        body = base64.urlsafe_b64decode(data).decode('utf-8')
    return body

def save_email(sender,subject,body):


def scan_inbox(service, app_start_time):
    print("\nScanning inbox for new Mail...")
    results = service.users().messages().list(userId='me', q=f'is:unread in:inbox after:{app_start_time}').execute()
    messages = results.get('messages', [])
    if not messages:
        print("Your inbox has no messages... Lucky Lad or Lonely?..")
        return
    for message in messages:
        msg = service.users().messages().get(userId='me', id=message['id'], format='full').execute()

        # Extract headers (Subject, Sender)
        headers = msg['payload']['headers']
        subject = next((header['value'] for header in headers if header['name'] == 'Subject'), "No Subject")
        sender = next((header['value'] for header in headers if header['name'] == 'From'), "Unknown Sender")

        # Extract the actual text body
        body = get_email_body(msg['payload'])

        # Skip empty emails (like image-only newsletters)
        if not body.strip():
            continue

        print(f"\n--- Analyzing New Email ---")
        print(f"From: {sender}")
        print(f"Subject: {subject}")

        # Run the model for a Prediction
        start_model_at = time.time()
        confidence = spamOrHam.predict_spam(body)
        model_answer_at = time.time()
        print(f"Time Wasted on \"Thinking\" {model_answer_at - start_model_at} seconds")
        if confidence >= 0.4:
            print("You got a spam email")
        else:
            print("You got a ham email")


if __name__ == '__main__':
    gmail_service = authenticate_gmail()

    try:
        while True:
            app_start_time = int(time.time())
            scan_inbox(gmail_service, app_start_time)
            time.sleep(10)
    except KeyboardInterrupt:
        print("Keyboard Interrupt. Bye Bye, may your inbox be only ham!")
