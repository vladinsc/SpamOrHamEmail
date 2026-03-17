import base64
import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.cloud import pubsub_v1
import time
import model as spamOrHam
from dotenv import load_dotenv
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
load_dotenv()
PROJECT_ID = os.getenv('PROJECT_ID')
TOPIC_NAME = os.getenv('TOPIC_NAME')
SUBSCRIPTION_ID = os.getenv('SUBSCRIPTION_ID')
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "secrets/pubsub_key.json"

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
    pass
def scan_inbox(service, start_time):
    print("\nScanning inbox for new Mail...")
    results = service.users().messages().list(userId='me', q=f'is:unread in:inbox after:{start_time}').execute()
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
def setup_gmail_push(service):
    request = {
        'labelIds': ["INBOX"],
        'labelFilterAction': 'include',
        'topicName': f'projects/{PROJECT_ID}/topics/{TOPIC_NAME}'
    }
    try:
        response = service.users().watch(userId='me', body=request).execute()
        print(f"Gmail Push Active! Cloud Response: {response}")
        return response
    except Exception as e:
        print(f"CRITICAL ERROR connecting to Pub/Sub: {e}")
def callback(message):
    print("Message Received!")
    message.ack()
    scan_inbox(gmail_service, int(time.time())-300)

if __name__ == '__main__':

    app_start_time = int(time.time())
    gmail_service = authenticate_gmail()
    setup_gmail_push(gmail_service)
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    with subscriber:
        try:
            streaming_pull_future.result()
        except KeyboardInterrupt or Exception as e :
            print(e)
            print("Keyboard Interrupt. Bye Bye, may your inbox be only ham!")
            streaming_pull_future.cancel()