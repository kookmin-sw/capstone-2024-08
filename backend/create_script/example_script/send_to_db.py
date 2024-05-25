import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("../.config_secret/serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def send_to_db(data):
    try:
        doc_ref = db.collection('example_script').document()
        doc_ref.set(data)

    except Exception as e:
        print(f"Firestore 전송 실패: {e}")