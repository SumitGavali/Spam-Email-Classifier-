import joblib
import numpy as np
import re
from scipy.sparse import hstack

model = joblib.load("models/spam_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

def predict_spam(email_text: str) -> float:
    text_features = tfidf.transform([email_text])

    extra_features = np.array([[
        email_text.count('!'),
        int(bool(re.search(r'[A-Z]{5,}', email_text))),
        int(bool(re.search(
            r'free|win|winner|cash|prize|urgent',
            email_text.lower()
        )))
    ]])

    combined = hstack([text_features, extra_features])
    return model.predict_proba(combined)[0][1]
