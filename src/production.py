from predict import predict_spam

emails = [
    "URGENT! You won FREE money!!!",
    "Hi team, let's meet tomorrow",
]

for e in emails:
    prob = predict_spam(e)
    print(f"{e[:30]} → {prob:.2%}")
