from xgboost import XGBClassifier

model = XGBClassifier()

def load():
    model.load_model('./xgboost_model.json')
    return model
