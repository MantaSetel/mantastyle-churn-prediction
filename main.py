from flask import Flask, jsonify, request
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import model

app = Flask(__name__)

label_encoder = LabelEncoder()

loaded_model = model.load()

# Contoh data sementara
tasks = [
    {
        'id': 1,
        'title': 'Belajar Flask',
        'done': False
    },
    {
        'id': 2,
        'title': 'Buat RESTful API',
        'done': False
    }
]

@app.route('/tasks', methods=['GET'])
def get_tasks():
    return jsonify({'tasks': tasks})

@app.route('/customer-churn-predict', methods=['POST'])
def predict_customer_churn():
    customer_data = {
        'TenureMonths': [request.json['TenureMonths']],
        'GamesProduct': [request.json['GamesProduct']],
        'MusicProduct': [request.json['MusicProduct']],
        'EducationProduct': [request.json['EducationProduct']],
        'CallCenter': [request.json['CallCenter']], 
        'VideoProduct': [request.json['VideoProduct']],
        'UseMyApp': [request.json['UseMyApp']], 
        'MonthlyPurchase': [request.json['MonthlyPurchase']], 
        'CLTV': [request.json['CLTV']],
        'DeviceClass_HighEnd': [request.json['DeviceClass_HighEnd']], 
        'DeviceClass_LowEnd': [request.json['DeviceClass_LowEnd']], 
        'DeviceClass_MidEnd': [request.json['DeviceClass_MidEnd']],
        'PaymentMethod_Credit': [request.json['PaymentMethod_Credit']],
        'PaymentMethod_Debit': [request.json['PaymentMethod_Debit']], 
        'PaymentMethod_DigitalWallet': [request.json['PaymentMethod_DigitalWallet']],
        'PaymentMethod_Pulsa': [request.json['PaymentMethod_Pulsa']]
    }

    dataFrame = pd.DataFrame(customer_data)
    dataFrame.columns = [
    'TenureMonths', 'GamesProduct', 'MusicProduct',
    'EducationProduct', 'CallCenter', 'VideoProduct',
    'UseMyApp', 'MonthlyPurchase', 'CLTV',
    'DeviceClass_HighEnd', 'DeviceClass_LowEnd',
    'DeviceClass_MidEnd', 'PaymentMethod_Credit',
    'PaymentMethod_Debit', 'PaymentMethod_DigitalWallet',
    'PaymentMethod_Pulsa']

    for column in dataFrame.columns:
        if dataFrame[column].dtype == 'object':
            dataFrame[column] = label_encoder.fit_transform(dataFrame[column])

    predictions = loaded_model.predict(dataFrame)

    predictions_list = predictions.tolist()

    return jsonify({'churn': predictions_list}), 200
    
if __name__ == '__main__':
    app.run(debug=True)