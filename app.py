from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('sales_model.pkl')

# ── 21 features — must match exactly what was used during model training ──
FEATURES = [
    'Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
    'StoreType', 'Assortment', 'CompetitionDistance', 'Promo2',
    'Year', 'Month', 'Day', 'WeekOfYear', 'IsWeekend',
    'IsMonthStart', 'IsMonthEnd',
    'Quarter',
    'Promo_Weekend', 'Promo_Month',
    'Store_DayOfWeek', 'NearCompetitor'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # ── Pull base values sent from the form ──────────────────────────────
        store        = float(data['Store'])
        day_of_week  = float(data['DayOfWeek'])
        promo        = float(data['Promo'])
        month        = float(data['Month'])
        is_weekend   = float(data['IsWeekend'])
        comp_dist    = float(data['CompetitionDistance'])

        # ── Auto-compute the 5 derived / interaction features ────────────────
        # These are computed here so the HTML form stays simple
        data['Promo_Weekend']   = promo * is_weekend          # Was there a promo on a weekend?
        data['Promo_Month']     = promo * month               # Which months run promos?
        data['Store_DayOfWeek'] = store * day_of_week         # Unique store+day pattern
        data['NearCompetitor']  = 1.0 if comp_dist < 1000 else 0.0  # Competitor within 1km?

        # Quarter is sent directly from the form dropdown
        # (already present in data['Quarter'])

        # ── Build feature vector in correct order ────────────────────────────
        features = [float(data[f]) for f in FEATURES]

        prediction = model.predict([features])[0]

        return jsonify({
            'prediction': round(float(prediction), 2),
            'status': 'success'
        })

    except KeyError as e:
        return jsonify({
            'error': f'Missing field: {str(e)} — make sure the form sent all required inputs.',
            'status': 'error'
        }), 400
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400


if __name__ == '__main__':
    app.run(debug=True)