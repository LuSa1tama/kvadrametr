"""
app.py — веб-приложение КвадраМетр (Flask)
Запуск: python app.py
Интерфейс: http://127.0.0.1:5000
"""

from flask import Flask, render_template_string, request
import joblib
import numpy as np
import json
import os

app = Flask(__name__)

# загрузка модели
model    = joblib.load('models/model.pkl')
scaler   = joblib.load('models/scaler.pkl')
le       = joblib.load('models/label_encoder.pkl')
features = joblib.load('models/features.pkl')
with open('models/results.json', encoding='utf-8') as f:
    meta = json.load(f)

DISTRICTS = meta['districts']

HTML = """
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>КвадраМетр — Прогноз цены</title>
<style>
  body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; color: #222; }
  h1   { font-size: 24px; margin-bottom: 4px; }
  p.sub { color: #666; font-size: 14px; margin-bottom: 24px; }
  label { display: block; margin-top: 12px; font-size: 14px; font-weight: bold; }
  input, select { width: 100%; padding: 7px 10px; margin-top: 4px;
                  border: 1px solid #ccc; border-radius: 4px; font-size: 14px; box-sizing: border-box; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .checks { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px; }
  .checks label { font-weight: normal; display: flex; align-items: center; gap: 6px; margin: 0; }
  .checks input { width: auto; }
  button { margin-top: 24px; width: 100%; padding: 12px; background: #2c5f8a;
           color: white; border: none; border-radius: 4px; font-size: 16px; cursor: pointer; }
  button:hover { background: #1e4470; }
  .result { margin-top: 24px; padding: 16px; background: #f0f7ff;
            border: 1px solid #b3d4f5; border-radius: 4px; }
  .result .price { font-size: 28px; font-weight: bold; color: #1a4f8a; }
  .result .detail { font-size: 13px; color: #555; margin-top: 6px; }
  .models { margin-top: 32px; border-top: 1px solid #ddd; padding-top: 16px; }
  .models h3 { font-size: 15px; margin-bottom: 10px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #eee; }
  th { background: #f5f5f5; }
  tr.best td { font-weight: bold; background: #eef7ee; }
  hr { margin: 28px 0; border: none; border-top: 1px solid #ddd; }
</style>
</head>
<body>

<h1>КвадраМетр</h1>
<p class="sub">Прогнозирование цен на недвижимость · Random Forest · R²={{ "%.3f"|format(best_r2) }}</p>

<form method="POST">
  <div class="row">
    <div>
      <label>Площадь (м²)</label>
      <input type="number" name="area" value="{{ form.area or 65 }}" min="15" max="300" step="0.5" required>
    </div>
    <div>
      <label>Комнат</label>
      <select name="rooms">
        {% for r in [1,2,3,4,5] %}
          <option value="{{ r }}" {{ 'selected' if (form.rooms|int == r) }}>{{ r }}</option>
        {% endfor %}
      </select>
    </div>
  </div>

  <div class="row">
    <div>
      <label>Этаж</label>
      <input type="number" name="floor" value="{{ form.floor or 5 }}" min="1" max="30" required>
    </div>
    <div>
      <label>Этажей в доме</label>
      <input type="number" name="total_floors" value="{{ form.total_floors or 16 }}" min="1" max="40" required>
    </div>
  </div>

  <div class="row">
    <div>
      <label>Год постройки</label>
      <input type="number" name="year_built" value="{{ form.year_built or 2015 }}" min="1960" max="2024" required>
    </div>
    <div>
      <label>Район</label>
      <select name="district">
        {% for d in districts %}
          <option value="{{ d }}" {{ 'selected' if form.district == d }}>{{ d }}</option>
        {% endfor %}
      </select>
    </div>
  </div>

  <label>Расстояние до метро (км)</label>
  <input type="number" name="metro" value="{{ form.metro or 3.0 }}" min="0.1" max="60" step="0.1" required>

  <label>До школы (км)</label>
  <input type="number" name="school" value="{{ form.school or 0.8 }}" min="0.1" max="10" step="0.1" required>

  <div class="row">
    <div>
      <label>До парка (км)</label>
      <input type="number" name="park" value="{{ form.park or 1.5 }}" min="0.1" max="15" step="0.1">
    </div>
    <div>
      <label>До больницы (км)</label>
      <input type="number" name="hospital" value="{{ form.hospital or 2.0 }}" min="0.1" max="20" step="0.1">
    </div>
  </div>

  <div class="checks">
    <label><input type="checkbox" name="parking"    {{ 'checked' if form.parking    }}> Парковка</label>
    <label><input type="checkbox" name="elevator"   {{ 'checked' if form.elevator   }} checked> Лифт</label>
    <label><input type="checkbox" name="renovation" {{ 'checked' if form.renovation }} checked> Ремонт</label>
    <label><input type="checkbox" name="new_building" {{ 'checked' if form.new_building }} checked> Новостройка</label>
  </div>

  <button type="submit">Рассчитать стоимость</button>
</form>

{% if result %}
<div class="result">
  <div class="price">{{ result.price_fmt }}</div>
  <div class="detail">{{ result.per_m2 }} ₽/м² · диапазон: {{ result.price_min }} — {{ result.price_max }}</div>
</div>
{% endif %}

<div class="models">
  <h3>Сравнение моделей</h3>
  <table>
    <tr><th>Модель</th><th>R²</th><th>MAPE</th></tr>
    {% for name, m in model_results.items() %}
    <tr class="{{ 'best' if name == best_name }}">
      <td>{{ name }}{{ ' ★' if name == best_name }}</td>
      <td>{{ "%.4f"|format(m.R2) }}</td>
      <td>{{ "%.2f"|format(m.MAPE) }}%</td>
    </tr>
    {% endfor %}
  </table>
</div>

</body>
</html>
"""

def make_prediction(form):
    district = form.get('district', DISTRICTS[0])
    try:
        district_enc = le.transform([district])[0]
    except Exception:
        district_enc = 0

    year_built = int(form.get('year_built', 2015))
    area       = float(form.get('area', 65))
    floor_n    = int(form.get('floor', 5))
    tfloors    = int(form.get('total_floors', 16))

    x = np.array([[
        area,
        int(form.get('rooms', 2)),
        floor_n,
        tfloors,
        2024 - year_built,
        district_enc,
        float(form.get('metro', 3.0)),
        1 if form.get('parking') else 0,
        1 if form.get('elevator') else 0,
        1 if form.get('renovation') else 0,
        1 if year_built >= 2010 else 0,
        float(form.get('school', 0.8)),
        float(form.get('park', 1.5)),
        float(form.get('hospital', 2.0)),
    ]])

    price = float(model.predict(x)[0])
    price = max(round(price / 1000) * 1000, 1_000_000)
    margin = price * 0.055
    per_m2 = int(price / area)

    def fmt(n):
        return f"{n/1e6:.2f} млн ₽" if n >= 1e6 else f"{n:,.0f} ₽"

    return {
        'price_fmt': fmt(price),
        'per_m2':    f"{per_m2:,}".replace(',', ' '),
        'price_min': fmt(round((price - margin) / 1000) * 1000),
        'price_max': fmt(round((price + margin) / 1000) * 1000),
    }


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    form   = {}
    if request.method == 'POST':
        form   = request.form
        result = make_prediction(form)

    best_r2 = meta['results'][meta['best']]['R2']
    model_results = {k: type('M', (), v)() for k, v in meta['results'].items()}

    return render_template_string(HTML,
        result=result,
        form=form,
        districts=DISTRICTS,
        model_results=model_results,
        best_name=meta['best'],
        best_r2=best_r2,
    )


if __name__ == '__main__':
    app.run(debug=True)
