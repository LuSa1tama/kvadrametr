"""
create_data.py — генерация датасета и обучение модели КвадраМетр
Запустить перед app.py
"""

import numpy as np
import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os, warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
N = 2000

districts = {
    'Центр':    {'base': 250000, 'coef': 1.8},
    'Запад':    {'base': 220000, 'coef': 1.5},
    'Север':    {'base': 200000, 'coef': 1.3},
    'Восток':   {'base': 180000, 'coef': 1.1},
    'Юг':       {'base': 170000, 'coef': 1.0},
    'Пригород': {'base': 130000, 'coef': 0.7},
}

district_col = np.random.choice(list(districts.keys()), N,
    p=[0.10, 0.15, 0.20, 0.20, 0.20, 0.15])
area        = np.clip(np.round(np.random.lognormal(4.2, 0.4, N), 1), 20, 250)
rooms       = np.clip(np.round(area / 30).astype(int), 1, 6)
floor       = np.random.randint(1, 25, N)
total_floors= np.clip(floor + np.random.randint(0, 10, N), floor, 30)
year_built  = np.random.choice(range(1960, 2024), N,
    p=np.concatenate([np.ones(20)*0.15/20, np.ones(20)*0.35/20, np.ones(24)*0.50/24]))
building_age = 2024 - year_built

metro_dist = np.array([
    np.random.exponential(3) if d == 'Центр'
    else np.random.uniform(15, 60) if d == 'Пригород'
    else np.random.exponential(8)
    for d in district_col])
metro_dist = np.clip(metro_dist, 0.3, 60)

has_parking   = np.random.binomial(1, 0.4, N)
has_elevator  = (total_floors >= 5).astype(int)
has_renovation= np.random.binomial(1, 0.55, N)
is_new_building = (year_built >= 2010).astype(int)
school_dist   = np.random.uniform(0.1, 5.0, N)
park_dist     = np.random.uniform(0.1, 8.0, N)
hospital_dist = np.random.uniform(0.2, 10.0, N)

prices = np.array([
    max(
        districts[district_col[i]]['base'] * districts[district_col[i]]['coef'] * area[i]
        - 800 * metro_dist[i] - 400 * building_age[i]
        + 15000 * has_renovation[i] + 20000 * has_parking[i]
        + 5000 * has_elevator[i] + 25000 * is_new_building[i]
        - 5000 * school_dist[i] - 3000 * park_dist[i]
        + (3000 if 1 < floor[i] < total_floors[i] else -5000)
        + np.random.normal(0, districts[district_col[i]]['base'] * area[i] * 0.05),
        1_000_000
    )
    for i in range(N)
])
prices = np.round(prices, -3).astype(int)

df = pd.DataFrame({
    'area': area, 'rooms': rooms, 'floor': floor,
    'total_floors': total_floors, 'year_built': year_built,
    'building_age': building_age, 'district': district_col,
    'metro_distance_km': np.round(metro_dist, 2),
    'has_parking': has_parking, 'has_elevator': has_elevator,
    'has_renovation': has_renovation, 'is_new_building': is_new_building,
    'school_distance_km': np.round(school_dist, 2),
    'park_distance_km': np.round(park_dist, 2),
    'hospital_distance_km': np.round(hospital_dist, 2),
    'price': prices,
})
df.to_csv('data/real_estate.csv', index=False)
print(f"Датасет сохранён: {len(df)} строк")

# --- обучение ---
le = LabelEncoder()
df['district_enc'] = le.fit_transform(df['district'])

features = ['area','rooms','floor','total_floors','building_age','district_enc',
            'metro_distance_km','has_parking','has_elevator','has_renovation',
            'is_new_building','school_distance_km','park_distance_km','hospital_distance_km']
X, y = df[features], df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_tr_sc, X_te_sc = scaler.fit_transform(X_train), scaler.transform(X_test)

models = {
    'Linear Regression':  (LinearRegression(),                                              True),
    'Ridge Regression':   (Ridge(alpha=10),                                                 True),
    'Decision Tree':      (DecisionTreeRegressor(max_depth=8, random_state=42),             False),
    'KNN':                (KNeighborsRegressor(n_neighbors=7),                              True),
    'Random Forest':      (RandomForestRegressor(n_estimators=200, max_depth=12,
                                                 random_state=42, n_jobs=-1),               False),
    'Gradient Boosting':  (GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                     learning_rate=0.05, random_state=42),  False),
}

results = {}
print("\n=== Сравнение моделей ===")
for name, (m, scaled) in models.items():
    m.fit(X_tr_sc if scaled else X_train, y_train)
    pred = m.predict(X_te_sc if scaled else X_test)
    r2   = r2_score(y_test, pred)
    mae  = mean_absolute_error(y_test, pred)
    mape = float(np.mean(np.abs((y_test - pred) / y_test)) * 100)
    results[name] = {'R2': r2, 'MAE': mae, 'MAPE': mape}
    print(f"  {name:22s}  R²={r2:.4f}  MAE={mae:,.0f}  MAPE={mape:.2f}%")

best_name = max(results, key=lambda x: results[x]['R2'])
best_model, best_scaled = models[best_name]
print(f"\nЛучшая модель: {best_name}")

joblib.dump(best_model, 'models/model.pkl')
joblib.dump(scaler,     'models/scaler.pkl')
joblib.dump(le,         'models/label_encoder.pkl')
joblib.dump(features,   'models/features.pkl')
with open('models/results.json', 'w', encoding='utf-8') as f:
    json.dump({'results': results, 'best': best_name,
               'districts': list(districts.keys())}, f, ensure_ascii=False, indent=2)
print("Модель сохранена в models/")
