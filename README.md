# Weather Prediction ML Pipeline
A fully containerized machine learning project that collects real-time weather data via the OpenWeatherMap API, preprocesses it, and trains predictive models inside a reproducible Docker DevContainer environment.

---

## 1. Overview
This document provides a complete technical guide for setting up a weather prediction machine learning project in a reproducible **Docker-based development environment** using **VS Code Dev Containers**. The system fetches weather data from the OpenWeatherMap API, preprocesses it, and trains a simple predictive model using scikit-learn.

---

## 2. Project Structure
```plaintext
weather-prediction-project/
│  ├── .devcontainer/
│  │    ├── devcontainer.json
│  │    └── Dockerfile
│  ├── src/
│  │    ├── data_ingest.py
│  │    ├── train.py
│  │    └── model.py
│  ├── requirements.txt
│  └── README.md
```

---

## 3. Dev Container Configuration

### 3.1 devcontainer.json
```json
{
  "name": "weather-ml-dev",
  "context": "..",
  "dockerFile": "Dockerfile",
  "workspaceFolder": "/workspace",
  "extensions": [
    "ms-python.python",
    "ms-python.vscode-pylance"
  ],
  "settings": {
    "python.pythonPath": "/usr/local/bin/python"
  }
}
```

### 3.2 Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /workspace

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

CMD ["bash"]
```

### 3.3 requirements.txt
```
pandas>=2.0
scikit-learn>=1.3
requests>=2.30
```

You may optionally add:
```
matplotlib
streamlit
```
for visualization and web interface purposes.

---

## 4. Data Ingestion (OpenWeatherMap API)

Create `src/data_ingest.py`:

```python
import requests
import pandas as pd

API_KEY = "YOUR_API_KEY"  # Replace with your OpenWeatherMap API key
BASE_URL = "https://api.openweathermap.org/data/2.5/onecall"

def fetch_weather(lat, lon):
    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY,
        "units": "metric"
    }
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    return response.json()

def to_dataframe(json_data):
    df = pd.json_normalize(json_data, record_path=["hourly"])
    return df

if __name__ == "__main__":
    lat, lon = 52.52, 13.405  # Berlin example
    data = fetch_weather(lat, lon)
    df = to_dataframe(data)
    df.to_csv("weather_hourly.csv", index=False)
    print("Weather data saved as weather_hourly.csv")
```

---

## 5. Model Training

Create `src/train.py`:

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("weather_hourly.csv")

# Feature and target selection
X = df[['temp', 'humidity', 'pressure', 'wind_speed']]
y = df['temp'].shift(-1).dropna()
X = X.iloc[:-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Model R² score: {score:.3f}")
```

---

## 6. Running in VS Code Dev Container

1. **Ensure Docker is installed and running.**  
2. Open the folder in **VS Code** → Command Palette → `Remote-Containers: Open Folder in Container`.  
3. Once the container builds, open the terminal and execute:
   ```bash
   python src/data_ingest.py
   python src/train.py
   ```
4. You may extend the project using **Streamlit** for real-time dashboarding:
   ```bash
   streamlit run src/dashboard.py
   ```

---

## 7. Best Practices
- Pin dependency versions in `requirements.txt` for reproducibility.
- Rebuild the container whenever you modify dependencies.
- Use Docker volumes for large datasets to improve performance.
- Store API keys securely via environment variables or `.env` files.
- For production, consider **multi-stage builds** to minimize image size.

---

## 8. References
- [OpenWeatherMap API Documentation](https://openweathermap.org/api)
- [VS Code Dev Containers Guide](https://code.visualstudio.com/docs/devcontainers/containers)
- [Docker Official Docs](https://docs.docker.com/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)

---

**Professional Recommendation:**  
Containerize and publish this project to GitHub, then connect it with GitHub Codespaces for instant cloud-based development and training anywhere.
