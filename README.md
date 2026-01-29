# Weather Prediction ML Pipeline

End-to-end machine learning workflow for fetching hourly weather data from the
OpenWeatherMap One Call API, preparing features, training a RandomForest model,
and persisting the trained model for later use.

## Highlights
- Fetches hourly weather observations from OpenWeatherMap (One Call 3.0).
- Stores cleaned data as a CSV with a UTC datetime index.
- Trains a RandomForest regressor to predict next-hour temperature.
- Includes test coverage for core utilities and data transformations.

## Project Structure
```plaintext
.
├── .devcontainer/
│   └── devcontainer.json
├── src/
│   ├── data_ingest.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── tests/
│   ├── test_data_ingest.py
│   ├── test_model.py
│   └── test_utils.py
├── requirements.txt
└── README.md
```

## Prerequisites
- Python 3.11+
- An OpenWeatherMap API key (for the One Call API 3.0 endpoint)

## Installation
Create a virtual environment, then install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration
Set your OpenWeatherMap API key in an environment variable or a `.env` file in
the project root:
```bash
export OPENWEATHERMAP_API_KEY="your_api_key_here"
```

> The ingestion script loads `.env` automatically when `python-dotenv` is
> installed (it is included in `requirements.txt`).

## Data Ingestion
Fetch hourly weather data and save it as a CSV file:
```bash
python src/data_ingest.py --lat 52.52 --lon 13.405 --output data/weather_hourly.csv
```

Key details:
- Endpoint: `https://api.openweathermap.org/data/3.0/onecall`
- Units: metric
- Output: CSV with a UTC datetime index

## Model Training
Train a RandomForest regressor and save the model artifact:
```bash
python src/train.py --data data/weather_hourly.csv --model models/weather_model.joblib
```

The training pipeline:
1. Reads the CSV into a DataFrame.
2. Generates features and the next-hour target value.
3. Splits the data into train/test sets.
4. Trains a RandomForest and prints the R² score.
5. Saves the model with `joblib`.

### Optional Training Parameters
```bash
python src/train.py \
  --data data/weather_hourly.csv \
  --model models/weather_model.joblib \
  --test-size 0.2 \
  --n-estimators 200 \
  --log-level INFO
```

## Development in VS Code Dev Container
This repository includes a `.devcontainer/devcontainer.json` configuration for
VS Code Dev Containers. Open the folder in VS Code and select:
`Dev Containers: Reopen in Container`.

## Testing
Run the test suite with:
```bash
pytest
```

## Notes & Best Practices
- Keep API keys out of source control by using `.env` files.
- Store generated data and model artifacts in dedicated folders (`data/`,
  `models/`) to keep the repository tidy.
- The One Call API endpoint requires a paid plan on OpenWeatherMap; confirm your
  account access before running ingestion.

## References
- [OpenWeatherMap API Documentation](https://openweathermap.org/api)
- [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
