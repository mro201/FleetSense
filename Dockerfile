FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# This installs uv and syncs dependencies before copying app code,
# so Docker can cache this layer when only the code changes, not deps.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

#copy over needed code
COPY fleetsense/ ./fleetsense/
COPY scripts/train_model.py ./scripts/train_model.py
COPY config.py ./config.py
COPY data/dataset/vessel_weekly_features_sample.csv ./data/dataset/vessel_weekly_features_sample.csv

RUN uv run scripts/train_model.py

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "fleetsense.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
