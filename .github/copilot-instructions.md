<!-- Copilot instructions for the Restaurant Recommendation System -->
# GitHub Copilot Instructions

Purpose: Quickly orient an AI coding agent to be productive in this repository.

- Repo root expectations: Python 3.10+, dependencies in `requirements.txt`.

Essential commands
- Install deps: `pip install -r requirements.txt`
- Build models & processed data: `python -m src.pipeline_build`
- Run app (Streamlit): `streamlit run app/app.py`
- Run tests: `pytest tests/`

Big-picture architecture
- Frontend: `app/` — Streamlit multi-page app. Pages are under `app/pages/` (e.g. `2_Recommender.py`).
- Core logic: `src/` — holds ingestion, preprocessing, feature engineering, pipeline build, recommender and utilities.
- Models/artifacts: `models/` — contains serialized artifacts used at runtime (e.g. `tfidf_vectorizer.joblib`, `tfidf_matrix.joblib`, `restaurant_index.json`).
- Data: `data/raw/` (source CSV) -> `src.pipeline_build` -> `data/processed/` and `models/`.

Key files to inspect when making changes
- `src/pipeline_build.py` — orchestrates dataset processing and model persistence. Update when changing training or processing steps.
- `src/preprocessing.py` and `src/feature_engineering.py` — text cleaning and feature creation used by both training and serving.
- `src/recommender.py` — recommendation logic that loads `models/` artifacts; use it as the canonical runtime path for serving predictions.
- `app/app.py` and `app/pages/2_Recommender.py` — UI wiring and calls into `src` for predictions; important for UX changes.
- `models/restaurant_index.json` — mapping/index used by the UI and recommender; keep formats stable when changing.

Project-specific patterns & conventions
- Artifacts are serialized with `joblib` and JSON in `models/`. Code expects those filenames; preserve names or update both save/load paths.
- The Streamlit app imports modules from `src/` directly rather than duplicating code in `app/`.
- Tests live in `tests/` (examples: `test_preprocessing.py`, `test_recommender.py`). Keep unit tests focused on `src/` functions.
- Numbers in page filenames (e.g. `1_EDA.py`) determine Streamlit order — preserve naming when adding pages.

Integration notes and runtime constraints
- `src/recommender.py` typically expects pre-built TF-IDF artifacts. If you change vectorizer or TF-IDF construction, run `python -m src.pipeline_build` to regenerate `models/` artifacts before running the app or tests.
- The recommender uses text similarity (TF-IDF) combined with rating/popularity — changes to scoring require updating both model save/load and any frontend displays that show weights.

Editing guidance for AI agents
- When changing serialization formats or filenames, update both `src/pipeline_build.py` (save) and `src/recommender.py` (load) together.
- Prefer small, targeted changes. Re-run `python -m src.pipeline_build` and `pytest tests/` after model or preprocessing edits.
- For UI changes, modify `app/pages/` files and test with `streamlit run app/app.py`.

Examples (common tasks)
- Regenerate models after preprocessing tweak:
  1. Edit `src/preprocessing.py`.
  2. Run `python -m src.pipeline_build`.
  3. Run `pytest tests/` and `streamlit run app/app.py` to validate.
- Add a new Streamlit page:
  1. Create `app/pages/4_MyPage.py` following the pattern in `app/pages/1_EDA.py`.
  2. Keep filename prefix for ordering.

What not to assume
- There is no upstream API server — the app is local Streamlit using on-disk `models/` artifacts.
- Tests are minimal; do not assume comprehensive coverage.

If you need clarification
- Ask: which file(s) will the change touch? Provide quick steps to reproduce locally (commands above).

---
Please review and tell me if you'd like more detail on any section or examples for common change types.
