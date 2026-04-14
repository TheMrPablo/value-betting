"""
xgboost_model.py — Modelo XGBoost calibrado para predecir Over/Under 2.5.

Usa los features construidos por feature_builder.py y aplica calibración
de probabilidades (Platt scaling) para que las probabilidades sean realistas
y comparables con las de Dixon-Coles.

Flujo:
  1. Carga features de match_features + target de fixtures
  2. Entrena XGBoost con walk-forward (sin data leakage)
  3. Calibra probabilidades con CalibratedClassifierCV (Platt)
  4. Guarda modelo serializado en models/xgb_calibrated_BL2.pkl
  5. Guarda predicciones en tabla predictions de la DB

Uso:
    python src/models/xgboost_model.py --league BL2
    python src/models/xgboost_model.py --league BL2 --backtest
    python src/models/xgboost_model.py --all
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

DB_PATH    = Path("data/betting.db")
MODELS_DIR = Path("models")

# Features que usa el modelo — en este orden exacto
FEATURE_COLS = [
    "home_goals_avg_5",
    "away_goals_avg_5",
    "home_conceded_avg_5",
    "away_conceded_avg_5",
    "home_ou25_rate_5",
    "away_ou25_rate_5",
    "h2h_ou25_rate",
    "h2h_matches",
    "home_rest_days",
    "away_rest_days",
    "home_position",
    "away_position",
]

TARGET_COL = "over_25"


# ──────────────────────────────────────────────────────────────────────────────
# Carga de datos
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(db_path: Path, league: str) -> pd.DataFrame:
    """
    Carga features + target desde la DB.
    Solo incluye partidos con features completas y resultado conocido.
    """
    query = """
        SELECT
            mf.fixture_id,
            f.league,
            f.season,
            f.kickoff_utc                   AS kickoff,
            mf.home_goals_avg_5,
            mf.away_goals_avg_5,
            mf.home_conceded_avg_5,
            mf.away_conceded_avg_5,
            mf.home_ou25_rate_5,
            mf.away_ou25_rate_5,
            mf.h2h_ou25_rate,
            mf.h2h_matches,
            mf.home_rest_days,
            mf.away_rest_days,
            mf.home_position,
            mf.away_position,
            f.over_25
        FROM match_features mf
        JOIN fixtures f ON mf.fixture_id = f.id
        WHERE f.league  = ?
          AND f.status  = 'finished'
          AND f.over_25 IS NOT NULL
        ORDER BY f.kickoff_utc ASC
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=[league])

    df["kickoff"] = pd.to_datetime(df["kickoff"])

    # Partidos sin suficientes datos históricos (primeras jornadas)
    # — los dejamos con NaN y XGBoost los maneja nativamente
    logger.info("[%s] Dataset: %d partidos cargados", league, len(df))
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Modelo
# ──────────────────────────────────────────────────────────────────────────────

def build_model() -> XGBClassifier:
    """
    Construye el clasificador XGBoost con hiperparámetros conservadores.

    Parámetros elegidos para evitar overfitting con datasets pequeños
    (~1,000–5,000 partidos):
    - n_estimators=300: suficientes árboles sin memorizar
    - max_depth=4: árboles poco profundos = menos overfitting
    - learning_rate=0.05: aprendizaje lento = mejor generalización
    - subsample=0.8: bagging de filas
    - colsample_bytree=0.8: bagging de features
    - enable_categorical=False: features son todas numéricas
    """
    return XGBClassifier(
        n_estimators      = 300,
        max_depth         = 4,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 5,
        gamma             = 1,
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        objective         = "binary:logistic",
        eval_metric       = "logloss",
        random_state      = 42,
        n_jobs            = -1,
        verbosity         = 0,
        enable_categorical = False,
    )


def train_calibrated(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> CalibratedClassifierCV:
    """
    Entrena XGBoost y aplica calibración de Platt (sigmoid).

    La calibración asegura que prob=0.60 signifique realmente
    que el evento ocurre ~60% de las veces, no solo que el modelo
    da más score a ese partido.

    Usa cv=5 folds para la calibración — con datasets pequeños
    esto es más robusto que un split train/calibration separado.
    """
    base_model = build_model()

    calibrated = CalibratedClassifierCV(
        base_model,
        method = "sigmoid",   # Platt scaling
        cv     = 5,
    )
    calibrated.fit(X_train, y_train)
    return calibrated


# ──────────────────────────────────────────────────────────────────────────────
# Walk-forward backtest
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class XGBBacktestResults:
    league: str
    test_seasons: list[str]
    brier_score: float = 0.0
    log_loss_val: float = 0.0
    roc_auc: float = 0.0
    n_test: int = 0
    predictions: list[dict] = field(default_factory=list)

    def print_summary(self) -> None:
        print(f"\n{'='*55}")
        print(f"  XGBoost Backtest — {self.league}")
        print(f"  Test seasons: {', '.join(self.test_seasons)}")
        print(f"{'='*55}")
        print(f"  Partidos evaluados: {self.n_test}")
        print(f"  Brier score:        {self.brier_score:.4f}  (DC fue: ver abajo)")
        print(f"  Log-loss:           {self.log_loss_val:.4f}")
        print(f"  ROC-AUC:            {self.roc_auc:.4f}  (0.5=random, 1.0=perfecto)")
        print(f"{'='*55}\n")


def walk_forward_backtest(
    df: pd.DataFrame,
    league: str,
    test_seasons: list[str],
) -> XGBBacktestResults:
    """
    Backtest walk-forward del modelo XGBoost.

    Para cada temporada de test:
    - Entrena con todas las temporadas anteriores
    - Predice sobre la temporada de test
    - Nunca ve el futuro
    """
    results = XGBBacktestResults(league=league, test_seasons=test_seasons)
    all_seasons = sorted(df["season"].unique())
    test_set    = set(test_seasons)

    all_preds = []
    all_true  = []

    for test_season in sorted(test_seasons):
        train_seasons = [s for s in all_seasons if s < test_season]
        if not train_seasons:
            logger.warning("[%s] Sin datos de entrenamiento para %s", league, test_season)
            continue

        train_df = df[df["season"].isin(train_seasons)].copy()
        test_df  = df[df["season"] == test_season].copy()

        # Filtrar filas con NaN en features críticas
        critical = ["home_goals_avg_5", "away_goals_avg_5",
                    "home_ou25_rate_5", "away_ou25_rate_5"]
        train_df = train_df.dropna(subset=critical)
        test_df  = test_df.dropna(subset=critical)

        if len(train_df) < 100 or len(test_df) == 0:
            logger.warning("[%s] Muestra insuficiente para %s", league, test_season)
            continue

        X_train = train_df[FEATURE_COLS].copy()
        y_train = train_df[TARGET_COL].astype(int)
        X_test  = test_df[FEATURE_COLS].copy()
        y_test  = test_df[TARGET_COL].astype(int)

        logger.info(
            "[%s] Test: %s | Train: %d partidos | Test: %d partidos",
            league, test_season, len(train_df), len(test_df),
        )

        model = train_calibrated(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]  # P(Over 2.5)

        # Guardar predicciones
        for i, (_, row) in enumerate(test_df.iterrows()):
            results.predictions.append({
                "fixture_id":  int(row["fixture_id"]),
                "season":      test_season,
                "prob_over":   float(probs[i]),
                "prob_under":  float(1 - probs[i]),
                "actual":      int(y_test.iloc[i]),
            })

        all_preds.extend(probs.tolist())
        all_true.extend(y_test.tolist())

        logger.info(
            "[%s] %s: Brier=%.4f",
            league, test_season,
            brier_score_loss(y_test, probs),
        )

    if all_preds:
        results.n_test       = len(all_preds)
        results.brier_score  = brier_score_loss(all_true, all_preds)
        results.log_loss_val = log_loss(all_true, all_preds)
        results.roc_auc      = roc_auc_score(all_true, all_preds)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Entrenamiento final y serialización
# ──────────────────────────────────────────────────────────────────────────────

def train_final_model(
    df: pd.DataFrame,
    league: str,
    train_seasons: Optional[list[str]] = None,
) -> CalibratedClassifierCV:
    """
    Entrena el modelo final con todos los datos disponibles.
    Este es el modelo que se usa en producción para predecir
    partidos futuros.
    """
    if train_seasons:
        train_df = df[df["season"].isin(train_seasons)].copy()
    else:
        train_df = df.copy()

    # Filtrar partidos sin features suficientes (primeras jornadas)
    critical = ["home_goals_avg_5", "away_goals_avg_5",
                "home_ou25_rate_5", "away_ou25_rate_5"]
    train_df = train_df.dropna(subset=critical)

    X = train_df[FEATURE_COLS].copy()
    y = train_df[TARGET_COL].astype(int)

    logger.info(
        "[%s] Entrenando modelo final con %d partidos (%d features)",
        league, len(X), len(FEATURE_COLS),
    )

    model = train_calibrated(X, y)

    # Cross-validation rápida para verificar
    base = build_model()
    scores = cross_val_score(base, X, y, cv=5, scoring="neg_brier_score")
    logger.info(
        "[%s] CV Brier score: %.4f ± %.4f",
        league, -scores.mean(), scores.std(),
    )

    return model


def save_model(model: CalibratedClassifierCV, league: str) -> Path:
    """Serializa el modelo a disco."""
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f"xgb_calibrated_{league}.pkl"
    with open(path, "wb") as f:
        pickle.dump({
            "model":        model,
            "feature_cols": FEATURE_COLS,
            "league":       league,
            "trained_at":   datetime.utcnow().isoformat(),
        }, f)
    logger.info("[%s] Modelo guardado en %s", league, path)
    return path


def load_model(league: str) -> tuple[CalibratedClassifierCV, list[str]]:
    """Carga un modelo serializado."""
    path = MODELS_DIR / f"xgb_calibrated_{league}.pkl"
    with open(path, "rb") as f:
        state = pickle.load(f)
    return state["model"], state["feature_cols"]


# ──────────────────────────────────────────────────────────────────────────────
# Predicción en producción
# ──────────────────────────────────────────────────────────────────────────────

def predict_upcoming(
    league: str,
    db_path: Path = DB_PATH,
    hours_ahead: float = 24.0,
) -> list[dict]:
    """
    Predice partidos próximos usando el modelo guardado.
    Llamado por value_engine.py en el pipeline de producción.
    """
    model, feature_cols = load_model(league)

    # Cargar features de fixtures próximos
    cutoff = datetime.utcnow()
    from datetime import timedelta
    limit = cutoff + timedelta(hours=hours_ahead)

    query = """
        SELECT
            mf.fixture_id,
            f.home_team_name AS home_team,
            f.away_team_name AS away_team,
            f.kickoff_utc,
            mf.home_goals_avg_5,
            mf.away_goals_avg_5,
            mf.home_conceded_avg_5,
            mf.away_conceded_avg_5,
            mf.home_ou25_rate_5,
            mf.away_ou25_rate_5,
            mf.h2h_ou25_rate,
            mf.h2h_matches,
            mf.home_rest_days,
            mf.away_rest_days,
            mf.home_position,
            mf.away_position
        FROM match_features mf
        JOIN fixtures f ON mf.fixture_id = f.id
        WHERE f.league     = ?
          AND f.status     = 'scheduled'
          AND f.kickoff_utc >= ?
          AND f.kickoff_utc <= ?
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            query, conn,
            params=[league, cutoff.isoformat(), limit.isoformat()],
        )

    if df.empty:
        logger.info("[%s] Sin fixtures próximos para predecir (XGB)", league)
        return []

    X = df[feature_cols].copy()
    probs = model.predict_proba(X)[:, 1]

    predictions = []
    for i, (_, row) in enumerate(df.iterrows()):
        predictions.append({
            "fixture_id": int(row["fixture_id"]),
            "home_team":  row["home_team"],
            "away_team":  row["away_team"],
            "prob_over":  round(float(probs[i]), 6),
            "prob_under": round(float(1 - probs[i]), 6),
            "model":      "xgboost",
        })

    return predictions


def save_predictions_to_db(predictions: list[dict], db_path: Path = DB_PATH) -> None:
    if not predictions:
        return
    now = datetime.utcnow().isoformat()
    
    fixture_ids = list({p["fixture_id"] for p in predictions})
    
    rows = []
    for p in predictions:
        rows.append((p["fixture_id"], "xgboost", "ou25", "over",  p["prob_over"],  None, None, now))
        rows.append((p["fixture_id"], "xgboost", "ou25", "under", p["prob_under"], None, None, now))
    
    with sqlite3.connect(db_path) as conn:
        # Borrar predicciones anteriores para estos fixtures
        placeholders = ",".join("?" * len(fixture_ids))
        conn.execute(f"""
            DELETE FROM predictions
            WHERE fixture_id IN ({placeholders}) AND model = 'xgboost'
        """, fixture_ids)
        
        conn.executemany("""
            INSERT INTO predictions
                (fixture_id, model, market, selection, prob, lambda_home, lambda_away, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
    
    logger.info("Guardadas %d predicciones XGB en DB", len(predictions))


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="XGBoost calibrado para Over/Under 2.5")
    parser.add_argument("--league",   choices=["BL2", "NL1"])
    parser.add_argument("--all",      action="store_true")
    parser.add_argument("--backtest", action="store_true", help="Correr walk-forward backtest")
    parser.add_argument("--test-seasons", nargs="+", default=["2023-24", "2024-25", "2025-26"])
    parser.add_argument("--db",       default=str(DB_PATH))
    args = parser.parse_args()

    leagues = ["BL2", "NL1"] if args.all else ([args.league] if args.league else None)
    if not leagues:
        parser.error("Especificá --league o --all")

    for league in leagues:
        print(f"\n{'='*55}")
        print(f"  XGBoost — {league}")
        print(f"{'='*55}")

        df = load_dataset(Path(args.db), league)

        if args.backtest:
            # Walk-forward backtest
            results = walk_forward_backtest(df, league, args.test_seasons)
            results.print_summary()

            # Comparación con Dixon-Coles (referencia del backtester.py)
            print("  Referencia Dixon-Coles (mismo período):")
            print("    BL2 Brier: 0.2405  (histórico) / 0.2515 (2025-26)")
            print("    Si XGB Brier < DC Brier → XGB añade valor\n")

        # Entrenar modelo final con todos los datos
        print("  Entrenando modelo final...")
        all_seasons = sorted(df["season"].unique())
        model = train_final_model(df, league)
        path  = save_model(model, league)

        print(f"\n  ✅ Modelo guardado: {path}")
        print(f"  Features usadas: {', '.join(FEATURE_COLS)}\n")
