"""
dixon_coles.py — Modelo Dixon-Coles para value-betting en fútbol europeo.

Implementa el modelo de Poisson bivariado con corrección tau (Dixon & Coles, 1997)
para estimar probabilidades de Over/Under 2.5 goles.

Uso:
    python src/models/dixon_coles.py --league BL2 --seasons 2022-23 2023-24 2024-25 2025-26 --hours 720
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

logger = logging.getLogger(__name__)


@dataclass
class DCParams:
    attack: dict[str, float] = field(default_factory=dict)
    defense: dict[str, float] = field(default_factory=dict)
    home_advantage: float = 0.0
    rho: float = 0.0
    fitted_at: Optional[datetime] = None
    league: str = ""
    seasons_used: list[str] = field(default_factory=list)
    n_matches: int = 0


@dataclass
class DCPrediction:
    fixture_id: int
    home_team: str
    away_team: str
    lambda_home: float
    lambda_away: float
    prob_over_25: float
    prob_under_25: float
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    prob_btts: float
    max_goals: int = 10
    predicted_at: datetime = field(default_factory=datetime.utcnow)


def _tau(goals_home: int, goals_away: int, lambda_h: float, lambda_a: float, rho: float) -> float:
    if goals_home == 0 and goals_away == 0:
        return 1.0 - lambda_h * lambda_a * rho
    elif goals_home == 1 and goals_away == 0:
        return 1.0 + lambda_a * rho
    elif goals_home == 0 and goals_away == 1:
        return 1.0 + lambda_h * rho
    elif goals_home == 1 and goals_away == 1:
        return 1.0 - rho
    else:
        return 1.0


def _time_weight(match_date: datetime, reference_date: datetime, xi: float = 0.0018) -> float:
    delta_days = max((reference_date - match_date).days, 0)
    return np.exp(-xi * delta_days)


class DixonColesModel:

    def __init__(self, league: str, db_path: str | Path = "data/betting.db",
                 xi: float = 0.0018, max_goals: int = 10):
        self.league = league
        self.db_path = Path(db_path)
        self.xi = xi
        self.max_goals = max_goals
        self.params: Optional[DCParams] = None
        self._teams: list[str] = []

    def _load_matches(self, seasons: list[str]) -> list[dict]:
        placeholders = ",".join("?" for _ in seasons)
        query = f"""
            SELECT f.home_team_name AS home_team, f.away_team_name AS away_team,
                   f.goals_home, f.goals_away, f.kickoff_utc
            FROM fixtures f
            WHERE f.league = ? AND f.season IN ({placeholders})
              AND f.status = 'finished'
              AND f.goals_home IS NOT NULL AND f.goals_away IS NOT NULL
              AND f.home_team_name IS NOT NULL AND f.away_team_name IS NOT NULL
            ORDER BY f.kickoff_utc ASC
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, [self.league] + seasons).fetchall()

        matches = []
        for row in rows:
            try:
                kickoff = datetime.fromisoformat(row["kickoff_utc"])
            except (TypeError, ValueError):
                continue
            matches.append({
                "home_team":  row["home_team"],
                "away_team":  row["away_team"],
                "goals_home": int(row["goals_home"]),
                "goals_away": int(row["goals_away"]),
                "kickoff":    kickoff,
            })

        logger.info("[%s] %d partidos cargados para temporadas %s", self.league, len(matches), seasons)
        return matches

    def _neg_log_likelihood(self, params_vec: np.ndarray, matches: list[dict],
                             teams: list[str], reference_date: datetime) -> float:
        n = len(teams)
        team_idx = {t: i for i, t in enumerate(teams)}

        alpha = np.exp(params_vec[:n])
        beta  = np.exp(params_vec[n:2 * n])
        gamma = np.exp(params_vec[2 * n])
        rho   = params_vec[2 * n + 1]

        total_nll = 0.0
        for match in matches:
            i = team_idx.get(match["home_team"])
            j = team_idx.get(match["away_team"])
            if i is None or j is None:
                continue

            gh = match["goals_home"]
            ga = match["goals_away"]
            lambda_h = alpha[i] * beta[j] * gamma
            lambda_a = alpha[j] * beta[i]

            tau_val = _tau(gh, ga, lambda_h, lambda_a, rho)
            if tau_val <= 0:
                return 1e10

            w = _time_weight(match["kickoff"], reference_date, self.xi)
            total_nll -= w * (poisson.logpmf(gh, lambda_h) + poisson.logpmf(ga, lambda_a) + np.log(tau_val))

        return total_nll

    def fit(self, seasons: list[str], reference_date: Optional[datetime] = None,
            max_iter: int = 2000) -> "DixonColesModel":
        if reference_date is None:
            reference_date = datetime.utcnow()

        matches = self._load_matches(seasons)
        if len(matches) < 50:
            raise ValueError(f"[{self.league}] Solo {len(matches)} partidos. Se necesitan al menos 50.")

        teams = sorted({m["home_team"] for m in matches} | {m["away_team"] for m in matches})
        n = len(teams)
        self._teams = teams

        logger.info("[%s] Ajustando modelo con %d equipos y %d partidos...", self.league, n, len(matches))

        x0 = np.zeros(2 * n + 2)
        x0[2 * n]     = np.log(1.3)
        x0[2 * n + 1] = -0.1

        result = minimize(self._neg_log_likelihood, x0,
                          args=(matches, teams, reference_date),
                          method="L-BFGS-B",
                          options={"maxiter": max_iter, "ftol": 1e-10, "gtol": 1e-8})

        if not result.success:
            logger.warning("[%s] Optimización no convergió: %s.", self.league, result.message)

        xopt  = result.x
        alpha = np.exp(xopt[:n])
        beta  = np.exp(xopt[n:2 * n])
        gamma = np.exp(xopt[2 * n])
        rho   = xopt[2 * n + 1]

        self.params = DCParams(
            attack         = dict(zip(teams, alpha.tolist())),
            defense        = dict(zip(teams, beta.tolist())),
            home_advantage = float(gamma),
            rho            = float(rho),
            fitted_at      = datetime.utcnow(),
            league         = self.league,
            seasons_used   = seasons,
            n_matches      = len(matches),
        )

        logger.info("[%s] Modelo ajustado: gamma=%.3f, rho=%.4f, NLL=%.2f",
                    self.league, gamma, rho, result.fun)
        return self

    def _joint_distribution(self, lambda_h: float, lambda_a: float) -> np.ndarray:
        M   = self.max_goals + 1
        rho = self.params.rho
        dist = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                p = poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a)
                dist[i, j] = max(p * _tau(i, j, lambda_h, lambda_a, rho), 0.0)
        total = dist.sum()
        if total > 0:
            dist /= total
        return dist

    def predict_match(self, home_team: str, away_team: str, fixture_id: int = -1) -> DCPrediction:
        if self.params is None:
            raise RuntimeError("Modelo no ajustado. Ejecutar fit() primero.")

        alpha_mean = np.mean(list(self.params.attack.values()))
        beta_mean  = np.mean(list(self.params.defense.values()))

        alpha_h = self.params.attack.get(home_team, alpha_mean)
        beta_h  = self.params.defense.get(home_team, beta_mean)
        alpha_a = self.params.attack.get(away_team, alpha_mean)
        beta_a  = self.params.defense.get(away_team, beta_mean)

        if home_team not in self.params.attack:
            logger.warning("[%s] Equipo '%s' no visto en entrenamiento. Usando medias.", self.league, home_team)
        if away_team not in self.params.attack:
            logger.warning("[%s] Equipo '%s' no visto en entrenamiento. Usando medias.", self.league, away_team)

        lambda_h = alpha_h * beta_a * self.params.home_advantage
        lambda_a = alpha_a * beta_h
        dist     = self._joint_distribution(lambda_h, lambda_a)

        prob_over_25 = prob_under_25 = prob_home_win = prob_draw = prob_away_win = prob_btts = 0.0
        M = self.max_goals + 1
        for i in range(M):
            for j in range(M):
                p = dist[i, j]
                if i + j > 2:
                    prob_over_25 += p
                else:
                    prob_under_25 += p
                if i > j:
                    prob_home_win += p
                elif i == j:
                    prob_draw += p
                else:
                    prob_away_win += p
                if i > 0 and j > 0:
                    prob_btts += p

        return DCPrediction(
            fixture_id=fixture_id, home_team=home_team, away_team=away_team,
            lambda_home=round(lambda_h, 4), lambda_away=round(lambda_a, 4),
            prob_over_25=round(prob_over_25, 6), prob_under_25=round(prob_under_25, 6),
            prob_home_win=round(prob_home_win, 6), prob_draw=round(prob_draw, 6),
            prob_away_win=round(prob_away_win, 6), prob_btts=round(prob_btts, 6),
            max_goals=self.max_goals,
        )

    def predict_fixtures(self, fixture_ids: list[int]) -> list[DCPrediction]:
        if not fixture_ids:
            return []
        placeholders = ",".join("?" for _ in fixture_ids)
        query = f"""
            SELECT f.id, f.home_team_name AS home_team, f.away_team_name AS away_team
            FROM fixtures f WHERE f.id IN ({placeholders}) AND f.league = ?
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, fixture_ids + [self.league]).fetchall()

        predictions = []
        for row in rows:
            try:
                predictions.append(self.predict_match(row["home_team"], row["away_team"], row["id"]))
            except Exception as exc:
                logger.error("Error prediciendo fixture %d: %s", row["id"], exc)
        return predictions

    def predict_upcoming(self, hours_ahead: float = 24.0,
                         reference_date: Optional[datetime] = None) -> list[DCPrediction]:
        if reference_date is None:
            reference_date = datetime.utcnow()
        cutoff = reference_date + timedelta(hours=hours_ahead)
        query = """
            SELECT f.id FROM fixtures f
            WHERE f.league = ? AND f.status = 'scheduled'
              AND f.kickoff_utc >= ? AND f.kickoff_utc <= ?
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, [self.league, reference_date.isoformat(), cutoff.isoformat()]).fetchall()

        fixture_ids = [row[0] for row in rows]
        logger.info("[%s] %d fixtures en ventana ±%.0fh.", self.league, len(fixture_ids), hours_ahead)
        return self.predict_fixtures(fixture_ids)

    def save_predictions(self, predictions: list[DCPrediction]) -> None:
        """
        Guarda predicciones en la DB.
        Borra primero las predicciones anteriores para evitar duplicados en JOINs.
        """
        if not predictions:
            return

        fixture_ids = list({p.fixture_id for p in predictions})
        now = datetime.utcnow().isoformat()

        rows = []
        for p in predictions:
            rows.append((p.fixture_id, "dixon_coles", "ou25", "over",
                         p.prob_over_25, p.lambda_home, p.lambda_away, now))
            rows.append((p.fixture_id, "dixon_coles", "ou25", "under",
                         p.prob_under_25, p.lambda_home, p.lambda_away, now))
            for sel, prob in [("home", p.prob_home_win), ("draw", p.prob_draw), ("away", p.prob_away_win)]:
                rows.append((p.fixture_id, "dixon_coles", "1x2", sel, prob,
                             p.lambda_home, p.lambda_away, now))
            rows.append((p.fixture_id, "dixon_coles", "btts", "btts_yes",
                         p.prob_btts, p.lambda_home, p.lambda_away, now))

        with sqlite3.connect(self.db_path) as conn:
            placeholders = ",".join("?" * len(fixture_ids))
            conn.execute(f"""
                DELETE FROM predictions
                WHERE fixture_id IN ({placeholders}) AND model = 'dixon_coles'
            """, fixture_ids)
            conn.executemany("""
                INSERT INTO predictions
                    (fixture_id, model, market, selection, prob,
                     lambda_home, lambda_away, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)

        logger.info("[%s] %d predicciones guardadas en DB.", self.league, len(predictions))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"params": self.params, "teams": self._teams,
                         "xi": self.xi, "max_goals": self.max_goals}, f)
        logger.info("[%s] Modelo guardado en %s", self.league, path)

    @classmethod
    def load(cls, path: str | Path, db_path: str | Path = "data/betting.db") -> "DixonColesModel":
        with open(Path(path), "rb") as f:
            state = pickle.load(f)
        instance = cls(league=state["params"].league, db_path=db_path,
                       xi=state.get("xi", 0.0018), max_goals=state.get("max_goals", 10))
        instance.params = state["params"]
        instance._teams = state.get("teams", [])
        logger.info("[%s] Modelo cargado desde %s", instance.league, path)
        return instance

    def summary(self) -> str:
        if self.params is None:
            return "Modelo no ajustado."
        p = self.params
        lines = [
            f"=== Dixon-Coles [{p.league}] ===",
            f"Ajustado:        {p.fitted_at.strftime('%Y-%m-%d %H:%M') if p.fitted_at else 'N/A'}",
            f"Temporadas:      {', '.join(p.seasons_used)}",
            f"Partidos usados: {p.n_matches}",
            f"Ventaja local:   {p.home_advantage:.4f}",
            f"Rho (corr tau):  {p.rho:.4f}",
            "", f"{'Equipo':<30} {'Ataque':>8} {'Defensa':>9}", "-" * 50,
        ]
        for team, alpha in sorted(p.attack.items(), key=lambda x: -x[1]):
            lines.append(f"{team:<30} {alpha:>8.4f} {p.defense.get(team, float('nan')):>9.4f}")
        return "\n".join(lines)

    def top_teams(self, n: int = 5) -> dict:
        if self.params is None:
            return {}
        return {
            "top_attack":  sorted(self.params.attack.items(),  key=lambda x: -x[1])[:n],
            "top_defense": sorted(self.params.defense.items(), key=lambda x:  x[1])[:n],
        }


def run_predictions(league: str, db_path: str = "data/betting.db",
                    model_path: Optional[str] = None, seasons: Optional[list[str]] = None,
                    hours_ahead: float = 24.0, persist: bool = True) -> list[DCPrediction]:
    if model_path and Path(model_path).exists():
        model = DixonColesModel.load(model_path, db_path=db_path)
    else:
        if not seasons:
            raise ValueError("Se necesita 'seasons' para ajustar el modelo desde cero.")
        model = DixonColesModel(league=league, db_path=db_path)
        model.fit(seasons=seasons)
        if model_path:
            model.save(model_path)

    predictions = model.predict_upcoming(hours_ahead=hours_ahead)
    if persist:
        model.save_predictions(predictions)
    return predictions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--league",     required=True, choices=["BL2", "NL1", "EFL_L1", "NOR1"])
    parser.add_argument("--db",         default="data/betting.db")
    parser.add_argument("--model",      default=None)
    parser.add_argument("--seasons",    nargs="+", default=["2022-23", "2023-24", "2024-25"])
    parser.add_argument("--hours",      type=float, default=24.0)
    parser.add_argument("--summary",    action="store_true")
    parser.add_argument("--no-persist", action="store_true")
    args = parser.parse_args()

    model_path = args.model or f"models/dc_params_{args.league}.pkl"
    preds = run_predictions(league=args.league, db_path=args.db, model_path=model_path,
                            seasons=args.seasons, hours_ahead=args.hours,
                            persist=not args.no_persist)

    if args.summary:
        m = DixonColesModel.load(model_path, db_path=args.db)
        print(m.summary())

    if preds:
        print(f"\n{'Partido':<40} {'λH':>5} {'λA':>5} {'Over%':>7} {'Under%':>7} {'1':>6} {'X':>6} {'2':>6} {'BTTS':>6}")
        print("-" * 95)
        for p in preds:
            partido = f"{p.home_team[:18]} vs {p.away_team[:18]}"
            print(f"{partido:<40} {p.lambda_home:>5.2f} {p.lambda_away:>5.2f} "
                  f"{p.prob_over_25*100:>6.1f}% {p.prob_under_25*100:>6.1f}% "
                  f"{p.prob_home_win*100:>5.1f}% {p.prob_draw*100:>5.1f}% "
                  f"{p.prob_away_win*100:>5.1f}% {p.prob_btts*100:>5.1f}%")
    else:
        print("No hay fixtures próximos para predecir.")
