"""
dixon_coles.py — Modelo Dixon-Coles para value-betting en fútbol europeo.

Implementa el modelo de Poisson bivariado con corrección tau (Dixon & Coles, 1997)
para estimar probabilidades de Over/Under 2.5 goles.

Referencia: Dixon, M. J., & Coles, S. G. (1997).
  "Modelling Association Football Scores and Inefficiencies in the Football Betting Market."
  Journal of the Royal Statistical Society Series C.

Uso:
    from src.models.dixon_coles import DixonColesModel

    model = DixonColesModel(league="BL2", db_path="data/betting.db")
    model.fit(seasons=["2223", "2324", "2425"])
    preds = model.predict_fixtures(fixture_ids=[101, 102, 103])
    model.save("models/dc_params_BL2.pkl")
"""

from __future__ import annotations

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


# ──────────────────────────────────────────────────────────────────────────────
# Estructuras de datos
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DCParams:
    """Parámetros ajustados del modelo Dixon-Coles."""
    attack: dict[str, float] = field(default_factory=dict)   # alpha_i por equipo
    defense: dict[str, float] = field(default_factory=dict)  # beta_i por equipo
    home_advantage: float = 0.0                              # gamma (ventaja local)
    rho: float = 0.0                                         # corrección tau/rho
    fitted_at: Optional[datetime] = None
    league: str = ""
    seasons_used: list[str] = field(default_factory=list)
    n_matches: int = 0


@dataclass
class DCPrediction:
    """Predicción por fixture."""
    fixture_id: int
    home_team: str
    away_team: str
    lambda_home: float          # goles esperados local
    lambda_away: float          # goles esperados visitante
    prob_over_25: float         # P(goles totales > 2.5)
    prob_under_25: float        # P(goles totales <= 2.5)
    prob_home_win: float        # P(1)
    prob_draw: float            # P(X)
    prob_away_win: float        # P(2)
    prob_btts: float            # P(ambos anotan)
    max_goals: int = 10         # truncamiento de la distribución
    predicted_at: datetime = field(default_factory=datetime.utcnow)


# ──────────────────────────────────────────────────────────────────────────────
# Corrección tau de Dixon-Coles
# ──────────────────────────────────────────────────────────────────────────────

def _tau(goals_home: int, goals_away: int, lambda_h: float, lambda_a: float, rho: float) -> float:
    """
    Factor de corrección tau para resultados de goles bajos (0-0, 1-0, 0-1, 1-1).

    Estos cuatro resultados tienen frecuencias que se desvían del Poisson puro,
    y tau los corrige. Para resultados con >= 2 goles en cualquier equipo, tau = 1.

    Dixon & Coles (1997), ecuación (1).
    """
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


# ──────────────────────────────────────────────────────────────────────────────
# Peso temporal (decaimiento exponencial)
# ──────────────────────────────────────────────────────────────────────────────

def _time_weight(match_date: datetime, reference_date: datetime, xi: float = 0.0018) -> float:
    """
    Peso temporal exponencial phi(t) = exp(-xi * delta_days).

    xi=0.0018 ≈ vida media de ~385 días (Dixon & Coles usan xi ≈ 0.002;
    aquí se usa un valor ligeramente más lento para ligas con menos partidos).

    Args:
        match_date:      Fecha del partido histórico.
        reference_date:  Fecha de referencia (normalmente hoy o último partido).
        xi:              Parámetro de decaimiento.

    Returns:
        Peso en (0, 1].
    """
    delta_days = max((reference_date - match_date).days, 0)
    return np.exp(-xi * delta_days)


# ──────────────────────────────────────────────────────────────────────────────
# Clase principal
# ──────────────────────────────────────────────────────────────────────────────

class DixonColesModel:
    """
    Modelo Dixon-Coles con decaimiento temporal y corrección tau.

    El modelo estima para cada equipo i:
      - alpha_i: fuerza atacante
      - beta_i:  solidez defensiva
    Más un parámetro global:
      - gamma:   ventaja de jugar en casa
      - rho:     corrección de correlación para resultados bajos

    La tasa esperada de goles es:
      lambda_home = alpha_home * beta_away * gamma
      lambda_away = alpha_away * beta_home
    """

    def __init__(
        self,
        league: str,
        db_path: str | Path = "data/betting.db",
        xi: float = 0.0018,
        max_goals: int = 10,
    ):
        self.league = league
        self.db_path = Path(db_path)
        self.xi = xi
        self.max_goals = max_goals
        self.params: Optional[DCParams] = None
        self._teams: list[str] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Carga de datos desde SQLite
    # ──────────────────────────────────────────────────────────────────────────

    def _load_matches(self, seasons: list[str]) -> list[dict]:
        """
        Carga partidos históricos desde betting.db para las temporadas indicadas.

        Filtra partidos con resultado conocido (status='finished') y excluye
        filas con goles nulos.

        Returns:
            Lista de dicts con claves: home_team, away_team, goals_home, goals_away,
            kickoff_utc (datetime).
        """
        placeholders = ",".join("?" for _ in seasons)
        query = f"""
            SELECT
                f.home_team_name AS home_team,
                f.away_team_name AS away_team,
                f.goals_home,
                f.goals_away,
                f.kickoff_utc
            FROM fixtures f
            WHERE f.league    = ?
              AND f.season    IN ({placeholders})
              AND f.status    = 'finished'
              AND f.goals_home  IS NOT NULL
              AND f.goals_away  IS NOT NULL
              AND f.home_team_name IS NOT NULL
              AND f.away_team_name IS NOT NULL
            ORDER BY f.kickoff_utc ASC
        """
        params = [self.league] + seasons

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        matches = []
        for row in rows:
            try:
                kickoff = datetime.fromisoformat(row["kickoff_utc"])
            except (TypeError, ValueError):
                logger.warning("Formato de fecha inválido para fila: %s", dict(row))
                continue
            matches.append({
                "home_team":  row["home_team"],
                "away_team":  row["away_team"],
                "goals_home": int(row["goals_home"]),
                "goals_away": int(row["goals_away"]),
                "kickoff":    kickoff,
            })

        logger.info(
            "[%s] %d partidos cargados para temporadas %s",
            self.league, len(matches), seasons,
        )
        return matches

    # ──────────────────────────────────────────────────────────────────────────
    # Log-likelihood con corrección tau y pesos temporales
    # ──────────────────────────────────────────────────────────────────────────

    def _neg_log_likelihood(
        self,
        params_vec: np.ndarray,
        matches: list[dict],
        teams: list[str],
        reference_date: datetime,
    ) -> float:
        """
        Función objetivo: -log-likelihood ponderada temporalmente.

        Estructura del vector de parámetros:
          params_vec[0..n_teams-1]              → log(alpha_i) para cada equipo
          params_vec[n_teams..2*n_teams-1]      → log(beta_i) para cada equipo
          params_vec[2*n_teams]                 → log(gamma) ventaja local
          params_vec[2*n_teams+1]               → rho (sin log, puede ser negativo)

        Usar log para forzar positividad en alpha, beta, gamma sin constraints.
        """
        n = len(teams)
        team_idx = {t: i for i, t in enumerate(teams)}

        # Extraer parámetros del vector plano
        log_alpha = params_vec[:n]
        log_beta  = params_vec[n:2 * n]
        log_gamma = params_vec[2 * n]
        rho       = params_vec[2 * n + 1]

        alpha = np.exp(log_alpha)
        beta  = np.exp(log_beta)
        gamma = np.exp(log_gamma)

        total_nll = 0.0

        for match in matches:
            i = team_idx.get(match["home_team"])
            j = team_idx.get(match["away_team"])
            if i is None or j is None:
                continue  # equipo no visto en el entrenamiento, ignorar

            gh = match["goals_home"]
            ga = match["goals_away"]

            lambda_h = alpha[i] * beta[j] * gamma
            lambda_a = alpha[j] * beta[i]

            # Poisson log-pmf
            log_p_h = poisson.logpmf(gh, lambda_h)
            log_p_a = poisson.logpmf(ga, lambda_a)

            # Corrección tau (solo para resultados 0-0, 1-0, 0-1, 1-1)
            tau_val = _tau(gh, ga, lambda_h, lambda_a, rho)

            if tau_val <= 0:
                # rho fuera del rango válido → penalidad grande
                return 1e10

            log_tau = np.log(tau_val)

            # Peso temporal
            w = _time_weight(match["kickoff"], reference_date, self.xi)

            total_nll -= w * (log_p_h + log_p_a + log_tau)

        return total_nll

    # ──────────────────────────────────────────────────────────────────────────
    # Ajuste del modelo
    # ──────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        seasons: list[str],
        reference_date: Optional[datetime] = None,
        max_iter: int = 2000,
    ) -> "DixonColesModel":
        """
        Ajusta los parámetros del modelo por máxima verosimilitud ponderada.

        Args:
            seasons:        Lista de temporadas a usar, e.g. ["2223", "2324", "2425"].
            reference_date: Fecha para el cálculo de pesos. Por defecto: hoy UTC.
            max_iter:       Máximo de iteraciones para el optimizador L-BFGS-B.

        Returns:
            self (para encadenar)

        Raises:
            ValueError: Si no hay suficientes datos o la optimización falla.
        """
        if reference_date is None:
            reference_date = datetime.utcnow()

        matches = self._load_matches(seasons)

        if len(matches) < 50:
            raise ValueError(
                f"[{self.league}] Solo {len(matches)} partidos. "
                "Se necesitan al menos 50 para ajustar el modelo."
            )

        # Recopilar todos los equipos del dataset
        teams_set: set[str] = set()
        for m in matches:
            teams_set.add(m["home_team"])
            teams_set.add(m["away_team"])
        teams = sorted(teams_set)
        n = len(teams)
        self._teams = teams

        logger.info("[%s] Ajustando modelo con %d equipos y %d partidos...", self.league, n, len(matches))

        # Valores iniciales: alpha=1, beta=1, gamma=1.3 (ventaja local típica), rho=-0.1
        x0 = np.zeros(2 * n + 2)
        x0[2 * n]     = np.log(1.3)   # log(gamma)
        x0[2 * n + 1] = -0.1          # rho

        # Restricción: para identificabilidad, fijar sum(log_alpha) = 0.
        # Implementado via constraint de scipy.
        constraints = {
            "type": "eq",
            "fun": lambda x: np.sum(x[:n]),   # sum(log_alpha) = 0
        }

        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(matches, teams, reference_date),
            method="L-BFGS-B",
            options={"maxiter": max_iter, "ftol": 1e-10, "gtol": 1e-8},
        )

        if not result.success:
            logger.warning(
                "[%s] Optimización no convergió: %s. Usando resultado parcial.",
                self.league, result.message,
            )

        # Extraer parámetros finales
        xopt = result.x
        log_alpha = xopt[:n]
        log_beta  = xopt[n:2 * n]
        log_gamma = xopt[2 * n]
        rho       = xopt[2 * n + 1]

        alpha = np.exp(log_alpha)
        beta  = np.exp(log_beta)
        gamma = np.exp(log_gamma)

        self.params = DCParams(
            attack          = dict(zip(teams, alpha.tolist())),
            defense         = dict(zip(teams, beta.tolist())),
            home_advantage  = float(gamma),
            rho             = float(rho),
            fitted_at       = datetime.utcnow(),
            league          = self.league,
            seasons_used    = seasons,
            n_matches       = len(matches),
        )

        logger.info(
            "[%s] Modelo ajustado: gamma=%.3f, rho=%.4f, NLL=%.2f",
            self.league, gamma, rho, result.fun,
        )
        return self

    # ──────────────────────────────────────────────────────────────────────────
    # Distribución conjunta de goles
    # ──────────────────────────────────────────────────────────────────────────

    def _joint_distribution(
        self, lambda_h: float, lambda_a: float
    ) -> np.ndarray:
        """
        Calcula la matriz de probabilidades conjuntas P(gh=i, ga=j) con corrección tau.

        Returns:
            Matriz (max_goals+1) x (max_goals+1) de probabilidades.
        """
        M = self.max_goals + 1
        rho = self.params.rho
        dist = np.zeros((M, M))

        for i in range(M):
            for j in range(M):
                p = poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a)
                p *= _tau(i, j, lambda_h, lambda_a, rho)
                dist[i, j] = max(p, 0.0)  # tau puede producir valores negativos en extremos

        # Renormalizar para que sume 1 (pequeña corrección numérica por truncamiento)
        total = dist.sum()
        if total > 0:
            dist /= total

        return dist

    # ──────────────────────────────────────────────────────────────────────────
    # Predicción individual
    # ──────────────────────────────────────────────────────────────────────────

    def predict_match(
        self, home_team: str, away_team: str, fixture_id: int = -1
    ) -> DCPrediction:
        """
        Genera predicción para un partido.

        Args:
            home_team:  Nombre del equipo local (debe coincidir con el de la DB).
            away_team:  Nombre del equipo visitante.
            fixture_id: ID del fixture en SQLite (-1 si es hipotético).

        Returns:
            DCPrediction con todas las probabilidades.

        Raises:
            RuntimeError: Si el modelo no ha sido ajustado.
            KeyError:     Si alguno de los equipos no está en el modelo.
        """
        if self.params is None:
            raise RuntimeError("Modelo no ajustado. Ejecutar fit() primero.")

        # Manejo de equipos no vistos: usar medias del modelo (fallback conservador)
        alpha_mean = np.mean(list(self.params.attack.values()))
        beta_mean  = np.mean(list(self.params.defense.values()))

        alpha_h = self.params.attack.get(home_team)
        beta_h  = self.params.defense.get(home_team)
        alpha_a = self.params.attack.get(away_team)
        beta_a  = self.params.defense.get(away_team)

        if alpha_h is None:
            logger.warning("[%s] Equipo '%s' no visto en entrenamiento. Usando medias.", self.league, home_team)
            alpha_h, beta_h = alpha_mean, beta_mean
        if alpha_a is None:
            logger.warning("[%s] Equipo '%s' no visto en entrenamiento. Usando medias.", self.league, away_team)
            alpha_a, beta_a = alpha_mean, beta_mean

        lambda_h = alpha_h * beta_a * self.params.home_advantage
        lambda_a = alpha_a * beta_h

        dist = self._joint_distribution(lambda_h, lambda_a)

        # Over/Under 2.5: total de goles > 2.5, es decir gh+ga >= 3
        prob_over_25  = 0.0
        prob_under_25 = 0.0
        prob_home_win = 0.0
        prob_draw     = 0.0
        prob_away_win = 0.0
        prob_btts     = 0.0

        M = self.max_goals + 1
        for i in range(M):
            for j in range(M):
                p = dist[i, j]
                total_goals = i + j
                if total_goals > 2:
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
            fixture_id    = fixture_id,
            home_team     = home_team,
            away_team     = away_team,
            lambda_home   = round(lambda_h, 4),
            lambda_away   = round(lambda_a, 4),
            prob_over_25  = round(prob_over_25, 6),
            prob_under_25 = round(prob_under_25, 6),
            prob_home_win = round(prob_home_win, 6),
            prob_draw     = round(prob_draw, 6),
            prob_away_win = round(prob_away_win, 6),
            prob_btts     = round(prob_btts, 6),
            max_goals     = self.max_goals,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Predicción por lote desde SQLite
    # ──────────────────────────────────────────────────────────────────────────

    def predict_fixtures(self, fixture_ids: list[int]) -> list[DCPrediction]:
        """
        Genera predicciones para una lista de fixture_ids desde betting.db.

        Args:
            fixture_ids: Lista de IDs en la tabla fixtures.

        Returns:
            Lista de DCPrediction. Los fixtures no encontrados se omiten con warning.
        """
        if not fixture_ids:
            return []

        placeholders = ",".join("?" for _ in fixture_ids)
        query = f"""
            SELECT
                f.id,
                f.home_team_name AS home_team,
                f.away_team_name AS away_team
            FROM fixtures f
            WHERE f.id IN ({placeholders})
              AND f.league = ?
        """

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, fixture_ids + [self.league]).fetchall()

        found_ids = {row["id"] for row in rows}
        for fid in fixture_ids:
            if fid not in found_ids:
                logger.warning("Fixture ID %d no encontrado en la DB para liga %s.", fid, self.league)

        predictions = []
        for row in rows:
            try:
                pred = self.predict_match(
                    home_team  = row["home_team"],
                    away_team  = row["away_team"],
                    fixture_id = row["id"],
                )
                predictions.append(pred)
            except Exception as exc:
                logger.error("Error prediciendo fixture %d: %s", row["id"], exc)

        return predictions

    # ──────────────────────────────────────────────────────────────────────────
    # Predicción de todos los fixtures del día o de una ventana
    # ──────────────────────────────────────────────────────────────────────────

    def predict_upcoming(
        self,
        hours_ahead: float = 24.0,
        reference_date: Optional[datetime] = None,
    ) -> list[DCPrediction]:
        """
        Predice todos los fixtures programados en las próximas `hours_ahead` horas.

        Útil para el pipeline automático: value_engine llama esto en lugar de
        pasar IDs manualmente.

        Args:
            hours_ahead:     Ventana de tiempo en horas.
            reference_date:  Fecha de referencia (default: ahora UTC).

        Returns:
            Lista de DCPrediction.
        """
        if reference_date is None:
            reference_date = datetime.utcnow()

        cutoff = reference_date + timedelta(hours=hours_ahead)

        query = """
            SELECT f.id
            FROM fixtures f
            WHERE f.league      = ?
              AND f.status      = 'scheduled'
              AND f.kickoff_utc >= ?
              AND f.kickoff_utc <= ?
        """

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                query,
                [self.league, reference_date.isoformat(), cutoff.isoformat()],
            ).fetchall()

        fixture_ids = [row[0] for row in rows]
        logger.info("[%s] %d fixtures en ventana ±%.0fh.", self.league, len(fixture_ids), hours_ahead)
        return self.predict_fixtures(fixture_ids)

    # ──────────────────────────────────────────────────────────────────────────
    # Guardar y cargar parámetros
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Serializa parámetros del modelo a un archivo .pkl."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "params": self.params,
                    "teams": self._teams,
                    "xi": self.xi,
                    "max_goals": self.max_goals,
                },
                f,
            )
        logger.info("[%s] Modelo guardado en %s", self.league, path)

    @classmethod
    def load(cls, path: str | Path, db_path: str | Path = "data/betting.db") -> "DixonColesModel":
        """Carga un modelo previamente serializado."""
        path = Path(path)
        with open(path, "rb") as f:
            state = pickle.load(f)

        instance = cls(
            league   = state["params"].league,
            db_path  = db_path,
            xi       = state.get("xi", 0.0018),
            max_goals = state.get("max_goals", 10),
        )
        instance.params  = state["params"]
        instance._teams  = state.get("teams", [])
        logger.info("[%s] Modelo cargado desde %s", instance.league, path)
        return instance

    # ──────────────────────────────────────────────────────────────────────────
    # Persistir predicciones en SQLite
    # ──────────────────────────────────────────────────────────────────────────

def save_predictions(self, predictions: list[DCPrediction]) -> None:
    if not predictions:
        return
    
    # Obtener fixture_ids únicos de esta tanda
    fixture_ids = list({p.fixture_id for p in predictions})
    
    now = datetime.utcnow().isoformat()
    rows = []
    for p in predictions:
        rows.append((p.fixture_id, "dixon_coles", "ou25", "over",
                     p.prob_over_25, p.lambda_home, p.lambda_away, now))
        rows.append((p.fixture_id, "dixon_coles", "ou25", "under",
                     p.prob_under_25, p.lambda_home, p.lambda_away, now))
        for sel, prob in [("home", p.prob_home_win), ("draw", p.prob_draw), ("away", p.prob_away_win)]:
            rows.append((p.fixture_id, "dixon_coles", "1x2", sel, prob, p.lambda_home, p.lambda_away, now))
        rows.append((p.fixture_id, "dixon_coles", "btts", "btts_yes",
                     p.prob_btts, p.lambda_home, p.lambda_away, now))

    with sqlite3.connect(self.db_path) as conn:
        # Borrar predicciones anteriores para estos fixtures — evita duplicados en JOINs
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

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO predictions
                    (fixture_id, model, market, selection, prob, lambda_home, lambda_away, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

        logger.info("[%s] %d predicciones guardadas en DB.", self.league, len(predictions))

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnóstico / resumen del modelo
    # ──────────────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Devuelve un string con los parámetros más relevantes del modelo."""
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
            "",
            f"{'Equipo':<30} {'Ataque':>8} {'Defensa':>9}",
            "-" * 50,
        ]
        # Ordenar equipos por fuerza de ataque descendente
        sorted_teams = sorted(p.attack.items(), key=lambda x: -x[1])
        for team, alpha in sorted_teams:
            beta = p.defense.get(team, float("nan"))
            lines.append(f"{team:<30} {alpha:>8.4f} {beta:>9.4f}")

        return "\n".join(lines)

    def top_teams(self, n: int = 5) -> dict:
        """Devuelve los n equipos con mayor ataque y mayor defensa."""
        if self.params is None:
            return {}
        sorted_attack  = sorted(self.params.attack.items(),  key=lambda x: -x[1])[:n]
        sorted_defense = sorted(self.params.defense.items(), key=lambda x: x[1])[:n]  # menor beta = mejor defensa
        return {
            "top_attack":  sorted_attack,
            "top_defense": sorted_defense,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Función de conveniencia para el pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_predictions(
    league: str,
    db_path: str = "data/betting.db",
    model_path: Optional[str] = None,
    seasons: Optional[list[str]] = None,
    hours_ahead: float = 24.0,
    persist: bool = True,
) -> list[DCPrediction]:
    """
    Entry point para el pipeline automático (llamado desde value_engine.py).

    1. Si existe un modelo guardado en model_path, lo carga.
    2. Si no, ajusta el modelo desde cero con las temporadas indicadas.
    3. Predice fixtures de las próximas hours_ahead horas.
    4. Opcionalmente guarda predicciones en DB.

    Args:
        league:      'BL2' o 'NL1'.
        db_path:     Ruta al archivo SQLite.
        model_path:  Ruta al .pkl del modelo. Si None, ajusta desde cero.
        seasons:     Temporadas para ajustar si no hay modelo guardado.
        hours_ahead: Ventana de predicción.
        persist:     Si True, guarda predicciones en la tabla predictions.

    Returns:
        Lista de DCPrediction.
    """
    if model_path and Path(model_path).exists():
        model = DixonColesModel.load(model_path, db_path=db_path)
        logger.info("[%s] Modelo cargado desde caché: %s", league, model_path)
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


# ──────────────────────────────────────────────────────────────────────────────
# CLI básico para correr manualmente
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Dixon-Coles model — value-betting system")
    parser.add_argument("--league",   required=True, choices=["BL2", "NL1", "EFL_L1", "NOR1"])
    parser.add_argument("--db",       default="data/betting.db")
    parser.add_argument("--model",    default=None, help="Ruta .pkl (carga si existe, guarda si no)")
    parser.add_argument("--seasons",  nargs="+",    default=["2223", "2324", "2425"])
    parser.add_argument("--hours",    type=float,   default=24.0)
    parser.add_argument("--summary",  action="store_true")
    parser.add_argument("--no-persist", action="store_true")
    args = parser.parse_args()

    model_path = args.model or f"models/dc_params_{args.league}.pkl"

    preds = run_predictions(
        league      = args.league,
        db_path     = args.db,
        model_path  = model_path,
        seasons     = args.seasons,
        hours_ahead = args.hours,
        persist     = not args.no_persist,
    )

    if args.summary:
        # Necesita cargar el modelo para el resumen
        m = DixonColesModel.load(model_path, db_path=args.db)
        print(m.summary())
        print()

    if preds:
        print(f"\n{'Partido':<40} {'λH':>5} {'λA':>5} {'Over%':>7} {'Under%':>7} {'1':>6} {'X':>6} {'2':>6} {'BTTS':>6}")
        print("-" * 95)
        for p in preds:
            partido = f"{p.home_team[:18]} vs {p.away_team[:18]}"
            print(
                f"{partido:<40} "
                f"{p.lambda_home:>5.2f} {p.lambda_away:>5.2f} "
                f"{p.prob_over_25*100:>6.1f}% {p.prob_under_25*100:>6.1f}% "
                f"{p.prob_home_win*100:>5.1f}% {p.prob_draw*100:>5.1f}% {p.prob_away_win*100:>5.1f}% "
                f"{p.prob_btts*100:>5.1f}%"
            )
    else:
        print("No hay fixtures próximos para predecir.")
