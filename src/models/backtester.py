"""
backtester.py — Walk-forward backtest del modelo Dixon-Coles y ensemble DC+XGB.

Simula apuestas históricas usando odds de cierre (closing odds) de bet365,
calcula ROI, Brier score y log-loss por temporada y liga.

Uso:
    python src/models/backtester.py --league BL2 --test-seasons 2023-24 2024-25 2025-26
    python src/models/backtester.py --league BL2 --compare
    python src/models/backtester.py --all --compare
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "features"))

from dixon_coles import DixonColesModel

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "home_goals_avg_5", "away_goals_avg_5",
    "home_conceded_avg_5", "away_conceded_avg_5",
    "home_ou25_rate_5", "away_ou25_rate_5",
    "h2h_ou25_rate", "h2h_matches",
    "home_rest_days", "away_rest_days",
    "home_position", "away_position",
]


# ──────────────────────────────────────────────────────────────────────────────
# Estructuras de datos
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BetRecord:
    fixture_id: int
    league: str
    season: str
    kickoff: str
    home_team: str
    away_team: str
    market: str
    selection: str
    prob_dc: float
    prob_xgb: float
    prob_ensemble: float
    odds: float
    implied_prob: float
    edge: float
    stake: float
    result: int
    pnl: float
    over_25_actual: int


@dataclass
class BacktestResults:
    league: str
    test_seasons: list[str]
    train_seasons: list[str]
    edge_threshold: float
    mode: str = "dc_only"
    total_bets: int = 0
    wins: int = 0
    losses: int = 0
    total_staked: float = 0.0
    total_pnl: float = 0.0
    roi_pct: float = 0.0
    brier_score: float = 0.0
    log_loss_val: float = 0.0
    avg_edge: float = 0.0
    avg_odds: float = 0.0
    bets_by_season: dict = field(default_factory=dict)
    bets: list[BetRecord] = field(default_factory=list)

    def compute_metrics(self) -> None:
        if not self.bets:
            return
        self.total_bets   = len(self.bets)
        self.wins         = sum(1 for b in self.bets if b.result == 1)
        self.losses       = self.total_bets - self.wins
        self.total_staked = sum(b.stake for b in self.bets)
        self.total_pnl    = sum(b.pnl for b in self.bets)
        self.roi_pct      = (self.total_pnl / self.total_staked * 100) if self.total_staked > 0 else 0.0
        self.avg_edge     = np.mean([b.edge for b in self.bets])
        self.avg_odds     = np.mean([b.odds for b in self.bets])
        y_true = [b.over_25_actual for b in self.bets]
        y_prob = [b.prob_ensemble for b in self.bets]
        if len(set(y_true)) > 1:
            self.brier_score  = brier_score_loss(y_true, y_prob)
            self.log_loss_val = log_loss(y_true, y_prob)
        seasons: dict[str, dict] = {}
        for b in self.bets:
            if b.season not in seasons:
                seasons[b.season] = {"bets": 0, "wins": 0, "pnl": 0.0, "staked": 0.0}
            seasons[b.season]["bets"]   += 1
            seasons[b.season]["wins"]   += b.result
            seasons[b.season]["pnl"]    += b.pnl
            seasons[b.season]["staked"] += b.stake
        for s, d in seasons.items():
            d["roi_pct"] = (d["pnl"] / d["staked"] * 100) if d["staked"] > 0 else 0.0
        self.bets_by_season = seasons


# ──────────────────────────────────────────────────────────────────────────────
# Carga de datos
# ──────────────────────────────────────────────────────────────────────────────

def load_fixtures_with_odds(db_path: str, league: str, seasons: list[str]) -> list[dict]:
    placeholders = ",".join("?" for _ in seasons)
    query = f"""
        SELECT f.id, f.league, f.season, f.kickoff_utc,
               f.home_team_name, f.away_team_name,
               f.goals_home, f.goals_away, f.over_25,
               o_over.odds_decimal AS odds_over,
               o_under.odds_decimal AS odds_under
        FROM fixtures f
        LEFT JOIN odds o_over  ON (o_over.fixture_id=f.id  AND o_over.market='ou25'  AND o_over.selection='over')
        LEFT JOIN odds o_under ON (o_under.fixture_id=f.id AND o_under.market='ou25' AND o_under.selection='under')
        WHERE f.league=? AND f.season IN ({placeholders})
          AND f.status='finished' AND f.over_25 IS NOT NULL
        ORDER BY f.kickoff_utc ASC
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, [league] + seasons).fetchall()
    fixtures = []
    for row in rows:
        if row["odds_over"] is None or row["odds_under"] is None:
            continue
        try:
            kickoff = datetime.fromisoformat(row["kickoff_utc"])
        except (TypeError, ValueError):
            continue
        fixtures.append({
            "id": row["id"], "league": row["league"], "season": row["season"],
            "kickoff": kickoff, "home_team": row["home_team_name"],
            "away_team": row["away_team_name"], "goals_home": row["goals_home"],
            "goals_away": row["goals_away"], "over_25": row["over_25"],
            "odds_over": row["odds_over"], "odds_under": row["odds_under"],
        })
    logger.info("[%s] %d partidos con odds para %s", league, len(fixtures), seasons)
    return fixtures


def load_features_df(db_path: str, league: str) -> pd.DataFrame:
    query = """
        SELECT mf.fixture_id,
               mf.home_goals_avg_5, mf.away_goals_avg_5,
               mf.home_conceded_avg_5, mf.away_conceded_avg_5,
               mf.home_ou25_rate_5, mf.away_ou25_rate_5,
               mf.h2h_ou25_rate, mf.h2h_matches,
               mf.home_rest_days, mf.away_rest_days,
               mf.home_position, mf.away_position
        FROM match_features mf
        JOIN fixtures f ON mf.fixture_id = f.id
        WHERE f.league = ?
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=[league])
    return df.set_index("fixture_id")


def get_all_seasons(db_path: str, league: str) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT season FROM fixtures WHERE league=? ORDER BY season", [league]
        ).fetchall()
    return [r[0] for r in rows]


# ──────────────────────────────────────────────────────────────────────────────
# Motor del backtest
# ──────────────────────────────────────────────────────────────────────────────

class WalkForwardBacktester:

    def __init__(self, db_path: str = "data/betting.db", xi: float = 0.0018,
                 max_goals: int = 10, ensemble: bool = False, dc_weight: float = 0.6):
        self.db_path    = db_path
        self.xi         = xi
        self.max_goals  = max_goals
        self.ensemble   = ensemble
        self.dc_weight  = dc_weight
        self.xgb_weight = 1.0 - dc_weight

    def run(self, league: str, test_seasons: list[str], edge_threshold: float = 0.05,
            selection: str = "best", min_odds: float = 1.30,
            max_odds: float = 4.00, max_overround: float = 0.08) -> BacktestResults:

        all_seasons   = get_all_seasons(self.db_path, league)
        test_set      = set(test_seasons)
        train_seasons = [s for s in all_seasons if s not in test_set]

        if not train_seasons:
            raise ValueError(f"Sin temporadas de entrenamiento. Disponibles: {all_seasons}")

        mode = "ensemble" if self.ensemble else "dc_only"
        logger.info("[%s] Backtest %s — Train: %s | Test: %s | Edge: %.1f%%",
                    league, mode, train_seasons, test_seasons, edge_threshold * 100)

        results = BacktestResults(league=league, test_seasons=test_seasons,
                                  train_seasons=train_seasons, edge_threshold=edge_threshold, mode=mode)

        features_df = load_features_df(self.db_path, league) if self.ensemble else None

        for test_season in sorted(test_seasons):
            available_train = [s for s in train_seasons if s < test_season]
            if not available_train:
                logger.warning("[%s] Sin train antes de %s. Saltando.", league, test_season)
                continue

            logger.info("[%s] Test: %s | Train DC: %s", league, test_season, available_train)

            dc_model = DixonColesModel(league=league, db_path=self.db_path,
                                       xi=self.xi, max_goals=self.max_goals)
            dc_model.fit(seasons=available_train)

            xgb_model = self._train_xgb(league, available_train, features_df) if self.ensemble else None

            test_fixtures = load_fixtures_with_odds(self.db_path, league, [test_season])
            season_bets = 0

            for fixture in test_fixtures:
                bet = self._evaluate_fixture(
                    fixture=fixture, dc_model=dc_model, xgb_model=xgb_model,
                    features_df=features_df, edge_threshold=edge_threshold,
                    selection=selection, min_odds=min_odds,
                    max_odds=max_odds, max_overround=max_overround,
                )
                if bet is not None:
                    results.bets.append(bet)
                    season_bets += 1

            logger.info("[%s] %s: %d apuestas de %d partidos",
                        league, test_season, season_bets, len(test_fixtures))

        results.compute_metrics()
        return results

    def _train_xgb(self, league: str, train_seasons: list[str],
                   features_df: Optional[pd.DataFrame]):
        try:
            from xgboost_model import train_calibrated
        except ImportError:
            logger.warning("xgboost_model no disponible.")
            return None

        placeholders = ",".join("?" for _ in train_seasons)
        query = f"""
            SELECT f.id AS fixture_id, f.over_25
            FROM fixtures f
            WHERE f.league=? AND f.season IN ({placeholders})
              AND f.status='finished' AND f.over_25 IS NOT NULL
        """
        with sqlite3.connect(self.db_path) as conn:
            targets = pd.read_sql_query(query, conn, params=[league] + train_seasons)
            targets = targets.set_index("fixture_id")

        train_idx = features_df.index.intersection(targets.index)
        X = features_df.loc[train_idx, FEATURE_COLS].copy()
        y = targets.loc[train_idx, "over_25"].astype(int)

        critical = ["home_goals_avg_5", "away_goals_avg_5",
                    "home_ou25_rate_5", "away_ou25_rate_5"]
        mask = X[critical].notna().all(axis=1)
        X, y = X[mask], y[mask]

        if len(X) < 100:
            logger.warning("[%s] Muestra insuficiente para XGB (%d).", league, len(X))
            return None

        model = train_calibrated(X, y)
        logger.info("[%s] XGB entrenado con %d partidos", league, len(X))
        return model

    def _get_xgb_prob(self, fixture_id: int, xgb_model,
                      features_df: Optional[pd.DataFrame]) -> Optional[float]:
        if xgb_model is None or features_df is None:
            return None
        if fixture_id not in features_df.index:
            return None
        row = features_df.loc[fixture_id, FEATURE_COLS]
        critical = ["home_goals_avg_5", "away_goals_avg_5",
                    "home_ou25_rate_5", "away_ou25_rate_5"]
        if pd.isna(row[critical]).any():
            return None
        try:
            return float(xgb_model.predict_proba(pd.DataFrame([row]))[0, 1])
        except Exception:
            return None

    def _evaluate_fixture(self, fixture: dict, dc_model, xgb_model,
                          features_df, edge_threshold: float, selection: str,
                          min_odds: float, max_odds: float,
                          max_overround: float) -> Optional[BetRecord]:
        try:
            pred = dc_model.predict_match(fixture["home_team"], fixture["away_team"], fixture["id"])
        except Exception:
            return None

        prob_dc_over  = pred.prob_over_25
        prob_dc_under = pred.prob_under_25
        prob_xgb_over = self._get_xgb_prob(fixture["id"], xgb_model, features_df)

        if self.ensemble and prob_xgb_over is not None:
            prob_over  = self.dc_weight * prob_dc_over  + self.xgb_weight * prob_xgb_over
            prob_under = self.dc_weight * prob_dc_under + self.xgb_weight * (1 - prob_xgb_over)
        else:
            prob_over, prob_under = prob_dc_over, prob_dc_under
            prob_xgb_over = 0.0

        odds_over, odds_under = fixture["odds_over"], fixture["odds_under"]
        implied_over  = 1 / odds_over
        implied_under = 1 / odds_under
        overround = implied_over + implied_under - 1

        if overround > max_overround:
            return None

        edge_over  = prob_over  - implied_over
        edge_under = prob_under - implied_under

        if selection == "over":
            best_edge, best_sel, best_odds, best_prob = edge_over,  "over",  odds_over,  prob_over
        elif selection == "under":
            best_edge, best_sel, best_odds, best_prob = edge_under, "under", odds_under, prob_under
        else:
            if edge_over >= edge_under:
                best_edge, best_sel, best_odds, best_prob = edge_over,  "over",  odds_over,  prob_over
            else:
                best_edge, best_sel, best_odds, best_prob = edge_under, "under", odds_under, prob_under

        if best_edge < edge_threshold or best_odds < min_odds or best_odds > max_odds:
            return None

        over_25_actual = fixture["over_25"]
        won = (best_sel == "over" and over_25_actual == 1) or \
              (best_sel == "under" and over_25_actual == 0)
        stake = 1.0
        pnl   = (best_odds - 1) * stake if won else -stake

        return BetRecord(
            fixture_id=fixture["id"], league=fixture["league"], season=fixture["season"],
            kickoff=fixture["kickoff"].isoformat(), home_team=fixture["home_team"],
            away_team=fixture["away_team"], market="ou25", selection=best_sel,
            prob_dc=round(prob_dc_over if best_sel == "over" else prob_dc_under, 6),
            prob_xgb=round(prob_xgb_over if best_sel == "over" else 1 - prob_xgb_over, 6),
            prob_ensemble=round(best_prob, 6), odds=best_odds,
            implied_prob=round(1/best_odds, 6), edge=round(best_edge, 6),
            stake=stake, result=int(won), pnl=round(pnl, 4),
            over_25_actual=over_25_actual,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Reporte
# ──────────────────────────────────────────────────────────────────────────────

def print_report(results: BacktestResults) -> None:
    mode_label = "DC + XGBoost (ensemble)" if results.mode == "ensemble" else "Dixon-Coles solo"
    print(f"\n{'='*60}")
    print(f"  BACKTEST — {results.league} — {mode_label}")
    print(f"{'='*60}")
    print(f"  Entrenamiento:  {', '.join(results.train_seasons)}")
    print(f"  Test:           {', '.join(results.test_seasons)}")
    print(f"  Edge threshold: {results.edge_threshold*100:.1f}%")
    print(f"{'-'*60}")
    if results.total_bets == 0:
        print("  Sin apuestas.")
    else:
        print(f"  Total apuestas: {results.total_bets}")
        print(f"  Ganadas:        {results.wins}  ({results.wins/results.total_bets*100:.1f}%)")
        print(f"  Perdidas:       {results.losses}")
        print(f"  P&L total:      {results.total_pnl:+.2f} u")
        print(f"  ROI:            {results.roi_pct:+.2f}%")
        print(f"  Cuota promedio: {results.avg_odds:.2f}")
        print(f"  Edge promedio:  {results.avg_edge*100:+.2f}%")
    print(f"{'-'*60}")
    print(f"  Brier score:    {results.brier_score:.4f}")
    print(f"  Log-loss:       {results.log_loss_val:.4f}")
    print(f"{'-'*60}")
    print("  Por temporada:")
    for season, d in sorted(results.bets_by_season.items()):
        print(f"    {season}: {d['bets']:3d} apuestas | ROI {d['roi_pct']:+6.1f}% | P&L {d['pnl']:+.2f}u")
    print(f"{'='*60}")
    if results.total_bets > 0:
        verdict = "✅ ROI positivo" if results.roi_pct >= 0 else "❌ ROI negativo"
        print(f"\n  {verdict} — {results.roi_pct:+.2f}%\n")


def print_comparison(dc: BacktestResults, ens: BacktestResults) -> None:
    print(f"\n{'='*60}")
    print("  COMPARACIÓN: Dixon-Coles vs Ensemble")
    print(f"{'='*60}")
    print(f"  {'Métrica':<22} {'DC solo':>10} {'Ensemble':>10} {'Δ':>8}")
    print(f"  {'-'*52}")
    print(f"  {'ROI %':<22} {dc.roi_pct:>+9.2f}% {ens.roi_pct:>+9.2f}% {ens.roi_pct-dc.roi_pct:>+7.2f}%")
    print(f"  {'Brier score':<22} {dc.brier_score:>10.4f} {ens.brier_score:>10.4f} {ens.brier_score-dc.brier_score:>+8.4f}")
    print(f"  {'Apuestas':<22} {dc.total_bets:>10} {ens.total_bets:>10}")
    print(f"  {'P&L (u)':<22} {dc.total_pnl:>+10.2f} {ens.total_pnl:>+10.2f} {ens.total_pnl-dc.total_pnl:>+8.2f}")
    print(f"{'='*60}")
    roi_mejor   = ens.roi_pct > dc.roi_pct
    brier_mejor = ens.brier_score < dc.brier_score
    if roi_mejor and brier_mejor:
        print("\n  ✅ Ensemble supera a DC en ROI y calibración.")
        print("     → Usar ensemble (DC 60% + XGB 40%) en producción.\n")
    elif roi_mejor:
        print("\n  ⚠️  Ensemble tiene mejor ROI pero peor calibración.")
        print("     → Proceder con cautela.\n")
    elif brier_mejor:
        print("\n  ⚠️  Ensemble calibra mejor pero ROI inferior.")
        print("     → Continuar monitoreando.\n")
    else:
        print("\n  ℹ️  DC solo rinde igual o mejor.")
        print("     → Continuar con DC solo por ahora.\n")


def save_report(results: BacktestResults, output_dir: str = "reports") -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    seasons_str = "_".join(results.test_seasons).replace("-", "")
    path = Path(output_dir) / f"backtest_{results.league}_{results.mode}_{seasons_str}.json"
    data = asdict(results)
    data["bets"] = [asdict(b) for b in results.bets]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Reporte guardado en %s", path)
    return str(path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser(description="Walk-forward backtest — DC y Ensemble")
    parser.add_argument("--league",        choices=["BL2", "NL1"])
    parser.add_argument("--all",           action="store_true")
    parser.add_argument("--test-seasons",  nargs="+", default=["2023-24", "2024-25", "2025-26"])
    parser.add_argument("--edge",          type=float, default=0.05)
    parser.add_argument("--selection",     choices=["over", "under", "best"], default="best")
    parser.add_argument("--min-odds",      type=float, default=1.30)
    parser.add_argument("--max-odds",      type=float, default=4.00)
    parser.add_argument("--max-overround", type=float, default=0.08)
    parser.add_argument("--db",            default="data/betting.db")
    parser.add_argument("--ensemble",      action="store_true")
    parser.add_argument("--compare",       action="store_true", help="Correr DC y ensemble y comparar")
    parser.add_argument("--dc-weight",     type=float, default=0.6)
    parser.add_argument("--no-save",       action="store_true")
    args = parser.parse_args()

    leagues = ["BL2", "NL1"] if args.all else ([args.league] if args.league else None)
    if not leagues:
        parser.error("Especificá --league o --all")

    for league in leagues:
        if args.compare:
            bt_dc  = WalkForwardBacktester(db_path=args.db, ensemble=False)
            res_dc = bt_dc.run(league=league, test_seasons=args.test_seasons,
                               edge_threshold=args.edge, selection=args.selection,
                               min_odds=args.min_odds, max_odds=args.max_odds,
                               max_overround=args.max_overround)
            print_report(res_dc)

            bt_ens  = WalkForwardBacktester(db_path=args.db, ensemble=True, dc_weight=args.dc_weight)
            res_ens = bt_ens.run(league=league, test_seasons=args.test_seasons,
                                 edge_threshold=args.edge, selection=args.selection,
                                 min_odds=args.min_odds, max_odds=args.max_odds,
                                 max_overround=args.max_overround)
            print_report(res_ens)
            print_comparison(res_dc, res_ens)

            if not args.no_save:
                save_report(res_dc)
                save_report(res_ens)
        else:
            bt = WalkForwardBacktester(db_path=args.db, ensemble=args.ensemble,
                                       dc_weight=args.dc_weight)
            res = bt.run(league=league, test_seasons=args.test_seasons,
                         edge_threshold=args.edge, selection=args.selection,
                         min_odds=args.min_odds, max_odds=args.max_odds,
                         max_overround=args.max_overround)
            print_report(res)
            if not args.no_save:
                print(f"  Reporte: {save_report(res)}\n")