"""
value_engine.py — Motor central de detección de value.

Combina las predicciones de Dixon-Coles y XGBoost, aplica los filtros
del spec §5, calcula el stake con Kelly fraccionario, y genera señales
BET / REVIEW / SKIP para el bot de Telegram.

Flujo:
  1. Cargar predicciones DC y XGB desde la DB (tabla predictions)
  2. Calcular probabilidad ensemble (DC 60% + XGB 40%)
  3. Calcular edge vs cuota de Betplay
  4. Construir FixtureContext con toda la info del partido
  5. Correr FilterEngine → status BET / REVIEW / SKIP
  6. Si BET o REVIEW: calcular stake con KellyCalculator
  7. Guardar señal en tabla signals
  8. Devolver lista de BettingSignal para el bot

Uso:
    python src/engine/value_engine.py --league BL2
    python src/engine/value_engine.py --all
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).parent.parent / "features"))

from filters import FilterEngine, FilterResult, FilterStatus, FixtureContext, run_filters
from kelly import KellyCalculator, KellyResult, compute_stake, recommend_profile

logger = logging.getLogger(__name__)

DB_PATH    = Path("data/betting.db")
CONFIG_DIR = Path("config")

# Pesos del ensemble (configurables)
DC_WEIGHT  = 0.60
XGB_WEIGHT = 0.40

# Thresholds por defecto (idealmente vienen de config/thresholds.yaml)
DEFAULT_THRESHOLDS = {
    "edge_min_conservador": 0.05,
    "edge_min_agresivo":    0.08,
    "max_overround":        0.11,
    "min_odds":             1.30,
    "max_odds":             4.00,
    "dc_weight":            0.60,
}


# ──────────────────────────────────────────────────────────────────────────────
# Estructura de señal
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BettingSignal:
    """Señal de apuesta generada por el value engine."""
    fixture_id: int
    league: str
    home_team: str
    away_team: str
    kickoff_utc: str
    market: str = "ou25"
    selection: str = ""           # 'over' o 'under'

    # Probabilidades
    prob_dc: float = 0.0
    prob_xgb: float = 0.0
    prob_ensemble: float = 0.0
    implied_prob: float = 0.0

    # Mercado
    betplay_odds: float = 0.0
    overround_pct: float = 0.0
    edge_pct: float = 0.0

    # Sizing
    stake_cop: float = 0.0
    kelly_fraction: float = 0.0
    profile: str = "burn_in"

    # Filtros
    status: str = "SKIP"          # 'BET', 'REVIEW', 'SKIP'
    flags: list[str] = field(default_factory=list)
    flags_detail: list[dict] = field(default_factory=list)

    # Meta
    generated_at: str = ""
    bankroll_snapshot: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Carga de datos
# ──────────────────────────────────────────────────────────────────────────────

def load_upcoming_fixtures(
    db_path: Path,
    league: str,
    hours_ahead: float = 24.0,
) -> list[dict]:
    """
    Carga fixtures próximos con sus predicciones DC y XGB desde la DB.
    Solo incluye fixtures que tienen predicciones de ambos modelos.
    """
    now    = datetime.utcnow()
    cutoff = now + timedelta(hours=hours_ahead)

    query = """
        SELECT
            f.id, f.league, f.season,
            f.home_team_name AS home_team,
            f.away_team_name AS away_team,
            f.kickoff_utc,
            f.matchweek,

            -- Predicciones DC
            p_dc_over.prob   AS prob_dc_over,
            p_dc_under.prob  AS prob_dc_under,

            -- Predicciones XGB
            p_xgb_over.prob  AS prob_xgb_over,
            p_xgb_under.prob AS prob_xgb_under,

            -- Features contextuales
            mf.lineup_confirmed,
            mf.home_rest_days,
            mf.away_rest_days,
            mf.home_key_attackers_out,
            mf.away_key_attackers_out,
            mf.home_keeper_out,
            0 AS heavy_rain,
            0 AS high_wind,
            0 AS snow

        FROM fixtures f

        -- DC predictions
        LEFT JOIN predictions p_dc_over ON (
            p_dc_over.fixture_id = f.id
            AND p_dc_over.model    = 'dixon_coles'
            AND p_dc_over.market   = 'ou25'
            AND p_dc_over.selection = 'over'
        )
        LEFT JOIN predictions p_dc_under ON (
            p_dc_under.fixture_id = f.id
            AND p_dc_under.model    = 'dixon_coles'
            AND p_dc_under.market   = 'ou25'
            AND p_dc_under.selection = 'under'
        )

        -- XGB predictions
        LEFT JOIN predictions p_xgb_over ON (
            p_xgb_over.fixture_id = f.id
            AND p_xgb_over.model    = 'xgboost'
            AND p_xgb_over.market   = 'ou25'
            AND p_xgb_over.selection = 'over'
        )
        LEFT JOIN predictions p_xgb_under ON (
            p_xgb_under.fixture_id = f.id
            AND p_xgb_under.model    = 'xgboost'
            AND p_xgb_under.market   = 'ou25'
            AND p_xgb_under.selection = 'under'
        )

        -- Features
        LEFT JOIN match_features mf ON mf.fixture_id = f.id

        WHERE f.league      = ?
          AND f.status      = 'scheduled'
          AND f.kickoff_utc >= ?
          AND f.kickoff_utc <= ?
          AND p_dc_over.prob IS NOT NULL

        ORDER BY f.kickoff_utc ASC
    """

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            query,
            [league, now.isoformat(), cutoff.isoformat()],
        ).fetchall()

    fixtures = []
    for row in rows:
        fixtures.append(dict(row))

    logger.info("[%s] %d fixtures próximos con predicciones", league, len(fixtures))
    return fixtures


def load_odds(db_path: Path, fixture_id: int) -> dict:
    """Carga las odds más recientes de Betplay para un fixture."""
    result = {"odds_over": None, "odds_under": None, "overround_pct": None}

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        over_row = conn.execute("""
            SELECT odds_decimal, overround_pct FROM odds
            WHERE fixture_id = ? AND bookmaker = 'betplay' AND market = 'ou25' AND selection = 'over'
            ORDER BY captured_at DESC LIMIT 1
        """, [fixture_id]).fetchone()
        under_row = conn.execute("""
            SELECT odds_decimal, overround_pct FROM odds
            WHERE fixture_id = ? AND bookmaker = 'betplay' AND market = 'ou25' AND selection = 'under'
            ORDER BY captured_at DESC LIMIT 1
        """, [fixture_id]).fetchone()

    if over_row:
        result["odds_over"] = over_row["odds_decimal"]
        if over_row["overround_pct"] is not None:
            result["overround_pct"] = over_row["overround_pct"] / 100  # viene en %
    if under_row:
        result["odds_under"] = under_row["odds_decimal"]
        if under_row["overround_pct"] is not None and result["overround_pct"] is None:
            result["overround_pct"] = under_row["overround_pct"] / 100

    # Si no hay odds de Betplay, usar las históricas de bet365 como referencia
    if result["odds_over"] is None:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            over_hist = conn.execute("""
                SELECT odds_decimal, overround_pct, bookmaker FROM odds
                WHERE fixture_id = ? AND market = 'ou25' AND selection = 'over'
                ORDER BY captured_at DESC LIMIT 1
            """, [fixture_id]).fetchone()
            under_hist = conn.execute("""
                SELECT odds_decimal, overround_pct, bookmaker FROM odds
                WHERE fixture_id = ? AND market = 'ou25' AND selection = 'under'
                ORDER BY captured_at DESC LIMIT 1
            """, [fixture_id]).fetchone()
        if over_hist:
            result["odds_over"]    = over_hist["odds_decimal"]
            result["odds_source"]  = over_hist["bookmaker"]  # "bet365_historical"
            logger.warning(
                "Fixture %d: sin odds Betplay en vivo — usando %s (%.3f). "
                "Cuota puede estar desactualizada.",
                fixture_id, over_hist["bookmaker"], over_hist["odds_decimal"],
            )
            if over_hist["overround_pct"] is not None and result["overround_pct"] is None:
                result["overround_pct"] = over_hist["overround_pct"] / 100
        if under_hist and result["odds_under"] is None:
            result["odds_under"] = under_hist["odds_decimal"]
            if under_hist["overround_pct"] is not None and result["overround_pct"] is None:
                result["overround_pct"] = under_hist["overround_pct"] / 100

    return result


def load_bankroll(db_path: Path) -> float:
    """
    Lee el bankroll actual:
    1. Desde el último snapshot en la DB (performance_snapshots).
    2. Si no hay snapshots, desde config/bankroll.yaml.
    3. Fallback final: 500_000 COP (valor real del proyecto).
    """
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("""
            SELECT bankroll FROM performance_snapshots
            ORDER BY snapshot_date DESC LIMIT 1
        """).fetchone()
    if row:
        return float(row[0])

    # Sin snapshots — leer bankroll.yaml
    try:
        import yaml
        yaml_path = Path("config/bankroll.yaml")
        if yaml_path.exists():
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f)
            return float(cfg.get("current_bankroll_cop", 500_000))
    except Exception as exc:
        logger.warning("No se pudo leer config/bankroll.yaml: %s", exc)

    return 500_000.0


def load_current_exposure(db_path: Path) -> float:
    """Suma de stakes de apuestas abiertas (GO pero sin resultado aún)."""
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("""
            SELECT COALESCE(SUM(b.stake_placed_cop), 0)
            FROM bets b
            JOIN fixtures f ON b.fixture_id = f.id
            WHERE b.user_decision = 'GO'
              AND f.status != 'finished'
        """).fetchone()
    return float(row[0]) if row else 0.0


def load_n_bets_placed(db_path: Path) -> int:
    """Total de apuestas reales colocadas (GO)."""
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("""
            SELECT COUNT(*) FROM bets WHERE user_decision = 'GO'
        """).fetchone()
    return int(row[0]) if row else 0


# ──────────────────────────────────────────────────────────────────────────────
# Motor principal
# ──────────────────────────────────────────────────────────────────────────────

class ValueEngine:

    def __init__(
        self,
        db_path: Path = DB_PATH,
        dc_weight: float = DC_WEIGHT,
        thresholds: Optional[dict] = None,
    ):
        self.db_path    = db_path
        self.dc_weight  = dc_weight
        self.xgb_weight = 1.0 - dc_weight
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.filter_engine = FilterEngine()

    def run(
        self,
        league: str,
        hours_ahead: float = 24.0,
    ) -> list[BettingSignal]:
        """
        Corre el pipeline completo para una liga.

        Returns:
            Lista de BettingSignal (BET y REVIEW incluidos, SKIP excluidos).
        """
        # Estado del bankroll
        bankroll         = load_bankroll(self.db_path)
        current_exposure = load_current_exposure(self.db_path)
        n_bets_placed    = load_n_bets_placed(self.db_path)

        logger.info(
            "[%s] Bankroll: %.0f COP | Exposición: %.0f COP | Apuestas: %d",
            league, bankroll, current_exposure, n_bets_placed,
        )

        # Cargar fixtures con predicciones
        fixtures = load_upcoming_fixtures(self.db_path, league, hours_ahead)
        if not fixtures:
            logger.info("[%s] Sin fixtures próximos.", league)
            return []

        signals = []
        for fixture in fixtures:
            signal = self._evaluate(
                fixture          = fixture,
                bankroll         = bankroll,
                current_exposure = current_exposure,
                n_bets_placed    = n_bets_placed,
            )
            if signal and signal.status != "SKIP":
                signals.append(signal)
                # Actualizar exposición estimada si es BET
                if signal.status == "BET":
                    current_exposure += signal.stake_cop

        # Guardar en DB
        self._save_signals(signals)

        logger.info(
            "[%s] %d señales generadas (%d BET, %d REVIEW)",
            league,
            len(signals),
            sum(1 for s in signals if s.status == "BET"),
            sum(1 for s in signals if s.status == "REVIEW"),
        )
        return signals

    def _evaluate(
        self,
        fixture: dict,
        bankroll: float,
        current_exposure: float,
        n_bets_placed: int,
    ) -> Optional[BettingSignal]:
        """Evalúa un fixture y genera una señal."""
        fid = fixture["id"]

        # Probabilidades DC
        prob_dc_over  = fixture.get("prob_dc_over")
        prob_dc_under = fixture.get("prob_dc_under")
        if prob_dc_over is None:
            logger.debug("Fixture %d sin predicción DC. Saltando.", fid)
            return None

        # Probabilidades XGB (opcionales)
        prob_xgb_over  = fixture.get("prob_xgb_over")
        prob_xgb_under = fixture.get("prob_xgb_under")

        # Ensemble
        if prob_xgb_over is not None:
            prob_over  = self.dc_weight * prob_dc_over  + self.xgb_weight * prob_xgb_over
            prob_under = self.dc_weight * prob_dc_under + self.xgb_weight * prob_xgb_under
        else:
            prob_over, prob_under = prob_dc_over, prob_dc_under
            prob_xgb_over = prob_dc_over  # fallback

        # Odds
        odds = load_odds(self.db_path, fid)
        odds_over    = odds.get("odds_over")
        odds_under   = odds.get("odds_under")
        overround    = odds.get("overround_pct") or 0.0
        odds_source  = odds.get("odds_source", "betplay")  # "betplay" | "bet365_historical"

        if odds_over is None or odds_under is None:
            logger.debug("Fixture %d sin odds disponibles. Saltando.", fid)
            return None

        # Calcular edges
        implied_over  = 1 / odds_over
        implied_under = 1 / odds_under
        edge_over  = prob_over  - implied_over
        edge_under = prob_under - implied_under

        # Seleccionar la mejor dirección
        if edge_over >= edge_under:
            selection  = "over"
            prob_ens   = prob_over
            prob_xgb_s = prob_xgb_over
            odds_sel   = odds_over
            edge       = edge_over
        else:
            selection  = "under"
            prob_ens   = prob_under
            prob_xgb_s = 1 - prob_xgb_over
            odds_sel   = odds_under
            edge       = edge_under

        # Acuerdo entre modelos
        dc_direction  = "over" if prob_dc_over > prob_dc_under else "under"
        xgb_direction = "over" if prob_xgb_over > (1 - prob_xgb_over) else "under"
        model_agreement = dc_direction == xgb_direction == selection

        # Horas hasta el kickoff
        try:
            kickoff = datetime.fromisoformat(fixture["kickoff_utc"])
            hours_before = (kickoff - datetime.utcnow()).total_seconds() / 3600
        except (TypeError, ValueError):
            hours_before = 24.0

        # Construir contexto para filtros
        ctx = FixtureContext(
            fixture_id              = fid,
            league                  = fixture["league"],
            home_team               = fixture["home_team"],
            away_team               = fixture["away_team"],
            kickoff_utc             = kickoff if isinstance(kickoff, datetime) else datetime.utcnow(),
            season                  = fixture.get("season", ""),
            odds_over               = odds_over,
            odds_under              = odds_under,
            overround_pct           = overround,
            hours_before_kickoff    = max(hours_before, 0),
            prob_ensemble           = prob_ens,
            edge_pct                = edge,
            prob_dc                 = prob_dc_over if selection == "over" else prob_dc_under,
            prob_xgb                = prob_xgb_s,
            model_agreement         = model_agreement,
            matchweek               = fixture.get("matchweek"),
            lineup_confirmed        = bool(fixture.get("lineup_confirmed", 0)),
            key_attacker_out        = bool(
                (fixture.get("home_key_attackers_out") or 0) > 0 or
                (fixture.get("away_key_attackers_out") or 0) > 0
            ),
            keeper_out              = bool(
                (fixture.get("home_keeper_out") or 0) > 0
            ),
            heavy_rain              = bool(fixture.get("heavy_rain", 0)),
            high_wind               = bool(fixture.get("high_wind", 0)),
            snow                    = bool(fixture.get("snow", 0)),
            current_exposure_pct    = current_exposure / bankroll if bankroll > 0 else 0,
            n_bets_placed           = n_bets_placed,
        )

        # Correr filtros
        filter_result = self.filter_engine.check(ctx)

        # Si SKIP, no generar señal
        if filter_result.status == FilterStatus.SKIP:
            logger.debug(
                "Fixture %d SKIP: %s",
                fid, [f.code for f in filter_result.flags],
            )
            return None

        # Recomendar perfil
        profile = recommend_profile(
            edge            = edge,
            lineup_confirmed = ctx.lineup_confirmed,
            model_agreement  = model_agreement,
            n_bets_placed    = n_bets_placed,
        )

        # Calcular Kelly
        kelly_result = compute_stake(
            prob             = prob_ens,
            odds             = odds_sel,
            bankroll         = bankroll,
            profile          = profile,
            current_exposure = current_exposure,
        )

        # Si Kelly inválido (edge insuficiente para el perfil), SKIP
        if not kelly_result.is_valid:
            logger.debug(
                "Fixture %d Kelly inválido: %s",
                fid, kelly_result.rejection_reason,
            )
            return None

        return BettingSignal(
            fixture_id      = fid,
            league          = fixture["league"],
            home_team       = fixture["home_team"],
            away_team       = fixture["away_team"],
            kickoff_utc     = fixture["kickoff_utc"],
            market          = "ou25",
            selection       = selection,
            prob_dc         = round(prob_dc_over if selection == "over" else prob_dc_under, 4),
            prob_xgb        = round(prob_xgb_s, 4),
            prob_ensemble   = round(prob_ens, 4),
            implied_prob    = round(1 / odds_sel, 4),
            betplay_odds    = odds_sel,
            overround_pct   = round(overround, 4),
            edge_pct        = round(edge, 4),
            stake_cop       = kelly_result.stake_final,
            kelly_fraction  = kelly_result.kelly_fraction,
            profile         = profile,
            status          = filter_result.status.value,
            flags           = filter_result.flag_codes,
            flags_detail    = [
                {"code": f.code, "message": f.message, "severity": f.severity}
                for f in filter_result.flags
            ] + (
                [{"code": "ODDS_HISTORICAS", "message": "⚠️ Odds de bet365 histórico — sin cuota Betplay en vivo", "severity": "warning"}]
                if odds_source != "betplay" else []
            ),
            generated_at    = datetime.utcnow().isoformat(),
            bankroll_snapshot = bankroll,
        )

    def _save_signals(self, signals: list[BettingSignal]) -> None:
        """
        Guarda las señales en la tabla signals de la DB.

        Lógica de upsert segura:
        - Si ya existe una señal enviada/descartada (telegram_sent > 0) para el mismo
          fixture+market+selection → no tocar (preservar historial).
        - Si existe una señal pendiente (telegram_sent = 0) → reemplazar con datos frescos.
        - Si no existe → insertar.

        Esto evita duplicados cuando el value engine corre múltiples veces
        (daily_pipeline + prematch_scan) sobre los mismos fixtures.
        """
        if not signals:
            return

        now = datetime.utcnow().isoformat()
        inserted = 0
        updated  = 0
        skipped  = 0

        with sqlite3.connect(self.db_path) as conn:
            for s in signals:
                flags_json = json.dumps(s.flags_detail, ensure_ascii=False)

                existing = conn.execute("""
                    SELECT id, telegram_sent FROM signals
                    WHERE fixture_id = ? AND market = ? AND selection = ?
                    ORDER BY created_at DESC LIMIT 1
                """, [s.fixture_id, s.market, s.selection]).fetchone()

                if existing:
                    sig_id, sent = existing
                    if sent != 0:
                        # Aprobada (-1), enviada (1), o descartada (2) — no sobreescribir
                        skipped += 1
                        continue
                    # Pendiente (telegram_sent=0) — actualizar con datos frescos
                    conn.execute("""
                        UPDATE signals SET
                            edge_pct=?, prob_ensemble=?, betplay_odds=?,
                            overround_pct=?, stake_cop=?, kelly_fraction=?,
                            profile=?, status=?, flags=?, created_at=?
                        WHERE id=?
                    """, [
                        s.edge_pct, s.prob_ensemble, s.betplay_odds,
                        s.overround_pct, s.stake_cop, s.kelly_fraction,
                        s.profile, s.status, flags_json, now, sig_id,
                    ])
                    updated += 1
                else:
                    conn.execute("""
                        INSERT INTO signals
                            (fixture_id, market, selection, edge_pct,
                             prob_ensemble, betplay_odds, overround_pct,
                             stake_cop, kelly_fraction, profile, status,
                             flags, telegram_sent, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                    """, (
                        s.fixture_id, s.market, s.selection,
                        s.edge_pct, s.prob_ensemble, s.betplay_odds,
                        s.overround_pct, s.stake_cop, s.kelly_fraction,
                        s.profile, s.status, flags_json, now,
                    ))
                    inserted += 1

        logger.info(
            "%d señales → %d nuevas, %d actualizadas, %d ya enviadas (sin cambios)",
            len(signals), inserted, updated, skipped,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Formato de señal para Telegram (preview)
# ──────────────────────────────────────────────────────────────────────────────

def format_signal(signal: BettingSignal, bankroll: float) -> str:
    """
    Formatea una señal como mensaje de Telegram.
    Replica el formato del spec (§6 del doc de arquitectura).
    """
    status_icon = "✅" if signal.status == "BET" else "⚠️"
    liga_names  = {"BL2": "BUNDESLIGA 2", "NL1": "EREDIVISIE",
                   "EFL_L1": "LEAGUE ONE", "NOR1": "ELITESERIEN"}
    liga_label  = liga_names.get(signal.league, signal.league)
    sel_label   = "Over 2.5 goles" if signal.selection == "over" else "Under 2.5 goles"

    try:
        kickoff = datetime.fromisoformat(signal.kickoff_utc)
        kickoff_str = kickoff.strftime("%A %d %b · %H:%M UTC")
    except (TypeError, ValueError):
        kickoff_str = signal.kickoff_utc

    exposure_pct = (signal.stake_cop / signal.bankroll_snapshot * 100) if signal.bankroll_snapshot > 0 else 0

    lines = [
        f"{status_icon} VALUE BET — {liga_label}",
        f"🏟 {signal.home_team} vs {signal.away_team}",
        f"📅 {kickoff_str}",
        f"⚽ Mercado: {sel_label}",
        "━" * 25,
        "📊 MODELO",
        f"Prob. ensemble: {signal.prob_ensemble*100:.1f}% (DC: {signal.prob_dc*100:.1f}% · XGB: {signal.prob_xgb*100:.1f}%)",
        f"Cuota: {signal.betplay_odds:.2f} (implícita: {signal.implied_prob*100:.1f}%)",
        f"Edge estimado: +{signal.edge_pct*100:.1f}%",
        f"Perfil: {signal.profile.upper()}",
        "━" * 25,
        "💰 STAKE RECOMENDADO",
        f"Kelly {signal.kelly_fraction:.3f}: {signal.stake_cop:,.0f} COP",
        f"Bankroll actual: {signal.bankroll_snapshot:,.0f} COP",
        f"Exposure: {exposure_pct:.1f}%",
        "━" * 25,
        "🔍 SEÑALES",
    ]

    # Flags
    if not signal.flags:
        lines.append("✅ Sin flags — todos los checks OK")
    else:
        for flag in signal.flags_detail:
            icon = "🟡" if flag["severity"] == "soft" else "🔴"
            lines.append(f"{icon} {flag['message']}")

    lines += [
        "━" * 25,
        "⏱️ Responde:",
        "GO → apostar  |  SKIP → descartar  |  HALF → apostar mitad",
    ]

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Value Engine — detección de apuestas con value")
    parser.add_argument("--league",  choices=["BL2", "NL1"], help="Liga específica")
    parser.add_argument("--all",     action="store_true")
    parser.add_argument("--hours",   type=float, default=24.0)
    parser.add_argument("--db",      default=str(DB_PATH))
    parser.add_argument("--preview", action="store_true", help="Mostrar preview del mensaje Telegram")
    args = parser.parse_args()

    leagues = ["BL2", "NL1"] if args.all else ([args.league] if args.league else None)
    if not leagues:
        parser.error("Especificá --league o --all")

    engine = ValueEngine(db_path=Path(args.db))
    bankroll = load_bankroll(Path(args.db))

    all_signals = []
    for league in leagues:
        signals = engine.run(league=league, hours_ahead=args.hours)
        all_signals.extend(signals)

    if not all_signals:
        print("\n  Sin señales para las próximas 24h.\n")
    else:
        print(f"\n  {len(all_signals)} señales generadas:\n")
        for signal in all_signals:
            if args.preview:
                print(format_signal(signal, bankroll))
                print()
            else:
                status_icon = "✅" if signal.status == "BET" else "⚠️"
                print(
                    f"  {status_icon} [{signal.league}] "
                    f"{signal.home_team} vs {signal.away_team} | "
                    f"{signal.selection.upper()} @ {signal.betplay_odds:.2f} | "
                    f"Edge: +{signal.edge_pct*100:.1f}% | "
                    f"Stake: {signal.stake_cop:,.0f} COP | "
                    f"Status: {signal.status} | "
                    f"Flags: {signal.flags or '—'}"
                )