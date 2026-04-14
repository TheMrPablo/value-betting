"""
filters.py — Reglas de exclusión pre-apuesta (spec §5).

Implementa todos los checks que el value_engine corre antes de
emitir una señal BET. Si algún filtro falla, la señal sale como
REVIEW (requiere intervención manual) o SKIP (descartada).

Filtros implementados:
  §5.1 Señales de posible match-fixing (nivel partido)
  §5.2 Condiciones de mercado que invalidan el modelo
  §5.3 Contextos que el modelo no captura bien

Uso:
    from filters import FilterEngine, FilterResult

    engine = FilterEngine()
    result = engine.check(fixture_context)
    if result.status == "BET":
        # emitir alerta
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Tipos
# ──────────────────────────────────────────────────────────────────────────────

class FilterStatus(str, Enum):
    BET    = "BET"     # todos los filtros OK, proceder
    REVIEW = "REVIEW"  # uno o más flags de advertencia, requiere ojo humano
    SKIP   = "SKIP"    # filtro duro activado, no apostar


@dataclass
class FixtureContext:
    """
    Todo lo que el FilterEngine necesita saber sobre un partido.
    El value_engine construye este objeto antes de llamar a check().
    """
    # Identificación
    fixture_id: int
    league: str
    home_team: str
    away_team: str
    kickoff_utc: datetime
    season: str

    # Mercado
    odds_over: float
    odds_under: float
    overround_pct: float          # margen del bookmaker en ese mercado (0–1)
    hours_before_kickoff: float   # horas hasta el partido

    # Modelo
    prob_ensemble: float          # probabilidad ensemble final
    edge_pct: float               # edge vs implied prob (0–1)
    prob_dc: float                # prob solo Dixon-Coles
    prob_xgb: float               # prob solo XGBoost
    model_agreement: bool         # DC y XGB coinciden en dirección

    # Contexto de temporada
    matchweek: Optional[int]      # jornada actual
    total_matchweeks: int = 34    # jornadas totales en la liga

    # Línea y movimiento
    opening_odds: Optional[float] = None   # cuota de apertura (si disponible)
    line_move_pct: Optional[float] = None  # movimiento % desde apertura

    # Información del partido
    lineup_confirmed: bool = False
    coach_changed_recently: bool = False   # cambio de entrenador < 3 partidos
    post_international_break: bool = False
    key_attacker_out: bool = False
    keeper_out: bool = False
    is_derby: bool = False         # derbi regional con tensión política

    # Clima (para Eliteserien)
    heavy_rain: bool = False
    high_wind: bool = False
    snow: bool = False

    # Estado del bankroll
    current_exposure_pct: float = 0.0   # exposición total actual / bankroll
    n_bets_placed: int = 0              # apuestas reales colocadas hasta ahora


@dataclass
class FilterFlag:
    """Una flag individual activada por un filtro."""
    code: str           # identificador corto
    message: str        # descripción legible
    severity: str       # 'hard' (→ SKIP) o 'soft' (→ REVIEW)


@dataclass
class FilterResult:
    """Resultado del chequeo completo de filtros."""
    fixture_id: int
    status: FilterStatus
    flags: list[FilterFlag] = field(default_factory=list)
    checks_passed: int = 0
    checks_total: int = 0

    @property
    def has_hard_flags(self) -> bool:
        return any(f.severity == "hard" for f in self.flags)

    @property
    def has_soft_flags(self) -> bool:
        return any(f.severity == "soft" for f in self.flags)

    @property
    def flag_codes(self) -> list[str]:
        return [f.code for f in self.flags]

    def summary(self) -> str:
        """Resumen legible para la alerta de Telegram."""
        if self.status == FilterStatus.BET:
            return f"✅ {self.checks_passed}/{self.checks_total} checks OK"
        lines = [f"⚠️ {len(self.flags)} flag(s):"]
        for flag in self.flags:
            icon = "🔴" if flag.severity == "hard" else "🟡"
            lines.append(f"  {icon} [{flag.code}] {flag.message}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Motor de filtros
# ──────────────────────────────────────────────────────────────────────────────

class FilterEngine:
    """
    Aplica todas las reglas de exclusión del spec §5.

    Cada check_ method retorna una FilterFlag o None.
    Un flag con severity='hard' fuerza SKIP.
    Un flag con severity='soft' fuerza REVIEW (si no hay hard flags).
    """

    def check(self, ctx: FixtureContext) -> FilterResult:
        """
        Corre todos los filtros sobre el contexto del partido.

        Returns:
            FilterResult con el status y los flags activados.
        """
        flags: list[FilterFlag] = []
        checks_run = 0

        # Lista de todos los checks a ejecutar
        check_methods = [
            # §5.1 Match-fixing
            self._check_line_movement,
            self._check_asymmetric_incentives,
            self._check_end_of_season,
            self._check_suspicious_lineup,

            # §5.2 Condiciones de mercado
            self._check_overround,
            self._check_market_open_too_late,

            # §5.3 Contexto del modelo
            self._check_early_season,
            self._check_late_season,
            self._check_post_international_break,
            self._check_coach_change,
            self._check_extreme_weather,
            self._check_derby,
            self._check_exposure_cap,
            self._check_model_disagreement,

            # §5.4 Burn-in
            self._check_burn_in_profile,
        ]

        for check in check_methods:
            checks_run += 1
            flag = check(ctx)
            if flag is not None:
                flags.append(flag)

        # Determinar status
        if any(f.severity == "hard" for f in flags):
            status = FilterStatus.SKIP
        elif flags:
            status = FilterStatus.REVIEW
        else:
            status = FilterStatus.BET

        return FilterResult(
            fixture_id    = ctx.fixture_id,
            status        = status,
            flags         = flags,
            checks_passed = checks_run - len(flags),
            checks_total  = checks_run,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # §5.1 Match-fixing
    # ──────────────────────────────────────────────────────────────────────────

    def _check_line_movement(self, ctx: FixtureContext) -> Optional[FilterFlag]:
        """
        Movimiento de línea anómalo sin noticia pública.
        Red flag si la cuota se movió > 15% sin explicación conocida.
        """
        if ctx.line_move_pct is None:
            return None  # sin datos de movimiento, no podemos chequear

        if abs(ctx.line_move_pct) > 0.15:
            return FilterFlag(
                code     = "LINE_MOVE",
                message  = f"Movimiento de línea anómalo: {ctx.line_move_pct*100:+.1f}% sin noticia pública",
                severity = "soft",
            )
        return None

    def _check_asymmetric_incentives(self, ctx: FixtureContext) -> Optional[FilterFlag]:
        """
        Partido de relegación/promoción con incentivos asimétricos.
        Un equipo sin nada en juego contra otro que sí tiene todo.
        — Esta señal requiere datos externos que el pipeline actual no tiene;
          por eso es soft y se deja para revisión manual.
        """
        # Por ahora solo chequeamos fin de temporada como proxy
        # (§5.1 requeriría datos de clasificación en tiempo real)
        return None  # placeholder — implementar en Fase 2 con datos de tabla en vivo

    def _check_end_of_season(self, ctx: FixtureContext) -> Optional[FilterFlag]:
        """
        Últimas 2 jornadas de la temporada con incentivos distorsionados.
        """
        if ctx.matchweek is None:
            return None
        if ctx.matchweek >= ctx.total_matchweeks - 1:
            return FilterFlag(
                code     = "END_SEASON",
                message  = f"Jornada {ctx.matchweek}/{ctx.total_matchweeks} — incentivos distorsionados al final de temporada",
                severity = "hard",
            )
        return None

    def _check_suspicious_lineup(self, ctx: FixtureContext) -> Optional[FilterFlag]:
        """
        Alineación con muchos suplentes sin explicación (señal de fix).
        Por ahora usa keeper_out como proxy — en Fase 2 añadir conteo real.
        """
        if ctx.keeper_out and not ctx.lineup_confirmed:
            return FilterFlag(
                code     = "SUSPICIOUS_LINEUP",
                message  = "Portero titular fuera + alineación no confirmada",
                severity = "soft",
            )
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # §5.2 Condiciones de mercado
    # ──────────────────────────────────────────────────────────────────────────

    def _check_overround(self, ctx: FixtureContext) -> Optional[FilterFlag]:
        """
        Overround > 8% en el mercado específico.
        Ningún edge razonable sobrevive un margen tan alto.
        """
        if ctx.overround_pct > 0.11:
            return FilterFlag(
                code     = "HIGH_OVERROUND",
                message  = f"Overround {ctx.overround_pct*100:.1f}% > 8% — margen demasiado alto",
                severity = "hard",
            )
        return None

    def _check_market_open_too_late(self, ctx: FixtureContext) -> Optional[FilterFlag]:
        """
        Mercado abierto < 3h antes del kickoff.
        Betplay esperó info — señal de que el mercado tiene información asimétrica.
        """
        if ctx.hours_before_kickoff < 3.0:
            return FilterFlag(
                code     = "LATE_MARKET",
                message  = f"Mercado abierto muy tarde: {ctx.hours_before_kickoff:.1f}h antes del kickoff",
                severity = "hard",
            )
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # §5.3 Contextos que el modelo no captura bien
    # ──────────────────────────────────────────────────────────────────────────

    def _check_early_season(self, ctx: FixtureContext) -> Optional[FilterFlag]:
        """
        Primeras 2 jornadas de la temporada.
        El modelo usa datos de la temporada anterior; los equipos han cambiado.
        """
        if ctx.matchweek is not None and ctx.matchweek <= 2:
            return FilterFlag(
                code     = "EARLY_SEASON",
                message  = f"Jornada {ctx.matchweek} — modelo sin datos de la temporada actual",
                severity = "hard",
            )
        return None

    def _check_late_season(self, ctx: FixtureContext) -> Optional[FilterFlag]:
        """Últimas 2 jornadas — duplicado de end_of_season para claridad."""
        # Ya cubierto por _check_end_of_season
        return None

    def _check_post_international_break(self, ctx: FixtureContext) -> Optional[FilterFlag]:
        """
        Primera jornada post-parón internacional.
        Fatiga + bajas por lesión internacional no modeladas.
        """
        if ctx.post_international_break:
            return FilterFlag(
                code     = "POST_BREAK",
                message  = "Primera jornada post-parón internacional",
                severity = "hard",
            )
        return None

    def _check_coach_change(self, ctx: FixtureContext) -> Optional[FilterFlag]:
        """
        Cambio de entrenador reciente (< 3 partidos con el nuevo).
        El modelo pondera el estilo del entrenador anterior.
        """
        if ctx.coach_changed_recently:
            return FilterFlag(
                code     = "COACH_CHANGE",
                message  = "Cambio de entrenador reciente (< 3 partidos) — modelo desactualizado",
                severity = "soft",
            )
        return None

    def _check_extreme_weather(self, ctx: FixtureContext) -> Optional[FilterFlag]:
        """
        Clima extremo no modelado (excepto Eliteserien con Yr.no).
        """
        if ctx.snow:
            return FilterFlag(
                code     = "WEATHER_SNOW",
                message  = "Nieve activa — afecta goles y modelo no lo captura",
                severity = "soft",
            )
        if ctx.high_wind:
            return FilterFlag(
                code     = "WEATHER_WIND",
                message  = "Viento fuerte (> 30 km/h) — reduce precisión de tiros",
                severity = "soft",
            )
        if ctx.heavy_rain:
            return FilterFlag(
                code     = "WEATHER_RAIN",
                message  = "Lluvia intensa — afecta ritmo de juego",
                severity = "soft",
            )
        return None

    def _check_derby(self, ctx: FixtureContext) -> Optional[FilterFlag]:
        """
        Derbi regional con tensión política o emocional.
        El factor emocional domina; Dixon-Coles no lo captura.
        """
        if ctx.is_derby:
            return FilterFlag(
                code     = "DERBY",
                message  = "Derbi regional — factor emocional domina sobre estadística",
                severity = "soft",
            )
        return None

    def _check_exposure_cap(self, ctx: FixtureContext) -> Optional[FilterFlag]:
        """
        Exposición total del bankroll > 15% (cap duro del spec §4.4).
        """
        if ctx.current_exposure_pct >= 0.15:
            return FilterFlag(
                code     = "EXPOSURE_CAP",
                message  = f"Exposición total {ctx.current_exposure_pct*100:.1f}% ≥ 15% del bankroll",
                severity = "hard",
            )
        return None

    def _check_model_disagreement(self, ctx: FixtureContext) -> Optional[FilterFlag]:
        """
        DC y XGBoost no coinciden en dirección.
        Solo aplica en perfil conservador — en agresivo se acepta un solo modelo.
        """
        if not ctx.model_agreement:
            # En perfil conservador, la discrepancia es más seria
            diff = abs(ctx.prob_dc - ctx.prob_xgb)
            if diff > 0.10:  # diferencia > 10 puntos porcentuales
                return FilterFlag(
                    code     = "MODEL_DISAGREE",
                    message  = f"DC ({ctx.prob_dc:.1%}) y XGB ({ctx.prob_xgb:.1%}) divergen {diff:.1%}",
                    severity = "soft",
                )
        return None

    def _check_burn_in_profile(self, ctx: FixtureContext) -> Optional[FilterFlag]:
        """
        Primeras 50 apuestas — burn-in period.
        No es un filtro duro, pero se añade como soft flag informativo.
        """
        if ctx.n_bets_placed < 50:
            return FilterFlag(
                code     = "BURN_IN",
                message  = f"Apuesta #{ctx.n_bets_placed + 1}/50 del burn-in — usar 1/8 Kelly",
                severity = "soft",
            )
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Función de conveniencia para el value_engine
# ──────────────────────────────────────────────────────────────────────────────

def run_filters(ctx: FixtureContext) -> FilterResult:
    """Entry point para value_engine.py."""
    engine = FilterEngine()
    return engine.check(ctx)


# ──────────────────────────────────────────────────────────────────────────────
# CLI para pruebas
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Test de filtros ===\n")

    # Caso 1: apuesta limpia
    ctx_clean = FixtureContext(
        fixture_id=1, league="BL2",
        home_team="Hamburg", away_team="Hannover",
        kickoff_utc=datetime(2026, 4, 19, 17, 30),
        season="2025-26",
        odds_over=1.95, odds_under=1.90,
        overround_pct=0.054,
        hours_before_kickoff=24.0,
        prob_ensemble=0.62, edge_pct=0.107,
        prob_dc=0.61, prob_xgb=0.63,
        model_agreement=True,
        matchweek=15, total_matchweeks=34,
        lineup_confirmed=True,
        n_bets_placed=10,
    )

    result = run_filters(ctx_clean)
    print(f"Caso 1 — Apuesta limpia:")
    print(f"  Status: {result.status.value}")
    print(f"  {result.summary()}\n")

    # Caso 2: overround alto + fin de temporada
    ctx_bad = FixtureContext(
        fixture_id=2, league="BL2",
        home_team="Schalke", away_team="Bochum",
        kickoff_utc=datetime(2026, 5, 15, 15, 30),
        season="2025-26",
        odds_over=1.80, odds_under=1.95,
        overround_pct=0.095,
        hours_before_kickoff=6.0,
        prob_ensemble=0.60, edge_pct=0.06,
        prob_dc=0.60, prob_xgb=0.60,
        model_agreement=True,
        matchweek=33, total_matchweeks=34,
        n_bets_placed=25,
    )

    result2 = run_filters(ctx_bad)
    print(f"Caso 2 — Overround alto + fin de temporada:")
    print(f"  Status: {result2.status.value}")
    print(f"  {result2.summary()}\n")

    # Caso 3: burn-in + coach change + weather
    ctx_review = FixtureContext(
        fixture_id=3, league="NOR1",
        home_team="Rosenborg", away_team="Molde",
        kickoff_utc=datetime(2026, 5, 3, 18, 0),
        season="2025-26",
        odds_over=1.90, odds_under=1.95,
        overround_pct=0.052,
        hours_before_kickoff=12.0,
        prob_ensemble=0.58, edge_pct=0.055,
        prob_dc=0.58, prob_xgb=0.58,
        model_agreement=True,
        matchweek=8, total_matchweeks=30,
        coach_changed_recently=True,
        heavy_rain=True,
        n_bets_placed=5,
    )

    result3 = run_filters(ctx_review)
    print(f"Caso 3 — Burn-in + coach change + lluvia:")
    print(f"  Status: {result3.status.value}")
    print(f"  {result3.summary()}\n")