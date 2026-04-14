"""
kelly.py — Calculadora de Kelly fraccionario con caps de seguridad.

Implementa el criterio de Kelly para sizing de apuestas en value betting.
Usa fracciones conservadoras (1/8 y 1/4) para compensar la incertidumbre
sobre el edge real del modelo.

Fórmula de Kelly:
    f* = (b*p - q) / b
    donde:
        b = cuota decimal - 1  (ganancia neta por unidad apostada)
        p = probabilidad estimada de ganar
        q = 1 - p              (probabilidad de perder)

Uso:
    from kelly import KellyCalculator

    calc = KellyCalculator(bankroll=150_000, kelly_multiplier=0.25)
    stake = calc.stake(prob=0.62, odds=1.95, profile="conservador")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# ──────────────────────────────────────────────────────────────────────────────
# Configuración de perfiles
# ──────────────────────────────────────────────────────────────────────────────

# Fracciones de Kelly por perfil (del spec §4.2)
KELLY_FRACTIONS = {
    "conservador": 0.25,   # 1/4 Kelly
    "agresivo":    0.125,  # 1/8 Kelly — mayor incertidumbre
    "burn_in":     0.125,  # 1/8 Kelly — primeras 50 apuestas
}

# Edge mínimo requerido por perfil (del spec §4.2)
MIN_EDGE = {
    "conservador": 0.05,   # 5%
    "agresivo":    0.08,   # 8%
    "burn_in":     0.05,   # 5%
}

# Caps duros (del spec §4.4)
MAX_STAKE_PCT   = 0.05    # nunca más del 5% del bankroll en una apuesta
MAX_EXPOSURE_PCT = 0.15   # nunca más del 15% del bankroll expuesto simultáneamente
MIN_STAKE_ABS   = 1_000   # stake mínimo absoluto en COP (para evitar apuestas ridículas)


# ──────────────────────────────────────────────────────────────────────────────
# Resultado del cálculo
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class KellyResult:
    """Resultado completo del cálculo de Kelly."""
    bankroll: float
    prob: float
    odds: float
    edge: float
    profile: str

    kelly_full: float        # Kelly completo (f*)
    kelly_fraction: float    # fracción aplicada (0.25 o 0.125)
    stake_raw: float         # stake antes de caps
    stake_final: float       # stake después de caps (lo que se apuesta)
    stake_pct: float         # stake como % del bankroll

    cap_applied: str         # qué cap limitó el stake ('none', 'max_pct', 'min_abs')
    is_valid: bool           # False si el edge es negativo o insuficiente
    rejection_reason: str    # por qué se rechazó (si is_valid=False)

    def __str__(self) -> str:
        if not self.is_valid:
            return f"❌ Apuesta inválida: {self.rejection_reason}"
        cap_str = f" [cap: {self.cap_applied}]" if self.cap_applied != "none" else ""
        return (
            f"✅ Stake: {self.stake_final:,.0f} COP "
            f"({self.stake_pct*100:.1f}% bankroll){cap_str} | "
            f"Kelly: {self.kelly_full*100:.2f}% × {self.kelly_fraction} = "
            f"{self.stake_raw:,.0f} → {self.stake_final:,.0f} | "
            f"Edge: {self.edge*100:.1f}% | Odds: {self.odds:.2f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Calculadora principal
# ──────────────────────────────────────────────────────────────────────────────

class KellyCalculator:
    """
    Calculadora de Kelly fraccionario con caps de seguridad.

    Args:
        bankroll:          Bankroll actual en COP.
        profile:           Perfil de apuesta ('conservador', 'agresivo', 'burn_in').
        current_exposure:  Suma de stakes de apuestas abiertas en COP.
                           Se usa para verificar el cap de exposición total.
    """

    def __init__(
        self,
        bankroll: float,
        profile: Literal["conservador", "agresivo", "burn_in"] = "conservador",
        current_exposure: float = 0.0,
    ):
        if bankroll <= 0:
            raise ValueError(f"Bankroll debe ser positivo. Recibido: {bankroll}")
        self.bankroll         = bankroll
        self.profile          = profile
        self.current_exposure = current_exposure
        self.kelly_fraction   = KELLY_FRACTIONS[profile]
        self.min_edge         = MIN_EDGE[profile]

    def calculate(self, prob: float, odds: float) -> KellyResult:
        """
        Calcula el stake óptimo para una apuesta.

        Args:
            prob:  Probabilidad estimada de ganar (0–1).
            odds:  Cuota decimal del bookmaker (ej. 1.95).

        Returns:
            KellyResult con el stake recomendado y todos los detalles.
        """
        # Validaciones básicas
        if not 0 < prob < 1:
            return self._invalid(prob, odds, 0.0, f"Probabilidad inválida: {prob}")
        if odds <= 1.0:
            return self._invalid(prob, odds, 0.0, f"Cuota inválida (debe ser > 1.0): {odds}")

        b = odds - 1.0          # ganancia neta por unidad
        q = 1.0 - prob          # probabilidad de perder
        edge = prob - (1 / odds)  # edge vs implied prob

        # Verificar edge mínimo
        if edge < self.min_edge:
            return self._invalid(
                prob, odds, edge,
                f"Edge {edge*100:.1f}% < mínimo {self.min_edge*100:.0f}% para perfil '{self.profile}'"
            )

        # Kelly completo: f* = (b*p - q) / b
        kelly_full = (b * prob - q) / b

        if kelly_full <= 0:
            return self._invalid(
                prob, odds, edge,
                f"Kelly negativo ({kelly_full*100:.2f}%) — no hay value real"
            )

        # Aplicar fracción de Kelly
        stake_raw = self.bankroll * kelly_full * self.kelly_fraction

        # Aplicar caps
        stake_final, cap_applied = self._apply_caps(stake_raw)

        return KellyResult(
            bankroll       = self.bankroll,
            prob           = prob,
            odds           = odds,
            edge           = edge,
            profile        = self.profile,
            kelly_full     = kelly_full,
            kelly_fraction = self.kelly_fraction,
            stake_raw      = stake_raw,
            stake_final    = stake_final,
            stake_pct      = stake_final / self.bankroll,
            cap_applied    = cap_applied,
            is_valid       = True,
            rejection_reason = "",
        )

    def _apply_caps(self, stake_raw: float) -> tuple[float, str]:
        """
        Aplica caps de seguridad al stake calculado.

        Orden de precedencia:
        1. Cap máximo: 5% del bankroll (cap duro, siempre aplica)
        2. Cap de exposición: si supera el 15% total, reduce proportionally
        3. Mínimo absoluto: si queda < 1,000 COP, retorna 0

        Returns:
            (stake_final, nombre_del_cap_aplicado)
        """
        max_stake = self.bankroll * MAX_STAKE_PCT
        stake = min(stake_raw, max_stake)
        cap = "max_pct" if stake < stake_raw else "none"

        # Cap de exposición total
        available_exposure = self.bankroll * MAX_EXPOSURE_PCT - self.current_exposure
        if available_exposure <= 0:
            return 0.0, "exposure_limit"
        if stake > available_exposure:
            stake = available_exposure
            cap = "exposure_limit"

        # Mínimo absoluto
        if stake < MIN_STAKE_ABS:
            return 0.0, "below_minimum"

        # Redondear a múltiplos de 500 COP para practicidad
        stake = round(stake / 500) * 500
        if stake < MIN_STAKE_ABS:
            stake = MIN_STAKE_ABS

        return stake, cap

    def _invalid(self, prob: float, odds: float, edge: float, reason: str) -> KellyResult:
        """Construye un KellyResult inválido."""
        return KellyResult(
            bankroll=self.bankroll, prob=prob, odds=odds, edge=edge,
            profile=self.profile, kelly_full=0.0, kelly_fraction=self.kelly_fraction,
            stake_raw=0.0, stake_final=0.0, stake_pct=0.0,
            cap_applied="none", is_valid=False, rejection_reason=reason,
        )

    def stake(self, prob: float, odds: float) -> float:
        """
        Versión simplificada — devuelve solo el stake en COP.
        Retorna 0.0 si la apuesta no es válida.
        """
        result = self.calculate(prob, odds)
        return result.stake_final if result.is_valid else 0.0

    def exposure_remaining(self) -> float:
        """Cuánto bankroll queda disponible para nuevas apuestas."""
        return max(0.0, self.bankroll * MAX_EXPOSURE_PCT - self.current_exposure)

    def exposure_pct(self) -> float:
        """Exposición actual como % del bankroll."""
        return self.current_exposure / self.bankroll if self.bankroll > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Función de conveniencia para el value_engine
# ──────────────────────────────────────────────────────────────────────────────

def compute_stake(
    prob: float,
    odds: float,
    bankroll: float,
    profile: Literal["conservador", "agresivo", "burn_in"] = "conservador",
    current_exposure: float = 0.0,
) -> KellyResult:
    """
    Entry point para value_engine.py.

    Calcula el stake con el perfil dado y devuelve el KellyResult completo.
    El value_engine usa result.stake_final para la alerta de Telegram.
    """
    calc = KellyCalculator(
        bankroll=bankroll,
        profile=profile,
        current_exposure=current_exposure,
    )
    return calc.calculate(prob, odds)


def recommend_profile(
    edge: float,
    lineup_confirmed: bool,
    model_agreement: bool,
    n_bets_placed: int,
) -> Literal["conservador", "agresivo", "burn_in"]:
    """
    Recomienda el perfil de sizing basado en el contexto de la apuesta.

    Lógica del spec §4.2:
    - burn_in: primeras 50 apuestas (verificar calibración)
    - conservador: modelos coinciden + alineación confirmada + edge ≥ 5%
    - agresivo: solo un modelo señala o alineación no confirmada + edge ≥ 8%

    Args:
        edge:              Edge estimado (0–1).
        lineup_confirmed:  Si la alineación está confirmada.
        model_agreement:   Si DC y XGBoost coinciden en dirección.
        n_bets_placed:     Apuestas reales colocadas hasta ahora.

    Returns:
        Perfil recomendado.
    """
    if n_bets_placed < 50:
        return "burn_in"

    if model_agreement and lineup_confirmed and edge >= MIN_EDGE["conservador"]:
        return "conservador"

    if edge >= MIN_EDGE["agresivo"]:
        return "agresivo"

    # Si llegamos acá, el edge no alcanza ningún umbral
    return "agresivo"  # el value_engine lo rechazará igual por edge insuficiente


# ──────────────────────────────────────────────────────────────────────────────
# CLI para pruebas manuales
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculadora Kelly fraccionario")
    parser.add_argument("--prob",     type=float, required=True, help="Probabilidad estimada (0-1)")
    parser.add_argument("--odds",     type=float, required=True, help="Cuota decimal")
    parser.add_argument("--bankroll", type=float, default=150_000, help="Bankroll en COP")
    parser.add_argument("--profile",  choices=["conservador", "agresivo", "burn_in"],
                        default="conservador")
    parser.add_argument("--exposure", type=float, default=0.0,
                        help="Exposición actual en COP (apuestas abiertas)")
    args = parser.parse_args()

    result = compute_stake(
        prob=args.prob,
        odds=args.odds,
        bankroll=args.bankroll,
        profile=args.profile,
        current_exposure=args.exposure,
    )

    print(f"\n{'='*60}")
    print(f"  Kelly Calculator — Perfil: {result.profile.upper()}")
    print(f"{'='*60}")
    print(f"  Bankroll:        {result.bankroll:>15,.0f} COP")
    print(f"  Probabilidad:    {result.prob:>15.1%}")
    print(f"  Cuota:           {result.odds:>15.2f}")
    print(f"  Implied prob:    {1/result.odds:>15.1%}")
    print(f"  Edge estimado:   {result.edge:>15.1%}")
    print(f"{'-'*60}")
    if result.is_valid:
        print(f"  Kelly completo:  {result.kelly_full:>15.2%}")
        print(f"  Fracción Kelly:  {result.kelly_fraction:>15.3f}  (×{1/result.kelly_fraction:.0f})")
        print(f"  Stake raw:       {result.stake_raw:>15,.0f} COP")
        print(f"  Cap aplicado:    {result.cap_applied:>15}")
        print(f"  STAKE FINAL:     {result.stake_final:>15,.0f} COP  ({result.stake_pct:.1%} bankroll)")
    else:
        print(f"  ❌ RECHAZADA: {result.rejection_reason}")
    print(f"{'='*60}\n")

    # Tabla de sensibilidad: cómo cambia el stake con distintos edges
    if result.is_valid:
        print("  Sensibilidad del stake a distintas probabilidades:")
        print(f"  {'Prob':>6} {'Edge':>7} {'Kelly':>7} {'Stake':>12}")
        print(f"  {'-'*35}")
        calc = KellyCalculator(args.bankroll, args.profile, args.exposure)
        for p in [0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70]:
            r = calc.calculate(p, args.odds)
            if r.is_valid:
                print(f"  {p:>6.0%} {r.edge:>+7.1%} {r.kelly_full:>7.2%} {r.stake_final:>11,.0f} COP")
        print()