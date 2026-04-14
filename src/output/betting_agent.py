"""
betting_agent.py — Agente Claude para análisis de señales de apuestas.

Antes de enviar una señal al usuario, el agente:
1. Busca noticias recientes del partido (lesiones, alineaciones, clima)
2. Evalúa si el edge del modelo es creíble o es ruido
3. Decide si recomendar la apuesta o descartarla silenciosamente
4. Genera un análisis en lenguaje simple para incluir en el mensaje

Cada semana revisa las señales descartadas y aprende de los resultados.

Uso:
    from src.output.betting_agent import analyze_signal, weekly_review
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

DB_PATH    = Path("data/betting.db")
CONFIG_DIR = Path("config")

# ──────────────────────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────────────────────

def load_api_key() -> str:
    """Carga la API key de Anthropic desde config/agent.json."""
    import os
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    config_path = CONFIG_DIR / "agent.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)["api_key"]
    raise RuntimeError("No se encontró API key. Crear config/agent.json")


# ──────────────────────────────────────────────────────────────────────────────
# Cliente Anthropic
# ──────────────────────────────────────────────────────────────────────────────

def call_claude(
    system_prompt: str,
    user_message: str,
    api_key: str,
    max_tokens: int = 1000,
    retries: int = 2,
) -> str:
    """Llama a la API de Anthropic con retry en caso de rate limit."""
    import time
    last_error = None
    for attempt in range(retries):
        try:
            response = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key":         api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                },
                json={
                    "model":      "claude-sonnet-4-5",
                    "max_tokens": max_tokens,
                    "system":     system_prompt,
                    "messages":   [{"role": "user", "content": user_message}],
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            return data["content"][0]["text"].strip()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < retries - 1:
                wait = 30 * (attempt + 1)
                logger.warning("Rate limit (429). Esperando %ds antes de reintentar...", wait)
                time.sleep(wait)
                last_error = e
            else:
                raise
    raise last_error


# ──────────────────────────────────────────────────────────────────────────────
# Búsqueda de noticias
# ──────────────────────────────────────────────────────────────────────────────

def search_news(query: str, api_key: str) -> str:
    """
    Usa Claude con web search para buscar noticias recientes del partido.
    Retorna un resumen de lo encontrado.
    """
    system = (
        "Eres un asistente especializado en fútbol europeo. "
        "Buscás noticias recientes sobre partidos de Bundesliga 2 en alemán e inglés. "
        "Respondés SOLO con hechos concretos relevantes para apostar: "
        "lesiones, alineaciones probables, estado de forma, clima. "
        "Si no encontrás nada relevante, decís 'Sin noticias relevantes'. "
        "Máximo 3 frases. Siempre en español."
    )

    response = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        },
        json={
            "model":      "claude-sonnet-4-5",
            "max_tokens": 500,
            "system":     system,
            "tools": [{
                "type": "web_search_20250305",
                "name": "web_search",
            }],
            "messages": [{
                "role":    "user",
                "content": f"Buscá noticias de las últimas 48 horas sobre: {query}. "
                           f"Enfocate en lesiones, alineaciones y estado de forma."
            }],
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    # Extraer texto de la respuesta (puede haber bloques de tool_use)
    text_parts = [
        block["text"]
        for block in data.get("content", [])
        if block.get("type") == "text"
    ]
    return " ".join(text_parts).strip() or "Sin noticias relevantes."


# ──────────────────────────────────────────────────────────────────────────────
# Análisis principal
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
Eres un analista experto en value betting de fútbol europeo, especializado en Bundesliga 2.

Tu trabajo es evaluar señales de apuestas generadas por modelos estadísticos (Dixon-Coles + XGBoost) 
y decidir si son apuestas reales o ruido del modelo.

REGLAS PARA DECIDIR:

DESCARTAR si:
- El edge es > 25%: edges tan altos casi siempre significan que el modelo está desactualizado 
  respecto al mercado, no que hay una oportunidad real.
- El mercado y el modelo divergen más de 20 puntos porcentuales en probabilidad sin una razón 
  clara (lesión masiva, clima extremo, cambio de entrenador).
- Las noticias muestran que algún equipo tiene 3+ titulares fuera por lesión.
- Es la primera o última jornada de la temporada.

RECOMENDAR si:
- El edge está entre 5% y 20%.
- Las noticias no muestran nada alarmante.
- El modelo y el mercado están relativamente cerca (diferencia < 15 puntos).
- Hay alguna razón contextual que explique el edge (equipo en buena racha, rival cansado, etc.)

FORMATO DE RESPUESTA — responde SOLO con este JSON exacto:
{
  "recomendar": true/false,
  "razon": "Una sola frase en español explicando la decisión. Lenguaje simple, sin jerga.",
  "confianza": "alta/media/baja"
}
"""


def analyze_signal(signal: dict, db_path: Path = DB_PATH) -> dict:
    """
    Analiza una señal y decide si recomendarla o descartarla.

    Args:
        signal: Dict con los datos de la señal (fixture, odds, edge, etc.)
        db_path: Ruta a la DB

    Returns:
        Dict con: recomendar (bool), razon (str), confianza (str), news (str)
    """
    api_key = load_api_key()

    home     = signal.get("home_team", "")
    away     = signal.get("away_team", "")
    league   = signal.get("league", "BL2")
    sel      = signal.get("selection", "over")
    edge     = signal.get("edge_pct", 0)
    prob_dc  = signal.get("prob_dc", 0)
    prob_xgb = signal.get("prob_xgb", 0)
    odds     = signal.get("betplay_odds", 0)
    kickoff  = signal.get("kickoff_utc", "")
    matchweek = signal.get("matchweek", "?")

    implied_prob = 1 / odds if odds > 0 else 0

    # 1. Buscar noticias
    liga_name = "Bundesliga 2" if league == "BL2" else "Eredivisie"
    news_query = f"{home} {away} {liga_name} lesiones alineación"
    logger.info("Buscando noticias: %s", news_query)

    try:
        news = search_news(news_query, api_key)
    except Exception as e:
        logger.warning("Error buscando noticias: %s", e)
        news = "Sin noticias disponibles."

    # 2. Construir contexto para el agente
    sel_label = "Over 2.5 goles" if sel == "over" else "Under 2.5 goles"
    divergencia = abs(prob_dc - implied_prob) * 100

    context = f"""
SEÑAL A ANALIZAR:
- Partido: {home} vs {away} ({liga_name}, Jornada {matchweek})
- Kickoff: {kickoff}
- Mercado: {sel_label}
- Cuota Betplay: {odds} (implica {implied_prob*100:.1f}% de probabilidad)
- Modelo Dixon-Coles: {prob_dc*100:.1f}% de probabilidad
- Modelo XGBoost: {prob_xgb*100:.1f}% de probabilidad  
- Edge estimado: +{edge*100:.1f}%
- Divergencia modelo vs mercado: {divergencia:.1f} puntos porcentuales

NOTICIAS RECIENTES:
{news}

¿Debería recomendarse esta apuesta?
"""

    # 3. Llamar al agente
    logger.info("Analizando señal: %s vs %s | Edge: +%.1f%%", home, away, edge * 100)

    try:
        response_text = call_claude(SYSTEM_PROMPT, context, api_key)
        # Limpiar el JSON (a veces viene con markdown)
        response_text = response_text.strip()
        if "```" in response_text:
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        result = json.loads(response_text)
    except Exception as e:
        logger.error("Error en análisis del agente: %s", e)
        # Fallback conservador: siempre SKIP si el agente no pudo analizar correctamente
        result = {
            "recomendar": False,
            "razon":      "No se pudo analizar correctamente — descartando por precaución.",
            "confianza":  "baja",
        }

    result["news"] = news
    result["edge_pct"] = edge

    logger.info(
        "Decisión: %s | Razón: %s | Confianza: %s",
        "RECOMENDAR" if result.get("recomendar") else "DESCARTAR",
        result.get("razon", ""),
        result.get("confianza", ""),
    )

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Revisión semanal
# ──────────────────────────────────────────────────────────────────────────────

WEEKLY_SYSTEM = """
Eres un analista de value betting que revisa el desempeño semanal del sistema.

Tu trabajo es:
1. Analizar las señales descartadas la semana pasada y ver si habrían ganado o perdido.
2. Identificar patrones: ¿qué tipo de señales descartadas habrían sido buenas apuestas?
3. Dar UNA recomendación concreta para mejorar el filtrado la próxima semana.

Sé directo y práctico. El usuario es un apostador individual con bankroll pequeño.
Responde en español, máximo 4 frases.
"""


def weekly_review(db_path: Path = DB_PATH) -> str:
    """
    Revisa las señales descartadas de la semana pasada y sus resultados.
    Retorna un análisis en texto para incluir en el resumen semanal.
    """
    api_key = load_api_key()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        # Señales descartadas por el agente (SKIP) con resultado conocido
        skipped = conn.execute("""
            SELECT
                s.edge_pct, s.prob_ensemble, s.betplay_odds, s.selection,
                f.home_team_name, f.away_team_name, f.over_25, f.goals_home, f.goals_away,
                b.user_decision
            FROM signals s
            JOIN fixtures f ON s.fixture_id = f.id
            LEFT JOIN bets b ON b.signal_id = s.id
            WHERE s.created_at >= datetime('now', '-7 days')
              AND f.status = 'finished'
              AND (b.user_decision = 'SKIP' OR b.id IS NULL)
            ORDER BY s.edge_pct DESC
        """).fetchall()

        # Señales apostadas con resultado
        played = conn.execute("""
            SELECT
                s.edge_pct, s.prob_ensemble, s.betplay_odds, s.selection,
                f.home_team_name, f.away_team_name, f.over_25,
                r.outcome, r.pnl_cop
            FROM signals s
            JOIN fixtures f ON s.fixture_id = f.id
            JOIN bets b ON b.signal_id = s.id
            LEFT JOIN results r ON r.bet_id = b.id
            WHERE s.created_at >= datetime('now', '-7 days')
              AND b.user_decision = 'GO'
              AND f.status = 'finished'
        """).fetchall()

    if not skipped and not played:
        return "Sin suficientes datos esta semana para el análisis."

    # Construir resumen para el agente
    skipped_summary = []
    for row in skipped:
        resultado_real = "OVER" if row["over_25"] == 1 else "UNDER"
        habria_ganado = (
            (row["selection"] == "over" and row["over_25"] == 1) or
            (row["selection"] == "under" and row["over_25"] == 0)
        )
        skipped_summary.append(
            f"- {row['home_team_name']} vs {row['away_team_name']}: "
            f"señal {row['selection'].upper()} @ {row['betplay_odds']} "
            f"(edge +{row['edge_pct']*100:.0f}%) | "
            f"Resultado: {resultado_real} | "
            f"{'✅ Habría GANADO' if habria_ganado else '❌ Habría PERDIDO'}"
        )

    played_summary = []
    for row in played:
        played_summary.append(
            f"- {row['home_team_name']} vs {row['away_team_name']}: "
            f"{row['selection'].upper()} @ {row['betplay_odds']} | "
            f"{'✅ GANÓ' if row['outcome'] == 'win' else '❌ PERDIÓ'} | "
            f"P&L: {row['pnl_cop']:+,.0f} COP"
        )

    context = f"""
SEÑALES DESCARTADAS ESTA SEMANA ({len(skipped)}):
{chr(10).join(skipped_summary) if skipped_summary else 'Ninguna'}

SEÑALES APOSTADAS ESTA SEMANA ({len(played)}):
{chr(10).join(played_summary) if played_summary else 'Ninguna'}

¿Qué aprendemos? ¿Deberíamos haber apostado en alguna de las descartadas?
"""

    try:
        analysis = call_claude(WEEKLY_SYSTEM, context, api_key, max_tokens=400)
    except Exception as e:
        logger.error("Error en revisión semanal: %s", e)
        analysis = "No se pudo generar el análisis semanal."

    return analysis


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline integrado: analizar todas las señales pendientes
# ──────────────────────────────────────────────────────────────────────────────

def process_pending_signals(db_path: Path = DB_PATH) -> tuple[int, int]:
    """
    Analiza todas las señales pendientes (telegram_sent=0) y:
    - Las que el agente aprueba: las deja con telegram_sent=0 para que el bot las envíe
    - Las que el agente descarta: las marca con telegram_sent=2 (descartada por agente)

    Returns:
        (aprobadas, descartadas)
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        signals = conn.execute("""
            SELECT
                s.id, s.fixture_id, s.selection, s.edge_pct,
                s.prob_ensemble, s.betplay_odds, s.profile, s.flags,
                f.league, f.home_team_name AS home_team,
                f.away_team_name AS away_team, f.kickoff_utc, f.matchweek,
                p_dc.prob  AS prob_dc,
                p_xgb.prob AS prob_xgb
            FROM signals s
            JOIN fixtures f ON s.fixture_id = f.id
            LEFT JOIN predictions p_dc  ON (
                p_dc.fixture_id=s.fixture_id AND p_dc.model='dixon_coles'
                AND p_dc.market='ou25' AND p_dc.selection=s.selection)
            LEFT JOIN predictions p_xgb ON (
                p_xgb.fixture_id=s.fixture_id AND p_xgb.model='xgboost'
                AND p_xgb.market='ou25' AND p_xgb.selection=s.selection)
            WHERE s.telegram_sent = 0
              AND s.status IN ('BET', 'REVIEW')
        """).fetchall()

    if not signals:
        logger.info("Sin señales pendientes para analizar.")
        return 0, 0

    aprobadas  = 0
    descartadas = 0

    for sig in signals:
        signal_dict = dict(sig)
        if signal_dict.get("prob_dc") is None:
            signal_dict["prob_dc"] = signal_dict["prob_ensemble"]
        if signal_dict.get("prob_xgb") is None:
            signal_dict["prob_xgb"] = signal_dict["prob_ensemble"]

        result = analyze_signal(signal_dict, db_path)

        with sqlite3.connect(db_path) as conn:
            if result.get("recomendar"):
                # Guardar el análisis en el campo flags para que el bot lo muestre
                existing_flags = []
                try:
                    existing_flags = json.loads(sig["flags"] or "[]")
                except Exception:
                    pass

                # Añadir análisis del agente como metadata
                conn.execute("""
                    UPDATE signals
                    SET flags = ?
                    WHERE id = ?
                """, [
                    json.dumps({
                        "flags":          existing_flags,
                        "agent_analysis": result.get("razon", ""),
                        "agent_news":     result.get("news", ""),
                        "agent_confidence": result.get("confianza", ""),
                    }),
                    sig["id"],
                ])
                aprobadas += 1
                logger.info("✅ Señal %d APROBADA: %s", sig["id"], result.get("razon"))
            else:
                # Marcar como descartada por agente (telegram_sent=2)
                conn.execute(
                    "UPDATE signals SET telegram_sent=2 WHERE id=?",
                    [sig["id"]],
                )
                descartadas += 1
                logger.info("❌ Señal %d DESCARTADA: %s", sig["id"], result.get("razon"))

    return aprobadas, descartadas


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

    parser = argparse.ArgumentParser(description="Agente Claude para análisis de señales")
    parser.add_argument("--analyze", action="store_true", help="Analizar señales pendientes")
    parser.add_argument("--weekly",  action="store_true", help="Revisión semanal")
    parser.add_argument("--db",      default=str(DB_PATH))
    args = parser.parse_args()

    db_path = Path(args.db)

    if args.analyze:
        print("\n🤖 Analizando señales pendientes...\n")
        aprobadas, descartadas = process_pending_signals(db_path)
        print(f"\n✅ Aprobadas para enviar: {aprobadas}")
        print(f"❌ Descartadas:          {descartadas}\n")

    elif args.weekly:
        print("\n📊 Generando revisión semanal...\n")
        analysis = weekly_review(db_path)
        print(analysis)

    else:
        parser.print_help()