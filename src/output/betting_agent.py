"""
betting_agent.py — Agente Claude para análisis de señales de apuestas.

Caché de noticias con TTL dinámico según distancia al kickoff:
  > 24h  → caché 6h
  6-24h  → caché 2h
  2-6h   → caché 1h
  < 2h   → sin caché (alineaciones oficiales — siempre fresh)
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

DB_PATH    = Path("data/betting.db")
CONFIG_DIR = Path("config")


def load_api_key() -> str:
    import os
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    config_path = CONFIG_DIR / "agent.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)["api_key"]
    raise RuntimeError("No se encontró API key. Crear config/agent.json")


def call_claude(
    system_prompt: str,
    user_message: str,
    api_key: str,
    max_tokens: int = 1000,
    retries: int = 2,
) -> str:
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
                    "model":      "claude-haiku-4-5-20251001",
                    "max_tokens": max_tokens,
                    "system":     system_prompt,
                    "messages":   [{"role": "user", "content": user_message}],
                },
                timeout=60,
            )
            response.raise_for_status()
            return response.json()["content"][0]["text"].strip()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < retries - 1:
                wait = 30 * (attempt + 1)
                logger.warning("Rate limit (429). Esperando %ds...", wait)
                time.sleep(wait)
                last_error = e
            else:
                raise
    raise last_error


def get_cache_ttl(hours_before_kickoff: float) -> int:
    """TTL del caché en horas según distancia al partido."""
    if hours_before_kickoff > 24:
        return 6
    elif hours_before_kickoff > 6:
        return 2
    elif hours_before_kickoff > 2:
        return 1
    else:
        return 0  # sin caché — siempre buscar fresh


def _ensure_cache_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS news_cache (
            query_hash TEXT PRIMARY KEY,
            query      TEXT,
            result     TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


def search_news(
    query: str,
    api_key: str,
    db_path: Path = DB_PATH,
    hours_before_kickoff: float = 24.0,
) -> str:
    """Busca noticias con caché dinámico según distancia al kickoff."""
    query_hash = hashlib.md5(query.encode()).hexdigest()
    ttl        = get_cache_ttl(hours_before_kickoff)

    # Verificar caché
    if ttl > 0:
        with sqlite3.connect(db_path) as conn:
            _ensure_cache_table(conn)
            cached = conn.execute("""
                SELECT result FROM news_cache
                WHERE query_hash = ?
                  AND created_at > datetime('now', ? || ' hours')
            """, [query_hash, f"-{ttl}"]).fetchone()
        if cached:
            logger.info("Caché hit (TTL %dh): %s", ttl, query)
            return cached[0]
    else:
        logger.info("Sin caché (< 2h kickoff) — buscando fresh: %s", query)

    # Llamar a la API
    system = (
        "You are a football betting analyst specialized in European football. "
        "Search recent news (last 48h) from sources like kicker.de, transfermarkt.com, "
        "bundesliga.com, and major sports outlets. "
        "Report ONLY concrete facts relevant for betting: injuries, confirmed lineups, "
        "recent form, suspensions, weather. "
        "If nothing relevant, say 'Sin noticias relevantes'. "
        "Max 3 sentences. Always respond in Spanish."
    )
    response = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        },
        json={
            "model":      "claude-haiku-4-5-20251001",
            "max_tokens": 500,
            "system":     system,
            "tools":      [{"type": "web_search_20250305", "name": "web_search"}],
            "messages":   [{
                "role":    "user",
                "content": f"Search for news in the last 48 hours about: {query}",
            }],
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    text_parts = [
        block["text"]
        for block in data.get("content", [])
        if block.get("type") == "text"
    ]
    result = " ".join(text_parts).strip() or "Sin noticias relevantes."

    # Guardar en caché
    with sqlite3.connect(db_path) as conn:
        _ensure_cache_table(conn)
        conn.execute("""
            INSERT OR REPLACE INTO news_cache (query_hash, query, result)
            VALUES (?, ?, ?)
        """, [query_hash, query, result])

    return result


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
- Las noticias muestran que algún equipo tiene 3+ JUGADORES CLAVE fuera por lesión.
- Es la primera o última jornada de la temporada.

RECOMENDAR si:
- El edge está entre 5% y 20%.
- Las noticias no muestran nada alarmante.
- El modelo y el mercado están relativamente cerca (diferencia < 15 puntos).
- Hay alguna razón contextual que explique el edge (equipo en buena racha, rival cansado, etc.)

SOBRE LESIONES — MUY IMPORTANTE:
- Solo son relevantes las lesiones de JUGADORES CLAVE que aparecen en la sección "JUGADORES CLAVE"
  de cada equipo (jugadores con ≥60% de minutos y ≥70% de titularidades en la temporada).
- Si el jugador lesionado NO aparece en la lista de jugadores clave, IGNORAR esa lesión completamente.
  Jugadores que apenas juegan no impactan los goles del partido.
- Si 3 o más JUGADORES CLAVE de un equipo están fuera → señal de alarma real.
- Si "No disponible" aparece en jugadores clave → evaluar lesiones con cautela sin poder
  determinar importancia real.

FORMATO DE RESPUESTA — responde SOLO con este JSON exacto:
{
  "recomendar": true/false,
  "razon": "Una sola frase en español explicando la decisión. Lenguaje simple, sin jerga.",
  "confianza": "alta/media/baja"
}
"""


def analyze_signal(signal: dict, db_path: Path = DB_PATH) -> dict:
    api_key = load_api_key()

    home      = signal.get("home_team", "")
    away      = signal.get("away_team", "")
    league    = signal.get("league", "BL2")
    sel       = signal.get("selection", "over")
    edge      = signal.get("edge_pct", 0)
    prob_dc   = signal.get("prob_dc", 0)
    prob_xgb  = signal.get("prob_xgb", 0)
    odds      = signal.get("betplay_odds", 0)
    kickoff   = signal.get("kickoff_utc", "")
    matchweek = signal.get("matchweek", "?")

    implied_prob = 1 / odds if odds > 0 else 0

    # Calcular horas hasta el kickoff para el TTL del caché
    try:
        kickoff_dt   = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
        hours_before = max((kickoff_dt - datetime.now(timezone.utc)).total_seconds() / 3600, 0)
    except (TypeError, ValueError):
        hours_before = 24.0

    # Descarte rápido sin gastar API
    if edge > 0.25:
        logger.info("Edge %.1f%% > 25%% — descarte directo sin API", edge * 100)
        return {
            "recomendar": False,
            "razon":      f"Edge de {edge*100:.0f}% demasiado alto — el mercado sabe algo que el modelo no ve.",
            "confianza":  "alta",
            "news":       "",
        }

    # 1. Buscar noticias con caché dinámico
    # Query en inglés para capturar fuentes alemanas/holandesas (kicker.de, transfermarkt, etc.)
    liga_map  = {"BL2": "2. Bundesliga", "NL1": "Eredivisie"}
    liga_name = liga_map.get(league, league)
    news_query = f"{home} vs {away} {liga_name} injury lineup team news"
    logger.info("Buscando noticias: %s (%.1fh al kickoff)", news_query, hours_before)

    try:
        news = search_news(news_query, api_key, db_path, hours_before_kickoff=hours_before)
    except Exception as e:
        logger.warning("Error buscando noticias: %s", e)
        news = "Sin noticias disponibles."

    # 2. Obtener jugadores clave de la DB
    key_players_section = ""
    try:
        from src.ingestion.player_stats import get_key_players_for_team
        home_keys = get_key_players_for_team(home, league, db_path)
        away_keys = get_key_players_for_team(away, league, db_path)

        def _format_key_players(team_name: str, keys: dict) -> str:
            if not keys["found"]:
                return f"JUGADORES CLAVE {team_name.upper()}: No disponible (ejecutar player_stats.py)"
            attackers = ", ".join(keys["attackers"]) if keys["attackers"] else "Ninguno identificado"
            return (
                f"JUGADORES CLAVE {team_name.upper()}:\n"
                f"  Atacantes/mediapuntas: {attackers}\n"
                f"  Portero: {keys['keeper']}"
            )

        key_players_section = (
            "\n" + _format_key_players(home, home_keys) +
            "\n" + _format_key_players(away, away_keys)
        )
        logger.info(
            "Jugadores clave cargados — %s: %d atacantes | %s: %d atacantes",
            home, len(home_keys.get("attackers", [])),
            away, len(away_keys.get("attackers", [])),
        )
    except Exception as e:
        logger.warning("No se pudo obtener jugadores clave: %s", e)
        key_players_section = "\nJUGADORES CLAVE: No disponible"

    # 3. Construir contexto
    sel_label   = "Over 2.5 goles" if sel == "over" else "Under 2.5 goles"
    divergencia = abs(prob_dc - implied_prob) * 100

    context = f"""
SEÑAL A ANALIZAR:
- Partido: {home} vs {away} ({liga_name}, Jornada {matchweek})
- Kickoff: {kickoff} ({hours_before:.1f}h restantes)
- Mercado: {sel_label}
- Cuota Betplay: {odds} (implica {implied_prob*100:.1f}% de probabilidad)
- Modelo Dixon-Coles: {prob_dc*100:.1f}% de probabilidad
- Modelo XGBoost: {prob_xgb*100:.1f}% de probabilidad
- Edge estimado: +{edge*100:.1f}%
- Divergencia modelo vs mercado: {divergencia:.1f} puntos porcentuales
{key_players_section}

NOTICIAS RECIENTES:
{news}

¿Debería recomendarse esta apuesta?
"""

    logger.info("Analizando señal: %s vs %s | Edge: +%.1f%%", home, away, edge * 100)

    try:
        response_text = call_claude(SYSTEM_PROMPT, context, api_key)
        response_text = response_text.strip()
        if "```" in response_text:
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        result = json.loads(response_text)
    except Exception as e:
        logger.error("Error en análisis del agente: %s", e)
        result = {
            "recomendar": False,
            "razon":      "No se pudo analizar correctamente — descartando por precaución.",
            "confianza":  "baja",
        }

    result["news"]     = news
    result["edge_pct"] = edge

    logger.info(
        "Decisión: %s | Razón: %s | Confianza: %s",
        "RECOMENDAR" if result.get("recomendar") else "DESCARTAR",
        result.get("razon", ""),
        result.get("confianza", ""),
    )
    return result


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
    api_key = load_api_key()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        skipped = conn.execute("""
            SELECT s.edge_pct, s.prob_ensemble, s.betplay_odds, s.selection,
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

        played = conn.execute("""
            SELECT s.edge_pct, s.prob_ensemble, s.betplay_odds, s.selection,
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

    skipped_summary = []
    for row in skipped:
        resultado_real = "OVER" if row["over_25"] == 1 else "UNDER"
        habria_ganado  = (
            (row["selection"] == "over"  and row["over_25"] == 1) or
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


MAX_SIGNALS_PER_RUN = 8  # BL2: 9 partidos/jornada × 2 selecciones, filtros reducen a <8


def process_pending_signals(db_path: Path = DB_PATH) -> tuple[int, int]:
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

    if len(signals) > MAX_SIGNALS_PER_RUN:
        raise RuntimeError(
            f"Se encontraron {len(signals)} señales pendientes — máximo permitido: "
            f"{MAX_SIGNALS_PER_RUN}. Probablemente hay señales duplicadas en la DB. "
            f"Revisar tabla signals antes de correr el agente."
        )

    aprobadas   = 0
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
                existing_flags = []
                try:
                    existing_flags = json.loads(sig["flags"] or "[]")
                except Exception:
                    pass
                conn.execute("UPDATE signals SET flags = ? WHERE id = ?", [
                    json.dumps({
                        "flags":            existing_flags,
                        "agent_analysis":   result.get("razon", ""),
                        "agent_news":       result.get("news", ""),
                        "agent_confidence": result.get("confianza", ""),
                    }),
                    sig["id"],
                ])
                aprobadas += 1
                logger.info("✅ Señal %d APROBADA: %s", sig["id"], result.get("razon"))
            else:
                conn.execute(
                    "UPDATE signals SET telegram_sent=2 WHERE id=?",
                    [sig["id"]],
                )
                descartadas += 1
                logger.info("❌ Señal %d DESCARTADA: %s", sig["id"], result.get("razon"))

        time.sleep(5)

    return aprobadas, descartadas


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--weekly",  action="store_true")
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
        print(weekly_review(db_path))
    else:
        parser.print_help()
