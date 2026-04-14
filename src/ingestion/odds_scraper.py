"""
odds_scraper.py — Scraper de cuotas desde la API de Kambi (Betplay).

Betplay usa la plataforma Kambi, que expone una API REST pública
sin autenticación. No se requiere Playwright ni login.

Endpoints usados:
  - Listado de eventos: /offering/v2018/betplay/listView/football/germany.json
  - Cuotas por evento: /offering/v2018/betplay/betoffer/event/{id}.json

Flujo:
  1. Obtener todos los eventos de Bundesliga 2 disponibles
  2. Para cada evento, obtener las cuotas de Over/Under 2.5
  3. Hacer match con los fixtures en la DB por equipos y fecha
  4. Guardar en tabla odds con bookmaker='betplay'

Uso:
    python src/ingestion/odds_scraper.py --league BL2
    python src/ingestion/odds_scraper.py --league BL2 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

DB_PATH = Path("data/betting.db")

# ──────────────────────────────────────────────────────────────────────────────
# Configuración de la API de Kambi
# ──────────────────────────────────────────────────────────────────────────────

KAMBI_BASE     = "https://us1.offering-api.kambicdn.com"
KAMBI_OPERATOR = "betplay"
KAMBI_PARAMS   = {
    "lang":       "es_CO",
    "market":     "CO",
    "client_id":  "200",
    "channel_id": "1",
}

# Headers realistas para no llamar la atención
HEADERS = {
    "User-Agent":       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept":           "application/json, text/plain, */*",
    "Accept-Language":  "es-CO,es;q=0.9,en;q=0.8",
    "Origin":           "https://www.betplay.com.co",
    "Referer":          "https://www.betplay.com.co/",
}

# Mapeo de nombres de equipos Kambi → nombres en la DB
# Kambi usa nombres completos; la DB usa nombres cortos del CSV de football-data.co.uk
TEAM_NAME_MAP = {
    # Bundesliga 2
    "Elversberg":                "Elversberg",
    "SV 07 Elversberg":          "Elversberg",
    "Karlsruher SC":             "Karlsruhe",
    "Holstein Kiel":             "Holstein Kiel",
    "1. FC Kaiserslautern":      "Kaiserslautern",
    "Arminia Bielefeld":         "Bielefeld",
    "1. FC Nürnberg":            "Nurnberg",
    "1. FC Nuremberg":           "Nurnberg",
    "1. FC Magdeburg":           "Magdeburg",
    "Fortuna Düsseldorf":        "Fortuna Dusseldorf",
    "Fortuna Dusseldorf":        "Fortuna Dusseldorf",
    "SG Dynamo Dresden":         "Dresden",
    "Dynamo Dresden":            "Dresden",
    "VfL Bochum":                "Bochum",
    "Hannover 96":               "Hannover",
    "SC Paderborn 07":           "Paderborn",
    "Paderborn":                 "Paderborn",
    "Eintracht Braunschweig":    "Braunschweig",
    "Hertha BSC":                "Hertha",
    "Hertha Berlin":             "Hertha",
    "Greuther Fürth":            "Greuther Furth",
    "SpVgg Greuther Fürth":      "Greuther Furth",
    "SV Darmstadt 98":           "Darmstadt",
    "Darmstadt":                 "Darmstadt",
    "FC Schalke 04":             "Schalke 04",
    "Schalke 04":                "Schalke 04",
    "Preußen Münster":           "PreuÃen MÃ¼nster",
    "SC Preußen Münster":        "PreuÃen MÃ¼nster",
    "Preussen Munster":          "PreuÃen MÃ¼nster",
    # Eredivisie (para Fase 2)
    "PSV Eindhoven":             "PSV Eindhoven",
    "Feyenoord":                 "Feyenoord",
    "AFC Ajax":                  "Ajax",
    "AZ Alkmaar":                "AZ Alkmaar",
    "FC Twente":                 "Twente",
    "FC Utrecht":                "Utrecht",
}


def normalize_team(kambi_name: str) -> str:
    """Convierte nombre de equipo de Kambi al formato de la DB."""
    # Buscar match exacto
    if kambi_name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[kambi_name]
    # Buscar match parcial
    for kambi, db in TEAM_NAME_MAP.items():
        if kambi.lower() in kambi_name.lower() or kambi_name.lower() in kambi.lower():
            return db
    # Fallback: devolver el nombre original limpio
    logger.warning("Equipo no mapeado: '%s' — usando nombre original", kambi_name)
    return kambi_name


# ──────────────────────────────────────────────────────────────────────────────
# Cliente de la API
# ──────────────────────────────────────────────────────────────────────────────

class KambiClient:
    """Cliente HTTP para la API de Kambi/Betplay."""

    def __init__(self, delay_min: float = 0.5, delay_max: float = 1.5):
        self.client    = httpx.Client(headers=HEADERS, timeout=30, follow_redirects=True)
        self.delay_min = delay_min
        self.delay_max = delay_max

    def _get(self, path: str, extra_params: dict = {}) -> dict:
        """Hace un GET a la API con delay aleatorio."""
        url    = f"{KAMBI_BASE}{path}"
        params = {**KAMBI_PARAMS, **extra_params}

        time.sleep(random.uniform(self.delay_min, self.delay_max))

        response = self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_events(self, sport_path: str = "football/germany") -> list[dict]:
        """
        Obtiene todos los eventos disponibles para un deporte/región.

        Returns:
            Lista de eventos con id, nombre de equipos, liga, kickoff.
        """
        path = f"/offering/v2018/{KAMBI_OPERATOR}/listView/{sport_path}.json"
        data = self._get(path, {"includeParticipants": "true"})

        events = []
        for item in data.get("events", []):
            ev = item.get("event", {})
            events.append({
                "id":        ev.get("id"),
                "name":      ev.get("name", ""),
                "league":    ev.get("group", ""),
                "kickoff":   ev.get("start", ""),
                "home_team": ev.get("homeName", ""),
                "away_team": ev.get("awayName", ""),
            })

        logger.info("Kambi: %d eventos encontrados en %s", len(events), sport_path)
        return events

    def get_ou25_odds(self, event_id: int) -> Optional[dict]:
        """
        Obtiene las cuotas de Over/Under 2.5 para un evento.

        Returns:
            Dict con odds_over, odds_under, overround_pct o None si no disponible.
        """
        path = f"/offering/v2018/{KAMBI_OPERATOR}/betoffer/event/{event_id}.json"
        try:
            data = self._get(path, {"includeParticipants": "true"})
        except httpx.HTTPStatusError as e:
            logger.warning("Error obteniendo odds para evento %d: %s", event_id, e)
            return None

        bet_offers = data.get("betOffers", [])

        odds_over  = None
        odds_under = None

        for offer in bet_offers:
            criterion = offer.get("criterion", {})
            label     = criterion.get("label", "")

            # Buscar "Total de goles" (tiempo completo, no por equipo ni parte)
            if "Total de goles" not in label:
                continue
            # Excluir mercados de equipo individual o partes
            if any(x in label for x in ["1.ª parte", "2.ª parte", "1ª parte", "2ª parte",
                                          "Elversberg", "Karlsruher", "equipo"]):
                continue

            # Buscar la línea 2.5 (en Kambi viene como 2500 en milésimas)
            outcomes = offer.get("outcomes", [])
            for outcome in outcomes:
                line = outcome.get("line", 0)
                if line != 2500:
                    continue
                outcome_label = outcome.get("label", "").lower()
                odds_decimal  = outcome.get("odds", 0) / 1000  # Kambi usa milésimas

                if "más" in outcome_label or "over" in outcome_label:
                    odds_over = round(odds_decimal, 3)
                elif "menos" in outcome_label or "under" in outcome_label:
                    odds_under = round(odds_decimal, 3)

        if odds_over is None or odds_under is None:
            logger.debug("Evento %d: sin cuotas O/U 2.5", event_id)
            return None

        overround = round((1 / odds_over + 1 / odds_under - 1) * 100, 2)

        return {
            "odds_over":    odds_over,
            "odds_under":   odds_under,
            "overround_pct": overround,
        }

    def close(self):
        self.client.close()


# ──────────────────────────────────────────────────────────────────────────────
# Matching con la DB
# ──────────────────────────────────────────────────────────────────────────────

def find_fixture_id(
    db_path: Path,
    home_team: str,
    away_team: str,
    kickoff_kambi: str,
    league: str,
    tolerance_hours: float = 6.0,
) -> Optional[int]:
    """
    Busca el fixture_id en la DB que corresponde al evento de Kambi.

    Hace match por nombre de equipo (normalizado) y fecha aproximada.
    """
    home_db = normalize_team(home_team)
    away_db = normalize_team(away_team)

    # Parsear kickoff de Kambi (viene en ISO 8601)
    try:
        kickoff_dt = datetime.fromisoformat(kickoff_kambi.replace("Z", "+00:00"))
        kickoff_dt = kickoff_dt.replace(tzinfo=None)  # quitar timezone para comparar
    except (TypeError, ValueError):
        logger.warning("Kickoff inválido: %s", kickoff_kambi)
        return None

    window_start = (kickoff_dt - timedelta(hours=tolerance_hours)).isoformat()
    window_end   = (kickoff_dt + timedelta(hours=tolerance_hours)).isoformat()

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("""
            SELECT id FROM fixtures
            WHERE home_team_name = ?
              AND away_team_name = ?
              AND kickoff_utc BETWEEN ? AND ?
              AND status = 'scheduled'
            LIMIT 1
        """, [home_db, away_db, window_start, window_end]).fetchone()

    if row:
        return row[0]

    # Segundo intento: solo por nombre de equipo sin ventana de tiempo
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("""
            SELECT id FROM fixtures
            WHERE home_team_name = ?
              AND away_team_name = ?
              AND status = 'scheduled'
            LIMIT 1
        """, [home_db, away_db]).fetchone()

    return row[0] if row else None


# ──────────────────────────────────────────────────────────────────────────────
# Guardado en DB
# ──────────────────────────────────────────────────────────────────────────────

def save_odds(
    db_path: Path,
    fixture_id: int,
    odds_over: float,
    odds_under: float,
    overround_pct: float,
    hours_before_kickoff: float,
) -> None:
    """Guarda o actualiza las cuotas en la tabla odds."""
    now = datetime.utcnow().isoformat()

    with sqlite3.connect(db_path) as conn:
        for selection, odds_val in [("over", odds_over), ("under", odds_under)]:
            # Verificar si ya existe una entrada reciente (última hora)
            existing = conn.execute("""
                SELECT id FROM odds
                WHERE fixture_id  = ?
                  AND bookmaker   = 'betplay'
                  AND market      = 'ou25'
                  AND selection   = ?
                  AND captured_at > datetime('now', '-1 hour')
            """, [fixture_id, selection]).fetchone()

            if existing:
                # Actualizar cuota existente
                conn.execute("""
                    UPDATE odds
                    SET odds_decimal = ?,
                        implied_prob = ?,
                        overround_pct = ?,
                        captured_at = ?,
                        hours_before_kickoff = ?
                    WHERE id = ?
                """, [odds_val, 1/odds_val, overround_pct, now,
                      hours_before_kickoff, existing[0]])
            else:
                # Insertar nueva entrada
                conn.execute("""
                    INSERT INTO odds
                        (fixture_id, bookmaker, market, selection,
                         odds_decimal, implied_prob, overround_pct,
                         captured_at, hours_before_kickoff)
                    VALUES (?, 'betplay', 'ou25', ?, ?, ?, ?, ?, ?)
                """, [fixture_id, selection, odds_val, 1/odds_val,
                      overround_pct, now, hours_before_kickoff])


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────────────────────────────────────

def run(
    db_path: Path = DB_PATH,
    league_filter: str = "BL2",
    dry_run: bool = False,
) -> dict:
    """
    Corre el pipeline completo de scraping de odds.

    Args:
        db_path:       Ruta a la DB.
        league_filter: 'BL2' o 'NL1'.
        dry_run:       Si True, no guarda en DB — solo muestra lo que encontraría.

    Returns:
        Dict con estadísticas: fixtures_found, odds_saved, errors.
    """
    # Mapeo de liga del sistema → nombre en Kambi
    kambi_paths = {
        "BL2": "football/germany",
        "NL1": "football/netherlands",
    }
    kambi_league_names = {
        "BL2": "2. Bundesliga",
        "NL1": "Eredivisie",
    }

    sport_path   = kambi_paths.get(league_filter, "football/germany")
    league_label = kambi_league_names.get(league_filter, "")

    client = KambiClient(delay_min=0.5, delay_max=1.2)
    stats  = {"fixtures_found": 0, "odds_saved": 0, "no_match": 0, "errors": 0}

    try:
        # 1. Obtener todos los eventos de la región
        all_events = client.get_events(sport_path)

        # 2. Filtrar por liga
        bl2_events = [
            e for e in all_events
            if league_label.lower() in e.get("league", "").lower()
        ]
        logger.info("[%s] %d eventos de '%s' encontrados en Kambi",
                    league_filter, len(bl2_events), league_label)

        # 3. Para cada evento, obtener O/U 2.5 y hacer match con DB
        for event in bl2_events:
            event_id  = event["id"]
            home_kambi = event.get("home_team", "")
            away_kambi = event.get("away_team", "")
            kickoff    = event.get("kickoff", "")

            logger.info("Procesando: %s vs %s (ID: %d)", home_kambi, away_kambi, event_id)

            # Obtener odds de Kambi
            odds = client.get_ou25_odds(event_id)
            if odds is None:
                logger.warning("Sin O/U 2.5 para %s vs %s", home_kambi, away_kambi)
                stats["errors"] += 1
                continue

            # Hacer match con fixture en DB
            fixture_id = find_fixture_id(
                db_path    = db_path,
                home_team  = home_kambi,
                away_team  = away_kambi,
                kickoff_kambi = kickoff,
                league     = league_filter,
            )

            if fixture_id is None:
                logger.warning(
                    "Sin match en DB para: %s vs %s (%s)",
                    normalize_team(home_kambi), normalize_team(away_kambi), kickoff[:10],
                )
                stats["no_match"] += 1
                continue

            stats["fixtures_found"] += 1

            # Calcular horas hasta el kickoff
            try:
                kickoff_dt = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
                kickoff_dt = kickoff_dt.replace(tzinfo=None)
                hours_before = max((kickoff_dt - datetime.utcnow()).total_seconds() / 3600, 0)
            except (TypeError, ValueError):
                hours_before = 24.0

            if dry_run:
                print(
                    f"  [DRY-RUN] Fixture #{fixture_id}: "
                    f"{normalize_team(home_kambi)} vs {normalize_team(away_kambi)} | "
                    f"Over: {odds['odds_over']} | Under: {odds['odds_under']} | "
                    f"Overround: {odds['overround_pct']:.1f}%"
                )
            else:
                save_odds(
                    db_path              = db_path,
                    fixture_id           = fixture_id,
                    odds_over            = odds["odds_over"],
                    odds_under           = odds["odds_under"],
                    overround_pct        = odds["overround_pct"],
                    hours_before_kickoff = hours_before,
                )
                stats["odds_saved"] += 1
                logger.info(
                    "Guardado fixture #%d: Over %.3f / Under %.3f (overround %.1f%%)",
                    fixture_id, odds["odds_over"], odds["odds_under"], odds["overround_pct"],
                )

    finally:
        client.close()

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Scraper de odds Kambi/Betplay")
    parser.add_argument("--league",   choices=["BL2", "NL1"], default="BL2")
    parser.add_argument("--db",       default=str(DB_PATH))
    parser.add_argument("--dry-run",  action="store_true", help="No guardar en DB")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  Odds scraper — {args.league}")
    print(f"  Modo: {'DRY-RUN' if args.dry_run else 'PRODUCCIÓN'}")
    print(f"{'='*55}\n")

    stats = run(
        db_path      = Path(args.db),
        league_filter = args.league,
        dry_run      = args.dry_run,
    )

    print(f"\n{'='*55}")
    print(f"  ✅ Fixtures encontrados: {stats['fixtures_found']}")
    print(f"  ✅ Odds guardadas:       {stats['odds_saved']}")
    print(f"  ⚠️  Sin match en DB:     {stats['no_match']}")
    print(f"  ❌ Errores:              {stats['errors']}")
    print(f"{'='*55}\n")