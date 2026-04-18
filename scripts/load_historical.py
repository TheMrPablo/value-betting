"""
load_historical.py — Descarga y carga datos de football-data.co.uk a SQLite.

Hace dos cosas en un solo script:
  1. Descarga CSVs que falten o que estén desactualizados desde football-data.co.uk
  2. Carga todos los CSVs a la DB (upsert — no duplica partidos existentes)

Diseñado para correr diariamente (GitHub Actions o manualmente).
La temporada actual siempre se re-descarga para capturar partidos nuevos.

Uso:
    python scripts/load_historical.py              # descarga + carga todo
    python scripts/load_historical.py --no-download # solo carga desde archivos locales
    python scripts/load_historical.py --current-only # solo actualiza temporada actual (uso diario)
    python scripts/load_historical.py --league BL2   # solo una liga
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Configuración — ÚNICO LUGAR donde cambiar cosas cada temporada
# ──────────────────────────────────────────────────────────────────────────────

DB_PATH         = Path("data/betting.db")
HISTORICAL_PATH = Path("data/historical")
BASE_URL        = "https://www.football-data.co.uk/mmz4281"

# ← Actualizar a "2627" en agosto 2026
CURRENT_SEASON_CODE = "2526"

# Todas las temporadas a mantener
ALL_SEASON_CODES = ["2122", "2223", "2324", "2425", "2526"]

LEAGUES = {
    "BL2": {"name": "2. Bundesliga", "csv_code": "D2"},
    "NL1": {"name": "Eredivisie",    "csv_code": "N1"},
}


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────────────────────

def season_code_to_label(code: str) -> str:
    """'2526' → '2025-26'"""
    return f"20{code[:2]}-{code[2:]}"


# Zona horaria de los equipos locales (para conversión a UTC)
LEAGUE_TZ = {
    "BL2": ZoneInfo("Europe/Berlin"),   # Bundesliga 2 → CET/CEST
    "NL1": ZoneInfo("Europe/Amsterdam"),  # Eredivisie → CET/CEST
}


def parse_kickoff_utc(date_str: str, time_str: str | None, league_tz: ZoneInfo | None) -> str | None:
    """
    Parsea fecha + hora local → ISO timestamp UTC.
    Si no hay hora, usa 12:00 local como estimado neutro (≈ medianoche UTC en verano).
    """
    date_clean = str(date_str).strip()
    for fmt in ["%d/%m/%y", "%d/%m/%Y"]:
        try:
            d = datetime.strptime(date_clean, fmt)
            break
        except ValueError:
            continue
    else:
        return None

    # Intentar parsear hora local (formato HH:MM)
    h, m = 12, 0
    if time_str:
        t = str(time_str).strip()
        if ":" in t and len(t) >= 4:
            try:
                parts = t.split(":")
                h, m = int(parts[0]), int(parts[1][:2])
            except (ValueError, IndexError):
                pass

    local_dt = d.replace(hour=h, minute=m, second=0, tzinfo=league_tz)
    utc_dt   = local_dt.astimezone(timezone.utc).replace(tzinfo=None)
    return utc_dt.strftime("%Y-%m-%d %H:%M:%S")


def parse_date(date_str: str) -> str | None:
    """Parsea fechas DD/MM/YY o DD/MM/YYYY → ISO timestamp (medianoche UTC, legacy)."""
    for fmt in ["%d/%m/%y", "%d/%m/%Y"]:
        try:
            return datetime.strptime(str(date_str).strip(), fmt).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Descarga
# ──────────────────────────────────────────────────────────────────────────────

def should_download(filepath: Path, season_code: str) -> bool:
    """
    Decide si hay que (re)descargar un CSV.
    - No existe → sí
    - Es la temporada actual → sí siempre (captura jornadas nuevas)
    - Temporada pasada y existe → no (datos inmutables)
    """
    if not filepath.exists():
        return True
    return season_code == CURRENT_SEASON_CODE


def download_csv(season_code: str, csv_code: str) -> Path | None:
    """Descarga un CSV desde football-data.co.uk."""
    filepath = HISTORICAL_PATH / f"{csv_code}_{season_code}.csv"
    url = f"{BASE_URL}/{season_code}/{csv_code}.csv"

    try:
        print(f"  ↓ Descargando {url}")
        response = httpx.get(url, timeout=30, follow_redirects=True)
        response.raise_for_status()

        content = response.text
        if "<html" in content.lower():
            print(f"    ⚠️  Respuesta HTML — temporada {season_code} no disponible aún")
            return None
        if len(content.strip()) < 100:
            print(f"    ⚠️  CSV vacío para temporada {season_code}")
            return None

        filepath.write_text(content, encoding="latin-1")
        lines = content.strip().count("\n")
        print(f"    ✅ Guardado: {filepath.name} ({lines} partidos)")
        return filepath

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            print(f"    ⚠️  404 — temporada {season_code} no existe aún en football-data.co.uk")
        else:
            print(f"    ❌ Error HTTP {e.response.status_code}")
        return None
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Carga a SQLite
# ──────────────────────────────────────────────────────────────────────────────

def load_csv_to_db(filepath: Path, league_key: str, season_label: str) -> tuple[int, int]:
    """
    Carga un CSV a la DB con upsert.

    - Partidos nuevos: INSERT
    - Partidos que ya existían sin resultado y ahora lo tienen: UPDATE
    - Partidos ya completos: IGNORE

    Returns:
        (insertados, ya_existian)
    """
    try:
        df = pd.read_csv(filepath, encoding="latin-1")
        df = df.dropna(how="all")
    except Exception as e:
        print(f"    ❌ Error leyendo {filepath.name}: {e}")
        return 0, 0

    if df.empty:
        print(f"    ⚠️  {filepath.name}: sin datos")
        return 0, 0

    league_tz = LEAGUE_TZ.get(league_key)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    inserted = 0
    skipped  = 0

    for _, row in df.iterrows():
        try:
            home     = str(row.get("HomeTeam", "")).strip()
            away     = str(row.get("AwayTeam", "")).strip()
            time_str = row.get("Time", None)
            date_str = parse_kickoff_utc(row.get("Date", ""), time_str, league_tz)

            if not home or not away or not date_str:
                skipped += 1
                continue

            goals_home = int(row["FTHG"]) if pd.notna(row.get("FTHG")) else None
            goals_away = int(row["FTAG"]) if pd.notna(row.get("FTAG")) else None
            result     = str(row.get("FTR", "")).strip() or None
            status     = "finished" if goals_home is not None else "scheduled"

            over_25 = btts = None
            if goals_home is not None and goals_away is not None:
                over_25 = 1 if (goals_home + goals_away) > 2 else 0
                btts    = 1 if (goals_home > 0 and goals_away > 0) else 0

            # Intentar insertar
            c.execute("""
                INSERT OR IGNORE INTO fixtures
                    (league, season, home_team_name, away_team_name,
                     kickoff_utc, status, goals_home, goals_away,
                     result, over_25, btts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                league_key, season_label, home, away, date_str,
                status, goals_home, goals_away, result, over_25, btts,
            ))

            fixture_id = c.lastrowid

            if fixture_id == 0:
                # Ya existía — actualizar kickoff y resultado si aplica
                if goals_home is not None:
                    c.execute("""
                        UPDATE fixtures
                        SET goals_home = ?, goals_away = ?, result = ?,
                            over_25 = ?, btts = ?, status = 'finished',
                            kickoff_utc = ?
                        WHERE league = ? AND season = ?
                          AND home_team_name = ? AND away_team_name = ?
                          AND status != 'finished'
                    """, (
                        goals_home, goals_away, result, over_25, btts,
                        date_str, league_key, season_label, home, away,
                    ))
                else:
                    # Partido aún no jugado — actualizar kickoff si cambió de medianoche
                    c.execute("""
                        UPDATE fixtures
                        SET kickoff_utc = ?
                        WHERE league = ? AND season = ?
                          AND home_team_name = ? AND away_team_name = ?
                          AND (kickoff_utc LIKE '% 00:00:00' OR kickoff_utc < ?)
                          AND status = 'scheduled'
                    """, (
                        date_str, league_key, season_label, home, away, date_str,
                    ))
                skipped += 1
                continue

            # Insertar odds B365 si existen
            if pd.notna(row.get("B365>2.5")) and pd.notna(row.get("B365<2.5")):
                odds_over  = float(row["B365>2.5"])
                odds_under = float(row["B365<2.5"])
                overround  = (1 / odds_over + 1 / odds_under - 1) * 100

                for sel, odds_val in [("over", odds_over), ("under", odds_under)]:
                    c.execute("""
                        INSERT OR IGNORE INTO odds
                            (fixture_id, bookmaker, market, selection,
                             odds_decimal, implied_prob, overround_pct,
                             captured_at, hours_before_kickoff)
                        VALUES (?, 'bet365_historical', 'ou25', ?, ?, ?, ?, ?, -1)
                    """, (fixture_id, sel, odds_val, 1/odds_val, overround, date_str))

            inserted += 1

        except Exception:
            skipped += 1
            continue

    conn.commit()
    conn.close()
    return inserted, skipped


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────────────────────────────────────

def run(leagues_to_run: list[str], season_codes: list[str], download: bool = True) -> None:

    HISTORICAL_PATH.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        total_before = conn.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0]

    print(f"\n{'='*55}")
    print(f"  Data updater — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Temporada actual: {season_code_to_label(CURRENT_SEASON_CODE)}")
    print(f"{'='*55}")
    print(f"\n  Partidos en DB antes: {total_before}\n")

    total_inserted = 0
    total_skipped  = 0

    for league_key in leagues_to_run:
        config = LEAGUES[league_key]
        print(f"🏆 {config['name']} ({league_key})")

        for season_code in season_codes:
            season_label = season_code_to_label(season_code)
            filepath = HISTORICAL_PATH / f"{config['csv_code']}_{season_code}.csv"

            # Paso 1: Descargar si hace falta
            if download and should_download(filepath, season_code):
                filepath = download_csv(season_code, config["csv_code"])
                if filepath is None:
                    continue
                time.sleep(0.5)  # cortesía con el servidor
            elif not filepath.exists():
                print(f"  ⚠️  {filepath.name} no encontrado — descargá manualmente o usá sin --no-download")
                continue
            else:
                print(f"  ✓  {filepath.name} (temporada pasada, no se re-descarga)")

            # Paso 2: Cargar a DB
            ins, skip = load_csv_to_db(filepath, league_key, season_label)
            print(f"     → {ins} nuevos insertados, {skip} ya existían\n")
            total_inserted += ins
            total_skipped  += skip

    # Resumen final
    with sqlite3.connect(DB_PATH) as conn:
        total_after = conn.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0]
        rows = conn.execute("""
            SELECT league, season, COUNT(*), ROUND(AVG(over_25)*100, 1)
            FROM fixtures
            WHERE status = 'finished' AND over_25 IS NOT NULL
            GROUP BY league, season
            ORDER BY league, season
        """).fetchall()

    print(f"\n{'='*55}")
    print(f"  ✅ Actualización completada")
    print(f"  Partidos nuevos:   {total_inserted}")
    print(f"  Ya existían:       {total_skipped}")
    print(f"  Total en DB ahora: {total_after}")
    print(f"\n  {'Liga':<6} {'Temporada':<12} {'Partidos':>9} {'Over2.5%':>9}")
    print(f"  {'-'*40}")
    for league, season, count, over_pct in rows:
        tag = " ← actual" if season == season_code_to_label(CURRENT_SEASON_CODE) else ""
        print(f"  {league:<6} {season:<12} {count:>9} {over_pct or 0:>8.1f}%{tag}")
    print(f"{'='*55}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga y carga datos de football-data.co.uk")
    parser.add_argument("--league",       choices=["BL2", "NL1"], help="Solo una liga (default: todas)")
    parser.add_argument("--seasons",      nargs="+", default=ALL_SEASON_CODES, help="Temporadas específicas")
    parser.add_argument("--no-download",  action="store_true", help="Solo cargar archivos locales, no descargar")
    parser.add_argument("--current-only", action="store_true", help="Solo actualizar temporada actual (uso diario)")
    args = parser.parse_args()

    leagues = [args.league] if args.league else list(LEAGUES.keys())
    seasons = [CURRENT_SEASON_CODE] if args.current_only else args.seasons

    run(
        leagues_to_run = leagues,
        season_codes   = seasons,
        download       = not args.no_download,
    )