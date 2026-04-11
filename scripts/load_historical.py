"""
Carga datos históricos desde archivos locales a SQLite.
Los CSVs deben estar en data/historical/ descargados de football-data.co.uk
"""
import pandas as pd
import sqlite3
import os
from pathlib import Path
from datetime import datetime

DB_PATH = Path("data/betting.db")
HISTORICAL_PATH = Path("data/historical")


def parse_date(date_str: str) -> str | None:
    """Parsea fechas en formato DD/MM/YY o DD/MM/YYYY."""
    for fmt in ["%d/%m/%y", "%d/%m/%Y"]:
        try:
            return datetime.strptime(str(date_str).strip(), fmt).strftime("%Y-%m-%d %H:%M:%S")
        except:
            continue
    return None


def load_from_local(league_code: str, season_code: str) -> pd.DataFrame | None:
    """Lee un CSV ya descargado desde data/historical/."""
    filepath = HISTORICAL_PATH / f"{league_code}_{season_code}.csv"

    if not filepath.exists():
        print(f"  ⚠️  No encontrado: {filepath.name}")
        return None

    try:
        df = pd.read_csv(filepath, encoding="latin-1")
        df = df.dropna(how="all")
        print(f"  ✅ {filepath.name}: {len(df)} partidos")
        return df
    except Exception as e:
        print(f"  ⚠️  {filepath.name}: {e}")
        return None


def load_to_db(df: pd.DataFrame, league_key: str, season: str):
    """Carga un DataFrame a SQLite."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    inserted = 0
    skipped = 0

    for _, row in df.iterrows():
        try:
            home = str(row.get("HomeTeam", "")).strip()
            away = str(row.get("AwayTeam", "")).strip()
            date_str = parse_date(row.get("Date", ""))

            if not home or not away or not date_str:
                skipped += 1
                continue

            goals_home = int(row["FTHG"]) if pd.notna(row.get("FTHG")) else None
            goals_away = int(row["FTAG"]) if pd.notna(row.get("FTAG")) else None
            result = str(row.get("FTR", "")).strip() or None

            over_25 = None
            if goals_home is not None and goals_away is not None:
                over_25 = 1 if (goals_home + goals_away) > 2 else 0

            btts = None
            if goals_home is not None and goals_away is not None:
                btts = 1 if (goals_home > 0 and goals_away > 0) else 0

            c.execute("""
                INSERT OR IGNORE INTO fixtures
                (league, season, home_team_name, away_team_name,
                 kickoff_utc, status, goals_home, goals_away,
                 result, over_25, btts)
                VALUES (?, ?, ?, ?, ?, 'finished', ?, ?, ?, ?, ?)
            """, (league_key, season, home, away, date_str,
                  goals_home, goals_away, result, over_25, btts))

            fixture_id = c.lastrowid

            if pd.notna(row.get("B365>2.5")) and pd.notna(row.get("B365<2.5")):
                odds_over = float(row["B365>2.5"])
                odds_under = float(row["B365<2.5"])
                overround = (1/odds_over + 1/odds_under - 1) * 100

                for selection, odds_val in [("over", odds_over), ("under", odds_under)]:
                    c.execute("""
                        INSERT INTO odds
                        (fixture_id, bookmaker, market, selection,
                         odds_decimal, implied_prob, overround_pct,
                         captured_at, hours_before_kickoff)
                        VALUES (?, 'bet365_historical', 'ou25', ?, ?, ?, ?, ?, -1)
                    """, (fixture_id, selection, odds_val,
                          1/odds_val, overround, date_str))

            inserted += 1

        except Exception:
            skipped += 1
            continue

    conn.commit()
    conn.close()
    print(f"     → {inserted} insertados, {skipped} omitidos")


def main():
    os.makedirs(HISTORICAL_PATH, exist_ok=True)

    print("📥 Cargando datos históricos desde archivos locales...\n")

    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0]
    conn.close()
    print(f"   Partidos en DB antes: {total}\n")

    SEASON_CODES = ["2122", "2223", "2324", "2425"]

    leagues = {
        "BL2": {"name": "2. Bundesliga", "code": "D2"},
        "NL1": {"name": "Eredivisie",    "code": "N1"},
    }

    for league_key, config in leagues.items():
        print(f"🏆 {config['name']} ({league_key})")
        for season_code in SEASON_CODES:
            season_label = f"20{season_code[:2]}-{season_code[2:]}"
            df = load_from_local(config["code"], season_code)
            if df is not None and len(df) > 0:
                load_to_db(df, league_key, season_label)

    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0]
    bl2 = conn.execute("SELECT COUNT(*) FROM fixtures WHERE league='BL2'").fetchone()[0]
    nl1 = conn.execute("SELECT COUNT(*) FROM fixtures WHERE league='NL1'").fetchone()[0]
    over25_bl2 = conn.execute("""
        SELECT ROUND(AVG(over_25)*100, 1) FROM fixtures
        WHERE league='BL2' AND over_25 IS NOT NULL
    """).fetchone()[0]
    over25_nl1 = conn.execute("""
        SELECT ROUND(AVG(over_25)*100, 1) FROM fixtures
        WHERE league='NL1' AND over_25 IS NOT NULL
    """).fetchone()[0]
    conn.close()

    print(f"\n✅ Carga completada:")
    print(f"   Total partidos en DB: {total}")
    print(f"   Bundesliga 2: {bl2} partidos | Over 2.5 rate: {over25_bl2}%")
    print(f"   Eredivisie:   {nl1} partidos | Over 2.5 rate: {over25_nl1}%")
    print(f"\n💡 Sanity check:")
    print(f"   Bundesliga 2 esperado: 52-58% | Eredivisie esperado: 56-62%")


if __name__ == "__main__":
    main()