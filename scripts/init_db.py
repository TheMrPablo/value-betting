"""
Inicializa el schema completo de la base de datos SQLite.
Correr una sola vez al inicio del proyecto.
"""
import sqlite3
import os
from pathlib import Path

DB_PATH = Path("data/betting.db")

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # --- EQUIPOS ---
    c.execute("""
    CREATE TABLE IF NOT EXISTS teams (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        name_normalized TEXT,
        league TEXT NOT NULL,
        country TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    # --- PARTIDOS ---
    c.execute("""
    CREATE TABLE IF NOT EXISTS fixtures (
        id INTEGER PRIMARY KEY,
        league TEXT NOT NULL,
        season TEXT NOT NULL,
        home_team_id INTEGER REFERENCES teams(id),
        away_team_id INTEGER REFERENCES teams(id),
        home_team_name TEXT,
        away_team_name TEXT,
        kickoff_utc TIMESTAMP NOT NULL,
        matchweek INTEGER,
        status TEXT DEFAULT 'scheduled',
        goals_home INTEGER,
        goals_away INTEGER,
        result TEXT,
        over_25 INTEGER,
        btts INTEGER,
        source_fixture_id TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    # --- CUOTAS ---
    c.execute("""
    CREATE TABLE IF NOT EXISTS odds (
        id INTEGER PRIMARY KEY,
        fixture_id INTEGER REFERENCES fixtures(id),
        bookmaker TEXT DEFAULT 'betplay',
        market TEXT NOT NULL,
        selection TEXT NOT NULL,
        odds_decimal REAL NOT NULL,
        implied_prob REAL,
        overround_pct REAL,
        captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        hours_before_kickoff REAL
    )""")

    # --- LESIONES ---
    c.execute("""
    CREATE TABLE IF NOT EXISTS injuries (
        id INTEGER PRIMARY KEY,
        team_id INTEGER REFERENCES teams(id),
        fixture_id INTEGER REFERENCES fixtures(id),
        player_name TEXT,
        status TEXT,
        is_key_attacker INTEGER DEFAULT 0,
        is_key_keeper INTEGER DEFAULT 0,
        source TEXT,
        confirmed INTEGER DEFAULT 0,
        captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    # --- EVENTOS DE NOTICIAS ---
    c.execute("""
    CREATE TABLE IF NOT EXISTS source_events (
        id INTEGER PRIMARY KEY,
        fixture_id INTEGER REFERENCES fixtures(id),
        source_name TEXT NOT NULL,
        event_type TEXT NOT NULL,
        raw_text TEXT,
        translated_text TEXT,
        language TEXT,
        reliability INTEGER,
        captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        hours_before_kickoff REAL
    )""")

    # --- CLIMA (Eliteserien fase 2) ---
    c.execute("""
    CREATE TABLE IF NOT EXISTS weather (
        id INTEGER PRIMARY KEY,
        fixture_id INTEGER REFERENCES fixtures(id),
        temperature_c REAL,
        wind_kmh REAL,
        precipitation_mm REAL,
        heavy_rain INTEGER DEFAULT 0,
        high_wind INTEGER DEFAULT 0,
        snow INTEGER DEFAULT 0,
        captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    # --- FEATURES PARA XGBOOST ---
    c.execute("""
    CREATE TABLE IF NOT EXISTS match_features (
        id INTEGER PRIMARY KEY,
        fixture_id INTEGER UNIQUE REFERENCES fixtures(id),
        home_goals_avg_5 REAL,
        away_goals_avg_5 REAL,
        home_conceded_avg_5 REAL,
        away_conceded_avg_5 REAL,
        home_ou25_rate_5 REAL,
        away_ou25_rate_5 REAL,
        h2h_ou25_rate REAL,
        h2h_matches INTEGER,
        home_rest_days INTEGER,
        away_rest_days INTEGER,
        home_key_attackers_out INTEGER DEFAULT 0,
        away_key_attackers_out INTEGER DEFAULT 0,
        home_keeper_out INTEGER DEFAULT 0,
        away_keeper_out INTEGER DEFAULT 0,
        home_position INTEGER,
        away_position INTEGER,
        lineup_confirmed INTEGER DEFAULT 0,
        home_goals_home_avg_5 REAL,
        home_conceded_home_avg_5 REAL,
        away_goals_away_avg_5 REAL,
        away_conceded_away_avg_5 REAL,
        computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    # --- PREDICCIONES ---
    c.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY,
        fixture_id INTEGER REFERENCES fixtures(id),
        model TEXT NOT NULL,
        market TEXT NOT NULL,
        selection TEXT NOT NULL,
        prob REAL NOT NULL,
        lambda_home REAL,
        lambda_away REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(fixture_id, model, market, selection)
    )""")

    # --- SEÑALES DE VALUE ---
    c.execute("""
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY,
        fixture_id INTEGER REFERENCES fixtures(id),
        market TEXT NOT NULL,
        selection TEXT NOT NULL,
        edge_pct REAL NOT NULL,
        prob_ensemble REAL NOT NULL,
        betplay_odds REAL NOT NULL,
        overround_pct REAL NOT NULL,
        stake_cop REAL NOT NULL,
        kelly_fraction REAL NOT NULL,
        profile TEXT NOT NULL,
        status TEXT NOT NULL,
        flags TEXT,
        telegram_sent INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(fixture_id, market, selection)
    )""")

    # --- APUESTAS EJECUTADAS ---
    c.execute("""
    CREATE TABLE IF NOT EXISTS bets (
        id INTEGER PRIMARY KEY,
        signal_id INTEGER REFERENCES signals(id),
        fixture_id INTEGER REFERENCES fixtures(id),
        user_decision TEXT NOT NULL,
        stake_placed_cop REAL,
        odds_taken REAL,
        bankroll_before REAL,
        decided_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        notes TEXT
    )""")

    # --- RESULTADOS Y P&L ---
    c.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY,
        bet_id INTEGER UNIQUE REFERENCES bets(id),
        fixture_id INTEGER REFERENCES fixtures(id),
        outcome TEXT NOT NULL,
        pnl_cop REAL NOT NULL,
        bankroll_after REAL,
        closing_odds REAL,
        closing_line_value REAL,
        settled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    # --- SNAPSHOTS SEMANALES ---
    c.execute("""
    CREATE TABLE IF NOT EXISTS performance_snapshots (
        id INTEGER PRIMARY KEY,
        snapshot_date DATE NOT NULL,
        league TEXT,
        total_bets INTEGER,
        wins INTEGER,
        losses INTEGER,
        roi_pct REAL,
        pnl_cop REAL,
        bankroll REAL,
        brier_score REAL,
        avg_edge REAL,
        avg_clv REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    # --- ÍNDICES ---
    c.execute("CREATE INDEX IF NOT EXISTS idx_fixtures_league_kickoff ON fixtures(league, kickoff_utc)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_odds_fixture_market ON odds(fixture_id, market, captured_at)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_source_events_fixture ON source_events(fixture_id, event_type)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_bets_decided ON bets(decided_at)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status, created_at)")

    # --- MIGRACIONES para DBs existentes ---
    # SQLite soporta ALTER TABLE ADD COLUMN (pero no ADD CONSTRAINT).
    # Aplicamos las columnas nuevas de venue-form de forma segura.
    venue_cols = [
        ("home_goals_home_avg_5",    "REAL"),
        ("home_conceded_home_avg_5", "REAL"),
        ("away_goals_away_avg_5",    "REAL"),
        ("away_conceded_away_avg_5", "REAL"),
    ]
    for col, col_type in venue_cols:
        try:
            c.execute(f"ALTER TABLE match_features ADD COLUMN {col} {col_type}")
        except Exception:
            pass  # columna ya existe

    conn.commit()
    conn.close()
    print("✅ Base de datos inicializada correctamente en data/betting.db")
    print(f"   Tablas creadas: teams, fixtures, odds, injuries, source_events,")
    print(f"   weather, match_features, predictions, signals, bets, results,")
    print(f"   performance_snapshots")

if __name__ == "__main__":
    init_db()