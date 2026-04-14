"""
feature_builder.py — Construye features contextuales para XGBoost.

Toma los datos crudos de la tabla fixtures y genera features
que Dixon-Coles no captura: forma reciente, descanso, H2H, posición.

Los features se guardan en la tabla match_features de betting.db.

Uso:
    python src/features/feature_builder.py --league BL2
    python src/features/feature_builder.py --all
    python src/features/feature_builder.py --league BL2 --season 2025-26
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH = Path("data/betting.db")

LEAGUES = ["BL2", "NL1"]


# ──────────────────────────────────────────────────────────────────────────────
# Carga de datos
# ──────────────────────────────────────────────────────────────────────────────

def load_fixtures(db_path: Path, league: str) -> pd.DataFrame:
    """
    Carga todos los partidos terminados de una liga, ordenados por fecha.
    Incluye partidos de todas las temporadas para poder calcular
    forma reciente incluso al inicio de una temporada nueva.
    """
    query = """
        SELECT
            id,
            league,
            season,
            home_team_name  AS home_team,
            away_team_name  AS away_team,
            kickoff_utc     AS kickoff,
            goals_home,
            goals_away,
            result,
            over_25,
            btts
        FROM fixtures
        WHERE league  = ?
          AND status  = 'finished'
          AND goals_home IS NOT NULL
          AND goals_away IS NOT NULL
        ORDER BY kickoff_utc ASC
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=[league])

    df["kickoff"] = pd.to_datetime(df["kickoff"])
    df["total_goals"] = df["goals_home"] + df["goals_away"]
    df["points_home"] = df["result"].map({"H": 3, "D": 1, "A": 0})
    df["points_away"] = df["result"].map({"H": 0, "D": 1, "A": 3})

    logger.info("[%s] %d partidos cargados", league, len(df))
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Features de forma reciente
# ──────────────────────────────────────────────────────────────────────────────

def compute_team_form(df: pd.DataFrame, n: int = 5) -> dict[tuple, dict]:
    """
    Para cada partido, calcula las estadísticas de forma de los últimos N
    partidos ANTERIORES al partido actual para cada equipo.

    Retorna un dict indexado por (fixture_id, rol) donde rol es 'home' o 'away'.
    """
    # Construir historial de cada equipo partido a partido
    # Cada fila del historial representa la perspectiva de un equipo
    records = []
    for _, row in df.iterrows():
        # Perspectiva del equipo local
        records.append({
            "fixture_id": row["id"],
            "kickoff":    row["kickoff"],
            "team":       row["home_team"],
            "rol":        "home",
            "goles_a_favor":   row["goals_home"],
            "goles_en_contra": row["goals_away"],
            "over_25":         row["over_25"],
            "btts":            row["btts"],
            "puntos":          row["points_home"],
            "es_local":        1,
        })
        # Perspectiva del equipo visitante
        records.append({
            "fixture_id": row["id"],
            "kickoff":    row["kickoff"],
            "team":       row["away_team"],
            "rol":        "away",
            "goles_a_favor":   row["goals_away"],
            "goles_en_contra": row["goals_home"],
            "over_25":         row["over_25"],
            "btts":            row["btts"],
            "puntos":          row["points_away"],
            "es_local":        0,
        })

    hist = pd.DataFrame(records).sort_values("kickoff")

    # Para cada partido, calcular forma de los N partidos anteriores
    form_cache: dict[tuple, dict] = {}

    for _, row in df.iterrows():
        fid     = row["id"]
        kickoff = row["kickoff"]

        for team, rol in [(row["home_team"], "home"), (row["away_team"], "away")]:
            # Partidos anteriores del equipo (antes de este kickoff)
            prev = hist[
                (hist["team"] == team) &
                (hist["kickoff"] < kickoff)
            ].tail(n)

            if len(prev) == 0:
                form_cache[(fid, rol)] = {
                    "goals_scored_avg":    None,
                    "goals_conceded_avg":  None,
                    "over_25_rate":        None,
                    "btts_rate":           None,
                    "points_avg":          None,
                    "n_games":             0,
                }
            else:
                form_cache[(fid, rol)] = {
                    "goals_scored_avg":    prev["goles_a_favor"].mean(),
                    "goals_conceded_avg":  prev["goles_en_contra"].mean(),
                    "over_25_rate":        prev["over_25"].mean(),
                    "btts_rate":           prev["btts"].mean(),
                    "points_avg":          prev["puntos"].mean(),
                    "n_games":             len(prev),
                }

    return form_cache


# ──────────────────────────────────────────────────────────────────────────────
# Features de descanso
# ──────────────────────────────────────────────────────────────────────────────

def compute_rest_days(df: pd.DataFrame) -> dict[tuple, int]:
    """
    Calcula los días de descanso de cada equipo antes de cada partido.
    (días desde su último partido)

    Retorna dict indexado por (fixture_id, 'home'|'away').
    """
    # Historial de partidos por equipo
    team_games: dict[str, list[datetime]] = {}
    for _, row in df.iterrows():
        for team in [row["home_team"], row["away_team"]]:
            if team not in team_games:
                team_games[team] = []
            team_games[team].append(row["kickoff"])

    # Ordenar
    for team in team_games:
        team_games[team] = sorted(team_games[team])

    rest_cache: dict[tuple, int] = {}

    for _, row in df.iterrows():
        fid     = row["id"]
        kickoff = row["kickoff"]

        for team, rol in [(row["home_team"], "home"), (row["away_team"], "away")]:
            games = [g for g in team_games.get(team, []) if g < kickoff]
            if games:
                rest = (kickoff - games[-1]).days
            else:
                rest = 14  # default si no hay partido previo (inicio de temporada)
            rest_cache[(fid, rol)] = rest

    return rest_cache


# ──────────────────────────────────────────────────────────────────────────────
# Features H2H
# ──────────────────────────────────────────────────────────────────────────────

def compute_h2h(df: pd.DataFrame, n: int = 10) -> dict[int, dict]:
    """
    Para cada partido, calcula el historial H2H entre los dos equipos
    en los N enfrentamientos anteriores (en cualquier dirección).

    Retorna dict indexado por fixture_id.
    """
    h2h_cache: dict[int, dict] = {}

    for _, row in df.iterrows():
        fid     = row["id"]
        kickoff = row["kickoff"]
        home    = row["home_team"]
        away    = row["away_team"]

        # Partidos previos entre estos dos equipos (cualquier dirección)
        prev = df[
            (df["kickoff"] < kickoff) &
            (
                ((df["home_team"] == home) & (df["away_team"] == away)) |
                ((df["home_team"] == away) & (df["away_team"] == home))
            )
        ].tail(n)

        if len(prev) == 0:
            h2h_cache[fid] = {
                "h2h_over_25_rate": None,
                "h2h_btts_rate":    None,
                "h2h_goals_avg":    None,
                "h2h_n_games":      0,
            }
        else:
            h2h_cache[fid] = {
                "h2h_over_25_rate": prev["over_25"].mean(),
                "h2h_btts_rate":    prev["btts"].mean(),
                "h2h_goals_avg":    prev["total_goals"].mean(),
                "h2h_n_games":      len(prev),
            }

    return h2h_cache


# ──────────────────────────────────────────────────────────────────────────────
# Features de posición en tabla
# ──────────────────────────────────────────────────────────────────────────────

def compute_table_position(df: pd.DataFrame) -> dict[tuple, dict]:
    """
    Para cada partido, calcula la posición en tabla de cada equipo
    basándose en los puntos acumulados en la temporada hasta ese momento.

    Retorna dict indexado por (fixture_id, 'home'|'away').
    """
    position_cache: dict[tuple, dict] = {}

    # Procesar por temporada
    for season in df["season"].unique():
        season_df = df[df["season"] == season].sort_values("kickoff")

        # Acumular puntos partido a partido
        team_points: dict[str, int] = {}
        team_played: dict[str, int] = {}

        for _, row in season_df.iterrows():
            fid   = row["id"]
            home  = row["home_team"]
            away  = row["away_team"]

            # Guardar posición ANTES de este partido
            all_teams = set(team_points.keys())
            if home not in all_teams and away not in all_teams:
                # Primer partido de la temporada
                for team, rol in [(home, "home"), (away, "away")]:
                    position_cache[(fid, rol)] = {
                        "team_points":   0,
                        "team_played":   0,
                        "team_position": None,
                        "opponent_points": 0,
                    }
            else:
                # Calcular ranking actual
                ranking = sorted(
                    team_points.items(), key=lambda x: -x[1]
                )
                pos_map = {team: i + 1 for i, (team, _) in enumerate(ranking)}

                for team, rol, opp in [
                    (home, "home", away),
                    (away, "away", home),
                ]:
                    position_cache[(fid, rol)] = {
                        "team_points":    team_points.get(team, 0),
                        "team_played":    team_played.get(team, 0),
                        "team_position":  pos_map.get(team, None),
                        "opponent_points": team_points.get(opp, 0),
                    }

            # Actualizar puntos después del partido
            team_points[home] = team_points.get(home, 0) + row["points_home"]
            team_points[away] = team_points.get(away, 0) + row["points_away"]
            team_played[home] = team_played.get(home, 0) + 1
            team_played[away] = team_played.get(away, 0) + 1

    return position_cache


# ──────────────────────────────────────────────────────────────────────────────
# Ensamblado y guardado
# ──────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combina todos los features en un DataFrame listo para XGBoost.
    """
    logger.info("Calculando forma reciente...")
    form = compute_team_form(df, n=5)

    logger.info("Calculando días de descanso...")
    rest = compute_rest_days(df)

    logger.info("Calculando H2H...")
    h2h = compute_h2h(df, n=10)

    logger.info("Calculando posición en tabla...")
    pos = compute_table_position(df)

    rows = []
    for _, row in df.iterrows():
        fid = row["id"]

        home_form = form.get((fid, "home"), {})
        away_form = form.get((fid, "away"), {})
        h2h_data  = h2h.get(fid, {})
        home_pos  = pos.get((fid, "home"), {})
        away_pos  = pos.get((fid, "away"), {})

        rows.append({
            "fixture_id": fid,
            "league":     row["league"],
            "season":     row["season"],
            "kickoff":    row["kickoff"].isoformat(),

            # Forma local
            "home_goals_scored_avg_5":   home_form.get("goals_scored_avg"),
            "home_goals_conceded_avg_5": home_form.get("goals_conceded_avg"),
            "home_over_25_rate_5":       home_form.get("over_25_rate"),
            "home_btts_rate_5":          home_form.get("btts_rate"),
            "home_points_avg_5":         home_form.get("points_avg"),
            "home_n_games_5":            home_form.get("n_games", 0),

            # Forma visitante
            "away_goals_scored_avg_5":   away_form.get("goals_scored_avg"),
            "away_goals_conceded_avg_5": away_form.get("goals_conceded_avg"),
            "away_over_25_rate_5":       away_form.get("over_25_rate"),
            "away_btts_rate_5":          away_form.get("btts_rate"),
            "away_points_avg_5":         away_form.get("points_avg"),
            "away_n_games_5":            away_form.get("n_games", 0),

            # Suma de promedios de goles (proxy de partido abierto)
            "total_goals_exp":           (
                (home_form.get("goals_scored_avg") or 0) +
                (away_form.get("goals_scored_avg") or 0)
            ),

            # Descanso
            "home_rest_days": rest.get((fid, "home"), 7),
            "away_rest_days": rest.get((fid, "away"), 7),
            "rest_diff":      rest.get((fid, "home"), 7) - rest.get((fid, "away"), 7),

            # H2H
            "h2h_over_25_rate": h2h_data.get("h2h_over_25_rate"),
            "h2h_btts_rate":    h2h_data.get("h2h_btts_rate"),
            "h2h_goals_avg":    h2h_data.get("h2h_goals_avg"),
            "h2h_n_games":      h2h_data.get("h2h_n_games", 0),

            # Posición en tabla
            "home_points":          home_pos.get("team_points", 0),
            "away_points":          away_pos.get("team_points", 0),
            "home_position":        home_pos.get("team_position"),
            "away_position":        away_pos.get("team_position"),
            "points_diff":          home_pos.get("team_points", 0) - away_pos.get("team_points", 0),

            # Target
            "over_25": row["over_25"],
            "btts":    row["btts"],
        })

    features_df = pd.DataFrame(rows)
    logger.info("Features construidas: %d partidos, %d columnas", len(features_df), len(features_df.columns))
    return features_df


def save_features(features_df: pd.DataFrame, db_path: Path) -> None:
    """
    Guarda los features en la tabla match_features usando upsert.
    Solo guarda los partidos que tienen target (over_25 no nulo).
    """
    df = features_df[features_df["over_25"].notna()].copy()

    with sqlite3.connect(db_path) as conn:
        # Limpiar registros anteriores de estas ligas/temporadas
        leagues  = df["league"].unique().tolist()
        seasons  = df["season"].unique().tolist()
        ph_l = ",".join("?" for _ in leagues)
        ph_s = ",".join("?" for _ in seasons)
        conn.execute(
            f"DELETE FROM match_features WHERE fixture_id IN "
            f"(SELECT id FROM fixtures WHERE league IN ({ph_l}) AND season IN ({ph_s}))",
            leagues + seasons,
        )

        rows = []
        for _, row in df.iterrows():
            rows.append((
                int(row["fixture_id"]),
                row.get("home_goals_scored_avg_5"),
                row.get("away_goals_scored_avg_5"),
                row.get("home_goals_conceded_avg_5"),
                row.get("away_goals_conceded_avg_5"),
                row.get("home_over_25_rate_5"),
                row.get("away_over_25_rate_5"),
                row.get("h2h_over_25_rate"),
                int(row.get("h2h_n_games", 0)),
                int(row.get("home_rest_days", 7)),
                int(row.get("away_rest_days", 7)),
                None,  # home_key_attackers_out (lesiones — Fase 2)
                None,  # away_key_attackers_out
                None,  # home_keeper_out
                None,  # away_keeper_out
                row.get("home_position"),
                row.get("away_position"),
                0,     # lineup_confirmed
                datetime.utcnow().isoformat(),
            ))

        conn.executemany("""
            INSERT OR REPLACE INTO match_features (
                fixture_id,
                home_goals_avg_5, away_goals_avg_5,
                home_conceded_avg_5, away_conceded_avg_5,
                home_ou25_rate_5, away_ou25_rate_5,
                h2h_ou25_rate, h2h_matches,
                home_rest_days, away_rest_days,
                home_key_attackers_out, away_key_attackers_out,
                home_keeper_out, away_keeper_out,
                home_position, away_position,
                lineup_confirmed,
                computed_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, rows)

    logger.info("Guardados %d registros en match_features", len(rows))


def save_features_csv(features_df: pd.DataFrame, league: str) -> Path:
    """Guarda los features en CSV para inspección manual."""
    Path("reports").mkdir(exist_ok=True)
    path = Path(f"reports/features_{league}.csv")
    features_df.to_csv(path, index=False)
    logger.info("Features exportadas a %s", path)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def run(league: str, db_path: Path = DB_PATH, export_csv: bool = False) -> pd.DataFrame:
    """Entry point para uso desde value_engine o CLI."""
    df = load_fixtures(db_path, league)
    features_df = build_features(df)

    try:
        save_features(features_df, db_path)
    except Exception as e:
        logger.warning("No se pudo guardar en match_features: %s. Continuando con CSV.", e)

    if export_csv:
        save_features_csv(features_df, league)

    return features_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Feature builder para XGBoost")
    parser.add_argument("--league",  choices=["BL2", "NL1"], help="Liga específica")
    parser.add_argument("--all",     action="store_true",    help="Todas las ligas")
    parser.add_argument("--db",      default=str(DB_PATH))
    parser.add_argument("--csv",     action="store_true",    help="Exportar CSV para inspección")
    args = parser.parse_args()

    leagues = LEAGUES if args.all else ([args.league] if args.league else None)
    if not leagues:
        parser.error("Especificá --league o --all")

    for league in leagues:
        print(f"\n{'='*50}")
        print(f"  Construyendo features: {league}")
        print(f"{'='*50}")
        features_df = run(league, Path(args.db), export_csv=args.csv)

        # Resumen
        df_fin = features_df[features_df["over_25"].notna()]
        print(f"\n  Partidos con features completas: {len(df_fin)}")
        print(f"  Over 2.5 rate real:              {df_fin['over_25'].mean()*100:.1f}%")
        print(f"  Temporadas cubiertas:            {sorted(df_fin['season'].unique())}")

        # Muestra de features del último partido
        last = df_fin.iloc[-1]
        print(f"\n  Último partido en features:")
        print(f"    Kickoff:              {last['kickoff'][:10]}")
        print(f"    Home goals avg (5):   {last['home_goals_scored_avg_5']:.2f}" if last['home_goals_scored_avg_5'] else "    Home goals avg (5):   N/A")
        print(f"    Away goals avg (5):   {last['away_goals_scored_avg_5']:.2f}" if last['away_goals_scored_avg_5'] else "    Away goals avg (5):   N/A")
        print(f"    Home rest days:       {last['home_rest_days']}")
        print(f"    Away rest days:       {last['away_rest_days']}")
        print(f"    H2H over rate:        {last['h2h_over_25_rate']:.2f}" if last['h2h_over_25_rate'] else "    H2H over rate:        N/A")
        print(f"    Points diff:          {last['points_diff']:.0f}")
        print()