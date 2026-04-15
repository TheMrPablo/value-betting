"""
player_stats.py — Descarga estadísticas de jugadores desde FBref (vía soccerdata).

Calcula is_key_player para cada jugador según:
  - >= 60% de los minutos posibles del equipo (threshold de Opus/StatsBomb)
  - >= 70% de los partidos como titular
  - Rol ofensivo (FW, AM) para is_key_attacker, o GK para is_key_keeper

Estos datos alimentan al agente Claude para filtrar lesiones irrelevantes
(jugadores que nunca juegan no impactan la predicción O/U 2.5).

Uso:
    python src/ingestion/player_stats.py --league BL2
    python src/ingestion/player_stats.py --league BL2 --season 2024-25
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

# Mapeo liga interna → código FBref usado por soccerdata
FBREF_LEAGUE_MAP = {
    "BL2": "GER-2",
    "NL1": "NED-Eredivisie",
}

# Thresholds recomendados por Opus / literatura StatsBomb
MIN_MINUTES_PCT  = 0.60   # >= 60% de minutos del equipo
MIN_STARTS_PCT   = 0.70   # >= 70% de partidos como titular

# Posiciones FBref que cuentan como "atacante clave" para O/U
KEY_ATTACKER_POSITIONS = {"FW", "MF,FW", "FW,MF", "AM", "MF"}


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Aplana MultiIndex de columnas de FBref a nombres simples."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    # Desduplicar columnas si las hay (FBref a veces repite nombres)
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        idxs = cols[cols == dup].index.tolist()
        for i, idx in enumerate(idxs):
            if i > 0:
                cols[idx] = f"{dup}_{i}"
    df.columns = cols.tolist()
    return df


def fetch_player_stats(league: str, season: str) -> pd.DataFrame:
    """
    Descarga estadísticas de jugadores desde FBref para la liga y temporada dadas.
    Retorna DataFrame con columnas: player, team, position, MP, Starts, Min, Gls, Ast.
    """
    try:
        import soccerdata as sd
    except ImportError:
        raise RuntimeError(
            "soccerdata no está instalado. Corré: pip install soccerdata"
        )

    fbref_league = FBREF_LEAGUE_MAP.get(league)
    if not fbref_league:
        raise ValueError(f"Liga '{league}' no soportada. Disponibles: {list(FBREF_LEAGUE_MAP)}")

    logger.info("[%s] Descargando stats de jugadores desde FBref (%s)...", league, season)

    fbref = sd.FBref(leagues=fbref_league, seasons=season)
    df_raw = fbref.read_player_season_stats(stat_type="standard")

    # Llevar índice a columnas
    df = df_raw.reset_index()
    df = _flatten_columns(df)

    # Normalizar nombres de columnas conocidas (FBref puede cambiarlos entre versiones)
    rename_map = {}
    col_lower = {c.lower(): c for c in df.columns}
    for target, candidates in {
        "player":  ["player"],
        "team":    ["team", "squad", "club"],
        "Pos":     ["pos", "position"],
        "MP":      ["mp", "matches", "matchesplayed"],
        "Starts":  ["starts"],
        "Min":     ["min", "minutes", "mins"],
        "Gls":     ["gls", "goals"],
        "Ast":     ["ast", "assists"],
    }.items():
        for cand in candidates:
            if cand in col_lower and target not in df.columns:
                rename_map[col_lower[cand]] = target
    df = df.rename(columns=rename_map)

    required = ["player", "team", "Pos", "MP", "Starts", "Min"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"FBref devolvió columnas inesperadas. Faltan: {missing}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    # Limpiar tipos
    for col in ["MP", "Starts", "Min", "Gls", "Ast"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    logger.info("[%s] %d jugadores descargados de FBref", league, len(df))
    return df[["player", "team", "Pos", "MP", "Starts", "Min",
               "Gls" if "Gls" in df.columns else "MP",
               "Ast" if "Ast" in df.columns else "MP"]].copy()


def compute_key_players(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega columnas is_key_attacker, is_key_keeper, is_key_player al DataFrame.
    Threshold: >= 60% minutos del equipo AND >= 70% titularidades AND rol ofensivo/GK.
    """
    # Máximo de partidos del equipo = proxy de partidos jugados en la temporada
    team_max_mp = df.groupby("team")["MP"].transform("max")
    team_max_mp = team_max_mp.replace(0, 1)  # evitar división por cero

    df = df.copy()
    df["minutes_pct"] = df["Min"] / (team_max_mp * 90)
    df["starts_pct"]  = df["Starts"] / team_max_mp

    base_threshold = (df["minutes_pct"] >= MIN_MINUTES_PCT) & (df["starts_pct"] >= MIN_STARTS_PCT)

    # Atacante/mediapunta creativo
    pos_series = df["Pos"].fillna("").str.upper()
    is_attacker_role = pos_series.apply(
        lambda p: any(role in p for role in ["FW", "AM"])
    )
    is_keeper_role = pos_series.str.contains("GK", na=False)

    df["is_key_attacker"] = (base_threshold & is_attacker_role).astype(int)
    df["is_key_keeper"]   = (base_threshold & is_keeper_role).astype(int)
    df["is_key_player"]   = ((df["is_key_attacker"] == 1) | (df["is_key_keeper"] == 1)).astype(int)

    n_key = df["is_key_player"].sum()
    logger.info(
        "Jugadores clave: %d (atacantes: %d, porteros: %d)",
        n_key, df["is_key_attacker"].sum(), df["is_key_keeper"].sum()
    )
    return df


def save_player_stats(df: pd.DataFrame, league: str, season: str,
                      db_path: Path = DB_PATH) -> int:
    """Guarda/reemplaza player_stats para la liga+temporada en la DB."""
    now = datetime.utcnow().isoformat()

    rows = []
    for _, row in df.iterrows():
        rows.append((
            str(row["player"]),
            str(row["team"]),
            league,
            season,
            str(row.get("Pos", "")),
            int(row.get("MP", 0)),
            int(row.get("Starts", 0)),
            int(row.get("Min", 0)),
            int(row.get("Gls", 0)),
            int(row.get("Ast", 0)),
            int(row.get("is_key_attacker", 0)),
            int(row.get("is_key_keeper", 0)),
            int(row.get("is_key_player", 0)),
            now,
        ))

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "DELETE FROM player_stats WHERE league = ? AND season = ?",
            [league, season],
        )
        conn.executemany("""
            INSERT OR IGNORE INTO player_stats
                (player_name, team_name, league, season, position,
                 appearances, starts, minutes_played, goals, assists,
                 is_key_attacker, is_key_keeper, is_key_player, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)

    logger.info("[%s] %d jugadores guardados en player_stats", league, len(rows))
    return len(rows)


def get_key_players_for_team(team_name: str, league: str,
                              db_path: Path = DB_PATH) -> dict:
    """
    Devuelve los jugadores clave de un equipo desde la DB.
    Usa búsqueda flexible por las primeras palabras del nombre del equipo.

    Returns:
        {
            "attackers": ["Jugador A", "Jugador B"],
            "keeper":    "Jugador C" | "N/A",
            "found":     True/False
        }
    """
    # Tomar primera palabra del nombre para búsqueda flexible
    # (ej. "Preußen Münster" → "Preußen", "Hannover 96" → "Hannover")
    keyword = team_name.split()[0]

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT player_name, is_key_attacker, is_key_keeper
            FROM player_stats
            WHERE league = ? AND is_key_player = 1
              AND team_name LIKE ?
            ORDER BY is_key_attacker DESC, player_name ASC
        """, [league, f"%{keyword}%"]).fetchall()

    if not rows:
        return {"attackers": [], "keeper": "N/A", "found": False}

    attackers = [r["player_name"] for r in rows if r["is_key_attacker"]]
    keepers   = [r["player_name"] for r in rows if r["is_key_keeper"]]

    return {
        "attackers": attackers,
        "keeper":    keepers[0] if keepers else "N/A",
        "found":     True,
    }


def run(league: str, season: str, db_path: Path = DB_PATH) -> None:
    try:
        df = fetch_player_stats(league, season)
    except Exception as e:
        logger.error("[%s] Error descargando stats de FBref: %s", league, e)
        logger.warning("[%s] Continuando sin actualizar player_stats", league)
        return

    df = compute_key_players(df)
    n = save_player_stats(df, league, season, db_path)

    # Imprimir resumen
    key = df[df["is_key_player"] == 1][["player", "team", "Pos", "Min", "Starts"]].copy()
    key["Min"] = key["Min"].apply(lambda x: f"{x:,}")
    print(f"\n{'='*55}")
    print(f"  Player stats — {league} {season}")
    print(f"{'='*55}")
    print(f"  Total jugadores: {len(df)}")
    print(f"  Jugadores clave: {df['is_key_player'].sum()}")
    print(f"    Atacantes clave: {df['is_key_attacker'].sum()}")
    print(f"    Porteros clave:  {df['is_key_keeper'].sum()}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--league",  default="BL2", choices=["BL2", "NL1"])
    parser.add_argument("--season",  default="2024-25")
    parser.add_argument("--db",      default=str(DB_PATH))
    args = parser.parse_args()

    run(args.league, args.season, Path(args.db))
