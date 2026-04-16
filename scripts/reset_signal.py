"""
Resetea señales descartadas por confianza baja para que el agente las re-analice.
Uso:
    python scripts/reset_signal.py --fixture "Elversberg"
    python scripts/reset_signal.py --all-low-confidence
"""
import argparse
import sqlite3
from pathlib import Path

DB_PATH = Path("data/betting.db")

def reset(db_path: Path, fixture_keyword: str = None, all_low: bool = False):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if fixture_keyword:
            rows = conn.execute("""
                SELECT s.id, f.home_team_name, f.away_team_name, s.flags, s.edge_pct
                FROM signals s JOIN fixtures f ON s.fixture_id = f.id
                WHERE s.telegram_sent = 2
                  AND (f.home_team_name LIKE ? OR f.away_team_name LIKE ?)
            """, [f"%{fixture_keyword}%", f"%{fixture_keyword}%"]).fetchall()
        elif all_low:
            rows = conn.execute("""
                SELECT s.id, f.home_team_name, f.away_team_name, s.flags, s.edge_pct
                FROM signals s JOIN fixtures f ON s.fixture_id = f.id
                WHERE s.telegram_sent = 2
                  AND s.flags LIKE '%confianza": "baja%'
                  AND f.kickoff_utc > datetime('now')
            """).fetchall()
        else:
            print("Especificar --fixture NOMBRE o --all-low-confidence")
            return

        if not rows:
            print("No se encontraron señales para resetear.")
            return

        for r in rows:
            print(f"  Reseteando: {r['home_team_name']} vs {r['away_team_name']} "
                  f"(id={r['id']}, edge={r['edge_pct']*100:.1f}%)")
            conn.execute("UPDATE signals SET telegram_sent=0, flags=NULL WHERE id=?", [r["id"]])

        print(f"\n✅ {len(rows)} señal(es) reseteada(s) — se re-analizarán en el próximo run.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", help="Palabra clave del equipo (ej: 'Elversberg')")
    parser.add_argument("--all-low-confidence", action="store_true",
                        help="Resetear todas las descartadas con confianza baja")
    parser.add_argument("--db", default=str(DB_PATH))
    args = parser.parse_args()
    reset(Path(args.db), args.fixture, args.all_low_confidence)
