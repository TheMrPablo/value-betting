"""
telegram_bot.py — Bot de Telegram para alertas de value betting.

Mensajes simplificados en lenguaje claro + botones interactivos inline.
El usuario ve exactamente qué apostar, cuánto, y por qué en 3 líneas.

Uso:
    python src/output/telegram_bot.py --send
    python src/output/telegram_bot.py --listen
    python src/output/telegram_bot.py --weekly
    python src/output/telegram_bot.py --test
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

DB_PATH    = Path("data/betting.db")
COT_OFFSET = timedelta(hours=-5)  # Colombia = UTC-5

DIAS_ES = {
    "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
    "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo",
}
MESES_ES = {
    "Jan": "Ene", "Feb": "Feb", "Mar": "Mar", "Apr": "Abr",
    "May": "May", "Jun": "Jun", "Jul": "Jul", "Aug": "Ago",
    "Sep": "Sep", "Oct": "Oct", "Nov": "Nov", "Dec": "Dic",
}


# ──────────────────────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    import os
    token   = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat_id:
        return {"token": token, "chat_id": chat_id}
    config_path = Path("config/telegram.json")
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    raise RuntimeError("No se encontró config/telegram.json")


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades de fecha
# ──────────────────────────────────────────────────────────────────────────────

def utc_to_cot(utc_str: str) -> str:
    """Convierte timestamp UTC a hora COT formateada en español."""
    try:
        dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        cot  = dt + COT_OFFSET
        dia  = DIAS_ES.get(cot.strftime("%A"), cot.strftime("%A"))
        mes  = MESES_ES.get(cot.strftime("%b"), cot.strftime("%b"))
        hora = cot.strftime("%I:%M%p").lstrip("0").lower()
        return f"{dia} {cot.day} {mes} · {hora} (COT)"
    except (TypeError, ValueError):
        return utc_str


# ──────────────────────────────────────────────────────────────────────────────
# Formato de mensajes
# ──────────────────────────────────────────────────────────────────────────────

def _parse_flags(raw_flags) -> tuple[list, str, str]:
    """
    Parsea el campo flags de la DB.
    Retorna (flags_detail, agent_analysis, agent_news).
    """
    if isinstance(raw_flags, str):
        try:
            raw_flags = json.loads(raw_flags)
        except Exception:
            return [], "", ""
    if isinstance(raw_flags, dict):
        return (
            raw_flags.get("flags", []),
            raw_flags.get("agent_analysis", ""),
            raw_flags.get("agent_news", ""),
        )
    return raw_flags if isinstance(raw_flags, list) else [], "", ""


def format_summary_message(signals: list[dict], bankroll: float, n_bets: int) -> str:
    """
    Un solo mensaje resumen con TODAS las señales del día.

    Estructura:
      Header (liga + jornada)
      ✅ APOSTAR — señales BET con instrucción clara
      ━━ separador ━━
      🟡 A TU CRITERIO — señales REVIEW con análisis de Haiku
      ━━ separador ━━
      📝 Contexto — noticias relevantes
      Footer (bankroll + exposición)
    """
    if not signals:
        return ""

    bet_signals    = [s for s in signals if s["status"] == "BET"]
    review_signals = [s for s in signals if s["status"] == "REVIEW"]

    league = signals[0]["league"]
    liga   = {"BL2": "Bundesliga 2", "NL1": "Eredivisie"}.get(league, league)
    mw     = signals[0].get("matchweek")

    header = f"🎯 <b>{liga}"
    if mw:
        header += f" — Jornada {mw}"
    header += "</b>"

    lines = [header, ""]

    # ── BET: instrucciones claras ──
    if bet_signals:
        lines.append("✅ <b>APOSTAR</b>")
        lines.append("")
        for s in bet_signals:
            sel_txt  = "Over 2.5 goles" if s["selection"] == "over" else "Under 2.5 goles"
            ganancia = round(s["stake_cop"] * (s["betplay_odds"] - 1))
            kickoff  = utc_to_cot(s.get("kickoff_utc", ""))
            flags_detail, _, _ = _parse_flags(s.get("flags"))
            flags_imp = [f for f in flags_detail
                         if isinstance(f, dict) and f.get("code") not in ("BURN_IN",)]

            lines.append(f"🔵 <b>{s['home_team']} vs {s['away_team']}</b>")
            lines.append(f"   {kickoff}")
            lines.append(f"   {sel_txt}  ·  Cuota <b>{s['betplay_odds']}</b>")
            lines.append(f"   Apostá <b>${s['stake_cop']:,.0f} COP</b>  →  "
                         f"ganás +${ganancia:,.0f}  /  perdés -${s['stake_cop']:,.0f}")
            if flags_imp:
                for f in flags_imp:
                    lines.append(f"   ⚠️ <i>{f.get('message', '')}</i>")
            lines.append("")

    # ── REVIEW: a tu criterio ──
    if review_signals:
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append("🟡 <b>A TU CRITERIO</b>")
        lines.append("")
        for s in review_signals:
            sel_txt       = "Over 2.5" if s["selection"] == "over" else "Under 2.5"
            kickoff       = utc_to_cot(s.get("kickoff_utc", ""))
            _, agent_txt, _ = _parse_flags(s.get("flags"))

            lines.append(f"<b>{s['home_team']} vs {s['away_team']}</b>")
            lines.append(f"   {kickoff}")
            lines.append(f"   {sel_txt}  ·  Cuota {s['betplay_odds']}  "
                         f"·  Edge +{s['edge_pct']*100:.0f}%")
            if agent_txt:
                lines.append(f"   <i>{agent_txt}</i>")
            lines.append("")

    # ── Contexto: noticias del agente ──
    context_lines = []
    for s in signals:
        _, _, agent_news = _parse_flags(s.get("flags"))
        if agent_news and "sin noticias" not in agent_news.lower():
            home_short = s["home_team"].split()[0]
            away_short = s["away_team"].split()[0]
            news_short = agent_news[:150].rstrip()
            context_lines.append(f"• <i>{home_short} vs {away_short}: {news_short}</i>")

    if context_lines:
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append("📝 <b>Contexto</b>")
        lines += context_lines
        lines.append("")

    # ── Footer ──
    total_stake  = sum(s["stake_cop"] for s in bet_signals)
    exposure_pct = total_stake / bankroll * 100 if bankroll > 0 else 0

    if n_bets < 50:
        lines.append(f"⚠️ <i>Burn-in: apuesta {n_bets + 1}/50 — stake a 1/8 Kelly</i>")

    lines.append(
        f"💰 Bankroll <b>${bankroll:,.0f} COP</b>  ·  "
        f"Apostando ${total_stake:,.0f} ({exposure_pct:.1f}%)"
    )

    return "\n".join(lines)


def build_summary_keyboard(signals: list[dict]) -> dict:
    """
    Teclado inline con una fila por señal BET.
    Cada fila: [✅ Equipo Sel $X  GO:id]  [½ $X/2  HALF:id]
    """
    rows = []
    for s in signals:
        if s["status"] != "BET":
            continue
        sid        = s["signal_id"]
        stake      = s["stake_cop"]
        half       = round(stake / 2 / 500) * 500
        home_short = s["home_team"].split()[0]
        sel_icon   = "↑" if s["selection"] == "over" else "↓"
        rows.append([
            {"text": f"✅ {home_short} {sel_icon} ${stake:,.0f}",
             "callback_data": f"GO:{sid}"},
            {"text": f"½ ${half:,.0f}",
             "callback_data": f"HALF:{sid}"},
        ])
    return {"inline_keyboard": rows}


def format_weekly_summary(stats: dict) -> str:
    pnl      = stats.get("pnl_cop", 0)
    roi      = stats.get("roi_pct", 0)
    roi_icon = "📈" if roi >= 0 else "📉"
    pnl_sign = "+" if pnl >= 0 else ""

    msg = (
        f"📊 <b>Resumen semanal</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Apuestas: {stats.get('total_bets', 0)} "
        f"({stats.get('wins', 0)} ganadas · {stats.get('losses', 0)} perdidas)\n"
        f"P&L: <b>{pnl_sign}${pnl:,.0f} COP</b>\n"
        f"{roi_icon} ROI: {roi:+.1f}%\n"
        f"\n"
        f"💰 Bankroll: ${stats.get('bankroll', 0):,.0f} COP\n"
    )
    for liga, d in stats.get("by_league", {}).items():
        msg += f"  {liga}: {d['bets']} apuestas · ROI {d['roi_pct']:+.1f}%\n"
    return msg


# ──────────────────────────────────────────────────────────────────────────────
# Cliente Telegram
# ──────────────────────────────────────────────────────────────────────────────

class TelegramClient:

    def __init__(self, token: str):
        self.token  = token
        self.base   = f"https://api.telegram.org/bot{token}"
        self.client = httpx.Client(timeout=30)

    def send_message(self, chat_id: str, text: str,
                     parse_mode: str = "HTML", reply_markup: dict = None) -> dict:
        payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
        if reply_markup:
            payload["reply_markup"] = json.dumps(reply_markup)
        r = self.client.post(f"{self.base}/sendMessage", json=payload)
        r.raise_for_status()
        return r.json()

    def answer_callback(self, callback_query_id: str, text: str = "") -> None:
        self.client.post(f"{self.base}/answerCallbackQuery",
                         json={"callback_query_id": callback_query_id, "text": text})

    def edit_message_reply_markup(self, chat_id: str, message_id: int,
                                  reply_markup: dict = None) -> None:
        self.client.post(f"{self.base}/editMessageReplyMarkup", json={
            "chat_id": chat_id, "message_id": message_id,
            "reply_markup": json.dumps(reply_markup or {"inline_keyboard": []}),
        })

    def get_updates(self, offset: int = 0, timeout: int = 5) -> list[dict]:
        r = self.client.get(f"{self.base}/getUpdates",
                            params={"offset": offset, "timeout": timeout})
        r.raise_for_status()
        return r.json().get("result", [])

    def close(self):
        self.client.close()


# ──────────────────────────────────────────────────────────────────────────────
# Envío de señales
# ──────────────────────────────────────────────────────────────────────────────

def send_pending_signals(db_path: Path, client: TelegramClient, chat_id: str) -> int:
    """
    Envía UN solo mensaje resumen por liga con todas las señales pendientes.
    BET = instrucción clara + botones.  REVIEW = a tu criterio, sin botones.
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT s.id AS signal_id, s.fixture_id, s.selection,
                   s.edge_pct, s.prob_ensemble, s.betplay_odds,
                   s.stake_cop, s.kelly_fraction, s.profile, s.status, s.flags,
                   f.league, f.matchweek,
                   f.home_team_name AS home_team,
                   f.away_team_name AS away_team,
                   f.kickoff_utc
            FROM signals s
            JOIN fixtures f ON s.fixture_id = f.id
            WHERE s.telegram_sent = 0 AND s.status IN ('BET', 'REVIEW')
            ORDER BY
                CASE WHEN s.status = 'BET' THEN 0 ELSE 1 END,
                f.kickoff_utc ASC
        """).fetchall()

        try:
            bankroll = conn.execute("""
                SELECT COALESCE(
                    (SELECT bankroll FROM performance_snapshots
                     ORDER BY snapshot_date DESC LIMIT 1),
                    500000)
            """).fetchone()[0]
        except Exception:
            bankroll = 500_000.0

        n_bets = conn.execute(
            "SELECT COUNT(*) FROM bets WHERE user_decision='GO'"
        ).fetchone()[0]

    if not rows:
        logger.info("Sin señales pendientes para enviar.")
        return 0

    # Agrupar por liga (por si en el futuro hay NL1 + BL2 en el mismo run)
    from collections import defaultdict
    by_league: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_league[row["league"]].append(dict(row))

    total_sent = 0
    for league, signals in by_league.items():
        try:
            msg     = format_summary_message(signals, bankroll, n_bets)
            buttons = build_summary_keyboard(signals)
            keyboard = buttons if buttons["inline_keyboard"] else None

            client.send_message(chat_id, msg, reply_markup=keyboard)

            signal_ids   = [s["signal_id"] for s in signals]
            placeholders = ",".join("?" * len(signal_ids))
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    f"UPDATE signals SET telegram_sent=1 WHERE id IN ({placeholders})",
                    signal_ids,
                )
            total_sent += len(signals)
            logger.info("[%s] Resumen enviado: %d señales (%d BET, %d REVIEW)",
                        league,
                        len(signals),
                        sum(1 for s in signals if s["status"] == "BET"),
                        sum(1 for s in signals if s["status"] == "REVIEW"))
        except Exception as e:
            logger.error("[%s] Error enviando resumen: %s", league, e)

    return total_sent


# ──────────────────────────────────────────────────────────────────────────────
# Procesamiento de botones
# ──────────────────────────────────────────────────────────────────────────────

def process_callback(callback: dict, db_path: Path,
                     client: TelegramClient, chat_id: str) -> bool:
    cb_id      = callback["id"]
    data       = callback.get("data", "")
    message    = callback.get("message", {})
    message_id = message.get("message_id")

    if ":" not in data:
        return False
    action, sid_str = data.split(":", 1)
    try:
        signal_id = int(sid_str)
    except ValueError:
        return False
    if action not in ("GO", "SKIP", "HALF"):
        return False

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        signal = conn.execute("SELECT * FROM signals WHERE id=?", [signal_id]).fetchone()
    if not signal:
        client.answer_callback(cb_id, "❌ Señal no encontrada")
        return False

    signal = dict(signal)
    if action == "GO":
        stake, decision = signal["stake_cop"], "GO"
    elif action == "HALF":
        stake, decision = round(signal["stake_cop"] / 2 / 500) * 500, "GO"
    else:
        stake, decision = 0, "SKIP"

    with sqlite3.connect(db_path) as conn:
        existing = conn.execute("SELECT id FROM bets WHERE signal_id=?",
                                [signal_id]).fetchone()
        if not existing:
            conn.execute("""
                INSERT INTO bets (signal_id, fixture_id, user_decision,
                    stake_placed_cop, odds_taken, decided_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
            """, (signal_id, signal["fixture_id"], decision,
                  stake if decision == "GO" else None,
                  signal["betplay_odds"] if decision == "GO" else None))

    # Toast de confirmación (aparece 2-3s en Telegram, no ocupa el chat)
    if decision == "GO":
        client.answer_callback(cb_id, f"✅ Registrado ${stake:,.0f} COP")
    else:
        client.answer_callback(cb_id, "❌ Pasado")

    # No editamos el teclado para que los otros botones del resumen sigan activos

    logger.info("Callback %s — señal %d — $%s COP", action, signal_id, f"{stake:,.0f}" if stake else "0")
    return True


def listen_for_responses(db_path: Path, client: TelegramClient, chat_id: str) -> int:
    offset_file = Path("data/.telegram_offset")
    offset      = int(offset_file.read_text().strip()) if offset_file.exists() else 0
    updates     = client.get_updates(offset=offset, timeout=5)
    processed   = 0

    for update in updates:
        offset = update["update_id"] + 1
        if "callback_query" in update:
            cb      = update["callback_query"]
            cb_chat = str(cb.get("message", {}).get("chat", {}).get("id", ""))
            if cb_chat == str(chat_id):
                if process_callback(cb, db_path, client, chat_id):
                    processed += 1

    offset_file.write_text(str(offset))
    return processed


# ──────────────────────────────────────────────────────────────────────────────
# Resumen semanal
# ──────────────────────────────────────────────────────────────────────────────

def _load_initial_bankroll() -> float:
    """Lee el bankroll inicial desde config/bankroll.yaml."""
    try:
        import yaml
        p = Path("config/bankroll.yaml")
        if p.exists():
            with open(p) as f:
                data = yaml.safe_load(f)
            return float(data.get("current_bankroll_cop", 500_000))
    except Exception:
        pass
    return 500_000.0


def _save_performance_snapshot(db_path: Path, stats: dict) -> None:
    """Persiste un snapshot semanal en performance_snapshots."""
    from datetime import date
    today = date.today().isoformat()

    # Brier score: promedio de (prob_ensemble - resultado)^2 para señales apostadas
    brier = None
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute("""
                SELECT s.prob_ensemble, f.over_25, s.selection
                FROM signals s
                JOIN fixtures f ON s.fixture_id = f.id
                JOIN bets b ON b.signal_id = s.id
                WHERE b.decided_at >= datetime('now', '-7 days')
                  AND b.user_decision = 'GO'
                  AND f.status = 'finished'
                  AND f.over_25 IS NOT NULL
            """).fetchall()
        if rows:
            sq_errors = []
            for prob, over25, sel in rows:
                outcome = 1 if (sel == "over" and over25 == 1) or (sel == "under" and over25 == 0) else 0
                sq_errors.append((prob - outcome) ** 2)
            brier = sum(sq_errors) / len(sq_errors)
    except Exception:
        pass

    avg_edge = stats.get("avg_edge")
    if avg_edge is None:
        try:
            with sqlite3.connect(db_path) as conn:
                row = conn.execute("""
                    SELECT AVG(s.edge_pct)
                    FROM signals s JOIN bets b ON b.signal_id = s.id
                    WHERE b.decided_at >= datetime('now', '-7 days')
                      AND b.user_decision = 'GO'
                """).fetchone()
            avg_edge = row[0] if row and row[0] is not None else 0.0
        except Exception:
            avg_edge = 0.0

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            INSERT OR IGNORE INTO performance_snapshots
                (snapshot_date, league, total_bets, wins, losses,
                 roi_pct, pnl_cop, bankroll, brier_score, avg_edge)
            VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            today,
            stats.get("total_bets", 0),
            stats.get("wins", 0),
            stats.get("losses", 0),
            stats.get("roi_pct", 0.0),
            stats.get("pnl_cop", 0.0),
            stats.get("bankroll", 0.0),
            brier,
            avg_edge,
        ])
    logger.info("Snapshot guardado para %s: ROI %.1f%% | Bankroll $%,.0f",
                today, stats.get("roi_pct", 0), stats.get("bankroll", 0))


def send_weekly_summary(db_path: Path, client: TelegramClient, chat_id: str) -> None:
    with sqlite3.connect(db_path) as conn:
        bets = conn.execute("""
            SELECT b.stake_placed_cop, r.outcome, r.pnl_cop, f.league
            FROM bets b JOIN fixtures f ON b.fixture_id=f.id
            LEFT JOIN results r ON r.bet_id=b.id
            WHERE b.decided_at >= datetime('now', '-7 days') AND b.user_decision='GO'
        """).fetchall()
        last_snapshot = conn.execute("""
            SELECT bankroll FROM performance_snapshots
            ORDER BY snapshot_date DESC LIMIT 1
        """).fetchone()

    # Bankroll: último snapshot → bankroll.yaml → 500K fallback
    prev_bankroll = last_snapshot[0] if last_snapshot else _load_initial_bankroll()

    total = len(bets)
    wins  = sum(1 for b in bets if b[1] == "win")
    pnl   = sum(b[2] or 0 for b in bets)
    stk   = sum(b[0] or 0 for b in bets)
    roi   = (pnl / stk * 100) if stk > 0 else 0.0
    new_bankroll = prev_bankroll + pnl

    by_league: dict = {}
    for b in bets:
        lg = b[3]
        if lg not in by_league:
            by_league[lg] = {"bets": 0, "pnl": 0.0, "staked": 0.0}
        by_league[lg]["bets"]   += 1
        by_league[lg]["pnl"]    += b[2] or 0
        by_league[lg]["staked"] += b[0] or 0
    for lg in by_league:
        s = by_league[lg]["staked"]
        by_league[lg]["roi_pct"] = (by_league[lg]["pnl"] / s * 100) if s > 0 else 0.0

    stats = {
        "total_bets": total, "wins": wins, "losses": total - wins,
        "pnl_cop": pnl, "roi_pct": roi, "bankroll": new_bankroll,
        "by_league": by_league,
    }
    client.send_message(chat_id, format_weekly_summary(stats), parse_mode="HTML")
    _save_performance_snapshot(db_path, stats)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--send",   action="store_true")
    parser.add_argument("--listen", action="store_true")
    parser.add_argument("--weekly", action="store_true")
    parser.add_argument("--test",   action="store_true")
    parser.add_argument("--db",     default=str(DB_PATH))
    args = parser.parse_args()

    config  = load_config()
    client  = TelegramClient(config["token"])
    chat_id = config["chat_id"]
    db_path = Path(args.db)

    try:
        if args.test:
            client.send_message(
                chat_id,
                f"🤖 <b>Bot activo</b>\n"
                f"Hora COT: {utc_to_cot(datetime.now(timezone.utc).isoformat())}",
                parse_mode="HTML",
            )
            print("✅ Test enviado")
        elif args.send:
            n = send_pending_signals(db_path, client, chat_id)
            print(f"✅ {n} señales enviadas")
        elif args.listen:
            n = listen_for_responses(db_path, client, chat_id)
            print(f"✅ {n} respuestas procesadas")
        elif args.weekly:
            send_weekly_summary(db_path, client, chat_id)
            print("✅ Resumen enviado")
        else:
            parser.print_help()
    finally:
        client.close()