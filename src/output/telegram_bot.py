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

def format_signal_message(signal: dict, bankroll: float, n_bets: int) -> str:
    """Mensaje claro y directo — sin jerga técnica."""
    sel      = signal["selection"]
    odds     = signal["betplay_odds"]
    stake    = signal["stake_cop"]
    prob     = signal["prob_ensemble"]
    edge     = signal["edge_pct"]
    kickoff  = utc_to_cot(signal.get("kickoff_utc", ""))
    home     = signal["home_team"]
    away     = signal["away_team"]
    liga     = {"BL2": "Bundesliga 2", "NL1": "Eredivisie"}.get(
                signal["league"], signal["league"])

    apuesta_txt = (
        "apostá a que el partido termina con\n   <b>3 o más goles (Over 2.5)</b>"
        if sel == "over" else
        "apostá a que el partido termina con\n   <b>2 o menos goles (Under 2.5)</b>"
    )
    ganancia = round(stake * (odds - 1))

    burn_txt = (
        f"\n\n⚠️ <i>Período de prueba — apuesta {n_bets + 1}/50.\n"
        f"Stake reducido a la mitad hasta validar el modelo.</i>"
        if n_bets < 50 else ""
    )

    flags_detail = signal.get("flags_detail", [])
    if isinstance(flags_detail, str):
        try:
            flags_detail = json.loads(flags_detail)
        except Exception:
            flags_detail = []

    flags_importantes = [f for f in flags_detail if f.get("code") != "BURN_IN"]
    flags_txt = ""
    if flags_importantes:
        flags_txt = "\n\n⚠️ <b>Atención:</b>\n"
        for f in flags_importantes:
            flags_txt += f"• {f.get('message', '')}\n"

    # Análisis del agente (si existe)
    agent_analysis = signal.get("agent_analysis", "")
    agent_txt = f"\n\n🤖 <b>Análisis:</b>\n{agent_analysis}" if agent_analysis else ""

    return (
        f"🎯 <b>{liga}</b>\n"
        f"<b>{home} vs {away}</b>\n"
        f"📅 {kickoff}\n"
        f"\n"
        f"👉 {apuesta_txt}\n"
        f"   Cuota Betplay: <b>{odds}</b>\n"
        f"\n"
        f"💰 Apostá: <b>${stake:,.0f} COP</b>\n"
        f"   Si ganás: +${ganancia:,.0f} COP\n"
        f"   Si perdés: -${stake:,.0f} COP\n"
        f"\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <i>El modelo estima {prob*100:.0f}% de probabilidad.\n"
        f"Betplay paga como si fuera {1/odds*100:.0f}%.\n"
        f"Ventaja estimada: +{edge*100:.0f}%</i>"
        f"{burn_txt}"
        f"{flags_txt}"
        f"{agent_txt}"
    )


def build_inline_keyboard(signal_id: int, stake: float) -> dict:
    """Botones: Aposté / Pasé / Aposté la mitad."""
    half = round(stake / 2 / 500) * 500
    return {
        "inline_keyboard": [[
            {"text": f"✅ Aposté ${stake:,.0f}",   "callback_data": f"GO:{signal_id}"},
            {"text": "❌ Pasé",                     "callback_data": f"SKIP:{signal_id}"},
            {"text": f"½ Aposté ${half:,.0f}",     "callback_data": f"HALF:{signal_id}"},
        ]]
    }


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
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        signals = conn.execute("""
            SELECT s.id AS signal_id, s.fixture_id, s.selection,
                   s.edge_pct, s.prob_ensemble, s.betplay_odds,
                   s.stake_cop, s.kelly_fraction, s.profile, s.status, s.flags,
                   f.league, f.home_team_name AS home_team,
                   f.away_team_name AS away_team, f.kickoff_utc,
                   1.0/s.betplay_odds AS implied_prob
            FROM signals s
            JOIN fixtures f ON s.fixture_id = f.id
            WHERE s.telegram_sent = 0 AND s.status IN ('BET', 'REVIEW')
            ORDER BY f.kickoff_utc ASC
        """).fetchall()

        bankroll = conn.execute("""
            SELECT COALESCE(
                (SELECT bankroll FROM performance_snapshots ORDER BY snapshot_date DESC LIMIT 1),
                150000)
        """).fetchone()[0]

        n_bets = conn.execute(
            "SELECT COUNT(*) FROM bets WHERE user_decision='GO'"
        ).fetchone()[0]

    sent = 0
    for sig in signals:
        sd = dict(sig)
        try:
            sd["flags_detail"] = json.loads(sd.get("flags") or "[]")
        except Exception:
            sd["flags_detail"] = []

        try:
            msg     = format_signal_message(sd, bankroll, n_bets)
            buttons = build_inline_keyboard(sd["signal_id"], sd["stake_cop"])
            client.send_message(chat_id, msg, reply_markup=buttons)
            with sqlite3.connect(db_path) as conn:
                conn.execute("UPDATE signals SET telegram_sent=1 WHERE id=?",
                             [sd["signal_id"]])
            sent += 1
            logger.info("Señal %d enviada", sd["signal_id"])
        except Exception as e:
            logger.error("Error enviando señal %d: %s", sd["signal_id"], e)

    return sent


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

    client.answer_callback(cb_id)
    if message_id:
        client.edit_message_reply_markup(chat_id, message_id)

    if decision == "GO":
        client.send_message(
            chat_id,
            f"✅ <b>Registrado</b>\n"
            f"Apostá <b>${stake:,.0f} COP</b> @ {signal['betplay_odds']} en Betplay.",
            parse_mode="HTML",
        )

    logger.info("Callback %s — señal %d", action, signal_id)
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

def send_weekly_summary(db_path: Path, client: TelegramClient, chat_id: str) -> None:
    with sqlite3.connect(db_path) as conn:
        bets = conn.execute("""
            SELECT b.stake_placed_cop, r.outcome, r.pnl_cop, f.league
            FROM bets b JOIN fixtures f ON b.fixture_id=f.id
            LEFT JOIN results r ON r.bet_id=b.id
            WHERE b.decided_at >= datetime('now', '-7 days') AND b.user_decision='GO'
        """).fetchall()
        bankroll = conn.execute("""
            SELECT COALESCE(
                (SELECT bankroll FROM performance_snapshots ORDER BY snapshot_date DESC LIMIT 1),
                150000)
        """).fetchone()[0]

    total = len(bets)
    wins  = sum(1 for b in bets if b[1] == "win")
    pnl   = sum(b[2] or 0 for b in bets)
    stk   = sum(b[0] or 0 for b in bets)
    roi   = (pnl / stk * 100) if stk > 0 else 0.0

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

    from datetime import date
    stats = {
        "total_bets": total, "wins": wins, "losses": total - wins,
        "pnl_cop": pnl, "roi_pct": roi, "bankroll": bankroll,
        "by_league": by_league,
    }
    client.send_message(chat_id, format_weekly_summary(stats), parse_mode="HTML")


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