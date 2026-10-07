"""Gold Signal Guide - Telegram alerts."""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

SYMBOL = os.getenv("SYMBOL", "GC=F")
INTERVAL = os.getenv("INTERVAL", "1h")
TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TEST = os.getenv("TEST_MESSAGE", "").lower() == "true"
STATE_FILE = "state.json"

PERIODS = {"15m": "59d", "30m": "59d", "1h": "365d"}
MINUTES = {"15m": 15, "30m": 30, "1h": 60}

EMA_F, EMA_M, EMA_S = 21, 55, 200
ADX_MIN = 20.0
SL_BUF = 0.3
MAX_RISK_ATR = 2.5
TP1_R, TP2_R = 1.0, 2.0
SESSION_START, SESSION_END = 7, 20
WARMUP = 250


def send(text: str) -> None:
    if not TOKEN or not CHAT_ID:
        raise RuntimeError("TELEGRAM_TOKEN / TELEGRAM_CHAT_ID secrets are missing")
    r = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        data={"chat_id": CHAT_ID, "text": text},
        timeout=30,
    )
    if not r.ok:
        raise RuntimeError(f"Telegram error: {r.status_code} {r.text}")


def fetch() -> pd.DataFrame:
    import yfinance as yf

    df = yf.Ticker(SYMBOL).history(period=PERIODS[INTERVAL], interval=INTERVAL, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError("No data returned from Yahoo Finance")
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l = df["Close"], df["High"], df["Low"]
    df["emaF"] = c.ewm(span=EMA_F, adjust=False).mean()
    df["emaM"] = c.ewm(span=EMA_M, adjust=False).mean()
    df["emaS"] = c.ewm(span=EMA_S, adjust=False).mean()

    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr = rma(tr, 14)
    df["atr"] = atr

    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = rma(gain, 14) / rma(loss, 14)
    df["rsi"] = 100 - 100 / (1 + rs)

    up = h.diff()
    down = -l.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    plus_di = 100 * rma(plus_dm, 14) / atr
    minus_di = 100 * rma(minus_dm, 14) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df["adx"] = rma(dx, 14)
    return df


def simulate(df: pd.DataFrame):
    events = []
    pos = 0
    entry = sl = tp1 = tp2 = 0.0
    tp1_hit = False

    for i, (t, r) in enumerate(df.iterrows()):
        if i < WARMUP or pd.isna(r["adx"]) or pd.isna(r["rsi"]) or pd.isna(r["atr"]):
            continue
        pos_prev = pos

        if pos_prev == 1:
            if r["Low"] <= sl:
                events.append({"time": t, "type": "sl", "side": 1, "tp1_hit": tp1_hit, "price": sl})
                pos = 0
            elif r["Close"] < r["emaM"]:
                events.append({"time": t, "type": "trend", "side": 1})
                pos = 0
            else:
                if not tp1_hit and r["High"] >= tp1:
                    tp1_hit = True
                    events.append({"time": t, "type": "tp1", "side": 1, "price": tp1})
                    sl = entry
                if r["High"] >= tp2:
                    events.append({"time": t, "type": "tp2", "side": 1, "price": tp2})
                    pos = 0
        elif pos_prev == -1:
            if r["High"] >= sl:
                events.append({"time": t, "type": "sl", "side": -1, "tp1_hit": tp1_hit, "price": sl})
                pos = 0
            elif r["Close"] > r["emaM"]:
                events.append({"time": t, "type": "trend", "side": -1})
                pos = 0
            else:
                if not tp1_hit and r["Low"] <= tp1:
                    tp1_hit = True
                    events.append({"time": t, "type": "tp1", "side": -1, "price": tp1})
                    sl = entry
                if r["Low"] <= tp2:
                    events.append({"time": t, "type": "tp2", "side": -1, "price": tp2})
                    pos = 0

        if pos_prev == 0:
            sess_ok = SESSION_START <= t.hour < SESSION_END
            if not (sess_ok and r["adx"] >= ADX_MIN):
                continue
            up_trend = r["Close"] > r["emaS"] and r["emaF"] > r["emaM"] > r["emaS"]
            down_trend = r["Close"] < r["emaS"] and r["emaF"] < r["emaM"] < r["emaS"]
            buy_raw = (up_trend and r["Low"] <= r["emaF"] and r["Close"] > r["emaF"]
                       and r["Close"] > r["Open"] and 45 < r["rsi"] < 70)
            sell_raw = (down_trend and r["High"] >= r["emaF"] and r["Close"] < r["emaF"]
                        and r["Close"] < r["Open"] and 30 < r["rsi"] < 55)
            if buy_raw:
                s = r["Low"] - r["atr"] * SL_BUF
                risk = r["Close"] - s
                if 0 < risk <= r["atr"] * MAX_RISK_ATR:
                    pos, entry, sl = 1, r["Close"], s
                    tp1, tp2 = entry + risk * TP1_R, entry + risk * TP2_R
                    tp1_hit = False
                    events.append({"time": t, "type": "entry", "side": 1, "entry": entry,
                                   "sl": sl, "tp1": tp1, "tp2": tp2, "risk": risk})
            elif sell_raw:
                s = r["High"] + r["atr"] * SL_BUF
                risk = s - r["Close"]
                if 0 < risk <= r["atr"] * MAX_RISK_ATR:
                    pos, entry, sl = -1, r["Close"], s
                    tp1, tp2 = entry - risk * TP1_R, entry - risk * TP2_R
                    tp1_hit = False
                    events.append({"time": t, "type": "entry", "side": -1, "entry": entry,
                                   "sl": sl, "tp1": tp1, "tp2": tp2, "risk": risk})

    last = df.iloc[-1]
    trend = ("UP" if last["Close"] > last["emaS"] and last["emaF"] > last["emaM"] > last["emaS"]
             else "DOWN" if last["Close"] < last["emaS"] and last["emaF"] < last["emaM"] < last["emaS"]
             else "MIXED")
    return events, {"pos": pos, "trend": trend, "price": float(last["Close"])}


def build_message(e: dict, cur_price: float) -> str:
    side = e["side"]
    t = e["type"]
    if t == "entry":
        risk = e["risk"]
        word = "شراء (BUY)" if side == 1 else "بيع (SELL)"
        icon = "🟢" if side == 1 else "🔴"
        below = "تحت" if side == 1 else "فوق"
        sign = "+" if side == 1 else "-"
        msg = (
            f"{icon} إشارة {word} - ذهب\n\n"
            f"سعر الإشارة: {e['entry']:.2f}\n"
            f"🛑 وقف الخسارة: {e['sl']:.2f} (على بعد {risk:.1f}$)\n"
            f"🎯 هدف 1: {e['tp1']:.2f} ({sign}{risk * TP1_R:.1f}$)\n"
            f"🎯 هدف 2: {e['tp2']:.2f} ({sign}{risk * TP2_R:.1f}$)\n\n"
            f"إزاي تنفذ:\n"
            f"1) افتح {'شراء' if side == 1 else 'بيع'} بسعر السوق دلوقتي.\n"
            f"2) حط وقف الخسارة {below} سعر دخولك بمسافة {risk:.1f}$.\n"
            f"3) عند الهدف الأول اقفل نص الصفقة وانقل الوقف لسعر الدخول.\n"
            f"4) عند الهدف الثاني اقفل الباقي.\n\n"
            f"⚠️ الأسعار من عقود الدهب الآجلة وقد تختلف عن سعر وسيطك بعدة دولارات، "
            f"فاعتمد على المسافات مش على الأرقام."
        )
        moved = (cur_price - e["entry"]) * side
        if abs(cur_price - e["entry"]) > 0.5 * risk:
            if moved > 0:
                msg += "\n\n⏳ السعر اتحرك كتير في اتجاه الصفقة من وقت الإشارة. الأفضل تتجاهلها."
            else:
                msg += "\n\n⏳ السعر اتحرك ضد الصفقة من وقت الإشارة. الأفضل تتجاهلها."
        return msg
    if t == "tp1":
        return "🎯 الهدف الأول اتحقق\nاقفل نص الصفقة وانقل وقف الخسارة لسعر الدخول."
    if t == "tp2":
        return "✅ الهدف الثاني اتحقق\nاقفل باقي الصفقة."
    if t == "sl":
        if e.get("tp1_hit"):
            return "🔒 الوقف (عند سعر الدخول) اتضرب\nالصفقة اتقفلت بدون خسارة على الجزء الباقي."
        return "🛑 وقف الخسارة اتضرب\nاقفل الصفقة لو لسه مفتوحة، واستنى الإشارة الجاية. الخسارة دي جزء طبيعي من الاستراتيجية."
    if t == "trend":
        return "⚠️ الاتجاه انكسر\nاقفل الصفقة دلوقتي."
    return str(e)


def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_state(last_bar: pd.Timestamp) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_bar": last_bar.isoformat()}, f)


def main() -> None:
    if TEST:
        send("✅ تجربة ناجحة: بوت الدهب متصل وهيبعتلك الإشارات هنا.")
        print("Test message sent")
        return

    df_all = fetch()
    now = datetime.now(timezone.utc)
    cur_price = float(df_all["Close"].iloc[-1])

    delta = timedelta(minutes=MINUTES[INTERVAL])
    closed = df_all[df_all.index + delta <= now].copy()
    if len(closed) < WARMUP + 10:
        raise RuntimeError("Not enough data")

    df = add_indicators(closed)
    events, status = simulate(df)
    last_bar = df.index[-1]

    state = load_state()
    if state is None:
        pos_txt = {1: "صفقة شراء مفتوحة", -1: "صفقة بيع مفتوحة", 0: "مفيش صفقة مفتوحة"}[status["pos"]]
        trend_txt = {"UP": "صاعد", "DOWN": "هابط", "MIXED": "مش واضح"}[status["trend"]]
        send(
            "🚀 بوت الدهب اشتغل\n\n"
            f"الفريم: {INTERVAL}\n"
            f"الاتجاه حاليًا: {trend_txt}\n"
            f"الحالة: {pos_txt}\n"
            f"آخر سعر: {status['price']:.2f}\n\n"
            "هتوصلك هنا إشارات الدخول والخروج أول بأول."
        )
        save_state(last_bar)
        print("First run - state initialised")
        return

    last_seen = pd.Timestamp(state["last_bar"])
    new_events = [e for e in events if e["time"] > last_seen]
    for e in new_events:
        send(build_message(e, cur_price))
        print("Sent:", e["type"], e["time"])

    if last_bar > last_seen:
        save_state(last_bar)
    print(f"Done. last_bar={last_bar}, new_events={len(new_events)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR:", exc, file=sys.stderr)
        raise
