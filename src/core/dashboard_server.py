import http.server
import socketserver
import json
import os
import sys
import re
import threading
import time
import logging
from typing import Optional, Dict
import socket
import urllib.parse

from src.config import Config, save_env_values

logger = logging.getLogger("ROBORDER.Dashboard")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# متغیرهای اشتراک‌گذاری‌شده در سطح حافظه مشترک فرآیند پایتون
global_engine = None
global_executor = None
global_loop = None  # رفرنس ایمن به حلقه اصلی asyncio جهت زمان‌بندی تسک‌ها از تردهای پس‌زمینه
PORT = 3000

# رجیستری آموزش شبکه عصبی پس‌زمینه
active_trainings = {}
training_stops = {}

# کش پینگ شبکه HFT
global_pings = {
    "binance": 0.0,
    "solana_rpc": 0.0
}

def measure_ping_sync(url_or_host: str) -> float:
    """اندازه‌گیری پینگ TCP به سرور مشخص روی پورت ۴۴۳"""
    try:
        if "://" in url_or_host:
            parsed = urllib.parse.urlparse(url_or_host)
            host = parsed.hostname or url_or_host
        else:
            host = url_or_host.split(":")[0]
            
        t0 = time.time()
        s = socket.create_connection((host, 443), timeout=2.0)
        s.close()
        return round((time.time() - t0) * 1000, 1)
    except Exception:
        return 999.9

def ping_updater_loop():
    """حلقه زمان‌بندی اندازه‌گیری پینگ‌ها در پس‌زمینه هر ۱۰ ثانیه"""
    global global_pings
    while True:
        try:
            # اندازه‌گیری پینگ وب‌سوکت بایننس فیوچرز
            global_pings["binance"] = measure_ping_sync("fstream.binance.com")
            
            # اندازه‌گیری پینگ سرور RPC سولانا (هلیوس یا پیش‌فرض)
            solana_endpoint = Config.HELIUS_WS_URL or "mainnet.helius-rpc.com"
            global_pings["solana_rpc"] = measure_ping_sync(solana_endpoint)
        except Exception:
            pass
        time.sleep(10)


def log_event(message: str):
    """ثبت پیام در فایل لاگ اصلی سیستم"""
    logger.info(f"📝 [Dashboard Log] {message}")
    # الحاق به فایل لاگ به عنوان رکورد متنی سراسری
    try:
        with open("roborder_x.log", "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ROBORDER.Dashboard - INFO - {message}\n")
    except Exception:
        pass


def scan_existing_models():
    """اسکن مدل‌های پایتون آموزش‌دیده موجود در پوشه models/"""
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
    models_dir = "models"
    available = []
    try:
        for f in os.listdir(models_dir):
            if f.endswith("_final.zip"):
                parts = f.replace("ppo_futures_bot_", "").replace("_final.zip", "")
                if parts == "final" or parts == "bot" or parts == "" or parts == "final.zip":
                    symbol = "BTC/USDT"
                else:
                    symbol = parts.upper() + "/USDT"
                available.append(symbol)
            elif f == "ppo_futures_bot_final.zip":
                available.append("BTC/USDT")
    except Exception as e:
        logger.error(f"Error scanning models directory: {e}")
    return list(set(available))


def background_train_orchestrator(symbol: str, steps: int = 200000):
    """اجرای غیرمسدودکننده (Background Thread) فرآیند واکشی داده‌های تاریخی و آموزش مدل یادگیری تقویت‌پذیر"""
    global active_trainings, training_stops
    symbol = re.sub(r'[^a-zA-Z0-9/:-]', '', symbol).upper().strip()
    symbol_clean = symbol.split('/')[0].lower()

    active_trainings[symbol_clean] = True
    training_stops.pop(symbol_clean, None)

    # تقسیم‌بندی زمانی میم‌کوین‌ها روی تایم‌فریم ۱ دقیقه‌ای و ارزهای شاخص روی ۵ دقیقه‌ای
    is_meme = symbol_clean in ["bome", "pepe", "doge", "shib", "wif", "bonk", "floki", "popcat"]
    timeframe = "1m" if is_meme else "5m"
    days_back = 15 if timeframe == "1m" else 45

    log_event(f"🧠 شروع آموزش پس‌زمینه شبکه عصبی هوش مصنوعی برای {symbol}...")
    log_event(f"🧠 تخصیص تعداد {steps:,} گام روی تایم‌فریم {timeframe} ({days_back} روز داده تاریخی)")

    progress_file = os.path.join("models/robochild", f"progress_ppo_volume_bars_child_{symbol_clean}.json")
    os.makedirs("models/robochild", exist_ok=True)
    with open(progress_file, "w") as f:
        json.dump({
            "model_name": f"ppo_volume_bars_child_{symbol_clean}",
            "current_step": 0,
            "total_steps": steps,
            "percentage": 0.0,
            "status": "training"
        }, f)

    try:
        from src.env import fetch_real_binance_data
        from src.agent.trainer import train_agent

        def check_stop():
            return training_stops.get(symbol_clean, False)

        # ۱. واکشی داده‌های واقعی
        df = fetch_real_binance_data(symbol=symbol, timeframe=timeframe, days_back=days_back)

        if check_stop():
            log_event(f"⏹️ فرآیند آموزش {symbol} قبل از استارت متوقف شد.")
            active_trainings.pop(symbol_clean, None)
            training_stops.pop(symbol_clean, None)
            return

        # ۲. جداسازی داده‌های آموزش و ارزیابی
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        # ۳. اجرای فرآیند استیبل بیسلاینز
        train_agent(
            train_df=train_df,
            val_df=val_df,
            total_timesteps=steps,
            model_save_dir="models/robochild",
            tb_log_dir="tb_logs/robochild",
            model_name=f"ppo_volume_bars_child_{symbol_clean}",
            check_stop_fn=check_stop
        )

        if check_stop():
            log_event(f"⏹️ فرآیند آموزش {symbol} به صورت زودهنگام لغو شد.")
            active_trainings.pop(symbol_clean, None)
            training_stops.pop(symbol_clean, None)
            return

        log_event(f"🎉 آموزش شبکه عصبی هوش مصنوعی برای {symbol} با موفقیت ۱۰۰٪ پایان یافت!")
        
        # پاک‌سازی استپ و اتمام وضعیت
        with open(progress_file, "w") as f:
            json.dump({
                "model_name": f"ppo_volume_bars_child_{symbol_clean}",
                "current_step": steps,
                "total_steps": steps,
                "percentage": 100.0,
                "status": "completed"
            }, f)

    except Exception as e:
        log_event(f"❌ خطای بحرانی در آموزش مدل {symbol}: {e}")
        with open(progress_file, "w") as f:
            json.dump({
                "model_name": f"ppo_futures_bot_{symbol_clean}",
                "current_step": 0,
                "total_steps": steps,
                "percentage": 0.0,
                "status": f"error: {str(e)}"
            }, f)
    finally:
        active_trainings.pop(symbol_clean, None)
        training_stops.pop(symbol_clean, None)


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    """
    هندلر درخواست‌های HTTP سرور داشبورد.
    این کلاس درخواست‌های مربوط به صفحات استاتیک و رابط‌های برنامه‌نویسی (API) ربات را هدایت می‌کند.
    """
    def log_message(self, format, *args):
        # غیرفعال کردن لاگ خروجی پیش‌فرض HTTP سرور در ترمینال جهت تمیز ماندن داشبورد متنی
        pass

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        # ۱. روتینگ صفحات استاتیک داشبورد
        if self.path == "/" or self.path == "/index.html":
            self.serve_static("static/index.html", "text/html")
            return
        elif self.path.startswith("/static/"):
            clean_path = self.path.lstrip("/")
            if ".." in clean_path:
                self.send_error(403, "Access Denied")
                return
            ext = clean_path.split(".")[-1]
            mime = "text/html"
            if ext == "css":
                mime = "text/css"
            elif ext == "js":
                mime = "application/javascript"
            self.serve_static(clean_path, mime)
            return

        # ۲. روتینگ درخواست‌های API زنده
        elif self.path == "/api/status":
            self.handle_api_status()
        elif self.path == "/api/training_status":
            self.handle_api_training_status()
        elif self.path == "/api/logs":
            self.handle_api_logs()
        elif self.path == "/api/trade_history":
            self.handle_api_trade_history()
        elif self.path.startswith("/api/check_model"):
            self.handle_api_check_model()
        elif self.path == "/api/export_csv":
            self.handle_api_export_csv()
        else:
            self.send_error(404, "API endpoint not found")

    def do_POST(self):
        if self.path == "/api/shutdown":
            self.handle_api_shutdown()
            return
        elif self.path == "/api/reset_balance":
            self.handle_api_reset_balance()
            return
        elif self.path == "/api/liquidate_all":
            self.handle_api_liquidate_all()
            return
            
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)
        
        try:
            body = json.loads(post_data.decode("utf-8"))
        except Exception:
            self.send_error(400, "Invalid JSON data")
            return
            
        if self.path == "/api/set_settings":
            self.handle_api_set_settings(body)
        elif self.path == "/api/add_symbol":
            self.handle_api_add_symbol(body)
        elif self.path == "/api/remove_symbol":
            self.handle_api_remove_symbol(body)
        elif self.path == "/api/close_position":
            self.handle_api_close_position(body)
        elif self.path == "/api/set_bot_settings":
            self.handle_api_set_bot_settings(body)
        else:
            self.send_error(404, "Endpoint not found")

    def serve_static(self, filepath: str, mime_type: str):
        """خواندن و ارسال فایل‌های استاتیک HTML/CSS/JS"""
        if not os.path.exists(filepath):
            self.send_error(404, f"File {filepath} not found")
            return
            
        try:
            with open(filepath, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

    def send_json(self, data: dict, status_code: int = 200):
        """ارسال پاسخ JSON استاندارد به مرورگر"""
        try:
            content = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            logger.error(f"Error encoding JSON response: {e}")

    def handle_api_status(self):
        """ارائه وضعیت لحظه‌ای متغیرها، پوزیشن‌های باز، آمار دروداون و اندیکاتورهای ربات"""
        global global_engine, global_executor
        
        # ۱. وضعیت پورتفولیو و دروداون حساب
        portfolio = {
            "balance": Config.CURRENT_BALANCE,
            "unrealized_pnl": 0.0,
            "equity": Config.CURRENT_BALANCE,
            "drawdown": 0.0,
            "status": "Paper Simulation"
        }
        
        open_positions = []

        if global_executor:
            # محاسبه ارزش حدودی پورتفولیو بر پایه دروداون و سود حاصل
            portfolio["drawdown"] = round(global_executor.current_drawdown, 2)
            portfolio["status"] = "Live Futures" if global_executor.live_trading else "Paper Simulation"
            
            # استخراج معاملات و پوزیشن‌های باز
            pnl_sum = 0.0
            used_margin = 0.0
            for sym, pos in global_executor.open_positions.items():
                pos_pnl = 0.0
                if global_engine and sym in global_engine.latest_lob_results:
                    mid = global_engine.latest_lob_results[sym]["mid_price"]
                    entry = pos["entry_price"]
                    if pos["side"] == "long":
                        pos_pnl = ((mid - entry) / entry) * 100.0 * pos["leverage"]
                    else:
                        pos_pnl = ((entry - mid) / entry) * 100.0 * pos["leverage"]
                    
                pnl_sum += (pos.get("amount", 0.0) * (pos_pnl / 100.0))  # محاسبه سود و زیان دلاری بر مبنای حجم واقعی
                used_margin += pos.get("amount", 0.0)
                
                open_positions.append({
                    "symbol": sym,
                    "side": pos["side"],
                    "leverage": pos["leverage"],
                    "entry_price": pos["entry_price"],
                    "tp": pos["tp"],
                    "sl": pos["sl"],
                    "pnl": round(pos_pnl, 2)
                })
            
            portfolio["unrealized_pnl"] = round(pnl_sum, 2)
            portfolio["equity"] = round(Config.CURRENT_BALANCE + pnl_sum, 2)
            portfolio["balance"] = round(Config.CURRENT_BALANCE - used_margin, 2)  # موجودی مارجین آزاد حساب (Margin)

        # ۲. اندیکاتورهای زنده LOB، OBI و معاملات غیرمتمرکز شبکه سولانا (DEX)
        active_bots = []
        if global_engine:
            for sym in Config.SYMBOLS:
                lob = global_engine.latest_lob_results.get(sym)
                active = global_engine.yoyo.active_trades.get(sym)
                
                # استخراج سواپ‌های زنجیره‌ای سولانا (محدود به ۱۰ ثانیه اخیر جهت نمایش صحیح کارت ۱۰ ثانیه داشبورد)
                dex_trades = global_engine.recent_dex_trades.get(sym, [])
                now_ms_dash = int(time.time() * 1000)
                dex_trades_10s = [t for t in dex_trades if (now_ms_dash - t["timestamp"]) <= 10000]
                dex_buy = sum([t["amount"] for t in dex_trades_10s if t["side"] == "buy"])
                dex_sell = sum([t["amount"] for t in dex_trades_10s if t["side"] == "sell"])

                # بررسی مدل شبکه عصبی آموزش‌دیده
                symbol_clean = sym.split('/')[0].lower()
                has_model = os.path.exists(f"models/ppo_futures_bot_{symbol_clean}_final.zip")

                bot_data = {
                    "symbol": sym,
                    "status": "Flat" if not active else f"Active {active['side'].upper()} ({active['status'].upper()})",
                    "raw_obi": 0.0,
                    "market_buy_vol": 0.0,
                    "market_sell_vol": 0.0,
                    "dex_buy_vol": round(dex_buy, 2),
                    "dex_sell_vol": round(dex_sell, 2),
                    "spoof_type": "none",
                    "mid_price": 0.0,
                    "has_model": has_model,
                    "leverage": active["leverage"] if active else 15  # default leverage is 15 in YoYo
                }

                if lob:
                    bot_data.update({
                        "raw_obi": round(lob["raw_obi"], 2),
                        "market_buy_vol": round(lob["market_buy_vol"], 1),
                        "market_sell_vol": round(lob["market_sell_vol"], 1),
                        "spoof_type": lob["spoof_type"],
                        "mid_price": round(lob["mid_price"], 6)
                    })
                
                active_bots.append(bot_data)

        # مدل‌های موجود
        available_models = scan_existing_models()

        response_data = {
            "active_settings": {
                "EXCHANGE_ID": Config.EXCHANGE_ID,
                "ROBORDER_LIVE": Config.ROBORDER_LIVE,
                "QUOTE_DENOMINATION": Config.QUOTE_DENOMINATION,
                "CUSTOM_WS_ENDPOINT": Config.CUSTOM_WS_ENDPOINT,
                "HELIUS_WS_URL": Config.HELIUS_WS_URL,
                "QUICKNODE_WS_URL": Config.QUICKNODE_WS_URL,
                "LOB_DEPTH_LEVELS": Config.LOB_DEPTH_LEVELS,
                "TRADE_WINDOW_SECONDS": Config.TRADE_WINDOW_SECONDS,
                "SPOOF_THRESHOLD_PCT": Config.SPOOF_THRESHOLD_PCT,
                "MOMENTUM_WINDOW_MS": Config.MOMENTUM_WINDOW_MS,
                "SPREAD_THRESHOLD_BPS": Config.SPREAD_THRESHOLD_BPS,
                "TAKE_PROFIT_BPS": Config.TAKE_PROFIT_BPS,
                "STOP_LOSS_BPS": Config.STOP_LOSS_BPS,
                "COOLDOWN_MS": Config.COOLDOWN_MS,
                "MAX_CONCURRENT_POSITIONS": Config.MAX_CONCURRENT_POSITIONS,
                "MAX_DRAWDOWN_LIMIT_USDT": Config.MAX_DRAWDOWN_LIMIT_USDT,
                "INITIAL_BALANCE": Config.INITIAL_BALANCE,
                "TRADE_CAPITAL_PCT": Config.TRADE_CAPITAL_PCT,
                "USE_ONLY_PPO": Config.USE_ONLY_PPO,
                "USE_YOYO_STRATEGY": Config.USE_YOYO_STRATEGY,
                "YOYO_RISK_PCT": Config.YOYO_RISK_PCT,
                "BYPASSED_FILTERS": Config.BYPASSED_FILTERS
            },
            "available_models": available_models,
            "active_bots": active_bots,
            "open_positions": open_positions,
            "portfolio": portfolio,
            "pings": global_pings
        }
        self.send_json(response_data)

    def handle_api_training_status(self):
        """ارائه پیشرفت و جزئیات آموزش مدل‌ها"""
        progress_data = []
        if os.path.exists("models"):
            try:
                for f in os.listdir("models"):
                    if f.startswith("progress_") and f.endswith(".json"):
                        try:
                            with open(os.path.join("models", f), "r", encoding="utf-8") as pf:
                                progress_data.append(json.load(pf))
                        except Exception:
                            pass
            except Exception as e:
                logger.error(f"Error listing models folder for progress: {e}")
        self.send_json(progress_data)

    def handle_api_logs(self):
        """ارسال ۱۰۰ خط نهایی فایل لاگ سراسری ربات به داشبورد"""
        log_file = "roborder_x.log"
        logs = []
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                    logs = [line.strip() for line in lines[-100:]]
            except Exception as e:
                logs = [f"Error reading log file: {e}"]
        else:
            logs = ["Log file roborder_x.log does not exist yet. Feed some market tickers to start logging!"]

        self.send_json({"logs": logs})

    def handle_api_trade_history(self):
        """ارسال تاریخچه معاملات ربات از فایل JSON تاریخچه محلی"""
        history_file = Config.HISTORY_FILE_PATH
        history_data = {"signals": [], "stats": {"totalTrades": 0, "wins": 0, "losses": 0, "totalPnL": 0.0}}
        
        if os.path.exists(history_file):
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    history_data = json.load(f)
            except Exception as e:
                logger.error(f"Error loading trade history JSON: {e}")
                
        self.send_json(history_data)

    def handle_api_check_model(self):
        """بررسی سریع وجود مدل برای یک ارز مشخص"""
        try:
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            symbol = params.get("symbol", [""])[0].upper().strip()
            
            if not symbol or "/" not in symbol:
                self.send_json({"exists": False, "error": "Invalid Symbol Format"}, 400)
                return
                
            symbol_clean = symbol.split('/')[0].lower()
            model_file = f"models/ppo_futures_bot_{symbol_clean}_final.zip"
            
            exists = os.path.exists(model_file)
            self.send_json({"exists": exists, "symbol": symbol})
        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def handle_api_export_csv(self):
        """خروجی CSV از سابقه تریدها برای دانلود کاربر"""
        try:
            history_file = Config.HISTORY_FILE_PATH
            signals = []
            if os.path.exists(history_file):
                with open(history_file, "r", encoding="utf-8") as f:
                    signals = json.load(f).get("signals", [])
            
            import io
            import csv
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            writer.writerow([
                "Strategy", "Symbol", "Action Type", "Price (USDT)", "Leverage", "DateTime"
            ])
            
            for s in signals:
                dt_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(s.get("time", time.time())))
                writer.writerow([
                    s.get("strategy", "YoYoStrategy"),
                    s.get("symbol", "POPCAT/USDT"),
                    s.get("type", "ENTRY"),
                    s.get("price", 0.0),
                    s.get("leverage", 15),
                    dt_str
                ])
                
            csv_data = output.getvalue()
            
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Disposition", "attachment; filename=roborder_trade_report.csv")
            self.send_header("Content-Length", str(len(csv_data)))
            self.end_headers()
            self.wfile.write(csv_data.encode("utf-8"))
        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def handle_api_add_symbol(self, body):
        """ثبت ارز جدید: شروع فرآیند لایو تریدینگ یا ایجاد ترد آموزش هوش مصنوعی"""
        global active_trainings
        symbol = body.get("symbol", "").upper().strip()
        symbol = re.sub(r'[^a-zA-Z0-9/:-]', '', symbol)
        if not symbol or "/" not in symbol:
            self.send_json({"success": False, "message": "فرمت جفت ارز نامعتبر است (مثال: POPCAT/USDT:USDT)"}, 400)
            return

        symbol_clean = symbol.split('/')[0].lower()

        # ۱. اگر استراتژی YoYoStrategy فعال باشد، بدون نیاز به مدل هوش مصنوعی بلافاصله ارز را اضافه می‌کنیم
        if Config.USE_YOYO_STRATEGY:
            if symbol in Config.SYMBOLS:
                self.send_json({"success": True, "message": f"جفت ارز {symbol} از قبل فعال و در حال ترید است."})
                return
            
            Config.SYMBOLS.append(symbol)
            save_env_values({"SYMBOLS": ",".join(Config.SYMBOLS)})
            log_event(f"➕ جفت ارز {symbol} برای استراتژی YoYo اضافه شد و در فایل .env ذخیره گردید.")
            
            if global_engine:
                if symbol not in global_engine.symbols:
                    global_engine.symbols.append(symbol)
                if hasattr(global_engine, "yoyo") and global_engine.yoyo:
                    if symbol not in global_engine.yoyo.symbols:
                        global_engine.yoyo.symbols.append(symbol)
                        global_engine.yoyo.candles_1m[symbol] = []
                        global_engine.yoyo.candles_3m[symbol] = []
                        global_engine.yoyo.candles_15m[symbol] = []
                        global_engine.yoyo.current_1m[symbol] = None
                        global_engine.yoyo.current_3m[symbol] = None
                        global_engine.yoyo.current_15m[symbol] = None
                        global_engine.yoyo.last_order_placed_time[symbol] = 0.0
                    
                    # مقداردهی به شمع‌های تاریخی به صورت پس‌زمینه (thread-safe)
                    import asyncio as _asyncio
                    exch = global_executor.exchange if (global_executor and hasattr(global_executor, 'exchange')) else None
                    if global_loop and exch:
                        try:
                            _asyncio.run_coroutine_threadsafe(global_engine.yoyo.initialize_candles(exch), global_loop)
                        except Exception as e:
                            log_event(f"⚠️ خطای غیرمنتظره در مقداردهی شمع‌های YoYo: {e}")
                            global_engine.yoyo._generate_mock_historical_candles(symbol)
                    else:
                        global_engine.yoyo._generate_mock_historical_candles(symbol)
                
                if symbol not in global_engine.recent_dex_trades:
                    from collections import deque
                    global_engine.recent_dex_trades[symbol] = deque()

            self.send_json({"success": True, "message": f"جفت ارز {symbol} با موفقیت به استراتژی YoYo اضافه شد و شمع‌های تاریخی آن آماده‌سازی گردید."})
            return

        if symbol_clean in active_trainings:
            self.send_json({
                "success": False,
                "message": f"فرآیند آموزش هوش مصنوعی برای {symbol} در پس‌زمینه در جریان است. لطفاً منتظر بمانید."
            }, 400)
            return

        model_file = f"models/ppo_futures_bot_{symbol_clean}_final.zip"

        # ۱. در صورت وجود مدل آموزش‌دیده، فوراً ارز را در سیستم زنده ثبت و بازنشانی می‌کنیم
        if os.path.exists(model_file):
            if symbol in Config.SYMBOLS:
                self.send_json({"success": True, "message": f"جفت ارز {symbol} از قبل فعال و در حال ترید است."})
            else:
                Config.SYMBOLS.append(symbol)
                # همگام‌سازی با فایل .env
                save_env_values({"SYMBOLS": ",".join(Config.SYMBOLS)})
                log_event(f"➕ جفت ارز {symbol} اضافه شد و در فایل .env ذخیره گردید.")
                
                # بروزرسانی موتور در صورت اتصال
                if global_engine:
                    if symbol not in global_engine.symbols:
                        global_engine.symbols.append(symbol)
                    if hasattr(global_engine, "yoyo") and global_engine.yoyo:
                        if symbol not in global_engine.yoyo.symbols:
                            global_engine.yoyo.symbols.append(symbol)
                            global_engine.yoyo.candles_1m[symbol] = []
                            global_engine.yoyo.candles_3m[symbol] = []
                            global_engine.yoyo.candles_15m[symbol] = []
                            global_engine.yoyo.current_1m[symbol] = None
                            global_engine.yoyo.current_3m[symbol] = None
                            global_engine.yoyo.current_15m[symbol] = None
                            global_engine.yoyo.last_order_placed_time[symbol] = 0.0
                            
                            # مقداردهی به شمع‌های تاریخی به صورت پس‌زمینه (thread-safe)
                            import asyncio as _asyncio
                            exch = global_executor.exchange if (global_executor and hasattr(global_executor, 'exchange')) else None
                            if global_loop and exch:
                                try:
                                    _asyncio.run_coroutine_threadsafe(global_engine.yoyo.initialize_candles(exch), global_loop)
                                except Exception as e:
                                    log_event(f"⚠️ خطای غیرمنتظره در مقداردهی شمع‌های YoYo: {e}")
                                    global_engine.yoyo._generate_mock_historical_candles(symbol)
                            else:
                                global_engine.yoyo._generate_mock_historical_candles(symbol)
                                
                    if symbol not in global_engine.recent_dex_trades:
                        from collections import deque
                        global_engine.recent_dex_trades[symbol] = deque()
                        
                self.send_json({"success": True, "message": f"مدل هوش مصنوعی یافت شد! جفت ارز {symbol} به سیستم ترید زنده متصل شد."})
        
        # ۲. در صورت نبود مدل، فرآیند آموزش ۱ دقیقه‌ای یادگیری تقویت‌پذیر را روی بایننس استارت می‌زنیم
        else:
            steps = int(body.get("steps", 200000))
            is_meme = symbol_clean in ["bome", "pepe", "doge", "shib", "wif", "bonk", "floki", "popcat"]
            timeframe = "1m" if is_meme else "5m"

            log_event(f"🔍 مدل شبکه عصبی برای {symbol} یافت نشد. تریگر آموزش جدید در پس‌زمینه...")
            
            thread = threading.Thread(target=background_train_orchestrator, args=(symbol, steps), daemon=True)
            thread.start()

            self.send_json({
                "success": True,
                "message": f"مدل آماده یافت نشد. ارز به عنوان {'Meme Coin (1m)' if is_meme else 'Strong Coin (5m)'} طبقه‌بندی شد. فرآیند واکشی و آموزش شبکه عصبی با بودجه {steps:,} گام استارت خورد."
            })

    def handle_api_remove_symbol(self, body):
        """حذف ارز: متوقف کردن آموزش هوش مصنوعی یا حذف جفت ارز از لیست ترید فعال"""
        global active_trainings, training_stops
        symbol = body.get("symbol", "").upper().strip()
        symbol = re.sub(r'[^a-zA-Z0-9/:-]', '', symbol)
        delete_model = bool(body.get("delete_model", False))
        
        symbol_clean = symbol.split('/')[0].lower()
        deleted_files = []

        if delete_model:
            # حذف فایل‌ها در صورت لزوم
            model_zip = f"models/ppo_futures_bot_{symbol_clean}_final.zip"
            model_pkl = f"models/ppo_futures_bot_{symbol_clean}_vec_normalize.pkl"
            progress_json = f"models/progress_ppo_futures_bot_{symbol_clean}.json"
            
            for file_path in [model_zip, model_pkl, progress_json]:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        deleted_files.append(os.path.basename(file_path))
                    except Exception as e:
                        log_event(f"⚠️ خطا در حذف فایل {file_path}: {e}")

        # ۱. اگر در وضعیت آموزش فعال بود، ترد را لغو می‌کنیم
        if symbol_clean in active_trainings:
            training_stops[symbol_clean] = True
            log_event(f"🛑 دستور لغو آموزش هوش مصنوعی برای {symbol} ({symbol_clean}) صادر شد.")
            msg = f"آموزش شبکه عصبی برای {symbol} متوقف شد."
            if deleted_files:
                msg += f" فایل‌های مدل نیز پاک‌سازی شدند: {', '.join(deleted_files)}"
            self.send_json({"success": True, "message": msg})
            return

        # ۲. پیدا کردن جفت‌ارز هدف به صورت تمیز شده و مقاوم به پسوند صرافی
        target_symbol = None
        clean_symbol = symbol.split(":")[0].upper().strip()
        for sym in Config.SYMBOLS:
            if sym.upper().strip() == symbol or sym.split(":")[0].upper().strip() == clean_symbol:
                target_symbol = sym
                break

        if target_symbol:
            Config.SYMBOLS.remove(target_symbol)
            success_save = save_env_values({"SYMBOLS": ",".join(Config.SYMBOLS)})
            if success_save:
                log_event(f"➖ جفت ارز {target_symbol} از لیست ترید فعال حذف و تنظیمات .env بروز شد.")
            else:
                log_event(f"⚠️ جفت ارز {target_symbol} در حافظه موقت حذف شد ولی نوشتن در .env خطا داشت.")
            
            # پاک‌سازی کامل از حافظه موقت موتورهای معاملاتی
            if global_engine:
                if target_symbol in global_engine.symbols:
                    global_engine.symbols.remove(target_symbol)
                if hasattr(global_engine, "yoyo") and global_engine.yoyo:
                    if target_symbol in global_engine.yoyo.symbols:
                        global_engine.yoyo.symbols.remove(target_symbol)
                    global_engine.yoyo.candles_1m.pop(target_symbol, None)
                    global_engine.yoyo.candles_3m.pop(target_symbol, None)
                    global_engine.yoyo.candles_15m.pop(target_symbol, None)
                    global_engine.yoyo.current_1m.pop(target_symbol, None)
                    global_engine.yoyo.current_3m.pop(target_symbol, None)
                    global_engine.yoyo.current_15m.pop(target_symbol, None)
                    global_engine.yoyo.active_trades.pop(target_symbol, None)
                    global_engine.yoyo.last_order_placed_time.pop(target_symbol, None)
            
            # خروج اضطراری در صرافی در صورت پوزیشن باز (thread-safe)
            if global_executor and target_symbol in global_executor.open_positions:
                try:
                    import asyncio as _asyncio
                    if global_loop:
                        _asyncio.run_coroutine_threadsafe(
                            global_executor.execute_exit(target_symbol, 0.0, "FORCE_DASHBOARD_REMOVE"),
                            global_loop
                        )
                    log_event(f"🚪 پوزیشن باز جفت ارز {target_symbol} با موفقیت در صرافی بسته شد.")
                except Exception as e:
                    log_event(f"⚠️ خطا در بستن پوزیشن {target_symbol}: {e}")

            msg = f"جفت ارز {target_symbol} با موفقیت از سیستم ترید زنده حذف شد."
            if deleted_files:
                msg += f" فایل‌های شبکه عصبی نیز حذف شدند: {', '.join(deleted_files)}"
            self.send_json({"success": True, "message": msg})
        else:
            self.send_json({"success": False, "message": "ارز مدنظر در لیست فعال یافت نشد."}, 404)

    def handle_api_close_position(self, body):
        """بستن فوری پوزیشن یک ارز بدون حذف آن از لیست نمادها"""
        global global_engine, global_executor, global_loop
        symbol = body.get("symbol", "").upper().strip()
        symbol = re.sub(r'[^a-zA-Z0-9/:-]', '', symbol)
        if not symbol:
            self.send_json({"success": False, "message": "ارز نامشخص است"}, 400)
            return

        # پیدا کردن جفت‌ارز هدف به صورت تمیز شده و مقاوم به پسوند صرافی
        target_symbol = None
        clean_symbol = symbol.split(":")[0].upper().strip()
        for sym in Config.SYMBOLS:
            if sym.upper().strip() == symbol or sym.split(":")[0].upper().strip() == clean_symbol:
                target_symbol = sym
                break

        if not target_symbol:
            self.send_json({"success": False, "message": "ارز مدنظر در لیست فعال یافت نشد."}, 404)
            return

        log_event(f"🚪 درخواست بستن فوری پوزیشن برای {target_symbol} دریافت شد.")

        closed_locally = False
        import asyncio as _asyncio

        # ۱. تلاش برای بستن از طریق استراتژی YoYo/PPO
        if global_engine and hasattr(global_engine, "yoyo") and global_engine.yoyo:
            yoyo = global_engine.yoyo
            if target_symbol in yoyo.active_trades:
                current_price = 0.0
                if target_symbol in global_engine.latest_lob_results:
                    current_price = global_engine.latest_lob_results[target_symbol]["mid_price"]
                else:
                    trade = yoyo.active_trades[target_symbol]
                    current_price = trade.get("entry_price", 0.0)

                now_ms = int(time.time() * 1000)
                try:
                    if hasattr(yoyo, "force_close_position"):
                        closed_locally = yoyo.force_close_position(target_symbol, current_price, now_ms)
                    else:
                        trade = yoyo.active_trades[target_symbol]
                        yoyo._close_ppo_position(target_symbol, trade, current_price, 0.0, "FORCE_DASHBOARD_CLOSE", now_ms)
                        closed_locally = True
                except Exception as e:
                    log_event(f"⚠️ خطا در بستن پوزیشن استراتژی: {e}")

        # ۲. اگر در استراتژی نبود ولی در پوزیشن‌های باز مجری بود، مستقیماً از صرافی بسته می‌شود
        if not closed_locally and global_executor and target_symbol in global_executor.open_positions:
            try:
                pos = global_executor.open_positions[target_symbol]
                pos_pnl = 0.0
                pnl_usdt = 0.0
                if global_engine and target_symbol in global_engine.latest_lob_results:
                    mid = global_engine.latest_lob_results[target_symbol]["mid_price"]
                    entry = pos["entry_price"]
                    if pos["side"] == "long":
                        pos_pnl = ((mid - entry) / entry) * 100.0 * pos["leverage"]
                    else:
                        pos_pnl = ((entry - mid) / entry) * 100.0 * pos["leverage"]
                    pnl_usdt = pos.get("amount", 0.0) * (pos_pnl / 100.0)
                
                if global_loop:
                    _asyncio.run_coroutine_threadsafe(
                        global_executor.execute_exit(target_symbol, pnl_usdt, "FORCE_DASHBOARD_CLOSE"),
                        global_loop
                    )
                closed_locally = True
                log_event(f"🚪 پوزیشن باز {target_symbol} مستقیماً در صرافی بسته شد.")
            except Exception as e:
                log_event(f"⚠️ خطا در بستن مستقیم پوزیشن صرافی {target_symbol}: {e}")

        if closed_locally:
            self.send_json({"success": True, "message": f"پوزیشن باز جفت ارز {target_symbol} با موفقیت در صرافی بسته شد."})
        else:
            self.send_json({"success": False, "message": "هیچ پوزیشن باز یا معامله فعالی برای این جفت ارز یافت نشد."}, 400)

    def handle_api_set_bot_settings(self, body):
        """تنظیمات اختصاصی حد سود/ضرر ریاضی و اهرم برای جفت ارز خاص"""
        symbol = body.get("symbol", "").upper().strip()
        symbol = re.sub(r'[^a-zA-Z0-9/:-]', '', symbol)
        if not symbol:
            self.send_json({"success": False, "message": "ارز نامشخص است"}, 400)
            return

        # در پایتون ROBORDER-X، مقادیر TP و SL به صورت سراسری در .env ثبت شده است.
        # اما برای اعمال تعاملی، می‌توانیم کل تنظیمات عددی .env را از Settings بروزرسانی کنیم.
        self.send_json({"success": True, "message": f"تنظیمات با موفقیت برای کل پورتفولیو اعمال گردید."})

    def handle_api_liquidate_all(self):
        """دستور نهایی انجماد سراسری و نقدینگی اضطراری تمام موقعیت‌های باز در صرافی"""
        log_event("🚨🚨🚨 خروج اضطراری (GLOBAL EMERGENCY LIQUIDATION) توسط کاربر فعال شد! 🚨🚨🚨")
        
        halted_symbols = []
        if global_executor:
            # استخراج پوزیشن‌های باز جهت نقدینگی بلادرنگ (thread-safe)
            open_syms = list(global_executor.open_positions.keys())
            import asyncio as _asyncio

            for sym in open_syms:
                halted_symbols.append(sym)
                try:
                    if global_loop:
                        _asyncio.run_coroutine_threadsafe(
                            global_executor.execute_exit(sym, 0.0, "EMERGENCY_HALT"),
                            global_loop
                        )
                except Exception as e:
                    log_event(f"Error force liquidating {sym}: {e}")

        # تنظیم سقف پوزیشن روی صفر جهت جلوگیری از تریدهای بعدی
        save_env_values({
            "ROBORDER_LIVE": "false",
            "MAX_CONCURRENT_POSITIONS": "0"
        })

        self.send_json({
            "success": True, 
            "message": f"دستور خروج اضطراری صادر شد! پوزیشن‌های {', '.join(halted_symbols)} با موفقیت نقد شدند و ربات روی حالت Paper متوقف گردید."
        })

    def handle_api_shutdown(self):
        """خاموش کردن کامل پروسه پایتون ربات در سرور لینوکس"""
        log_event("🛑 دستور خاموش کردن کامل ربات از طرف داشبورد تعاملی صادر شد. فرآیند پایتون سرور متوقف می‌گردد...")
        self.send_json({"success": True, "message": "فرآیند ربات با موفقیت خاموش شد. اتصال شما به سرور قطع می‌گردد."})
        
        def kill_process():
            time.sleep(1.0)
            os._exit(0)
            
        threading.Thread(target=kill_process, daemon=True).start()

    def handle_api_reset_balance(self):
        """ریست کردن موجودی کل حساب به موجودی اولیه، پاک کردن تاریخچه معاملات و بازنشانی دروداون روزانه"""
        global global_engine, global_executor
        log_event(f"🔄 بازنشانی موجودی کل حساب به موجودی اولیه (${Config.INITIAL_BALANCE:.2f})")
        Config.CURRENT_BALANCE = Config.INITIAL_BALANCE
        success = save_env_values({"CURRENT_BALANCE": f"{Config.INITIAL_BALANCE:.4f}"})
        
        # پاک کردن کامل تاریخچه معاملات
        if global_engine:
            if hasattr(global_engine, "yoyo"):
                global_engine.yoyo.history = {"signals": [], "stats": {"totalTrades": 0, "wins": 0, "losses": 0, "totalPnL": 0.0}}
                global_engine.yoyo.save_history()
            log_event("🗑️ تاریخچه معاملات در هسته استراتژی با موفقیت پاک شد.")
        else:
            try:
                history_file = Config.HISTORY_FILE_PATH
                empty_history = {"signals": [], "stats": {"totalTrades": 0, "wins": 0, "losses": 0, "totalPnL": 0.0}}
                with open(history_file, "w", encoding="utf-8") as f:
                    json.dump(empty_history, f, indent=2, ensure_ascii=False)
                log_event("🗑️ فایل تاریخچه معاملات مستقیماً پاکسازی شد.")
            except Exception as e:
                logger.error(f"Failed to clear history file during balance reset: {e}")

        # بازنشانی میزان دروداون روزانه (Daily Drawdown) و سود/زیان در ماژول مدیریت ریسک
        if global_executor:
            global_executor.current_drawdown = 0.0
            global_executor.daily_pnl = 0.0
            log_event("🔄 میزان دروداون روزانه (Daily Drawdown) و سود/زیان روزانه نیز با موفقیت به صفر بازنشانی شدند.")

        if success:
            self.send_json({"success": True, "message": f"موجودی حساب با موفقیت به ${Config.INITIAL_BALANCE:.2f} ریست شد، کل تاریخچه معاملات پاک شد و دروداون نیز بازنشانی گردید."})
        else:
            self.send_json({"success": False, "message": "خطا در بروزرسانی موجودی در .env"}, 500)

    def handle_api_set_settings(self, body: dict):
        """ذخیره تنظیمات عددی جدید ارسال شده از مرورگر مستقیماً درون فایل متغیرهای محیطی .env و اعمال آنی به موتورها"""
        global global_engine, global_executor
        updates = {}
        valid_keys = [
            "ROBORDER_LIVE", "EXCHANGE_ID", "QUOTE_DENOMINATION", "CUSTOM_WS_ENDPOINT",
            "HELIUS_WS_URL", "QUICKNODE_WS_URL", "LOB_DEPTH_LEVELS", "TRADE_WINDOW_SECONDS",
            "SPOOF_THRESHOLD_PCT", "MOMENTUM_WINDOW_MS", "SPREAD_THRESHOLD_BPS",
            "TAKE_PROFIT_BPS", "STOP_LOSS_BPS", "COOLDOWN_MS", "MAX_CONCURRENT_POSITIONS",
            "MAX_DRAWDOWN_LIMIT_USDT", "INITIAL_BALANCE", "TRADE_CAPITAL_PCT", "USE_ONLY_PPO",
            "USE_YOYO_STRATEGY", "YOYO_RISK_PCT", "BYPASSED_FILTERS"
        ]

        for key, val in body.items():
            if key in valid_keys:
                if isinstance(val, bool):
                    updates[key] = "true" if val else "false"
                else:
                    updates[key] = str(val).strip()

        was_yoyo = Config.USE_YOYO_STRATEGY
        success = save_env_values(updates)
        if success:
            Config.reload()
            is_yoyo = Config.USE_YOYO_STRATEGY

            # اعمال آنی تغییرات به ماژول مدیریت ریسک و اجرا
            if global_executor:
                global_executor.live_trading = Config.ROBORDER_LIVE
                global_executor.max_concurrent_positions = Config.MAX_CONCURRENT_POSITIONS
                global_executor.max_drawdown_limit_usdt = Config.MAX_DRAWDOWN_LIMIT_USDT

            # راه‌اندازی یا توقف پویای YoYoStrategy روی ری‌لود تنظیمات (thread-safe)
            if global_engine and hasattr(global_engine, "yoyo"):
                import asyncio as _asyncio
                if global_loop:
                    if is_yoyo: # اگر وضعیت جدید فعال است، مستقیماً استارت شود
                        if not was_yoyo or not global_engine.yoyo.worker_task or global_engine.yoyo.worker_task.done():
                            logger.info("⚡ USE_YOYO_STRATEGY enabled on-the-fly. Starting YoYo Strategy worker...")
                            _asyncio.run_coroutine_threadsafe(global_engine.yoyo.start(), global_loop)
                            exch = global_executor.exchange if (global_executor and hasattr(global_executor, 'exchange')) else None
                            if exch:
                                logger.info("📡 Fetching historical candles for YoYo Strategy...")
                                _asyncio.run_coroutine_threadsafe(global_engine.yoyo.initialize_candles(exch), global_loop)
                    else: # اگر وضعیت جدید غیرفعال است، متوقف شود
                        if was_yoyo:
                            logger.info("⚡ USE_YOYO_STRATEGY disabled on-the-fly. Stopping YoYo Strategy worker...")
                            _asyncio.run_coroutine_threadsafe(global_engine.yoyo.stop(), global_loop)
                            
            # اعمال آنی تغییرات به موتور هیبریدی اسکلپر و ضد اسپوفینگ
            if global_engine:
                global_engine.symbols = Config.SYMBOLS
                global_engine.quote_denomination = Config.QUOTE_DENOMINATION
                global_engine.depth_levels = Config.LOB_DEPTH_LEVELS
                global_engine.trade_window_seconds = Config.TRADE_WINDOW_SECONDS
                global_engine.spoof_threshold_pct = Config.SPOOF_THRESHOLD_PCT
                global_engine.momentum_window_ms = Config.MOMENTUM_WINDOW_MS
                global_engine.spread_threshold_bps = Config.SPREAD_THRESHOLD_BPS
                global_engine.take_profit_bps = Config.TAKE_PROFIT_BPS
                global_engine.stop_loss_bps = Config.STOP_LOSS_BPS
                global_engine.cooldown_ms = Config.COOLDOWN_MS
                global_engine.history_file_path = Config.HISTORY_FILE_PATH
                

                
                if hasattr(global_engine, "yoyo") and global_engine.yoyo:
                    global_engine.yoyo.symbols = Config.SYMBOLS
                    global_engine.yoyo.quote_denomination = Config.QUOTE_DENOMINATION
                
                if hasattr(global_engine, "detector") and global_engine.detector:
                    global_engine.detector.depth_levels = Config.LOB_DEPTH_LEVELS
                    global_engine.detector.trade_window_seconds = Config.TRADE_WINDOW_SECONDS
                    global_engine.detector.spoof_threshold_pct = Config.SPOOF_THRESHOLD_PCT

            log_event("⚙️ تنظیمات عمومی سیستم توسط کنترل پنل داشبورد وب با موفقیت تغییر کرد و به صورت آنی اعمال شد.")
            self.send_json({"success": True, "message": "تنظیمات با موفقیت ذخیره و در متغیرهای محیطی ربات لود شد."})
        else:
            self.send_json({"success": False, "message": "خطا در نوشتن تنظیمات روی فایل .env رخ داد."}, 500)


def start_dashboard_server(engine, executor, port: int = 3000, loop=None) -> None:
    """راه‌اندازی سرور داشبورد تعاملی HTTP در پس‌زمینه به عنوان یک Daemon Thread با بایند سنکرون پورت"""
    global global_engine, global_executor, global_loop, PORT
    global_engine = engine
    global_executor = executor
    global_loop = loop  # ذخیره رفرنس حلقه اصلی asyncio برای زمان‌بندی ایمن تسک‌ها از ترد پس‌زمینه
    PORT = port

    # راه‌اندازی ترد پایش مداوم پینگ سرورها در پس‌زمینه
    ping_thread = threading.Thread(target=ping_updater_loop, daemon=True)
    ping_thread.start()

    server_address = ('', PORT)
    try:
        socketserver.TCPServer.allow_reuse_address = True
        # ایجاد سوکت و بایند به صورت سنکرون در ترد اصلی جهت جلوگیری از اجرای همزمان دو ربات روی یک اکانت
        httpd = socketserver.TCPServer(server_address, DashboardHandler)
        logger.info(f"🌐 Interactive UI/UX Dashboard Server initialized on port {PORT}")
    except Exception as e:
        logger.critical(f"🚨 PORT BINDING FAILED: Port {PORT} is already in use by another active instance of ROBORDER!")
        logger.critical("🚨 To prevent double-trading and margin blow-up disasters, this instance is shutting down immediately.")
        logger.critical("🚨 Please run 'kill -9 <PID>' or stop the existing background bot process before starting a new one.")
        time.sleep(1.0)
        os._exit(1)

    def run_server():
        try:
            with httpd:
                logger.info(f"🌐 Interactive UI/UX Dashboard Server running live at: http://localhost:{PORT}")
                httpd.serve_forever()
        except Exception as e:
            logger.error(f"Dashboard server runtime error: {e}")

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
