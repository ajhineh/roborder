import sys
import io
import os
import asyncio
import logging
import time
from typing import List
import ccxt.pro as ccxt

# پیکربندی خروجی ترمینال برای پشتیبانی از کاراکترهای فارسی در ویندوز
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# تنظیم سطح لاگینگ کلی پروژه ROBORDER-X به صورت هوشمند و بدون خطای قطع ترمینال
log_handlers = [
    logging.FileHandler("roborder_x.log", encoding="utf-8")
]

# تنها در صورتی لاگ‌ها را به کنسول می‌فرستیم که ترمینال تعاملی (TTY) متصل باشد
if sys.stdout and sys.stdout.isatty():
    log_handlers.append(logging.StreamHandler(sys.stdout))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=log_handlers
)
logger = logging.getLogger("ROBORDER.Main")

from src.config import Config
from src.core.hybrid_engine import HybridEngine, ActiveTrade
from src.core.dex_tracker import SolanaDEXTracker
from src.execution.order_executor import OrderExecutor
from src.core.dashboard_server import start_dashboard_server


async def watch_single_ticker(exchange, symbol: str, engine: HybridEngine):
    """شنود موازی تیکرهای یک جفت‌ارز خاص"""
    while True:
        try:
            ticker = await exchange.watch_ticker(symbol)
            if ticker and ticker.get('last') is not None:
                timestamp = ticker.get('timestamp') or int(asyncio.get_event_loop().time() * 1000)
                engine.feed_tick(symbol, ticker['last'], timestamp)
            await asyncio.sleep(0.001)
        except Exception as e:
            logger.error(f"Error in Ticker WebSocket stream for {symbol}: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)


async def watch_tickers_task(exchange, symbols: List[str], engine: HybridEngine):
    """مدیریت موازی وظیفه دریافت تیکرهای قیمت و پشتیبانی از افزودن پویا"""
    logger.info("📡 Starting WebSocket Ticker stream listener...")
    active_tasks = {}
    while True:
        try:
            for symbol in list(symbols):
                if symbol not in active_tasks or active_tasks[symbol].done():
                    logger.info(f"📡 Launching parallel Ticker listener task for {symbol}...")
                    active_tasks[symbol] = asyncio.create_task(watch_single_ticker(exchange, symbol, engine))
            await asyncio.sleep(1.0)
        except Exception as e:
            logger.error(f"Error in Ticker manager loop: {e}")
            await asyncio.sleep(5)


async def watch_single_trade(exchange, symbol: str, engine: HybridEngine):
    """شنود موازی حجم معاملات مارکت یک جفت‌ارز خاص"""
    while True:
        try:
            trades = await exchange.watch_trades(symbol)
            for trade in trades:
                engine.feed_trade(
                    symbol=symbol,
                    price=trade['price'],
                    amount=trade['amount'],
                    side=trade['side'],
                    timestamp=trade['timestamp']
                )
            await asyncio.sleep(0.001)
        except Exception as e:
            logger.error(f"Error in Trades WebSocket stream for {symbol}: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)


async def watch_trades_task(exchange, symbols: List[str], engine: HybridEngine):
    """مدیریت موازی دریافت بلادرنگ معاملات نهایی شده صرافی و پشتیبانی از افزودن پویا"""
    logger.info("📡 Starting WebSocket Market Trades stream listener...")
    active_tasks = {}
    while True:
        try:
            for symbol in list(symbols):
                if symbol not in active_tasks or active_tasks[symbol].done():
                    logger.info(f"📡 Launching parallel Trades listener task for {symbol}...")
                    active_tasks[symbol] = asyncio.create_task(watch_single_trade(exchange, symbol, engine))
            await asyncio.sleep(1.0)
        except Exception as e:
            logger.error(f"Error in Trades manager loop: {e}")
            await asyncio.sleep(5)


async def watch_single_order_book(exchange, symbol: str, engine: HybridEngine):
    """شنود موازی دفترچه سفارشات LOB یک جفت‌ارز خاص"""
    while True:
        try:
            orderbook = await exchange.watch_order_book(symbol)
            bids = orderbook['bids']
            asks = orderbook['asks']
            timestamp = orderbook.get('timestamp') or int(asyncio.get_event_loop().time() * 1000)
            engine.feed_order_book(symbol, bids, asks, timestamp)
            await asyncio.sleep(0.001)
        except Exception as e:
            logger.error(f"Error in Order Book WebSocket stream for {symbol}: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)


async def watch_order_books_task(exchange, symbols: List[str], engine: HybridEngine):
    """مدیریت موازی دریافت بلادرنگ عمق بازار دفترچه سفارشات و پشتیبانی از افزودن پویا"""
    logger.info("📡 Starting WebSocket Order Book stream listener...")
    active_tasks = {}
    while True:
        try:
            for symbol in list(symbols):
                if symbol not in active_tasks or active_tasks[symbol].done():
                    logger.info(f"📡 Launching parallel Order Book listener task for {symbol}...")
                    active_tasks[symbol] = asyncio.create_task(watch_single_order_book(exchange, symbol, engine))
            await asyncio.sleep(1.0)
        except Exception as e:
            logger.error(f"Error in Order Book manager loop: {e}")
            await asyncio.sleep(5)


async def display_status_loop(engine: HybridEngine, executor: OrderExecutor):
    """حلقه چاپی بلادرنگ وضعیت سلامت ربات، دروداون روزانه و پوزیشن‌های باز"""
    await asyncio.sleep(5)  # تاخیر اولیه جهت لود شدن استریم‌ها
    while True:
        try:
            # بررسی اینکه آیا به ترمینال تعاملی متصل هستیم یا در پس‌زمینه اجرا می‌شویم
            # اگر TTY فعال نباشد، از اجرای پرینت کنسول جهت جلوگیری از خطای Broken Pipe اجتناب می‌کنیم
            if not sys.stdout or not sys.stdout.isatty():
                await asyncio.sleep(10)
                continue

            os.system('cls' if os.name == 'nt' else 'clear')
            print("=========================================================================")
            print("                🤖 ROBORDER-X Hybrid Scalper Engine Active               ")
            print("=========================================================================")
            print(f"وضعیت معامله: {'زنده (LIVE)' if executor.live_trading else 'شبیه‌ساز (PAPER TRADING)'}")
            print(f"ارز مرجع معاملات: {Config.QUOTE_DENOMINATION}")
            print(f"حد ضرر روزانه حساب: ${executor.max_drawdown_limit_usdt:.2f} USDT")
            print(f"میزان دروداون روزانه فعلی: ${executor.current_drawdown:.2f} USDT")
            print(f"تعداد پوزیشن‌های باز همزمان: {len(executor.open_positions)} / {executor.max_concurrent_positions}")
            
            print("\n📌 پوزیشن‌های باز زنده:")
            if not executor.open_positions:
                print("   هیچ موقعیت بازی وجود ندارد.")
            else:
                for sym, pos in executor.open_positions.items():
                    print(f"   • {sym} | جهت: {pos['side'].upper()} | اهرم: {pos['leverage']}x | قیمت ورود: ${pos['entry_price']:.6f} | حد سود: ${pos['tp']:.6f} | حد ضرر: ${pos['sl']:.6f}")

            print("\n📊 آخرین آمار اندیکاتورهای ریاضی، فیلترهای اردر بوک و نقدینگی زنجیره‌ای:")
            for sym in engine.symbols:
                lob = engine.latest_lob_results.get(sym)
                active = engine.yoyo.active_trades.get(sym)
                
                # استخراج حجم سواپ‌های بلاکچین سولانا (DEX) در ۱۰ ثانیه اخیر
                dex_trades = engine.recent_dex_trades.get(sym, [])
                dex_buy = sum([t["amount"] for t in dex_trades if t["side"] == "buy"])
                dex_sell = sum([t["amount"] for t in dex_trades if t["side"] == "sell"])

                if active:
                    status_str = f"ACTIVE {active['side'].upper()} ({active['status'].upper()})"
                else:
                    status_str = "FLAT"
                
                if lob:
                    print(
                        f"   • {sym:<18} | وضعیت: {status_str:<12} | OBI: {lob['raw_obi']:+.2f} | "
                        f"خرید ۱۰ث: {lob['market_buy_vol']:<7.1f} | فروش ۱۰ث: {lob['market_sell_vol']:<7.1f} | "
                        f"DEX خرید: ${dex_buy:<6.1f} | DEX فروش: ${dex_sell:<6.1f} | "
                        f"فریب: {lob['spoof_type']}"
                    )
                else:
                    print(f"   • {sym:<18} | وضعیت: {status_str:<12} | در حال انتظار برای دریافت دیتای وب‌سوکت...")

            print("=========================================================================")
            print("توقف ربات با فشردن Ctrl+C")
            await asyncio.sleep(1.0)
        except Exception as e:
            err_msg = str(e).lower()
            if "broken pipe" in err_msg or "errno 32" in err_msg or (hasattr(e, 'errno') and e.errno in (32, 107, 109)):
                # نادیده گرفتن و خواب موقت در زمان قطع ناگهانی ترمینال خروجی استاندارد
                await asyncio.sleep(10)
            else:
                logger.error(f"Error in status loop: {e}")
                await asyncio.sleep(2)


async def macro_news_sync_loop():
    """تسک پس‌زمینه جهت به‌روزرسانی خودکار تقویم اقتصادی اخبار ماکرو به صورت روزانه"""
    from src.core.macro_news_updater import update_macro_news_file
    while True:
        try:
            logger.info("📅 Running scheduled economic calendar update (Macro News Sync)...")
            await asyncio.get_event_loop().run_in_executor(None, update_macro_news_file)
        except Exception as e:
            logger.error(f"Error in macro news sync task: {e}")
            
        # هر ۲۴ ساعت یکبار اجرا می‌شود (86400 ثانیه)
        await asyncio.sleep(86400)


async def main():
    logger.info("==========================================================")
    logger.info("       🚀 Welcome to ROBORDER-X Production Engine 🚀       ")
    logger.info("==========================================================")

    # Auto-configure crontab for macro news updates on Linux
    if sys.platform != "win32":
        try:
            import subprocess
            logger.info("🔍 Checking system crontab configuration...")
            result = subprocess.run("crontab -l", shell=True, capture_output=True, text=True)
            current_cron = result.stdout or ""
            if "macro_news_updater.py" not in current_cron:
                logger.info("🔧 Auto-configuring crontab for macro_news_updater.py...")
                cron_cmd = '(crontab -l 2>/dev/null; echo "0 0 * * * /home/ubuntu/opt/ROBORDER/venv/bin/python /home/ubuntu/opt/ROBORDER/src/core/macro_news_updater.py >> /home/ubuntu/opt/ROBORDER/roborder_x.log 2>&1") | crontab -'
                subprocess.run(cron_cmd, shell=True, check=True)
                logger.info("✅ Crontab configured successfully!")
            else:
                logger.info("📅 Crontab is already configured for macro news updates.")
        except Exception as e:
            logger.warning(f"⚠️ Failed to auto-configure crontab: {e}")

    # ۱. راه‌اندازی ماژول اجرای فرامین و کنترل ریسک پیش‌معاملاتی (PTRC) از روی متغیرهای محیطی
    executor = OrderExecutor(
        exchange_id=Config.EXCHANGE_ID,
        live_trading=Config.ROBORDER_LIVE,
        max_concurrent_positions=Config.MAX_CONCURRENT_POSITIONS,
        max_drawdown_limit_usdt=Config.MAX_DRAWDOWN_LIMIT_USDT
    )

    # ۲. راه‌اندازی هسته هوشمند هیبریدی از روی متغیرهای محیطی
    engine = HybridEngine(
        symbols=Config.SYMBOLS,
        quote_denomination=Config.QUOTE_DENOMINATION,
        depth_levels=Config.LOB_DEPTH_LEVELS,
        trade_window_seconds=Config.TRADE_WINDOW_SECONDS,
        spoof_threshold_pct=Config.SPOOF_THRESHOLD_PCT,
        momentum_window_ms=Config.MOMENTUM_WINDOW_MS,
        spread_threshold_bps=Config.SPREAD_THRESHOLD_BPS,
        take_profit_bps=Config.TAKE_PROFIT_BPS,
        stop_loss_bps=Config.STOP_LOSS_BPS,
        cooldown_ms=Config.COOLDOWN_MS,
        history_file_path=Config.HISTORY_FILE_PATH
    )

    # راه‌اندازی وب سرور داشبورد تعاملی در ترد پس‌زمینه مستقل
    # ارسال رفرنس حلقه اصلی asyncio جهت زمان‌بندی ایمن تسک‌ها از ترد HTTP
    _main_loop = asyncio.get_running_loop()
    start_dashboard_server(engine, executor, port=6006, loop=_main_loop)

    # ۳. تعریف کالبک ناهمگام و اتصال خروجی‌های معتبر موتور هیبریدی به کلاینت صرافی
    async def handle_execute_entry(trade):
        import time
        success = await executor.execute_entry(
            symbol=trade["symbol"],
            side=trade["side"],
            amount_usdt=trade["amount"],
            leverage=trade["leverage"],
            take_profit_quote=trade["take_profit_quote"],
            stop_loss_quote=trade["stop_loss_quote"],
            entry_price=trade["entry_price_usdt"]
        )
        if not success:
            logger.warning(f"⚠️ Exchange order execution failed. Immediately cancelling local pending trade for {trade['symbol']}...")
            engine.yoyo._cancel_trade(trade["symbol"], int(time.time() * 1000))
        else:
            logger.info(f"🟢 Trade execution verified successfully for {trade['symbol']}. Logging entry and activating monitor...")
            if hasattr(engine, "yoyo") and trade["symbol"] in engine.yoyo.active_trades:
                local_trade = engine.yoyo.active_trades[trade["symbol"]]
                local_trade["status"] = "filled"
                local_trade["timestamp"] = int(time.time() * 1000)
                
                # ثبت رویداد ورود در تاریخچه سیگنال‌های ربات
                engine.yoyo.log_signal({
                    "symbol": trade["symbol"],
                    "type": "BUY" if trade["side"] == "long" else "SELL",
                    "price": trade["entry_price_usdt"],
                    "leverage": trade["leverage"],
                    "time": int(time.time()),
                    "strategy": "PurePPOStrategy"
                })

    engine.set_execution_callbacks(
        on_execute_entry=lambda trade: asyncio.create_task(handle_execute_entry(trade)),
        on_execute_exit=lambda sym, trade, exit_price, pnl, reason: asyncio.create_task(
            executor.execute_exit(
                symbol=sym,
                pnl_usdt=pnl,
                reason=reason
            )
        )
    )

    # ۴. راه‌اندازی کلاینت پایش و ردیابی معاملات زنجیره‌ای صرافی غیرمتمرکز سولانا (DEX)
    dex_tracker = SolanaDEXTracker(symbols=Config.SYMBOLS)
    # اتصال تراکنش‌های بلاکچینی شنود شده به بافرهای هیبرید انجین
    dex_tracker.set_callback(lambda sym, side, amt: engine.feed_dex_trade(sym, side, amt))
    await dex_tracker.start()

    # ۵. راه‌اندازی پشته وب‌سوکت CCXT Pro به صورت داینامیک بر اساس آدرس وب‌سوکت سفارشی و شناسه صرافی
    exchange_class = getattr(ccxt, Config.EXCHANGE_ID)
    exchange_options = {
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future'
        }
    }
    
    # اعمال آدرس وب‌سوکت سفارشی در صورت تعریف در فایل تنظیمات (مثلاً آدرس QuickNode)
    if Config.CUSTOM_WS_ENDPOINT:
        exchange_options['options']['ws'] = Config.CUSTOM_WS_ENDPOINT
        logger.info(f"🔗 Using custom WebSocket endpoint: {Config.CUSTOM_WS_ENDPOINT}")

    exchange = exchange_class(exchange_options)
    
    # همگام‌سازی زمان سرور با ساعت جهانی صرافی (NTP Synchronization)
    try:
        logger.info("⏱️ Synchronizing system clock with exchange time...")
        local_before = int(time.time() * 1000)
        server_time = await exchange.fetch_time()
        local_after = int(time.time() * 1000)
        
        # محاسبه دقیق اختلاف با احتساب تاخیر شبکه (Network RTT)
        rtt = (local_after - local_before) // 2
        local_midpoint = local_before + rtt
        drift = server_time - local_midpoint
        Config.CLOCK_DRIFT_MS = drift
        logger.info(f"⏱️ Time sync completed. Clock Drift: {drift:+} ms | Network RTT: {rtt * 2} ms")
    except Exception as e:
        logger.error(f"⚠️ Failed to sync clock with exchange: {e}. Defaulting to system time.")
        Config.CLOCK_DRIFT_MS = 0
    
    # ۵. راه‌اندازی شمع‌های تاریخی برای استراتژی فعال (YoYo / PPO)
    yoyo_task = None
    await engine.yoyo.initialize_candles(exchange)
    await engine.yoyo.start()
    yoyo_task = engine.yoyo.worker_task

    try:
        # لیست تسک‌های اصلی برای اجرای موازی
        tasks = [
            watch_tickers_task(exchange, Config.SYMBOLS, engine),
            watch_trades_task(exchange, Config.SYMBOLS, engine),
            watch_order_books_task(exchange, Config.SYMBOLS, engine),
            display_status_loop(engine, executor),
            macro_news_sync_loop()
        ]
        
        # در صورتی که تسک یویو فعال است، آن را به لوپ ناظر موازی اضافه کن تا روشن بماند
        if yoyo_task and not yoyo_task.done():
            tasks.append(yoyo_task)

        # اجرای موازی و همگام تمام موتورهای معاملاتی بدون ریسک تاخیر یا لغو ناخواسته
        await asyncio.gather(*tasks)

    except KeyboardInterrupt:
        logger.info("User requested shutdown.")
    except Exception as e:
        logger.error(f"Critical error in main loop: {e}")
    finally:
        await dex_tracker.stop()
        if hasattr(engine, "yoyo") and engine.yoyo:
            await engine.yoyo.stop()
        await exchange.close()
        await executor.close_connections()
        logger.info("👋 ROBORDER-X shut down successfully.")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
