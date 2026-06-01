import os
import json
import time
import logging
import asyncio
import random
from collections import deque
from typing import Dict, List, Optional, Callable, Literal, TypedDict
import numpy as np

from src.config import Config
from src.core.rl_shared.state_parser import RLStateParser
from src.core.rl_shared.model_loader import RLModelLoader

logger = logging.getLogger("ROBORDER.PurePPOStrategy")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class ActivePPOTrade(TypedDict, total=False):
    symbol: str
    side: Literal["long", "short"]
    entry_price: float
    amount: float
    leverage: int
    sl: float
    tp: float
    tp1: float
    tp2: float
    tp1_hit: bool
    timestamp: int
    status: Literal["pending", "filled"]
    diagnostic_report: Optional[dict]


class PurePPOStrategy:
    """
    استراتژی معاملاتی خالص مبتنی بر یادگیری تقویت‌پذیر پایان‌به‌پایان (End-to-End PPO-LSTM).
    این کلاس با تفکیک کاملا ناهمگام بخش پردازش داده از ترد اصلی صرافی، سرعت فوق‌العاده بالا
    و تصمیم‌گیری بلادرنگ عصبی بر مبنای بردار وضعیت ۱۲ بعدی را ارائه می‌دهد.
    """
    def __init__(
        self,
        symbols: List[str],
        quote_denomination: Literal["USDT", "SOL"] = "USDT",
        history_file_path: str = "microtick_history.json"
    ):
        self.symbols = symbols
        self.quote_denomination = quote_denomination
        self.history_file_path = os.path.abspath(history_file_path)

        # صف دریافت ناهمگام تیک‌های قیمتی جهت پیشگیری از تأخیر در دریافت پیام‌های وب‌سوکت
        self.tick_queue: asyncio.Queue = asyncio.Queue()
        self.worker_task: Optional[asyncio.Task] = None
        self.should_run = True

        # بافرهای تاریخچه تیک‌های قیمتی برای هر نماد (نگهداری آخرین ۶۰ ثانیه تیک‌ها)
        self.prices: Dict[str, deque] = {sym: deque() for sym in symbols}
        
        # مدیریت مدل‌های یادگیری تقویت‌پذیر و آمار نرمال‌سازی
        self.loader = RLModelLoader()
        self.models: Dict[str, Any] = {}
        self.normalized_envs: Dict[str, Any] = {}
        self.lstm_states: Dict[str, Optional[tuple]] = {sym: None for sym in symbols}

        # مدیریت موقعیت‌ها و سفارشات فعال ربات
        self.active_trades: Dict[str, ActivePPOTrade] = {}
        self.last_exit_times: Dict[str, int] = {}
        self.last_order_placed_time: Dict[str, float] = {sym: 0.0 for sym in symbols}

        # متغیرهای مربوط به اتصال LOB و تراکنش‌های DEX که از هسته هیبریدی تغذیه می‌شوند
        self.latest_lob: Optional[dict] = None
        self.recent_dex_trades: List[dict] = []

        # تاریخچه سیگنال‌ها و آمارهای مالی داشبورد ربات
        self.history = {
            "signals": [],
            "stats": {
                "totalTrades": 0,
                "wins": 0,
                "losses": 0,
                "totalPnL": 0.0
            }
        }

        # کالبک‌های خروجی جهت ارسال پیام ورود/خروج به هسته صرافی/مدیریت ریسک
        self.on_entry_callback: Optional[Callable[[dict], None]] = None
        self.on_exit_callback: Optional[Callable[[str, dict, float, float, str], None]] = None

        self.load_history()

    async def start(self) -> None:
        """راه‌اندازی ترد ناهمگام استراتژی و بارگذاری مدل‌های PPO"""
        self.should_run = True
        
        # بارگذاری پویای مدل‌های PPO برای تمام جفت‌ارزها
        for sym in self.symbols:
            model, env = self.loader.load_ppo_model(sym)
            if model is not None:
                self.models[sym] = model
                self.normalized_envs[sym] = env
                logger.info(f"🧠 PPO Model loaded successfully for {sym}")
            else:
                logger.warning(f"⚠️ Could not load PPO model weights for {sym}. Artificial mock actions will be used.")

        if not self.worker_task or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._worker_loop())
        logger.info("🚀 PurePPOStrategy background worker loop started.")

    def set_callbacks(
        self,
        on_entry: Callable[[dict], None],
        on_exit: Callable[[str, dict, float, float, str], None]
    ):
        """تنظیم کالبک‌ها جهت تعامل با هسته صرافی"""
        self.on_entry_callback = on_entry
        self.on_exit_callback = on_exit

    def load_history(self) -> None:
        try:
            if os.path.exists(self.history_file_path):
                with open(self.history_file_path, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load PPO history: {e}")

    def save_history(self) -> None:
        try:
            with open(self.history_file_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save PPO history: {e}")

    def log_signal(self, signal: dict) -> None:
        self.history["signals"].append(signal)
        self.save_history()

    async def initialize_candles(self, exchange) -> None:
        """متد تطبیق رابط کاربری جهت سازگاری کامل با هسته اصلی ربات"""
        logger.info("📡 [PurePPOStrategy] Dynamic model agent initialization completed.")
        
    def feed_tick(self, symbol: str, price: float, timestamp: int, lob_result: Optional[dict] = None, dex_trades: Optional[list] = None) -> None:
        """تغذیه ناهمگام تیک‌های قیمتی زمان‌واقعی به صف دریافت بدون بلاک کردن ترد فراخوان"""
        try:
            self.tick_queue.put_nowait({
                "symbol": symbol,
                "price": price,
                "timestamp": timestamp,
                "lob_result": lob_result,
                "dex_trades": dex_trades
            })
        except Exception as e:
            logger.error(f"Error putting tick into PPO queue: {e}")

    async def _worker_loop(self) -> None:
        """حلقه همگام‌ساز ناهمگام برای استخراج داده‌ها و اجرای پیش‌بینی مدل عصبی"""
        while self.should_run:
            try:
                tick = await self.tick_queue.get()
                await self._handle_tick_async(tick)
                self.tick_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in PPO async worker: {e}")
                await asyncio.sleep(0.1)

    async def _handle_tick_async(self, tick: dict) -> None:
        symbol = tick["symbol"]
        price = tick["price"]
        now = tick["timestamp"]
        lob_result = tick.get("lob_result")
        dex_trades = tick.get("dex_trades", [])

        # ۱. راه‌اندازی و مقداردهی تنبل (Lazy Initialization) نمادهای پویا
        if symbol not in self.prices:
            self.prices[symbol] = deque()
        if symbol not in self.lstm_states:
            self.lstm_states[symbol] = None
        if symbol not in self.last_order_placed_time:
            self.last_order_placed_time[symbol] = 0.0

        # ۲. بارگذاری تنبل مدل عصبی در صورت لزوم
        if symbol not in self.models and symbol not in self.normalized_envs:
            model, env = self.loader.load_ppo_model(symbol)
            if model is not None:
                self.models[symbol] = model
                self.normalized_envs[symbol] = env
                logger.info(f"🧠 PPO Model loaded dynamically via tick for {symbol}")
            else:
                self.models[symbol] = None
                self.normalized_envs[symbol] = None
                logger.warning(f"⚠️ Could not load PPO model weights for {symbol}. Artificial mock actions will be used.")

        # ۳. به‌روزرسانی بافر محلی قیمت‌ها
        history = self.prices[symbol]
        history.append({"price": price, "timestamp": now})
        
        # پاکسازی تیک‌های قدیمی‌تر از ۶۰ ثانیه
        cutoff = now - 60000
        while history and history[0]["timestamp"] < cutoff:
            history.popleft()

        # ۲. بررسی فیلترهای فاندامنتال (اخبار کلان و فاندینگ ریت) و لغو سفارشات معلق PPO
        active_trade = self.active_trades.get(symbol)
        if hasattr(self, "engine") and self.engine:
            drift_now_ms = now + Config.CLOCK_DRIFT_MS
            macro_ok = self.engine._check_macro_news_window(drift_now_ms)
            funding_ok = self.engine._check_funding_rate_window(drift_now_ms)
            
            if not macro_ok or not funding_ok:
                if active_trade:
                    if active_trade["status"] == "pending":
                        logger.warning(
                            f"🚨 Pending limit order for {symbol} cancelled due to fundamental block! "
                            f"Macro News: {'OK' if macro_ok else 'BLOCKED'} | Funding Rate: {'OK' if funding_ok else 'BLOCKED'}"
                        )
                        self._cancel_trade(symbol, now)
                        return
                else:
                    # هیچ موقعیت بازی نداریم و به دلیل شرایط بلاک خبری/فاندینگ ریت مجاز به ثبت سفارش جدید نیستیم
                    return

        # ۳. بررسی پوزیشن باز فعال
        if active_trade:
            await self._monitor_position_async(symbol, price, now, active_trade)
            return

        # ۳. بررسی دوره استراحت (Cooldown)
        last_exit = self.last_exit_times.get(symbol, 0)
        if (now - last_exit) < Config.COOLDOWN_MS:
            return

        # جلوگیری از ارسال سفارشات مکرر در فواصل خیلی کوتاه
        now_sec = time.time()
        if (now_sec - self.last_order_placed_time[symbol]) < 5.0:
            return

        # ۴. نمونه‌گیری وضعیت و استخراج ویژگی‌های ۱۲ بعدی
        # تخمین نوسانات متحرک ساده در بافر قیمت‌ها
        volatility_ratio = 0.02
        if len(history) >= 5:
            prices_list = [p["price"] for p in history]
            volatility_ratio = float(np.std(prices_list) / np.mean(prices_list))

        # استفاده از پارسر ویژگی برای ساخت بردار ورودی
        obs = RLStateParser.parse_market_state(
            symbol=symbol,
            lob_result=lob_result,
            dex_trades=dex_trades,
            volatility_ratio=volatility_ratio,
            account_position=0.0,  # پوزیشن فعلی صفر (آماده ورود)
            max_inventory=10.0,
            mid_price=price,
            funding_rate=0.0001,
            basis_ratio=0.0
        )

        # ۵. نرمال‌سازی بردار ورودی
        normalized_env = self.normalized_envs.get(symbol)
        obs_normalized = RLModelLoader.normalize_observation(obs, normalized_env)

        # ۶. پیش‌بینی سیاست بهینه توسط مدل عصبی PPO
        model = self.models.get(symbol)
        action_ratio = 0.0
        
        if model is not None:
            try:
                # اجرای گام پیش‌بینی به صورت کاملاً دترمینستیک با لایه حافظه‌دار LSTM
                last_state = self.lstm_states[symbol]
                # اگر مدل از نوع RecurrentPPO باشد، state را می‌گیرد
                if hasattr(model, "policy") and "Lstm" in type(model.policy).__name__:
                    # در لایو بازار گام‌ها به صورت متوالی ۱ هستند
                    episode_start = np.array([last_state is None])
                    action, next_state = model.predict(
                        obs_normalized,
                        state=last_state,
                        episode_start=episode_start,
                        deterministic=True
                    )
                    self.lstm_states[symbol] = next_state
                    action_ratio = float(action[0])
                else:
                    action, _ = model.predict(obs_normalized, deterministic=True)
                    action_ratio = float(action[0])
            except Exception as pred_err:
                logger.error(f"Error predicting action with PPO for {symbol}: {pred_err}")
                action_ratio = 0.0
        else:
            # شبیه‌سازی رفتار در صورت عدم بارگذاری وزنه مدل جهت تست یکپارچگی خط لوله اجرا
            if random.random() < 0.01:
                action_ratio = random.choice([0.8, -0.8])

        # ۷. ارزیابی حد آستانه ورود سیگنال (Trigger Threshold)
        # هر اکشنی که قدر مطلق آن بالای ۰.۲۵ باشد به عنوان سیگنال ورود تعبیر می‌شود
        if abs(action_ratio) >= 0.25:
            side: Literal["long", "short"] = "long" if action_ratio > 0 else "short"
            await self._trigger_ppo_trade(symbol, price, side, action_ratio, now)

    async def _trigger_ppo_trade(
        self,
        symbol: str,
        price: float,
        side: Literal["long", "short"],
        action_ratio: float,
        timestamp: int
    ) -> None:
        """ایجاد پوزیشن معاملاتی شبیه‌سازی شده و ارسال سیگنال پیشنهادی به هسته صرافی جهت کنترل فیلترهای ۲۹ گانه"""
        self.last_order_placed_time[symbol] = time.time()

        # ۱. محاسبه حجم داینامیک بر اساس درصد سرمایه مجاز
        max_capital = (Config.TRADE_CAPITAL_PCT / 100.0) * Config.CURRENT_BALANCE
        amount_usdt = abs(action_ratio) * max_capital
        
        # اطمینان از قرارگیری حجم در محدوده‌های ایمن
        amount_usdt = min(amount_usdt, max_capital)
        if amount_usdt <= 5.0:  # حداقل حجم معامله ۵ تتر
            return

        # ۲. محاسبه اهرم بر اساس درصد ریسک حساب و حد ضرر فعال
        try:
            sl_pct = Config.STOP_LOSS_BPS / 100.0  # حد ضرر به درصد (مثلا 0.1% برای 10 BPS)
            # اهرم به گونه‌ای محاسبه می‌شود که ضربدر حد ضرر درصد ریسک مجاز را پوشش دهد
            calculated_leverage = int(Config.YOYO_RISK_PCT / sl_pct)
            if calculated_leverage <= 0:
                calculated_leverage = Config.DEFAULT_LEVERAGE
        except Exception:
            calculated_leverage = Config.DEFAULT_LEVERAGE
            
        leverage = min(max(calculated_leverage, 1), Config.MAX_LEVERAGE)

        # ۳. محاسبه تارگت‌ها بر مبنای BPS تنظیم شده در پیکربندی پروژه
        tp_ratio = Config.TAKE_PROFIT_BPS / 10000.0
        sl_ratio = Config.STOP_LOSS_BPS / 10000.0
        tp1_ratio = (Config.STOP_LOSS_BPS * 1.5) / 10000.0 # TP1 is 1.5x risk

        if side == "long":
            tp1 = price * (1 + tp1_ratio)
            tp2 = price * (1 + tp_ratio)
            sl = price * (1 - sl_ratio)
        else:
            tp1 = price * (1 - tp1_ratio)
            tp2 = price * (1 - tp_ratio)
            sl = price * (1 + sl_ratio)

        # ۴. ایجاد پوزیشن در حالت pending
        new_trade: ActivePPOTrade = {
            "symbol": symbol,
            "side": side,
            "entry_price": price,
            "amount": amount_usdt,
            "leverage": leverage,
            "sl": sl,
            "tp": tp2,
            "tp1": tp1,
            "tp2": tp2,
            "tp1_hit": False,
            "timestamp": timestamp,
            "status": "pending",
            "diagnostic_report": None
        }

        self.active_trades[symbol] = new_trade
        logger.info(f"🏹 Neural Network PPO proposed {side.upper()} order for {symbol} at ${price:.6f} | Weight: {action_ratio:+.2f}")

        # ۴. فید کردن سیگنال به موتور تصمیم‌گیر جهت اعمال فیلترهای ۲۹ گانه
        if self.on_entry_callback:
            try:
                # آماده‌سازی دیتا منطبق با ورودی handle_execute_entry در main.py
                self.on_entry_callback({
                    "symbol": symbol,
                    "side": side,
                    "entry_price_quote": price,
                    "entry_price_usdt": price,
                    "amount": amount_usdt,
                    "leverage": leverage,
                    "take_profit_quote": tp2,
                    "stop_loss_quote": sl,
                    "timestamp": timestamp,
                    "is_yoyo": True,  # جهت تطبیق کامل با معماری کالبک بدون تغییر main
                    "yoyo_data": new_trade
                })
            except Exception as e:
                logger.error(f"Error in PPO on_entry callback execution: {e}")

    async def _monitor_position_async(self, symbol: str, price: float, now: int, trade: ActivePPOTrade) -> None:
        """پایش مداوم وضعیت تارگت‌های معامله فعال"""
        if trade["status"] == "pending":
            # در بازار لایو وضعیت به filled تغییر می‌کند، در شبیه‌ساز با برخورد قیمت پر می‌شود
            if trade["side"] == "long" and price <= trade["entry_price"]:
                trade["status"] = "filled"
                trade["timestamp"] = now
                logger.info(f"💥 PPO LONG Limit Filled for {symbol} at ${trade['entry_price']:.6f}!")
            elif trade["side"] == "short" and price >= trade["entry_price"]:
                trade["status"] = "filled"
                trade["timestamp"] = now
                logger.info(f"💥 PPO SHORT Limit Filled for {symbol} at ${trade['entry_price']:.6f}!")
            return

        # ۱. بررسی لمس حد سود اول (TP1) و تبدیل به ریسک‌فری (Break-Even) تعدیل شده با کارمزد
        tp1 = trade.get("tp1", trade["tp"])
        tp2 = trade.get("tp2", trade["tp"])
        tp1_hit = trade.get("tp1_hit", False)

        if not tp1_hit:
            # بررسی لمس TP1 برای خروج ۵۰٪ حجم
            if (trade["side"] == "long" and price >= tp1) or (trade["side"] == "short" and price <= tp1):
                half_amount = trade["amount"] * 0.5
                pnl_pct = ((tp1 - trade["entry_price"]) / trade["entry_price"]) * 100 * trade["leverage"]
                if trade["side"] == "short":
                    pnl_pct = -pnl_pct
                
                # محاسبه سود تتر
                pnl_usdt = half_amount * (pnl_pct / 100.0)
                
                logger.info(f"🎯 TP1 Hit for {symbol} | Exiting 50% volume | PnL: {pnl_pct:.2f}% ({pnl_usdt:.4f} USDT)")
                
                # بروزرسانی حجم باقی‌مانده و پرچم TP1
                trade["amount"] = half_amount
                trade["tp1_hit"] = True
                
                # انتقال حد ضرر (SL) به نقطه ورود واقعی تعدیل شده با کارمزد رفت و برگشت (0.08% برای 2 * 0.04% صرافی)
                fee_rate = 0.0004
                round_trip_fee = 2.0 * fee_rate
                if trade["side"] == "long":
                    trade["sl"] = trade["entry_price"] * (1.0 + round_trip_fee)
                else:
                    trade["sl"] = trade["entry_price"] * (1.0 - round_trip_fee)
                
                logger.info(f"🛡️ SL moved to Fee-Adjusted Break-Even for remaining 50% volume at ${trade['sl']:.6f}")
                
                # ثبت رویداد خروج پله‌ای در تاریخچه محلی
                self.log_signal({
                    "symbol": symbol,
                    "type": "SELL_EXIT" if trade["side"] == "long" else "BUY_EXIT",
                    "price": tp1,
                    "time": int(now / 1000),
                    "exitReason": "TP1",
                    "fullyExited": False,
                    "pnl": pnl_usdt,
                    "strategy": "PurePPOStrategy",
                    "diagnostic_report": trade.get("diagnostic_report")
                })
                
                # بروزرسانی آمارهای معاملاتی برای خروج پله‌ای اول
                stats = self.history["stats"]
                stats["totalTrades"] += 1
                if pnl_usdt > 0:
                    stats["wins"] += 1
                else:
                    stats["losses"] += 1
                stats["totalPnL"] = stats.get("totalPnL", 0.0) + pnl_usdt
                self.save_history()

                # فراخوانی کالبک خروج صرافی برای ۵۰٪ پوزیشن
                if self.on_exit_callback:
                    try:
                        self.on_exit_callback(symbol, trade, tp1, pnl_usdt, "TP1")
                    except Exception as e:
                        logger.error(f"Error in PPO TP1 callback: {e}")
                
                return

        # ۲. بررسی حد ضرر (SL یا BE)
        if (trade["side"] == "long" and price <= trade["sl"]) or (trade["side"] == "short" and price >= trade["sl"]):
            pnl_pct = ((trade["sl"] - trade["entry_price"]) / trade["entry_price"]) * 100 * trade["leverage"]
            if trade["side"] == "short":
                pnl_pct = -pnl_pct
            
            pnl_usdt = trade["amount"] * (pnl_pct / 100.0)
            exit_reason = "BE" if tp1_hit else "SL"
            
            logger.info(f"🚪 EXIT ({exit_reason}) PPO {trade['side'].upper()} for {symbol} at ${price:.6f} | PnL: {pnl_pct:.2f}% ({pnl_usdt:.4f} USDT)")
            self._close_ppo_position(symbol, trade, price, pnl_usdt, exit_reason, now)
            return

        # ۳. بررسی حد سود نهایی (TP2 یا همان TP اصلی)
        if (trade["side"] == "long" and price >= tp2) or (trade["side"] == "short" and price <= tp2):
            pnl_pct = ((tp2 - trade["entry_price"]) / trade["entry_price"]) * 100 * trade["leverage"]
            if trade["side"] == "short":
                pnl_pct = -pnl_pct
                
            pnl_usdt = trade["amount"] * (pnl_pct / 100.0)
            logger.info(f"🚪 EXIT (Take-Profit 2) PPO {trade['side'].upper()} for {symbol} at ${price:.6f} | PnL: {pnl_pct:.2f}% ({pnl_usdt:.4f} USDT)")
            self._close_ppo_position(symbol, trade, price, pnl_usdt, "TP2" if tp1_hit else "TP", now)
            return

    def _cancel_trade(self, symbol: str, now: int) -> None:
        """ابطال معامله در صورت رد شدن توسط فیلترهای کنترلی"""
        if symbol in self.active_trades:
            trade = self.active_trades[symbol]
            del self.active_trades[symbol]
            self.last_exit_times[symbol] = now
            self.lstm_states[symbol] = None  # بازنشانی حافظه LSTM برای معامله بعدی
            
            if self.on_exit_callback:
                try:
                    self.on_exit_callback(symbol, trade, trade["entry_price"], 0.0, "CANCEL")
                except Exception as e:
                    logger.error(f"Error in PPO cancel callback: {e}")

    def _close_ppo_position(
        self,
        symbol: str,
        trade: ActivePPOTrade,
        exit_price: float,
        pnl_usdt: float,
        reason: str,
        now: int
    ) -> None:
        """بستن کامل پوزیشن و به‌روزرسانی داشبورد مالی"""
        del self.active_trades[symbol]
        self.last_exit_times[symbol] = now
        self.lstm_states[symbol] = None

        # ثبت خروج در تاریخچه محلی
        self.log_signal({
            "symbol": symbol,
            "type": "SELL_EXIT" if trade["side"] == "long" else "BUY_EXIT",
            "price": exit_price,
            "time": int(now / 1000),
            "exitReason": reason,
            "fullyExited": True,
            "pnl": pnl_usdt,
            "strategy": "PurePPOStrategy",
            "diagnostic_report": trade.get("diagnostic_report")
        })

        # بروزرسانی آمارهای معاملاتی
        stats = self.history["stats"]
        stats["totalTrades"] += 1
        if pnl_usdt > 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1
        stats["totalPnL"] = stats.get("totalPnL", 0.0) + pnl_usdt
        self.save_history()

        # فعال‌سازی کالبک خروج صرافی
        if self.on_exit_callback:
            try:
                self.on_exit_callback(symbol, trade, exit_price, pnl_usdt, reason)
            except Exception as e:
                logger.error(f"Error in PPO on_exit callback: {e}")

    def force_close_position(self, symbol: str, current_price: float, now: int) -> bool:
        """بستن فوری و دستی یک موقعیت معاملاتی توسط کاربر از داشبورد"""
        if symbol not in self.active_trades:
            return False
            
        trade = self.active_trades[symbol]
        entry = trade["entry_price"]
        side = trade["side"]
        leverage = trade["leverage"]
        amount = trade["amount"]
        
        if side == "long":
            pnl_pct = ((current_price - entry) / entry) * 100.0 * leverage
        else:
            pnl_pct = ((entry - current_price) / entry) * 100.0 * leverage
            
        pnl_usdt = amount * (pnl_pct / 100.0)
        logger.info(f"🚪 FORCE CLOSE PPO {side.upper()} for {symbol} at ${current_price:.6f} | PnL: {pnl_pct:+.2f}% ({pnl_usdt:+.4f} USDT)")
        
        self._close_ppo_position(
            symbol=symbol,
            trade=trade,
            exit_price=current_price,
            pnl_usdt=pnl_usdt,
            reason="FORCE_DASHBOARD_CLOSE",
            now=now
        )
        return True

    async def stop(self) -> None:
        """توقف ایمن ترد ناهمگام استراتژی"""
        self.should_run = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        logger.info("🔌 PurePPOStrategy background task stopped.")
