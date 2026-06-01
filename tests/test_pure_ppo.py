import sys
import os
import unittest
import asyncio
import time
import numpy as np

# اضافه کردن مسیر پروژه به PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.hybrid_engine import HybridEngine
from src.config import Config
from src.core.rl_shared.state_parser import RLStateParser
from src.core.rl_shared.model_loader import RLModelLoader


class TestPurePPOAndFilters(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.symbols = ["POPCAT/USDT:USDT", "SOL/USDT:USDT"]
        self.history_file = "test_ppo_history.json"
        
        if os.path.exists(self.history_file):
            os.remove(self.history_file)

        # فعال‌سازی تنظیمات اجرای خالص PPO
        Config.USE_ONLY_PPO = True
        Config.USE_YOYO_STRATEGY = False
        Config.ROBORDER_LIVE = False  # Paper Trading به عنوان پیش‌فرض
        Config.BYPASSED_FILTERS = ""
        Config.BYPASSED_FILTERS_SET = set()
        Config.CURRENT_BALANCE = 10000.0

        # ساخت نمونه موتور هیبریدی
        self.engine = HybridEngine(
            symbols=self.symbols,
            quote_denomination="USDT",
            depth_levels=3,
            trade_window_seconds=10,
            spoof_threshold_pct=0.15,
            history_file_path=self.history_file
        )

        # شروع ترد پس‌زمینه Pure PPO
        await self.engine.yoyo.start()

        self.approved_entries = []
        self.approved_exits = []

        def on_execute_entry(trade):
            self.approved_entries.append(trade)

        def on_execute_exit(symbol, trade, exit_price, pnl, reason):
            self.approved_exits.append((symbol, trade, exit_price, pnl, reason))

        self.engine.set_execution_callbacks(on_execute_entry, on_execute_exit)
        self.symbol = "POPCAT/USDT:USDT"

    async def asyncTearDown(self):
        await self.engine.yoyo.stop()
        if os.path.exists(self.history_file):
            os.remove(self.history_file)

    def test_dependency_injection(self):
        """تایید صحت تزریق وابستگی و بارگذاری استراتژی PurePPOStrategy به جای YoYoStrategy"""
        self.assertEqual(type(self.engine.yoyo).__name__, "PurePPOStrategy")

    def test_state_parser_output_shape(self):
        """تایید صحت پردازشگر بردار ویژگی ۱۲ بعدی"""
        lob_result = {
            "best_bid": 1.48,
            "best_ask": 1.49,
            "depth_imbalance": 0.3
        }
        dex_trades = [
            {"timestamp": int(time.time() * 1000), "side": "buy", "amount": 1000.0},
            {"timestamp": int(time.time() * 1000), "side": "sell", "amount": 400.0}
        ]
        
        obs = RLStateParser.parse_market_state(
            symbol=self.symbol,
            lob_result=lob_result,
            dex_trades=dex_trades,
            volatility_ratio=0.03,
            account_position=0.0,
            max_inventory=10.0,
            mid_price=1.485
        )
        
        # ابعاد بردار ویژگی ورودی به مدل باید دقیقاً ۱۲ باشد
        self.assertEqual(obs.shape, (12,))
        self.assertEqual(obs[0], 0.0)      # Position ratio
        self.assertEqual(obs[1], 0.5)      # Progress
        self.assertAlmostEqual(obs[2], 0.01 / 1.485, places=5)  # Spread ratio
        self.assertEqual(obs[3], 0.3)      # Depth Imbalance (OBI)
        
        # تایید محاسبه احساسات نقدینگی زنجیره‌ای سولانا (DEX Sentiment)
        # (1000 - 400) / (1000 + 400) = 600 / 1400 = 0.4285
        self.assertAlmostEqual(obs[9], 0.42857, places=4)

    def test_solana_dex_buffer_expansion(self):
        """تایید افزایش مدت نگهداری تراکنش‌های DEX به ۵ دقیقه (۳۰۰ ثانیه) در موتور هیبریدی"""
        now_ms = int(time.time() * 1000)
        
        # تراکنش ۱: ۵ دقیقه و ۱۰ ثانیه پیش (خارج از بافر - باید پاکسازی شود)
        self.engine.feed_dex_trade(self.symbol, "buy", 500.0)
        self.engine.recent_dex_trades[self.symbol][-1]["timestamp"] = now_ms - 310000
        
        # تراکنش ۲: ۴ دقیقه پیش (باید در بافر حفظ شود)
        self.engine.feed_dex_trade(self.symbol, "sell", 300.0)
        self.engine.recent_dex_trades[self.symbol][-1]["timestamp"] = now_ms - 240000

        # تراکنش ۳: ۳۰ ثانیه پیش (باید حفظ شود)
        self.engine.feed_dex_trade(self.symbol, "buy", 800.0)
        self.engine.recent_dex_trades[self.symbol][-1]["timestamp"] = now_ms - 30000

        # فراخوانی مجدد جهت اعمال روتین پاکسازی بافر
        self.engine.feed_dex_trade(self.symbol, "buy", 100.0)
        
        # تایید تعداد تراکنش‌های معتبر در بافر (تراکنش اول پاک شده و تراکنش‌های بعدی به اضافه تراکنش آخر وجود دارند)
        self.assertEqual(len(self.engine.recent_dex_trades[self.symbol]), 3)

    async def test_paper_trading_filters_bypass(self):
        """بررسی استثنای Paper Trading: فیلترهای ۲۷ گانه سنتی نباید مانع ترید شوند، اما تراکنش‌های ۵ دقیقه و ۱ دقیقه پایش می‌شوند"""
        now_ms = int(time.time() * 1000)
        
        # پیش‌فرض Paper Trading: خاموش بودن ۲۷ فیلتر و روشن بودن ۲۸ و ۲۹
        Config.ROBORDER_LIVE = False
        
        # تغذیه داده‌های تراز تراکنش DEX با برآیند کاملاً مثبت در ۵ دقیقه گذشته
        self.engine.feed_dex_trade(self.symbol, "buy", 2000.0)
        
        # پیشنهاد دستی سفارش LONG (درحالی که دیتای اردر بوک LOB اصلاً وجود ندارد که فیلتر اول را رد کند)
        trade_proposal = {
            "symbol": self.symbol,
            "side": "long",
            "entry_price_quote": 1.5,
            "entry_price_usdt": 1.5,
            "amount": 100.0,
            "leverage": 15,
            "take_profit_quote": 1.6,
            "stop_loss_quote": 1.4,
            "timestamp": now_ms,
            "is_yoyo": True,
            "yoyo_data": {
                "symbol": self.symbol,
                "side": "long",
                "entry_price": 1.5,
                "amount": 100.0,
                "leverage": 15,
                "sl": 1.4,
                "tp": 1.6,
                "timestamp": now_ms,
                "status": "pending",
                "diagnostic_report": None
            }
        }
        
        # فراخوانی فیلتر موتور هیبریدی
        self.engine._handle_yoyo_entry(trade_proposal)
        
        # تایید اینکه سفارش با وجود شکست فیلترهای اردر بوک (LOB_DATA_REQUIRED)، به دلیل Paper Trading تایید شده است
        self.assertEqual(len(self.approved_entries), 1)
        self.assertIn("diagnostic_report", trade_proposal["yoyo_data"])
        
        report = trade_proposal["yoyo_data"]["diagnostic_report"]
        # بررسی ثبت وضعیت کل ۲۹ فیلتر در گزارش
        self.assertIn("filter_results", report)
        self.assertFalse(report["filter_results"]["LOB_DATA_REQUIRED"])  # ثبت فیلتر ۱ به عنوان Fail
        self.assertTrue(report["filter_results"]["DEX_FLOW_BALANCE_LAST_30S"])  # ثبت فیلتر ۲۹ به عنوان Pass

    async def test_paper_trading_flow_balance_rejection(self):
        """بررسی رد معامله در Paper Trading در صورت عدم همسویی تراز نقدینگی شبکه زنجیره‌ای (تراز منفی در سیگنال خرید)"""
        now_ms = int(time.time() * 1000)
        Config.ROBORDER_LIVE = False
        
        # برآیند تراکنش‌های DEX منفی است (فروش سنگین)
        self.engine.feed_dex_trade(self.symbol, "sell", 5000.0)
        
        # پیشنهاد سفارش LONG
        trade_proposal = {
            "symbol": self.symbol,
            "side": "long",
            "entry_price_quote": 1.5,
            "entry_price_usdt": 1.5,
            "amount": 100.0,
            "leverage": 15,
            "take_profit_quote": 1.6,
            "stop_loss_quote": 1.4,
            "timestamp": now_ms,
            "is_yoyo": True,
            "yoyo_data": {
                "symbol": self.symbol,
                "side": "long",
                "entry_price": 1.5,
                "amount": 100.0,
                "leverage": 15,
                "sl": 1.4,
                "tp": 1.6,
                "timestamp": now_ms,
                "status": "pending",
                "diagnostic_report": None
            }
        }
        
        # فراخوانی فیلترها
        self.engine._handle_yoyo_entry(trade_proposal)
        
        # تایید رد شدن سفارش به دلیل منفی بودن جریان حجمی در بلاکچین
        self.assertEqual(len(self.approved_entries), 0)

    async def test_advance_dex_flow_balance_logic(self):
        """بررسی دقیق منطق ۴ حالته جدید فیلتر تراز جریان برای پوزیشن‌های LONG"""
        # حالت اول: هر دو مثبت (1m = +3000, 30s = +2000)
        now_ms = int(time.time() * 1000)
        self.engine.recent_dex_trades[self.symbol].clear()
        self.engine.feed_dex_trade(self.symbol, "buy", 1000.0)
        self.engine.recent_dex_trades[self.symbol][-1]["timestamp"] = now_ms - 45000
        self.engine.feed_dex_trade(self.symbol, "buy", 2000.0)
        self.engine.recent_dex_trades[self.symbol][-1]["timestamp"] = now_ms - 15000
        
        trade_proposal = self._create_test_proposal(now_ms)
        self.approved_entries.clear()
        self.engine._handle_yoyo_entry(trade_proposal)
        self.assertEqual(len(self.approved_entries), 1)  # باید تایید شود

        # حالت دوم: هر دو منفی (1m = -3000, 30s = -2000)
        self.engine.recent_dex_trades[self.symbol].clear()
        self.engine.feed_dex_trade(self.symbol, "sell", 1000.0)
        self.engine.recent_dex_trades[self.symbol][-1]["timestamp"] = now_ms - 45000
        self.engine.feed_dex_trade(self.symbol, "sell", 2000.0)
        self.engine.recent_dex_trades[self.symbol][-1]["timestamp"] = now_ms - 15000
        
        trade_proposal = self._create_test_proposal(now_ms)
        self.approved_entries.clear()
        self.engine._handle_yoyo_entry(trade_proposal)
        self.assertEqual(len(self.approved_entries), 0)  # باید رد شود

        # حالت سوم: ۱ دقیقه منفی و ۳۰ ثانیه مثبت (1m = -3000, 30s = +2000) -> روند رو به صعود
        self.engine.recent_dex_trades[self.symbol].clear()
        self.engine.feed_dex_trade(self.symbol, "sell", 5000.0)
        self.engine.recent_dex_trades[self.symbol][-1]["timestamp"] = now_ms - 45000
        self.engine.feed_dex_trade(self.symbol, "buy", 2000.0)
        self.engine.recent_dex_trades[self.symbol][-1]["timestamp"] = now_ms - 15000
        
        trade_proposal = self._create_test_proposal(now_ms)
        self.approved_entries.clear()
        self.engine._handle_yoyo_entry(trade_proposal)
        self.assertEqual(len(self.approved_entries), 1)  # باید تایید شود

        # حالت چهارم: ۱ دقیقه مثبت و ۳۰ ثانیه منفی (1m = +3000, 30s = -2000) -> روند رو به نزول
        self.engine.recent_dex_trades[self.symbol].clear()
        self.engine.feed_dex_trade(self.symbol, "buy", 5000.0)
        self.engine.recent_dex_trades[self.symbol][-1]["timestamp"] = now_ms - 45000
        self.engine.feed_dex_trade(self.symbol, "sell", 2000.0)
        self.engine.recent_dex_trades[self.symbol][-1]["timestamp"] = now_ms - 15000
        
        trade_proposal = self._create_test_proposal(now_ms)
        self.approved_entries.clear()
        self.engine._handle_yoyo_entry(trade_proposal)
        self.assertEqual(len(self.approved_entries), 0)  # باید رد شود

    def _create_test_proposal(self, now_ms):
        return {
            "symbol": self.symbol,
            "side": "long",
            "entry_price_quote": 1.5,
            "entry_price_usdt": 1.5,
            "amount": 100.0,
            "leverage": 15,
            "take_profit_quote": 1.6,
            "stop_loss_quote": 1.4,
            "timestamp": now_ms,
            "is_yoyo": True,
            "yoyo_data": {
                "symbol": self.symbol,
                "side": "long",
                "entry_price": 1.5,
                "amount": 100.0,
                "leverage": 15,
                "sl": 1.4,
                "tp": 1.6,
                "timestamp": now_ms,
                "status": "pending",
                "diagnostic_report": None
            }
        }
    async def test_cooldown_period_enforcement(self):
        """تایید اجرای فیزیکی دوره خنک‌سازی در معاملات شبیه‌سازی شده"""
        now_ms = int(time.time() * 1000)
        Config.ROBORDER_LIVE = False
        Config.COOLDOWN_MS = 30000  # ۳۰ ثانیه
        
        # تغذیه داده‌های تراز تراکنش DEX با برآیند کاملاً مثبت
        self.engine.feed_dex_trade(self.symbol, "buy", 2000.0)
        
        # ثبت یک خروج معامله فرضی در همان لحظه
        self.engine.yoyo.last_exit_times[self.symbol] = now_ms
        
        # پیشنهاد سفارش LONG بلافاصله بعد از خروج (زیر ۳۰ ثانیه)
        trade_proposal = self._create_test_proposal(now_ms)
        
        # فراخوانی فیلترها
        self.approved_entries.clear()
        self.engine._handle_yoyo_entry(trade_proposal)
        
        # تایید رد شدن سفارش به دلیل قرار داشتن در دوره خنک‌سازی (حتی در پیپر تریدینگ)
        self.assertEqual(len(self.approved_entries), 0)
        
        # شبیه‌سازی گذر زمان بیش از ۳۰ ثانیه
        self.engine.yoyo.last_exit_times[self.symbol] = now_ms - 35000
        
        # فراخوانی دوباره فیلترها
        self.approved_entries.clear()
        self.engine._handle_yoyo_entry(trade_proposal)
        
        # تایید قبولی سفارش پس از اتمام دوره خنک‌سازی
        self.assertEqual(len(self.approved_entries), 1)

    def test_funding_rate_filter_block(self):
        """تایید مسدودسازی تراکنش‌ها در بازه ۳ دقیقه‌ای منتهی به پرداخت فاندینگ ریت چرخه ۸ ساعته UTC"""
        # شبیه‌سازی زمانی: ۱ دقیقه و ۳۰ ثانیه قبل از ساعت ۰۸:۰۰ UTC (یعنی ۰۷:۵۸:۳۰ UTC)
        import datetime
        mock_date = datetime.datetime(2026, 6, 1, 7, 58, 30, tzinfo=datetime.timezone.utc)
        mock_now_ms = int(mock_date.timestamp() * 1000)
        
        # بررسی فعال شدن فیلتر و رد کردن
        self.assertFalse(self.engine._check_funding_rate_window(mock_now_ms))
        
        # شبیه‌سازی زمانی: ۱۰ دقیقه قبل از ساعت ۰۸:۰۰ UTC (یعنی ۰۷:۵۰:۰۰ UTC)
        mock_date_safe = datetime.datetime(2026, 6, 1, 7, 50, 0, tzinfo=datetime.timezone.utc)
        mock_now_ms_safe = int(mock_date_safe.timestamp() * 1000)
        
        # بررسی غیرفعال بودن فیلتر و قبولی
        self.assertTrue(self.engine._check_funding_rate_window(mock_now_ms_safe))

    def test_macro_news_filter_block_and_garbage_collection(self):
        """تایید مسدودسازی در بازه ۵ دقیقه اخبار کلان و پاکسازی خودکار رویدادهای گذشته از حافظه"""
        # زمان انتشار خبر فرضی
        news_time_ms = int(time.time() * 1000)
        self.engine.macro_news_events = [
            {"name": "US CPI Release", "timestamp": news_time_ms}
        ]
        
        # ۱. شبیه‌سازی زمان در بازه ممنوعه (۲ دقیقه قبل از خبر)
        mock_now_active = news_time_ms - 120000
        self.assertFalse(self.engine._check_macro_news_window(mock_now_active))
        # تایید عدم پاکسازی خبر چون هنوز منقضی نشده است
        self.assertEqual(len(self.engine.macro_news_events), 1)
        
        # ۲. شبیه‌سازی زمان بعد از بازه ممنوعه و انقضای خبر (۶ دقیقه بعد از خبر - یعنی ۳۶۰,۰۰۰ میلی‌ثانیه)
        mock_now_expired = news_time_ms + 360000
        self.assertTrue(self.engine._check_macro_news_window(mock_now_expired))
        # تایید پاکسازی خودکار رویداد منقضی شده از آرایه حافظه موقت (Garbage Collection)
        self.assertEqual(len(self.engine.macro_news_events), 0)

    async def test_pending_order_cancellation_during_news_block(self):
        """تایید لغو خودکار سفارشات معلق (pending) به محض ورود به بازه ممنوعه فاندامنتال"""
        now_ms = int(time.time() * 1000)
        
        # ایجاد سفارش معلق در استراتژی
        trade_proposal = self._create_test_proposal(now_ms)
        self.engine.yoyo.active_trades[self.symbol] = trade_proposal["yoyo_data"]
        
        # ایجاد یک خبر کلان فعال در ردیاب موتور هیبریدی
        self.engine.macro_news_events = [
            {"name": "FOMC Interest Rate Decision", "timestamp": now_ms}
        ]
        
        # تغذیه تیک قیمتی جدید در بازه خبری (ساعت جاری)
        tick = {
            "symbol": self.symbol,
            "price": 1.5,
            "timestamp": now_ms,
            "lob_result": None,
            "dex_trades": []
        }
        
        # فراخوانی متد پردازش تیک ناهمگام
        await self.engine.yoyo._handle_tick_async(tick)
        
        # تایید اینکه سفارش معلق به دلیل وقوع خبر لغو و از لیست حذف شده است
        self.assertNotIn(self.symbol, self.engine.yoyo.active_trades)
        # تایید ثبت زمان خروج در ردیاب استراحت استراتژی
        self.assertEqual(self.engine.yoyo.last_exit_times[self.symbol], now_ms)


if __name__ == "__main__":
    unittest.main()
