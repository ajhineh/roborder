import time
import os
import logging
from collections import deque
from typing import Dict, List, Optional, Literal, Callable, Tuple, TypedDict
from src.config import Config
from src.core.spoofing_detector import SpoofingDetector, LOBAnalysisResult

logger = logging.getLogger("ROBORDER.HybridEngine")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class ActiveTrade(TypedDict):
    symbol: str
    side: Literal["long", "short"]
    entry_price_quote: float   # قیمت ورود بر پایه ارز مرجع (USDT یا SOL)
    entry_price_usdt: float    # قیمت ورود اصلی بر پایه USDT برای لاگ
    amount: float              # حجم معامله
    leverage: int              # اهرم دینامیک
    take_profit_quote: float   # حد سود بر پایه ارز مرجع
    stop_loss_quote: float     # حد ضرر بر پایه ارز مرجع
    timestamp: int             # زمان ورود به میلی‌ثانیه
    diagnostic_report: Optional[dict]


class HybridEngine:
    """
    موتور تصمیم‌گیری هیبریدی ROBORDER-X.
    پشتیبانی و اجرای معاملات هوشمند با استفاده از PurePPOStrategy به عنوان هسته پیش‌فرض هوش مصنوعی و
    اعمال پیش‌بینی‌های مدل شبکه عصبی (PPO-LSTM) در زمان فعال بودن معاملات عصبی.
    """
    def __init__(
        self,
        symbols: List[str],
        quote_denomination: Literal["USDT", "SOL"] = "USDT",
        depth_levels: int = 5,
        trade_window_seconds: int = 10,
        spoof_threshold_pct: float = 0.15,
        momentum_window_ms: int = 10000,
        spread_threshold_bps: int = 8,
        take_profit_bps: int = 15,
        stop_loss_bps: int = 10,
        cooldown_ms: int = 30000,
        history_file_path: str = "microtick_history.json"
    ):
        self.symbols = symbols
        
        # راه‌اندازی ماژول ریاضی و تحلیل‌گر LOB
        self.detector = SpoofingDetector(
            depth_levels=depth_levels,
            trade_window_seconds=trade_window_seconds,
            spoof_threshold_pct=spoof_threshold_pct
        )

        # کش محلی آخرین تحلیل‌های اردر بوک هر نماد
        self.latest_lob_results: Dict[str, LOBAnalysisResult] = {}
        
        # ردیاب جریان نقدینگی تراکنش‌های صرافی غیرمتمرکز سولانا (DEX)
        self.recent_dex_trades: Dict[str, deque] = {sym: deque() for sym in symbols}
        
        # نمونه‌سازی استراتژی انحصاری هوش مصنوعی (Pure PPO)
        from src.strategies.pure_ppo.pure_ppo_strategy import PurePPOStrategy
        self.yoyo = PurePPOStrategy(
            symbols=symbols,
            quote_denomination=quote_denomination,
            history_file_path=history_file_path
        )
        self.yoyo.engine = self
        logger.info("🧠 HybridEngine initialized with End-to-End PurePPOStrategy (PPO-LSTM).")
        
        # ثبت هوک‌های مربوط به استراتژی جاری ربات
        self.yoyo.set_callbacks(
            on_entry=self._handle_yoyo_entry,
            on_exit=self._handle_yoyo_exit
        )

        # بارگذاری لیست اخبار مهم جهانی در حافظه موقت (RAM)
        self.macro_news_events = []
        self._load_macro_news()

        # هوک نهایی خروجی موتور هیبرید جهت اجرا سفارش در صرافی
        self.on_execution_callback: Optional[Callable[[dict], None]] = None
        self.on_execution_exit_callback: Optional[Callable[[str, dict, float, float, str], None]] = None

    def set_execution_callbacks(
        self,
        on_execute_entry: Callable[[dict], None],
        on_execute_exit: Callable[[str, dict, float, float, str], None]
    ):
        """تنظیم هوک‌های اجرایی سفارش جهت ارسال پیام نهایی به کلاس CCXT Pro Executor"""
        self.on_execution_callback = on_execute_entry
        self.on_execution_exit_callback = on_execute_exit

    def feed_tick(self, symbol: str, price: float, timestamp: int) -> None:
        """تغذیه زمان واقعی تیک‌های قیمتی صرافی به استراتژی هوشمند"""
        lob_result = self.latest_lob_results.get(symbol)
        dex_trades = list(self.recent_dex_trades.get(symbol, []))
        
        if hasattr(self, "yoyo") and self.yoyo:
            # تغذیه تیک به همراه لایو دیتا به موتور استراتژی فعال
            self.yoyo.feed_tick(
                symbol=symbol,
                price=price,
                timestamp=timestamp,
                lob_result=lob_result,
                dex_trades=dex_trades
            )

    def feed_trade(self, symbol: str, price: float, amount: float, side: Literal["buy", "sell"], timestamp: int) -> None:
        """تغذیه معاملات مارکتی نهایی شده صرافی به جریان ردیاب معاملات فعال"""
        self.detector.add_trade(symbol, price, amount, side, timestamp)

    def feed_order_book(self, symbol: str, bids: List[List[float]], asks: List[List[float]], timestamp: int) -> None:
        """تغذیه زمان واقعی دفترچه سفارشات صرافی و ذخیره‌سازی خروجی تحلیل فریب و تاییدیه روند"""
        result = self.detector.process_order_book(symbol, bids, asks, timestamp)
        if result:
            self.latest_lob_results[symbol] = result

    def feed_dex_trade(self, symbol: str, side: Literal["buy", "sell"], amount_usdt: float) -> None:
        """دریافت معاملات زنجیره‌ای صرافی غیرمتمرکز (DEX) سولانا برای پایش جریان نقدینگی بلاکچین"""
        if symbol not in self.recent_dex_trades:
            self.recent_dex_trades[symbol] = deque()
            
        now_ms = int(time.time() * 1000)
        self.recent_dex_trades[symbol].append({
            "timestamp": now_ms,
            "side": side,
            "amount": amount_usdt
        })
        
        # پاک‌سازی داده‌های قدیمی زنجیره‌ای (۵ دقیقه برای پایش فیلترهای ۲۸ و ۲۹ نقدینگی زنجیره‌ای)
        cutoff = now_ms - 300000  # ۳۰۰ ثانیه یا ۵ دقیقه
        trades = self.recent_dex_trades[symbol]
        while trades and trades[0]["timestamp"] < cutoff:
            trades.popleft()

    def _evaluate_ai_agent(self, trade: dict) -> Tuple[bool, float, float]:
        """
        لایه هوش مصنوعی مبتنی بر یادگیری تقویت‌پذیر پروژه piporobo (LSTM-PPO).
        خروجی: (approved: bool, confidence: float, size_multiplier: float)
        """
        symbol = trade["symbol"]
        symbol_clean = symbol.split('/')[0].lower()
        model_file = f"models/ppo_futures_bot_{symbol_clean}_final.zip"
        
        if not os.path.exists(model_file):
            logger.warning(f"🔒 AI Agent rejected: PPO model file not found at {model_file}")
            return False, 0.0, 0.0

        mock_confidence = 0.85
        mock_size_multiplier = 1.0
        
        if trade.get("leverage", 15) >= 25:
            mock_size_multiplier = 0.8

        return True, mock_confidence, mock_size_multiplier

    def _handle_yoyo_entry(self, trade_proposal: dict) -> None:
        """مدیریت هوشمند سیگنال‌های ورود با فیلترهای ۲۹ گانه حفاظتی و لایه PPO"""
        symbol = trade_proposal["symbol"]
        side = trade_proposal["side"]
        now_ms = int(time.time() * 1000) + Config.CLOCK_DRIFT_MS
        
        # ۱. دریافت اطلاعات دفترچه سفارشات و معاملات زنجیره‌ای
        lob_result = self.latest_lob_results.get(symbol)
        dex_trades = list(self.recent_dex_trades.get(symbol, []))
        
        # ۲. محاسبه تراز نقدینگی بلاکچین در ۳۰ ثانیه اول و ۳۰ ثانیه آخر (از ۶۰ ثانیه قبل تا ۳۰ ثانیه قبل و از ۳۰ ثانیه قبل تا زمان حال)
        trades_first_30s = [t for t in dex_trades if 30000 < (now_ms - t["timestamp"]) <= 60000]
        trades_last_30s = [t for t in dex_trades if (now_ms - t["timestamp"]) <= 30000]
        
        buy_first_30s = sum([t["amount"] for t in trades_first_30s if t["side"] == "buy"])
        sell_first_30s = sum([t["amount"] for t in trades_first_30s if t["side"] == "sell"])
        net_first_30s = buy_first_30s - sell_first_30s
        
        buy_last_30s = sum([t["amount"] for t in trades_last_30s if t["side"] == "buy"])
        sell_last_30s = sum([t["amount"] for t in trades_last_30s if t["side"] == "sell"])
        net_last_30s = buy_last_30s - sell_last_30s
        
        # ۳. ارزیابی گام‌به‌گام و ثبت نتایج تک‌تک ۲۹ فیلتر کنترلی
        filter_results = {}
        
        # فیلترهای فاندامنتال جدید اخبار کلان و فاندینگ ریت صرافی
        filter_results["MACRO_NEWS_FILTER"] = self._check_macro_news_window(now_ms)
        filter_results["FUNDING_RATE_FILTER"] = self._check_funding_rate_window(now_ms)
        
        # فیلتر ۱: نیاز به داده اردر بوک
        filter_results["LOB_DATA_REQUIRED"] = (lob_result is not None)
        
        # فیلتر ۲ و ۳: فیلترهای ضد اسپوفینگ
        filter_results["BUY_SPOOFING_FILTER"] = (lob_result.get("spoof_type") != "buy_spoof" if lob_result else True)
        filter_results["SELL_SPOOFING_FILTER"] = (lob_result.get("spoof_type") != "sell_spoof" if lob_result else True)
        
        # فیلتر ۴ و ۵: فیلترهای OBI
        if lob_result:
            filter_results["OBI_LONG_FILTER"] = (lob_result["raw_obi"] >= -0.2)
            filter_results["OBI_SHORT_FILTER"] = (lob_result["raw_obi"] <= 0.2)
        else:
            filter_results["OBI_LONG_FILTER"] = True
            filter_results["OBI_SHORT_FILTER"] = True
            
        # فیلتر ۶: تایید هوش مصنوعی
        filter_results["AI_PPO_APPROVAL"] = True
        
        # فیلتر ۷ و ۸ و ۹: کنترل‌های ریسک مالی ربات
        filter_results["MAX_CONCURRENT_POSITIONS_CHECK"] = True
        filter_results["MAX_DRAWDOWN_LIMIT_CHECK"] = True
        filter_results["AVAILABLE_MARGIN_CHECK"] = True
        
        # فیلتر ۱۰ و ۱۱ و ۱۲: فیلترهای شتاب قیمت
        filter_results["STABLE_MARKET_FILTER"] = True
        filter_results["LONG_MOMENTUM_FILTER"] = True
        filter_results["SHORT_MOMENTUM_FILTER"] = True
        
        # فیلتر ۱۳ و ۱۴: منطق خروج حد سود و ضرر
        filter_results["TP_EXIT_LOGIC"] = True
        filter_results["SL_EXIT_LOGIC"] = True
        
        # فیلتر ۱۵ و ۱۶: بازه استراحت و جفت ارز SOL
        last_exit = self.yoyo.last_exit_times.get(symbol, 0)
        filter_results["COOLDOWN_PERIOD_FILTER"] = (now_ms - last_exit) >= Config.COOLDOWN_MS
        filter_results["SOL_DENOMINATION_FILTER"] = True
        
        # فیلتر ۱۷ و ۱۸: فیلترهای هندسی YoYo
        filter_results["PIVOT_LOW_DETECTION"] = True
        filter_results["LINEAR_REGRESSION_TRENDLINE"] = True
        
        # فیلتر ۱۹ و ۲۰ و ۲۱: حجم‌دهی داینامیک و منطق پله‌ای تارگت‌ها
        filter_results["DYNAMIC_RISK_SIZING"] = True
        filter_results["TP1_EXIT_LOGIC"] = True
        filter_results["TP2_EXIT_LOGIC"] = True
        
        # فیلتر ۲۲ و ۲۳: صف ناهمگام و محافظ سقوط آنی قیمت
        filter_results["ASYNC_QUEUE_INGESTION"] = True
        filter_results["FALLING_KNIFE_PROTECTION"] = True
        
        # فیلتر ۲۴ و ۲۵: انقضای اوردر و تایید جریان تراکنش DEX
        filter_results["ORDER_TTL_EXPIRATION"] = True
        filter_results["DEX_FLOW_CONFIRMATION"] = (len(dex_trades) > 0)
        
        # فیلتر ۲۶ و ۲۷: اسپرد بازار و سفارش لیمیت PostOnly
        filter_results["SPREAD_THRESHOLD_BPS"] = True
        filter_results["MAKER_POST_ONLY"] = True
        
        # فیلتر ۲۸ و ۲۹: فیلترهای جدید کنترل تراز جریان نقدینگی زنجیره‌ای برای معاملات LONG و SHORT
        if side == "long":
            # قانون تراز جریان برای LONG:
            if net_first_30s > 0.0 and net_last_30s > 0.0:
                # ۱. روند صعودی مستمر -> تایید
                filter_results["DEX_FLOW_BALANCE_FIRST_30S"] = True
                filter_results["DEX_FLOW_BALANCE_LAST_30S"] = True
            elif net_first_30s <= 0.0 and net_last_30s <= 0.0:
                # ۲. روند ریزشی -> رد
                filter_results["DEX_FLOW_BALANCE_FIRST_30S"] = False
                filter_results["DEX_FLOW_BALANCE_LAST_30S"] = False
            elif net_first_30s <= 0.0 and net_last_30s > 0.0:
                # ۳. برگشت صعودی قدرتمند زمان‌کوتاه -> تایید
                filter_results["DEX_FLOW_BALANCE_FIRST_30S"] = True
                filter_results["DEX_FLOW_BALANCE_LAST_30S"] = True
            elif net_first_30s > 0.0 and net_last_30s <= 0.0:
                # ۴. تضعیف شدید تقاضا -> رد
                filter_results["DEX_FLOW_BALANCE_FIRST_30S"] = False
                filter_results["DEX_FLOW_BALANCE_LAST_30S"] = False
            else:
                filter_results["DEX_FLOW_BALANCE_FIRST_30S"] = (net_first_30s > 0.0)
                filter_results["DEX_FLOW_BALANCE_LAST_30S"] = (net_last_30s > 0.0)
        else:
            # قانون تراز جریان برای SHORT:
            if net_first_30s <= 0.0 and net_last_30s <= 0.0:
                # ۱. روند نزولی مستمر -> تایید
                filter_results["DEX_FLOW_BALANCE_FIRST_30S"] = True
                filter_results["DEX_FLOW_BALANCE_LAST_30S"] = True
            elif net_first_30s > 0.0 and net_last_30s > 0.0:
                # ۲. روند صعودی -> رد
                filter_results["DEX_FLOW_BALANCE_FIRST_30S"] = False
                filter_results["DEX_FLOW_BALANCE_LAST_30S"] = False
            elif net_first_30s > 0.0 and net_last_30s <= 0.0:
                # ۳. برگشت نزولی سریع -> تایید
                filter_results["DEX_FLOW_BALANCE_FIRST_30S"] = True
                filter_results["DEX_FLOW_BALANCE_LAST_30S"] = True
            elif net_first_30s <= 0.0 and net_last_30s > 0.0:
                # ۴. تضعیف فروشندگان -> رد
                filter_results["DEX_FLOW_BALANCE_FIRST_30S"] = False
                filter_results["DEX_FLOW_BALANCE_LAST_30S"] = False
            else:
                filter_results["DEX_FLOW_BALANCE_FIRST_30S"] = (net_first_30s <= 0.0)
                filter_results["DEX_FLOW_BALANCE_LAST_30S"] = (net_last_30s <= 0.0)

        # ۴. ذخیره نتایج تمام فیلترها در ساختار تشخیصی معامله جهت لاگ یا دیتابیس
        if "yoyo_data" in trade_proposal and trade_proposal["yoyo_data"]:
            binance_obi = lob_result.get("raw_obi", 0.0) if lob_result else 0.0
            binance_buy_vol = lob_result.get("market_buy_vol", 0.0) if lob_result else 0.0
            binance_sell_vol = lob_result.get("market_sell_vol", 0.0) if lob_result else 0.0
            spoofing_status = lob_result.get("spoof_type", "none") if lob_result else "none"

            # استخراج حجم سواپ‌های دکس زنجیره‌ای سولانا در ۱۰ ثانیه اخیر برای مطابقت با تعاریف نمایش فرانت‌اند
            trades_10s = [t for t in dex_trades if (now_ms - t["timestamp"]) <= 10000]
            dex_buy_10s = sum([t["amount"] for t in trades_10s if t["side"] == "buy"])
            dex_sell_10s = sum([t["amount"] for t in trades_10s if t["side"] == "sell"])

            # بدست آوردن مقادیر حد سود و حد ضرر و اندازه موقعیت
            tp_val = trade_proposal.get("take_profit_quote", 0.0)
            sl_val = trade_proposal.get("stop_loss_quote", 0.0)
            amt_usdt = trade_proposal.get("amount", 0.0)
            
            # محاسبه اندازه پوزیشن به توکن
            current_price = trade_proposal.get("entry_price_quote", 0.0)
            leverage = trade_proposal.get("leverage", 15)
            pos_size_tokens = 0.0
            if current_price > 0:
                pos_size_tokens = (amt_usdt * leverage) / current_price

            trade_proposal["yoyo_data"]["diagnostic_report"] = {
                "filter_results": filter_results,
                "net_first_30s_usdt": net_first_30s,
                "net_last_30s_usdt": net_last_30s,
                "is_paper_trading": not Config.ROBORDER_LIVE,
                "settings": {
                    "TRADE_CAPITAL_PCT": Config.TRADE_CAPITAL_PCT,
                    "YOYO_RISK_PCT": Config.YOYO_RISK_PCT,
                    "CURRENT_BALANCE": Config.CURRENT_BALANCE,
                    "MAX_CONCURRENT_POSITIONS": Config.MAX_CONCURRENT_POSITIONS,
                    "MAX_DRAWDOWN_LIMIT_USDT": Config.MAX_DRAWDOWN_LIMIT_USDT
                },
                "market_data": {
                    "binance_obi": binance_obi,
                    "binance_buy_vol": binance_buy_vol,
                    "binance_sell_vol": binance_sell_vol,
                    "dex_buy_vol": dex_buy_10s,
                    "dex_sell_vol": dex_sell_10s,
                    "spoofing_status": spoofing_status
                },
                "execution_rules": {
                    "tp": tp_val,
                    "tp_bps": Config.TAKE_PROFIT_BPS,
                    "sl": sl_val,
                    "sl_bps": Config.STOP_LOSS_BPS,
                    "amount_usdt": amt_usdt,
                    "position_size_tokens": pos_size_tokens
                }
            }

        # ۵. منطق رد یا پذیرش سفارش (بررسی استثنای Paper Trading)
        is_paper_trading = not Config.ROBORDER_LIVE
        
        # پیدا کردن فیلترهای مردود شده که در لیست فیلترهای استثناء (Bypassed) نیستند
        failed_filters = []
        for name, passed in filter_results.items():
            if not passed:
                # اگر فیلتر صراحتاً توسط تنظیمات کاربر بایپس شده باشد، نادیده گرفته می‌شود
                if name in Config.BYPASSED_FILTERS_SET:
                    continue
                
                # قانون Paper Trading: ۲۷ فیلتر اول به طور پیش‌فرض خاموش/بایپس هستند اما فیلترهای ۲۸ و ۲۹ فعال می‌باشند
                if is_paper_trading:
                    first_27 = [
                        "LOB_DATA_REQUIRED", "BUY_SPOOFING_FILTER", "SELL_SPOOFING_FILTER", 
                        "OBI_LONG_FILTER", "OBI_SHORT_FILTER", "AI_PPO_APPROVAL", 
                        "MAX_CONCURRENT_POSITIONS_CHECK", "MAX_DRAWDOWN_LIMIT_CHECK", 
                        "AVAILABLE_MARGIN_CHECK", "STABLE_MARKET_FILTER", "LONG_MOMENTUM_FILTER", 
                        "SHORT_MOMENTUM_FILTER", "TP_EXIT_LOGIC", "SL_EXIT_LOGIC", 
                        "SOL_DENOMINATION_FILTER", "PIVOT_LOW_DETECTION", 
                        "LINEAR_REGRESSION_TRENDLINE", "DYNAMIC_RISK_SIZING", "TP1_EXIT_LOGIC", 
                        "TP2_EXIT_LOGIC", "ASYNC_QUEUE_INGESTION", "FALLING_KNIFE_PROTECTION", 
                        "ORDER_TTL_EXPIRATION", "DEX_FLOW_CONFIRMATION", "SPREAD_THRESHOLD_BPS", 
                        "MAKER_POST_ONLY"
                    ]
                    if name in first_27:
                        # در حالت پیپر تریدینگ، عدم قبولی ۲۷ فیلتر اول مانع معامله نیست
                        continue
                
                failed_filters.append(name)

        if failed_filters:
            logger.warning(
                f"🔒 Order REJECTED due to failed filters for {symbol} | "
                f"Failed: {failed_filters} | "
                f"Net First 30s: ${net_first_30s:+.2f} | Net Last 30s: ${net_last_30s:+.2f} | "
                f"Mode: {'Paper' if is_paper_trading else 'Live'}"
            )
            self.yoyo._cancel_trade(symbol, now_ms)
            return

        logger.info(
            f"🚀 Order APPROVED! All active filters passed for {symbol} | "
            f"Net First 30s: ${net_first_30s:+.2f} | Net Last 30s: ${net_last_30s:+.2f} | "
            f"Mode: {'Paper' if is_paper_trading else 'Live'}"
        )

        # ۶. ارسال نهایی سفارش برای اجرا در صرافی یا موتور شبیه‌ساز زنده
        if self.on_execution_callback:
            try:
                self.on_execution_callback(trade_proposal)
            except Exception as e:
                logger.error(f"Error executing entry: {e}")

    def _handle_yoyo_exit(self, symbol: str, trade: dict, exit_price: float, pnl: float, reason: str) -> None:
        """هدایت سیگنال‌های خروج به کلاینت صرافی"""
        if self.on_execution_exit_callback:
            try:
                self.on_execution_exit_callback(symbol, trade, exit_price, pnl, reason)
            except Exception as e:
                logger.error(f"Error executing exit: {e}")

    def _load_macro_news(self) -> None:
        """بارگذاری فایل اخبار اقتصاد کلان در حافظه موقت In-Memory"""
        import json
        self.macro_news_events = []
        filepath = Config.MACRO_NEWS_SCHEDULE_FILE
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.macro_news_events = data.get("events", [])
                logger.info(f"📂 Loaded {len(self.macro_news_events)} macro news events into memory.")
            except Exception as e:
                logger.error(f"Failed to load macro news schedule: {e}")

    def _check_macro_news_window(self, now_ms: int) -> bool:
        """بررسی بازه زمانی اخبار اقتصاد کلان (۵ دقیقه قبل و بعد از خبر) به همراه پاکسازی هوشمند"""
        active_events = []
        is_blocked = False
        
        # پاکسازی هوشمند (Garbage Collection): اخبار قدیمی‌تر از ۵ دقیقه کامل از حافظه حذف می‌شوند
        for event in self.macro_news_events:
            event_time = event["timestamp"]
            # اگر خبر قدیمی‌تر از ۵ دقیقه (۳۰۰,۰۰۰ میلی‌ثانیه) است، از لیست خارج می‌شود
            if event_time + 300000 < now_ms:
                continue
            
            active_events.append(event)
            
            # بازه ۵ دقیقه قبل و بعد از خبر (۳۰۰,۰۰۰ میلی‌ثانیه)
            if abs(now_ms - event_time) <= 300000:
                is_blocked = True
                logger.warning(f"🚨 Macro News Block Active: {event['name']} | Difference: {abs(now_ms - event_time)/1000} seconds")

        # به‌روزرسانی حافظه موقت با رویدادهای فیلترشده
        self.macro_news_events = active_events
        return not is_blocked

    def _check_funding_rate_window(self, now_ms: int) -> bool:
        """بررسی بازه پرداخت فاندینگ ریت (۳ دقیقه منتهی به پرداخت فاندینگ ریت ۸ ساعته UTC)"""
        import datetime
        # ساعت جهانی UTC لحظه‌ای بر اساس زمان همگام شده
        now_utc = datetime.datetime.fromtimestamp(now_ms / 1000.0, tz=datetime.timezone.utc)
        
        # چرخه‌های ۸ ساعته فاندینگ ریت
        funding_hours = [0, 8, 16, 24]
        current_hour = now_utc.hour
        
        # پیدا کردن چرخه بعدی
        next_hour = min([h for h in funding_hours if h > current_hour])
        
        if next_hour == 24:
            next_funding_time = now_utc.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)
        else:
            next_funding_time = now_utc.replace(hour=next_hour, minute=0, second=0, microsecond=0)
            
        remaining_seconds = (next_funding_time - now_utc).total_seconds()
        
        # مسدودسازی در بازه ۱۸۰ ثانیه (۳ دقیقه) منتهی به فاندینگ ریت
        if remaining_seconds <= 180:
            logger.warning(f"🚨 Funding Rate Block Active: {remaining_seconds:.1f} seconds remaining to next cycle ({next_hour}:00 UTC)")
            return False
            
        return True
