"""Reproducible, read-only research steps and their evidence."""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import re
from uuid import uuid4

from .asset_state import build_asset_state, ledger_number
from .foundation import ServiceReadError, failure_message
from .portfolio_risk import cash_budget
from .quote_contract import MarketQuote, QuoteStatus


RESEARCH_VERSION = "1"
TICKER_PATTERN = re.compile(r"\^?[A-Z0-9][A-Z0-9.=^\-]{0,23}")
EXPECTED_FAILURES = (RuntimeError, TypeError, ValueError, KeyError, OSError, ArithmeticError)


@dataclass(frozen=True)
class ResearchStep:
    name: str
    status: str
    source: str
    evidence: str


@dataclass(frozen=True)
class ResearchRun:
    run_id: str
    started_at: datetime
    symbol: str
    demo: bool
    steps: tuple[ResearchStep, ...]
    quote: MarketQuote | None


def research_symbol(value):
    symbol = value.strip().upper()
    if not TICKER_PATTERN.fullmatch(symbol):
        raise ValueError("請輸入可辨識的股票代碼，例如 2330.TW 或 AAPL。")
    return symbol


def run_research(symbol, read_quote, *, read_ledger=None, proposed=None, demo=False):
    symbol = research_symbol(symbol)
    started = datetime.now(timezone.utc)
    quote = None
    steps = []
    try:
        quote = read_quote(symbol)
        if not isinstance(quote, MarketQuote) or quote.symbol != symbol:
            raise ValueError("Unexpected quote")
        if quote.status is QuoteStatus.UNAVAILABLE:
            steps.append(ResearchStep("日線行情", "缺資料", quote.source, "目前沒有可用的日 K 棒。"))
        else:
            status = "沿用最後有效值" if quote.status is QuoteStatus.LAST_VALID else "完成"
            currency = quote.currency or "幣別未驗證"
            steps.append(ResearchStep("日線行情", status, quote.source,
                                      f"{quote.as_of.isoformat()} 日 K 棒收盤 {quote.price:,f}（{currency}）。"))
    except EXPECTED_FAILURES:
        quote = None
        steps.append(ResearchStep("日線行情", "失敗", "日線資料服務", "暫時無法取得行情。"))

    if read_ledger is not None:
        try:
            amount = ledger_number(proposed)
            if amount is None or amount < 0:
                raise ValueError("Invalid proposed amount")
            positions, settings, source = read_ledger()
            budget = cash_budget(build_asset_state(positions), settings or {})
            decision = budget.covers(amount)
            if decision is None:
                status, detail = "缺資料", "需核對帳本現金與已存最低保留額。"
            else:
                status = "預算內" if decision else "超出預算或低於保留額"
                detail = f"試算投入 NT$ {amount:,.2f}，可用預算 NT$ {budget.available:,.2f}。"
            steps.append(ResearchStep("現金預算", status, source, detail))
        except EXPECTED_FAILURES as error:
            steps.append(ResearchStep("現金預算", "失敗", "投資組合",
                                      failure_message(error, "暫時無法核對帳本現金。")))
    return ResearchRun(uuid4().hex, started, symbol, demo, tuple(steps), quote)


def public_ai_prompt(run):
    quote = run.quote
    if run.demo or quote is None or quote.status is QuoteStatus.UNAVAILABLE:
        raise ValueError("目前沒有可用的公開行情。")
    return (
        "請用繁體中文整理這一筆公開日線行情。先列可核對的事實，再列至少兩項待查資料。"
        "只可使用提供的證據。不可推論投資人的持倉、不可給買賣指令，也不可捏造即時價格或其他數據。\n"
        f"標的={quote.symbol}；資料日={quote.as_of.isoformat()}；日 K 棒收盤={quote.price}；"
        f"幣別={quote.currency or '未驗證'}；來源={quote.source}；狀態={quote.status.value}"
    )
