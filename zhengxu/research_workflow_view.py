"""Workstation entry point for the research workflow preview."""

from datetime import date
from decimal import Decimal

import streamlit as st

from .chart_quotes import load_chart_market_quote
from .foundation import AccountScope
from .portfolio_research import DEMO_POSITIONS
from .quote_contract import MarketQuote, QuoteKind, QuoteStatus
from .research_ai import generate_content_with_retry
from .research_workflow import RESEARCH_VERSION, public_ai_prompt, run_research
from .risk_review import DEMO_SETTINGS


WORKFLOW_STATE = "workbench_research_runs"


def _demo_quote(symbol):
    return MarketQuote(symbol, "固定示範", QuoteKind.CHART_BAR, QuoteStatus.AVAILABLE,
                       Decimal("150"), date(2026, 1, 2), "USD")


def _signed_in_runs(account):
    owner = AccountScope.from_session(account)
    state = st.session_state.get(WORKFLOW_STATE)
    if state is None or state["owner"] != owner:
        state = {"owner": owner, "runs": [], "ai": {}}
        st.session_state[WORKFLOW_STATE] = state
    return state


def _gemini_key():
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except (KeyError, FileNotFoundError):
        return ""


def render_research_workflow(account, read_ledger, *, demo_only=False):
    st.subheader("研究工作流（測試版）")
    state = _signed_in_runs(account)
    with st.form("workbench_research_run"):
        demo = demo_only or st.checkbox("使用固定示範資料", value=True, key="workbench_research_demo")
        symbol = st.text_input("研究股票代碼", value="AAPL", max_chars=24, key="workbench_research_symbol", disabled=demo_only,
                               help="示範模式固定使用 AAPL。")
        check_budget = st.checkbox("一併檢查現金預算", value=True, key="workbench_research_budget")
        proposed = st.number_input("預計投入金額（TWD）", min_value=0.0, value=100.0, step=50.0,
                                   key="workbench_research_amount")
        submitted = st.form_submit_button("執行研究")
    if submitted:
        quote_loader = _demo_quote if demo else lambda ticker: load_chart_market_quote(ticker, {"p": "1mo", "i": "1d"})[1]
        ledger_loader = (lambda: (DEMO_POSITIONS, DEMO_SETTINGS, "固定示範帳本")) if demo else read_ledger
        try:
            run = run_research("AAPL" if demo else symbol, quote_loader,
                               read_ledger=ledger_loader if check_budget else None, proposed=proposed, demo=demo)
        except ValueError as error:
            st.warning(str(error))
        else:
            state["runs"] = [run, *state["runs"][:9]]
            st.session_state["workbench_research_selected"] = run.run_id
            visible_ids = {item.run_id for item in state["runs"]}
            state["ai"] = {key: value for key, value in state["ai"].items() if key in visible_ids}
    if not state["runs"]:
        return

    runs = {item.run_id: item for item in state["runs"]}
    if st.session_state.get("workbench_research_selected") not in runs:
        st.session_state["workbench_research_selected"] = next(iter(runs))
    selected = st.selectbox("本次登入的研究紀錄", tuple(runs), key="workbench_research_selected",
                            format_func=lambda key: f"{runs[key].symbol} · "
                            f"{'示範' if runs[key].demo else '實際資料'} · {key[:8]}")
    run = runs[selected]
    st.caption(f"規則版本 {RESEARCH_VERSION}。示範與實際資料各自標示，登出後清除研究紀錄。")
    st.dataframe([{
        "步驟": step.name, "狀態": step.status, "來源": step.source, "證據": step.evidence,
    } for step in run.steps], hide_index=True, width="stretch")

    if not run.demo and run.quote is not None and run.quote.status is not QuoteStatus.UNAVAILABLE:
        st.caption("Gemini 解讀使用此任務的公開行情資料。")
        key = _gemini_key()
        if not key:
            st.info("目前未設定 Gemini。可繼續查看上方的程式查核結果。")
        elif st.button("請 Gemini 解讀公開行情", key="workbench_research_generate"):
            try:
                result = generate_content_with_retry(public_ai_prompt(run), key)
                if not isinstance(result, str) or not result.strip():
                    raise ValueError("Empty response")
            except Exception:
                st.error("Gemini 暫時無法完成解讀，可稍後重試。")
            else:
                state["ai"][run.run_id] = result
        if run.run_id in state["ai"]:
            st.markdown("#### Gemini 解讀（待覆核）")
            st.write(state["ai"][run.run_id])
