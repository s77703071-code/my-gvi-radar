"""TAIFEX daily institutional futures positions."""

from datetime import datetime, timedelta
from io import StringIO
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st


TAIFEX_HISTORY_URL = "https://www.taifex.com.tw/cht/3/futContractsDateView"
TAIFEX_CSV_URL = "https://www.taifex.com.tw/cht/3/futContractsDateDown"
TAIFEX_DAILY_URL = (
    "https://openapi.taifex.com.tw/v1/"
    "MarketDataOfMajorInstitutionalTradersDetailsOfFuturesContractsBytheDate"
)
INVESTORS = ("外資及陸資", "投信", "自營商")
PRIORITY_CONTRACTS = ("臺股期貨", "小型臺指期貨", "微型臺指期貨")
POSITION_COLUMNS = {
    "多方未平倉口數": "多方未平倉（口）",
    "空方未平倉口數": "空方未平倉（口）",
    "多空未平倉口數淨額": "淨未平倉（口）",
    "多空交易口數淨額": "當日交易淨口數",
}
KEYS = ["日期", "商品名稱", "法人"]
NET_POSITION = "多空未平倉口數淨額"


def parse_taifex_positions(payload):
    try:
        decoded = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        decoded = payload.decode("cp950")
    frame = pd.read_csv(StringIO(decoded), dtype=str, keep_default_na=False)
    required = ["日期", "商品名稱", "身份別", *POSITION_COLUMNS]
    if any(column not in frame for column in required):
        raise ValueError("TAIFEX futures response lacks required columns")
    frame = frame[required].rename(columns={"身份別": "法人"})
    for column in frame:
        frame[column] = frame[column].str.strip()
    date_text = frame["日期"].str.replace("/", "", regex=False)
    frame["日期"] = pd.to_datetime(date_text, format="%Y%m%d", errors="coerce")
    frame["法人"] = frame["法人"].replace({"外資": "外資及陸資"})
    frame = frame.loc[frame["日期"].notna() & frame["商品名稱"].ne("") & frame["法人"].isin(INVESTORS)].copy()
    for column in POSITION_COLUMNS:
        value = frame[column].str.replace(",", "", regex=False)
        frame[column] = pd.to_numeric(value.where(value.str.fullmatch(r"[+-]?\d+")), errors="coerce").astype("Int64")
    long_positions = frame["多方未平倉口數"]
    short_positions = frame["空方未平倉口數"]
    frame["多方未平倉口數"] = long_positions.mask(long_positions.lt(0).fillna(False))
    frame["空方未平倉口數"] = short_positions.mask(short_positions.lt(0).fillna(False))
    computed_net = frame["多方未平倉口數"] - frame["空方未平倉口數"]
    mismatch = frame[NET_POSITION].ne(computed_net).fillna(False)
    frame[NET_POSITION] = frame[NET_POSITION].mask(mismatch)
    frame = frame.drop_duplicates().drop_duplicates(subset=KEYS, keep=False)
    if frame.dropna(subset=list(POSITION_COLUMNS), how="all").empty:
        raise ValueError("TAIFEX futures response contains no valid position observations")
    return frame.sort_values(KEYS).reset_index(drop=True)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_taifex_positions(end_date):
    # The historical download rejects weekend end dates.
    history_end = end_date - timedelta(days=max(0, end_date.weekday() - 4))
    start_date = history_end - timedelta(days=14)
    try:
        response = requests.get(
            TAIFEX_CSV_URL,
            params={
                "queryStartDate": start_date.strftime("%Y/%m/%d"),
                "queryEndDate": history_end.strftime("%Y/%m/%d"),
                "commodityId": "",
            },
            timeout=10,
        )
        response.raise_for_status()
        frame = parse_taifex_positions(response.content)
        frame = frame.loc[frame["日期"].between(pd.Timestamp(start_date), pd.Timestamp(history_end))]
        if frame.dropna(subset=list(POSITION_COLUMNS), how="all").empty:
            raise ValueError("TAIFEX did not provide valid positions for the requested dates")
        return frame
    except (requests.RequestException, ValueError):
        response = requests.get(TAIFEX_DAILY_URL, timeout=8)
        response.raise_for_status()
        frame = parse_taifex_positions(response.content)
        frame = frame.loc[frame["日期"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))]
        if frame.dropna(subset=list(POSITION_COLUMNS), how="all").empty:
            raise ValueError("TAIFEX open data did not provide a recent valid position snapshot")
        return frame


def summarize_taifex_contract(frame, contract):
    days = pd.DatetimeIndex(frame["日期"].unique()).sort_values()
    positions = frame.loc[frame["商品名稱"].eq(contract)].set_index(["日期", "法人"])
    latest = positions.reindex(pd.MultiIndex.from_product([days[-1:], INVESTORS], names=["日期", "法人"]))
    latest = latest.droplevel("日期")[list(POSITION_COLUMNS)].copy()
    trend = positions[NET_POSITION].unstack("法人").reindex(index=days, columns=INVESTORS)
    previous = days[-2] if len(days) >= 2 else None
    latest["淨未平倉較前日（口）"] = trend.iloc[-1] - trend.iloc[-2] if previous is not None else pd.NA
    latest = latest.rename(columns=POSITION_COLUMNS).reset_index()
    return latest, trend, days[-1], previous


def render_taifex_positions():
    st.markdown("#### 法人期貨布局")
    st.caption(f"資料來源：[臺灣期貨交易所｜三大法人各期貨契約]({TAIFEX_HISTORY_URL})。依期交所公布日期顯示。")
    state_key = "taifex_last_futures_positions"
    prior = st.session_state.get(state_key)
    try:
        today = datetime.now(ZoneInfo("Asia/Taipei")).date()
        with st.spinner("讀取期交所三大法人期貨資料…"):
            frame = fetch_taifex_positions(today)
        if prior is not None and frame["日期"].max() < prior["日期"].max():
            raise ValueError("TAIFEX returned an older reporting date")
        st.session_state[state_key] = frame
    except (requests.RequestException, ValueError):
        if prior is None:
            st.info("目前未取得期交所期貨資料。")
            return
        frame = prior
        st.warning("期交所資料暫時無法更新，以下顯示本次工作階段最後取得的資料。")

    available = set(frame["商品名稱"])
    contracts = [contract for contract in PRIORITY_CONTRACTS if contract in available]
    contracts.extend(sorted(available.difference(PRIORITY_CONTRACTS)))
    contract = st.selectbox("期貨契約", contracts, key="taifex_institutional_contract")
    latest, trend, reporting_day, previous = summarize_taifex_contract(frame, contract)
    comparison = f"。對照日 {previous:%Y/%m/%d}" if previous is not None else "。目前僅取得一個資料日"
    st.caption(f"資料日 {reporting_day:%Y/%m/%d}{comparison}。表中空白表示資料暫缺。")
    if latest[list(POSITION_COLUMNS.values())].isna().all().all():
        st.info("這項契約在最新資料日未取得法人資料，可查看下方已有的歷史紀錄。")
    st.dataframe(latest, hide_index=True, use_container_width=True)
    if len(trend.dropna(how="all")) >= 2:
        st.markdown("##### 淨未平倉走勢（口）")
        st.line_chart(trend)
    st.caption(
        "淨未平倉為多方口數減空方口數。當日交易淨口數是成交統計，"
        "未平倉變化則以兩個資料日的部位相減。法人數字是同類機構的合計，解讀時可一併考慮避險與結算。"
    )
