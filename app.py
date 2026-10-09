# ------------------------------------------------------------------
# 【美股專屬基本面備援邏輯】當 yfinance.info 缺值時自動解析財務報表
# ------------------------------------------------------------------
if (book_value is None or roe_val is None) and (".TW" not in t and ".TWO" not in t):
    try:
        # 1. 嘗試從 Balance Sheet 抓取總股東權益與總股數計算每股淨值
        bs = stock.quarterly_balance_sheet
        if bs.empty: bs = stock.balance_sheet
        if not bs.empty:
            for key in ['Total Stockholder Equity', 'Stockholders Equity', 'Common Stock Equity']:
                if key in bs.index:
                    tot_equity = bs.loc[key].iloc[0]
                    shares = info.get('sharesOutstanding') or info.get('impliedSharesOutstanding')
                    if tot_equity and shares and shares > 0 and book_value is None:
                        book_value = float(tot_equity) / float(shares)
                        bv_source = "yfinance 美股資產負債表推算"
                    break

        # 2. 嘗試從 Income Statement 淨利與股東權益計算 ROE (Net Income / Equity)
        inc = stock.quarterly_financials
        if inc.empty: inc = stock.financials
        if not inc.empty and not bs.empty:
            for net_key in ['Net Income', 'Net Income Common Stockholders']:
                if net_key in inc.index:
                    # 計算近四季累計淨利 (TTM Net Income)
                    net_income_ttm = inc.loc[net_key].iloc[:4].sum() if len(inc.loc[net_key]) >= 4 else inc.loc[net_key].iloc[0] * 4
                    if tot_equity and tot_equity > 0 and roe_val is None:
                        roe_val = parse_roe(net_income_ttm / tot_equity)
                        roe_source = "yfinance 美股財報推算 ROE"
                    break
    except Exception as us_e:
        pass