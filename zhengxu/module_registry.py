"""Shared navigation catalog for the existing analysis modules."""

from types import MappingProxyType


# Entries with the same category stay together while preserving the
# module names used by session state and page dispatch.
MODULE_GROUPS = MappingProxyType({
    "📊 彩色 K 線圖與成交量": "📈 圖表與籌碼",
    "📦 籌碼流向參考": "📈 圖表與籌碼",
    "🤖 網格自動生成器": "⚖️ 策略與回測",
    "⚖️ 網格／持有／定期定額比較": "⚖️ 策略與回測",
    "🧪 技術指標自訂策略回測": "⚖️ 策略與回測",
    "💸 配息複利策略": "⚖️ 策略與回測",
    "🧠 智慧型雙策略網格系統": "⚖️ 策略與回測",
    "🚀 華爾街機構級三核心策略雷達": "🏦 選股與市場掃描",
    "🏦 台股三大法人獨立選股策略": "🏦 選股與市場掃描",
    "📊 當日全市場三大法人佔股本比 Top 10": "🏦 選股與市場掃描",
    "🧭 投資組合理財": "🧭 投資組合",
})

DUAL_GRID_MODULE = "🧠 智慧型雙策略網格系統"
