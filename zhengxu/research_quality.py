"""Build auditable report sections separately from provider-generated interpretation."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real
import re


@dataclass(frozen=True)
class RadarEvidence:
    name: str
    matched: bool | None
    reason: str
    source: str


def _plain_markdown(value):
    # External source labels are printed as text, never as Markdown links or headings.
    return re.sub(r"([\\`*_{}\[\]<>#!|])", r"\\\1", " ".join(str(value).split()))


def build_research_evidence(*, price, quote_time, timeframe, radar_evidence, risk_findings):
    lines = ["### 程式可核對的研究資料"]
    if isinstance(price, Real) and isfinite(price) and price > 0 and quote_time:
        lines.append(f"- 研究採用價格 {_plain_markdown(f'{price:,.2f}')}，來源為 Yahoo Finance "
                     f"{_plain_markdown(timeframe)} K 棒，行情時間 {_plain_markdown(quote_time)}。")
    else:
        lines.append("- 行情資料不足，價格或行情時間無法核對。")
    lines.append("- 財報公告日未由本次資料取得。基本面快取與擷取時間只說明更新狀態。")

    groups = (
        ("支持證據", [item for item in radar_evidence if item.matched is True],
         "本次沒有可列出的達標條件。"),
        ("反對證據與風險提醒", [item for item in radar_evidence if item.matched is False]
         + [item for item in risk_findings if item.category == "風險提醒"],
         "本次規則未產生未達標條件或風險提醒。"),
        ("資料不足與待查證", [item for item in radar_evidence if item.matched is None]
         + [item for item in risk_findings if item.category == "資料不足"],
         "本次雷達與風險規則未標示缺值，仍需核對財報公告日與行情期間。"),
    )
    for title, entries, fallback in groups:
        lines.extend(("", f"#### {title}"))
        if entries:
            for entry in entries:
                name = f"{_plain_markdown(entry.name)}｜" if isinstance(entry, RadarEvidence) else ""
                lines.append(f"- {name}{_plain_markdown(entry.reason)}（來源 {_plain_markdown(entry.source)}）")
        else:
            lines.append(f"- {fallback}")
    return "\n".join(lines)


def compose_research_report(evidence, provider, analysis, sources):
    if not isinstance(analysis, str) or not analysis.strip():
        raise ValueError("AI 沒有回傳可顯示的分析內容。")
    note = "這一節為 AI 生成的解讀與推論。請用上方程式資料及原始來源核對新增的敘述與數字。"
    if provider == "ChatGPT" and not sources:
        note += " 本次服務未回傳可追溯的網路引文。"
    return f"{evidence}\n\n### AI 分析與推論（待查證）\n{note}\n\n{analysis.strip()}"
