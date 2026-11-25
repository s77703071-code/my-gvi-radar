"""Standalone demonstration with fixed values, without an account or live services."""
import streamlit as st

from zhengxu.research_workflow_view import render_research_workflow

st.set_page_config(page_title="正序研究工作流｜離線示範", layout="wide")
render_research_workflow({"id": 1}, None, demo_only=True)
