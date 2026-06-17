"""Streamlit dashboard: upload customer feedback, discover topics, summarize.

Run:
    streamlit run app.py

Bilingual UI (Vietnamese / English) via a sidebar language switch. Reuses the
src/ pipeline modules. The LLM summary backend and all parameters come from
config.yaml; the API key (if any) is read from the environment, never the UI.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src import embed, preprocess, summarize, topic_model
from src.config import Config

st.set_page_config(
    page_title="Customer Feedback Analysis",
    page_icon=":material/storefront:",
    layout="wide",
)

CONFIG_PATH = "config.yaml"
RESULT_PATH = Path("output/last_result.json")  # survives page reloads
INK, VIOLET = "#1B1726", "#5B2EFF"


def save_result(out: dict) -> None:
    """Persist the latest analysis so a full page reload doesn't lose it."""
    try:
        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass  # persistence is best-effort; never break the run


def load_result() -> dict | None:
    if RESULT_PATH.exists():
        try:
            return json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

# ----------------------------------------------------------------- i18n ------
I18N = {
    "vi": {
        "eyebrow": "Voice of customer · Gen-Z fashion",
        "h1a": "Lắng nghe khách hàng,",
        "h1b": "gọn thành chủ đề",
        "hero_p": "Tải phản hồi thô lên — hệ thống tự gom nhóm và tóm tắt điều khách "
        "hàng thực sự quan tâm, từ chất vải đến giao hàng và thanh toán.",
        "sb_data": "Dữ liệu",
        "sb_options": "Tùy chọn",
        "uploader": "File phản hồi (.xlsx / .csv)",
        "col_select": "Cột chứa nội dung phản hồi",
        "toggle_sum": "Tóm tắt chủ đề bằng AI",
        "key_offline": "Tóm tắt chạy offline (`{x}`), không cần API key.",
        "key_ready": "`{x}`: đã sẵn sàng",
        "key_missing": "`{x}`: chưa được đặt",
        "key_caption": "Tắt mục này để chạy nhanh chỉ với từ khoá, hoặc đặt API key rồi tải lại.",
        "btn_analyze": "Phân tích",
        "spinner": "Đang phân tích — gom nhóm và tóm tắt phản hồi…",
        "err_analyze": "Không phân tích được: {e}",
        "err_read": "Không đọc được file: {e}",
        "err_few": "Cần ít nhất khoảng 5 dòng phản hồi hợp lệ để gom nhóm.",
        "empty": "Tải file phản hồi ở thanh bên trái để bắt đầu.",
        "stat_total": "Phản hồi",
        "stat_topics": "Chủ đề",
        "stat_noise": "Chưa gom nhóm",
        "sec_chart": "Quy mô các chủ đề",
        "sec_table": "Bảng chủ đề",
        "sec_details": "Chi tiết từng chủ đề",
        "c_topic": "Chủ đề",
        "c_count": "Số feedback",
        "c_keywords": "Từ khoá",
        "c_summary": "Tóm tắt",
        "c_sentiment": "Cảm xúc",
        "c_priority": "Ưu tiên",
        "btn_download": "Tải kết quả (CSV)",
        "exp": "Chủ đề {id} · {n} phản hồi · {kw}",
        "rep": "Phản hồi tiêu biểu",
    },
    "en": {
        "eyebrow": "Voice of customer · Gen-Z fashion",
        "h1a": "Listen to your customers,",
        "h1b": "organized into topics",
        "hero_p": "Upload raw feedback — the system automatically groups and summarizes "
        "what customers really care about, from fabric to delivery and payment.",
        "sb_data": "Data",
        "sb_options": "Options",
        "uploader": "Feedback file (.xlsx / .csv)",
        "col_select": "Feedback text column",
        "toggle_sum": "Summarize topics with AI",
        "key_offline": "Summaries run offline (`{x}`), no API key needed.",
        "key_ready": "`{x}`: ready",
        "key_missing": "`{x}`: not set",
        "key_caption": "Turn this off for a fast keyword-only run, or set the API key and reload.",
        "btn_analyze": "Analyze",
        "spinner": "Analyzing — grouping and summarizing feedback…",
        "err_analyze": "Analysis failed: {e}",
        "err_read": "Couldn't read file: {e}",
        "err_few": "Need at least ~5 valid feedback rows to cluster.",
        "empty": "Upload a feedback file in the left sidebar to start.",
        "stat_total": "Feedback",
        "stat_topics": "Topics",
        "stat_noise": "Ungrouped",
        "sec_chart": "Topic sizes",
        "sec_table": "Topic table",
        "sec_details": "Topic details",
        "c_topic": "Topic",
        "c_count": "Feedback count",
        "c_keywords": "Keywords",
        "c_summary": "Summary",
        "c_sentiment": "Sentiment",
        "c_priority": "Priority",
        "btn_download": "Download results (CSV)",
        "exp": "Topic {id} · {n} feedback · {kw}",
        "rep": "Representative feedback",
    },
}

# canonical sentiment/priority -> (vi label, en label, color)
SENT = {
    "Tích cực": ("Tích cực", "Positive", "#16A34A"),
    "Tiêu cực": ("Tiêu cực", "Negative", "#E11D48"),
    "Trung lập": ("Trung lập", "Neutral", "#6B6577"),
}
PRIO = {
    "Cao": ("Cao", "High", "#E11D48"),
    "Trung bình": ("Trung bình", "Medium", "#D97706"),
    "Thấp": ("Thấp", "Low", "#16A34A"),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0&display=block');

:root{ --ink:#1B1726; --muted:#6B6577; --canvas:#F4F2F7; --surface:#FFFFFF;
  --line:#ECE9F2; --violet:#5B2EFF; }
html, body, [data-testid="stAppViewContainer"]{ background:var(--canvas); }
body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"]{
  font-family:'Inter', system-ui, sans-serif; color:var(--ink); }
.block-container{ max-width:1080px; padding-top:2.4rem; padding-bottom:4rem; }
h1,h2,h3{ font-family:'Space Grotesk', sans-serif; color:var(--ink); letter-spacing:-.02em; }
.msr{ font-family:'Material Symbols Rounded'; line-height:1; vertical-align:middle; }

.hero{ border-bottom:1px solid var(--line); padding-bottom:22px; margin-bottom:26px; }
.hero .eyebrow{ font-family:'Space Grotesk'; font-weight:600; font-size:.72rem;
  letter-spacing:.2em; text-transform:uppercase; color:var(--violet); }
.hero h1{ font-size:2.6rem; font-weight:700; line-height:1.04; margin:.4rem 0 .55rem; }
.hero h1 span{ color:var(--violet); }
.hero p{ color:var(--muted); font-size:1.03rem; max-width:52ch; margin:0; }

.stats{ display:flex; gap:14px; margin:2px 0 8px; }
.stat{ flex:1; background:var(--surface); border:1px solid var(--line);
  border-radius:18px; padding:18px 20px; }
.stat .k{ font-family:'Space Grotesk'; font-size:2.1rem; font-weight:700; line-height:1; }
.stat .l{ font-size:.74rem; color:var(--muted); margin-top:8px;
  text-transform:uppercase; letter-spacing:.1em; }
.stat.accent{ border-color:transparent; background:var(--violet); }
.stat.accent .k, .stat.accent .l{ color:#fff; }

.sec{ display:flex; align-items:center; gap:10px; font-family:'Space Grotesk';
  font-weight:600; font-size:1.12rem; color:var(--ink); margin:30px 0 12px; }
.sec .msr{ color:var(--violet); font-size:24px; }

.stButton>button{ border-radius:12px; font-family:'Space Grotesk'; font-weight:600; padding:.5rem 1.1rem; }
[data-testid="stExpander"]{ border:1px solid var(--line) !important;
  border-left:3px solid var(--violet) !important; border-radius:14px !important;
  background:var(--surface); margin-bottom:10px; }
[data-testid="stExpander"] summary{ font-weight:600; font-family:'Space Grotesk'; }
[data-testid="stExpander"] summary:hover{ color:var(--violet); }
[data-testid="stDataFrame"]{ border:1px solid var(--line); border-radius:14px; overflow:hidden; }
[data-testid="stSidebar"]{ border-right:1px solid var(--line); }
[data-testid="stSidebar"] h2{ font-size:1rem; }
#MainMenu, footer{ visibility:hidden; }
</style>
"""


@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def read_uploaded(file) -> pd.DataFrame:
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


def api_key_status(cfg: Config):
    """Return (present, message_key, value) for the configured summary backend."""
    backend = cfg.summarizer.get("backend")
    if backend == "anthropic":
        env = "ANTHROPIC_API_KEY"
    elif backend == "openai_compatible":
        env = cfg.summarizer.get("openai_compatible", {}).get("api_key_env", "OPENROUTER_API_KEY")
    else:  # llama_cpp
        return True, "key_offline", backend
    present = bool(os.getenv(env))
    return present, ("key_ready" if present else "key_missing"), env


def analyze(df: pd.DataFrame, text_column: str, cfg: Config, do_summarize: bool, t):
    """Run the full pipeline on an in-memory DataFrame. Returns a results dict."""
    work = df[[text_column]].rename(columns={text_column: "text"})
    work["text"] = work["text"].fillna("").astype(str)

    work = preprocess.preprocess(work, cfg.preprocess)
    texts = work["clean"].tolist()
    if len(texts) < 5:
        raise ValueError(t("err_few"))

    device = embed.resolve_device(cfg.embedding.get("device", "auto"))
    model = load_embedding_model(cfg.embedding["model_name"], device)
    embeddings = model.encode(
        texts, batch_size=cfg.embedding.get("batch_size", 32), show_progress_bar=False
    )

    tmodel = topic_model.build_model(cfg, embedding_model=model)
    topics, _ = topic_model.fit(tmodel, texts, embeddings, cfg)

    info = tmodel.get_topic_info()
    info = info[info.Topic != -1]
    n_noise = int(sum(1 for t_ in topics if t_ == -1))

    smap: dict[int, dict] = {}
    if do_summarize:
        sumdf = summarize.make_summarizer(cfg.summarizer).summarize_topics(tmodel)
        smap = {int(r["Topic"]): r for _, r in sumdf.iterrows()}

    rows = []
    for _, r in info.iterrows():
        tid = int(r["Topic"])
        kws = [w for w, _ in tmodel.get_topic(tid)][: cfg.summarizer.get("max_keywords", 10)]
        s = smap.get(tid)
        rows.append(
            {
                "topic": tid,
                "count": int(r["Count"]),
                "keywords": " · ".join(kws[:6]),
                "summary": s["Summary"] if s is not None else "",
                "sentiment": s["Sentiment"] if s is not None else "",
                "priority": s["Priority"] if s is not None else "",
                "docs": tmodel.get_representative_docs(tid)[:5],
            }
        )
    return {"results": rows, "total": len(texts), "n_topics": len(rows), "n_noise": n_noise}


def icon(name: str) -> str:
    return f'<span class="msr">{name}</span>'


# ---------------------------------------------------------------- UI ---------
cfg = Config.load(CONFIG_PATH)
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    lang_label = st.radio(
        "Language", ["Tiếng Việt", "English"], horizontal=True, label_visibility="collapsed"
    )
LANG = "vi" if lang_label == "Tiếng Việt" else "en"


def t(key: str) -> str:
    return I18N[LANG].get(key, key)


def sent_disp(canon: str) -> str:
    e = SENT.get(canon)
    return e[0 if LANG == "vi" else 1] if e else ""


def prio_disp(canon: str) -> str:
    e = PRIO.get(canon)
    return e[0 if LANG == "vi" else 1] if e else ""


def badge(text: str, color: str) -> str:
    return (
        f'<span style="background:{color}1A;color:{color};border:1px solid {color}55;'
        f'padding:2px 10px;border-radius:999px;font-size:.78rem;font-weight:600;'
        f'font-family:Space Grotesk;">{text}</span>'
    )


st.markdown(
    f"""
    <div class="hero">
      <div class="eyebrow">{t('eyebrow')}</div>
      <h1>{t('h1a')}<br><span>{t('h1b')}</span></h1>
      <p>{t('hero_p')}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header(f":material/database: {t('sb_data')}")
    uploaded = st.file_uploader(t("uploader"), type=["xlsx", "xls", "csv"])

    text_column = None
    if uploaded is not None:
        try:
            preview = read_uploaded(uploaded)
            default_col = cfg.data.get("text_column")
            cols = list(preview.columns)
            idx = cols.index(default_col) if default_col in cols else 0
            text_column = st.selectbox(t("col_select"), cols, index=idx)
        except Exception as e:
            st.error(t("err_read").format(e=e))

    st.header(f":material/tune: {t('sb_options')}")
    do_summarize = st.toggle(t("toggle_sum"), value=cfg.summarizer.get("enabled", True))
    present, msg_key, value = api_key_status(cfg)
    if do_summarize:
        text = t(msg_key).format(x=value)
        (st.success if present else st.warning)(text, icon=":material/key:")
        if not present:
            st.caption(t("key_caption"))

    run = st.button(
        t("btn_analyze"),
        icon=":material/insights:",
        type="primary",
        use_container_width=True,
        disabled=uploaded is None or text_column is None,
    )

if run and uploaded is not None and text_column:
    try:
        with st.spinner(t("spinner")):
            uploaded.seek(0)
            df = read_uploaded(uploaded)
            result = analyze(df, text_column, cfg, do_summarize, t)
            st.session_state["out"] = result
            save_result(result)
    except Exception as e:
        st.error(t("err_analyze").format(e=e))

# Resolve current result: in-session first, else the last one saved to disk
# (so a full page reload / new session still shows the latest analysis).
out = st.session_state.get("out") or load_result()
if not out:
    st.info(t("empty"), icon=":material/upload_file:")
    st.stop()

# ---- stats ----
st.markdown(
    f"""
    <div class="stats">
      <div class="stat accent"><div class="k">{out['total']}</div><div class="l">{t('stat_total')}</div></div>
      <div class="stat"><div class="k">{out['n_topics']}</div><div class="l">{t('stat_topics')}</div></div>
      <div class="stat"><div class="k">{out['n_noise']}</div><div class="l">{t('stat_noise')}</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

full = pd.DataFrame(out["results"])
has_sentiment = bool(full.get("sentiment", pd.Series([], dtype=str)).astype(bool).any())
table = full[["topic", "count", "keywords", "summary"]].copy()

# ---- chart ----
st.markdown(f'<div class="sec">{icon("bar_chart")} {t("sec_chart")}</div>', unsafe_allow_html=True)
chart_df = table.copy()
chart_df["label"] = "#" + chart_df["topic"].astype(str) + "  " + chart_df["keywords"].str.split(" · ").str[0]
chart = (
    alt.Chart(chart_df)
    .mark_bar(color=VIOLET, cornerRadiusEnd=5, size=18)
    .encode(
        x=alt.X("count:Q", title=None, axis=alt.Axis(grid=False)),
        y=alt.Y("label:N", sort="-x", title=None),
        tooltip=["topic", "count", "keywords"],
    )
    .properties(height=max(180, len(chart_df) * 30))
    .configure_view(stroke=None)
    .configure_axis(labelColor=INK, labelFont="Inter", labelFontSize=12)
)
st.altair_chart(chart, use_container_width=True)

# ---- table ----
st.markdown(f'<div class="sec">{icon("table_view")} {t("sec_table")}</div>', unsafe_allow_html=True)
display = table.copy()
col_cfg = {
    "topic": st.column_config.NumberColumn(t("c_topic"), width="small"),
    "count": st.column_config.ProgressColumn(
        t("c_count"), format="%d", min_value=0, max_value=int(table["count"].max())
    ),
    "keywords": st.column_config.TextColumn(t("c_keywords"), width="medium"),
    "summary": st.column_config.TextColumn(t("c_summary"), width="large"),
}
order = ["topic", "count", "keywords", "summary"]
if has_sentiment:
    display["priority"] = full["priority"].map(prio_disp)
    display["sentiment"] = full["sentiment"].map(sent_disp)
    col_cfg["priority"] = st.column_config.TextColumn(t("c_priority"), width="small")
    col_cfg["sentiment"] = st.column_config.TextColumn(t("c_sentiment"), width="small")
    order = ["topic", "count", "priority", "sentiment", "keywords", "summary"]
st.dataframe(display[order], use_container_width=True, hide_index=True, column_config=col_cfg)

export = display[order].rename(columns={k: t(f"c_{k}") for k in order})
st.download_button(
    t("btn_download"),
    export.to_csv(index=False).encode("utf-8-sig"),
    file_name="topic_summaries.csv",
    mime="text/csv",
    icon=":material/download:",
)

# ---- details ----
st.markdown(f'<div class="sec">{icon("search")} {t("sec_details")}</div>', unsafe_allow_html=True)
for row in out["results"]:
    label = t("exp").format(id=row["topic"], n=row["count"], kw=row["keywords"])
    with st.expander(label):
        badges = []
        if row.get("sentiment") in SENT:
            badges.append(badge(sent_disp(row["sentiment"]), SENT[row["sentiment"]][2]))
        if row.get("priority") in PRIO:
            badges.append(badge(f"{t('c_priority')}: {prio_disp(row['priority'])}", PRIO[row["priority"]][2]))
        if badges:
            st.markdown(" ".join(badges), unsafe_allow_html=True)
        if row["summary"]:
            st.markdown(row["summary"])
        st.markdown(f"**{t('rep')}**")
        for d in row["docs"]:
            st.markdown(f"- {d}")
