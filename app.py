"""Streamlit dashboard: upload customer feedback, discover topics, summarize.

Run:
    streamlit run app.py

Reuses the src/ pipeline modules. The LLM summary backend and all parameters
come from config.yaml; the API key (if any) is read from the environment, never
entered in the UI.
"""
from __future__ import annotations

import os

import altair as alt
import pandas as pd
import streamlit as st

from src import embed, preprocess, summarize, topic_model
from src.config import Config

st.set_page_config(
    page_title="Tiếng nói khách hàng",
    page_icon=":material/storefront:",
    layout="wide",
)

CONFIG_PATH = "config.yaml"

# Palette
INK = "#1B1726"
VIOLET = "#5B2EFF"
CORAL = "#FF5C8A"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0&display=block');

:root{
  --ink:#1B1726; --muted:#6B6577; --canvas:#F4F2F7;
  --surface:#FFFFFF; --line:#ECE9F2; --violet:#5B2EFF;
  --violet-soft:#EEEAFF; --coral:#FF5C8A;
}

html, body, [data-testid="stAppViewContainer"]{ background:var(--canvas); }
body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"]{
  font-family:'Inter', system-ui, sans-serif; color:var(--ink);
}
.block-container{ max-width:1080px; padding-top:2.4rem; padding-bottom:4rem; }
h1,h2,h3{ font-family:'Space Grotesk', sans-serif; color:var(--ink); letter-spacing:-.02em; }

.msr{ font-family:'Material Symbols Rounded'; font-weight:normal; line-height:1;
  vertical-align:middle; -webkit-font-feature-settings:'liga'; }

/* ---- hero ---- */
.hero{ border-bottom:1px solid var(--line); padding-bottom:22px; margin-bottom:26px; }
.hero .eyebrow{ font-family:'Space Grotesk'; font-weight:600; font-size:.72rem;
  letter-spacing:.2em; text-transform:uppercase; color:var(--violet); }
.hero h1{ font-size:2.6rem; font-weight:700; line-height:1.04; margin:.4rem 0 .55rem; }
.hero h1 span{ color:var(--violet); }
.hero p{ color:var(--muted); font-size:1.03rem; max-width:52ch; margin:0; }

/* ---- stat cards ---- */
.stats{ display:flex; gap:14px; margin:2px 0 8px; }
.stat{ flex:1; background:var(--surface); border:1px solid var(--line);
  border-radius:18px; padding:18px 20px; }
.stat .k{ font-family:'Space Grotesk'; font-size:2.1rem; font-weight:700; line-height:1; }
.stat .l{ font-size:.74rem; color:var(--muted); margin-top:8px;
  text-transform:uppercase; letter-spacing:.1em; }
.stat.accent{ border-color:transparent; background:var(--violet); }
.stat.accent .k, .stat.accent .l{ color:#fff; }

/* ---- section header ---- */
.sec{ display:flex; align-items:center; gap:10px; font-family:'Space Grotesk';
  font-weight:600; font-size:1.12rem; color:var(--ink); margin:30px 0 12px; }
.sec .msr{ color:var(--violet); font-size:24px; }

/* ---- buttons ---- */
.stButton>button{ border-radius:12px; font-family:'Space Grotesk'; font-weight:600;
  padding:.5rem 1.1rem; }

/* ---- expanders as topic cards ---- */
[data-testid="stExpander"]{ border:1px solid var(--line) !important;
  border-left:3px solid var(--violet) !important; border-radius:14px !important;
  background:var(--surface); margin-bottom:10px; }
[data-testid="stExpander"] summary{ font-weight:600; font-family:'Space Grotesk'; }
[data-testid="stExpander"] summary:hover{ color:var(--violet); }

/* ---- dataframe ---- */
[data-testid="stDataFrame"]{ border:1px solid var(--line); border-radius:14px; overflow:hidden; }

/* ---- sidebar ---- */
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


def api_key_status(cfg: Config) -> tuple[bool, str]:
    """Return (key_present, human_message) for the configured summary backend."""
    backend = cfg.summarizer.get("backend")
    if backend == "anthropic":
        env = "ANTHROPIC_API_KEY"
    elif backend == "openai_compatible":
        env = cfg.summarizer.get("openai_compatible", {}).get("api_key_env", "OPENROUTER_API_KEY")
    else:  # llama_cpp
        return True, f"Tóm tắt chạy offline (`{backend}`), không cần API key."
    present = bool(os.getenv(env))
    msg = f"`{env}`: " + ("đã sẵn sàng" if present else "chưa được đặt")
    return present, msg


def analyze(df: pd.DataFrame, text_column: str, cfg: Config, do_summarize: bool):
    """Run the full pipeline on an in-memory DataFrame. Returns a results dict."""
    work = df[[text_column]].rename(columns={text_column: "text"})
    work["text"] = work["text"].fillna("").astype(str)

    work = preprocess.preprocess(work, cfg.preprocess)
    texts = work["clean"].tolist()
    if len(texts) < 5:
        raise ValueError("Cần ít nhất khoảng 5 dòng phản hồi hợp lệ để gom nhóm.")

    device = embed.resolve_device(cfg.embedding.get("device", "auto"))
    model = load_embedding_model(cfg.embedding["model_name"], device)
    embeddings = model.encode(
        texts, batch_size=cfg.embedding.get("batch_size", 32), show_progress_bar=False
    )

    tmodel = topic_model.build_model(cfg, embedding_model=model)
    topics, _ = topic_model.fit(tmodel, texts, embeddings, cfg)

    info = tmodel.get_topic_info()
    info = info[info.Topic != -1]
    n_noise = int(sum(1 for t in topics if t == -1))

    summaries: dict[int, str] = {}
    if do_summarize:
        summaries = {
            int(r["Topic"]): r["Summary"]
            for _, r in summarize.make_summarizer(cfg.summarizer)
            .summarize_topics(tmodel)
            .iterrows()
        }

    rows = []
    for _, r in info.iterrows():
        tid = int(r["Topic"])
        keywords = [w for w, _ in tmodel.get_topic(tid)][: cfg.summarizer.get("max_keywords", 10)]
        rows.append(
            {
                "Topic": tid,
                "Số feedback": int(r["Count"]),
                "Keywords": " · ".join(keywords[:6]),
                "Tóm tắt": summaries.get(tid, ""),
                "_docs": tmodel.get_representative_docs(tid)[:5],
            }
        )
    return {"results": rows, "total": len(texts), "n_topics": len(rows), "n_noise": n_noise}


def icon(name: str) -> str:
    return f'<span class="msr">{name}</span>'


# ---------------------------------------------------------------- UI ---------
cfg = Config.load(CONFIG_PATH)
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Voice of customer · Gen-Z fashion</div>
      <h1>Lắng nghe khách hàng,<br><span>gọn thành chủ đề</span></h1>
      <p>Tải phản hồi thô lên — hệ thống tự gom nhóm và tóm tắt điều khách hàng
      thực sự quan tâm, từ chất vải đến giao hàng và thanh toán.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header(":material/database: Dữ liệu")
    uploaded = st.file_uploader("File phản hồi (.xlsx / .csv)", type=["xlsx", "xls", "csv"])

    text_column = None
    if uploaded is not None:
        try:
            preview = read_uploaded(uploaded)
            default_col = cfg.data.get("text_column")
            cols = list(preview.columns)
            idx = cols.index(default_col) if default_col in cols else 0
            text_column = st.selectbox("Cột chứa nội dung phản hồi", cols, index=idx)
        except Exception as e:
            st.error(f"Không đọc được file: {e}")

    st.header(":material/tune: Tùy chọn")
    do_summarize = st.toggle("Tóm tắt chủ đề bằng AI", value=cfg.summarizer.get("enabled", True))
    key_ok, key_msg = api_key_status(cfg)
    if do_summarize:
        (st.success if key_ok else st.warning)(key_msg, icon=":material/key:")
        if not key_ok:
            st.caption("Tắt mục này để chạy nhanh chỉ với từ khoá, hoặc đặt API key rồi tải lại.")

    run = st.button(
        "Phân tích",
        icon=":material/insights:",
        type="primary",
        use_container_width=True,
        disabled=uploaded is None or text_column is None,
    )

if run and uploaded is not None and text_column:
    try:
        with st.spinner("Đang phân tích — gom nhóm và tóm tắt phản hồi…"):
            uploaded.seek(0)
            df = read_uploaded(uploaded)
            st.session_state["out"] = analyze(df, text_column, cfg, do_summarize)
    except Exception as e:
        st.error(f"Không phân tích được: {e}")

out = st.session_state.get("out")
if not out:
    st.info("Tải file phản hồi ở thanh bên trái để bắt đầu.", icon=":material/upload_file:")
    st.stop()

# ---- stats ----
st.markdown(
    f"""
    <div class="stats">
      <div class="stat accent"><div class="k">{out['total']}</div><div class="l">Phản hồi</div></div>
      <div class="stat"><div class="k">{out['n_topics']}</div><div class="l">Chủ đề</div></div>
      <div class="stat"><div class="k">{out['n_noise']}</div><div class="l">Chưa gom nhóm</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

table = pd.DataFrame(out["results"])[["Topic", "Số feedback", "Keywords", "Tóm tắt"]]

# ---- chart ----
st.markdown(f'<div class="sec">{icon("bar_chart")} Quy mô các chủ đề</div>', unsafe_allow_html=True)
chart_df = table.copy()
chart_df["label"] = "#" + chart_df["Topic"].astype(str) + "  " + chart_df["Keywords"].str.split(" · ").str[0]
chart = (
    alt.Chart(chart_df)
    .mark_bar(color=VIOLET, cornerRadiusEnd=5, size=18)
    .encode(
        x=alt.X("Số feedback:Q", title=None, axis=alt.Axis(grid=False)),
        y=alt.Y("label:N", sort="-x", title=None),
        tooltip=["Topic", "Số feedback", "Keywords"],
    )
    .properties(height=max(180, len(chart_df) * 30))
    .configure_view(stroke=None)
    .configure_axis(labelColor=INK, labelFont="Inter", labelFontSize=12)
)
st.altair_chart(chart, use_container_width=True)

# ---- table ----
st.markdown(f'<div class="sec">{icon("table_view")} Bảng chủ đề</div>', unsafe_allow_html=True)
maxv = int(table["Số feedback"].max())
st.dataframe(
    table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Topic": st.column_config.NumberColumn("Chủ đề", width="small"),
        "Số feedback": st.column_config.ProgressColumn(
            "Số feedback", format="%d", min_value=0, max_value=maxv
        ),
        "Keywords": st.column_config.TextColumn("Từ khoá", width="medium"),
        "Tóm tắt": st.column_config.TextColumn("Tóm tắt", width="large"),
    },
)
st.download_button(
    "Tải kết quả (CSV)",
    table.to_csv(index=False).encode("utf-8-sig"),
    file_name="topic_summaries.csv",
    mime="text/csv",
    icon=":material/download:",
)

# ---- details ----
st.markdown(f'<div class="sec">{icon("search")} Chi tiết từng chủ đề</div>', unsafe_allow_html=True)
for row in out["results"]:
    with st.expander(f"Chủ đề {row['Topic']} · {row['Số feedback']} phản hồi · {row['Keywords']}"):
        if row["Tóm tắt"]:
            st.markdown(row["Tóm tắt"])
        st.markdown("**Phản hồi tiêu biểu**")
        for d in row["_docs"]:
            st.markdown(f"- {d}")
