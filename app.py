# app_gemini.py
# Streamlit UI for StageDetector with Gemini embeddings (default) and local TF-IDF fallback.
# Run locally with:
#   pip install streamlit google-generativeai numpy pandas scikit-learn plotly python-dotenv
#   export GOOGLE_API_KEY=...  (or set in your system / .env)
#   streamlit run app_gemini.py
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

# Plotly & analysis imports
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

import time
from typing import List
import pandas as pd

try:
    import streamlit as st
except Exception:
    st = None  # type: ignore

from stage_detector_gemini import (
    StageDetectorGemini, DetectorConfig, Message, STAGES
)

# Optional fallback import (local TF-IDF) if user toggles
try:
    from stage_detector import StageDetector as StageDetectorLocal
except Exception:
    StageDetectorLocal = None  # type: ignore


# ------------------------------------------------------------
# Advanced Plotly visualizations (10 charts: 2D + 3D)
# ------------------------------------------------------------
def render_advanced_graphs(det, res, messages: List[Message]):
    """
    Build 10 advanced Plotly charts (2D & 3D) from StageDetector results.
    - det: StageDetectorGemini or fallback detector (needs .encoder for embeddings plots)
    - res: SessionResult from det.process(...)
    - messages: list[Message]
    """
    STAGE_LIST = ["clarify", "plan", "execute", "verify", "reflect"]

    # ---------- Prepare tidy data ----------
    rows = []
    for i, r in enumerate(res.per_message, start=1):
        dom = max(r.p_smoothed, key=r.p_smoothed.get)
        probs = [r.p_smoothed[s] for s in STAGE_LIST]
        probs_sorted = sorted(probs, reverse=True)
        margin = probs_sorted[0] - probs_sorted[1] if len(probs_sorted) >= 2 else probs_sorted[0]
        # tokens: simple whitespace split; feel free to swap to a proper tokenizer
        tokens = max(1, len((messages[i-1].text or "").split()))
        # dwell-like: timestamp difference (capped to 300s)
        if i == 1 or messages[i-1].timestamp is None or messages[i-2].timestamp is None:
            dwell_like = 1.0
        else:
            dwell_like = min(max(0.0, messages[i-1].timestamp - messages[i-2].timestamp), 300.0)

        rows.append({
            "t": i,
            "dominant": dom,
            "ambiguous": r.ambiguous,
            "weight": float(r.weight),
            **{f"p[{k}]": float(r.p_smoothed[k]) for k in STAGE_LIST},
            "entropy": float(-sum(r.p_smoothed[k]*np.log(r.p_smoothed[k] + 1e-12) for k in STAGE_LIST)),
            "margin": float(margin),
            "tokens": int(tokens),
            "event_flag": 1 if messages[i-1].has_event else 0,
            "dwell_like": dwell_like
        })
    df = pd.DataFrame(rows)

    # Session-level vector (radar)
    pi = res.pi
    pi_df = pd.DataFrame({"stage": STAGE_LIST, "percentage": [pi[s] for s in STAGE_LIST]})

    # Probability matrix for heatmap/surface
    prob_mat = df[[f"p[{s}]" for s in STAGE_LIST]].to_numpy()  # shape (T, 5)

    # Try to compute embeddings via Gemini encoder (3D/2D scatter)
    embeddings = None
    can_embed = hasattr(det, "encoder") and hasattr(det.encoder, "embed")
    if can_embed:
        try:
            embeddings = np.vstack([det.encoder.embed(m.text or "") for m in messages])
        except Exception:
            embeddings = None

    # ---------- 1) Heatmap of per-message stage probabilities ----------
    fig1 = px.imshow(
        prob_mat.T,
        labels=dict(x="Message index (t)", y="Stage", color="p_t"),
        x=df["t"],
        y=STAGE_LIST,
        aspect="auto",
        title="Per-Message Stage Probabilities — Heatmap"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ---------- 2) Stacked area chart over t (probability mass by stage) ----------
    area_df = df.melt(
        id_vars=["t"],
        value_vars=[f"p[{s}]" for s in STAGE_LIST],
        var_name="stage",
        value_name="p"
    )
    area_df["stage"] = area_df["stage"].str.replace("p[", "", regex=False).str.replace("]", "", regex=False)
    fig2 = px.area(
        area_df, x="t", y="p", color="stage",
        title="Stage Trajectory — Stacked Area (Smoothed p_t)"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ---------- 3) Radar (polar) of session percentages ----------
    fig3 = go.Figure()
    fig3.add_trace(go.Scatterpolar(
        r=pi_df["percentage"].values.tolist(),
        theta=pi_df["stage"].values.tolist(),
        fill="toself",
        name="Session π"
    ))
    fig3.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(0.001, max(pi_df['percentage'])*1.1)])),
        title="Session Stage Distribution — Radar (π)"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ---------- 4) 3D scatter of message embeddings (PCA→3D), colored by dominant stage ----------
    if embeddings is not None and embeddings.shape[0] >= 3:
        try:
            pca3 = PCA(n_components=3)
            xyz = pca3.fit_transform(embeddings)
            fig4 = px.scatter_3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                color=df["dominant"],
                symbol=df["ambiguous"].map({True: "ambiguous", False: "clear"}),
                hover_name=df["t"].astype(str),
                title="3D Embedding Projection (PCA-3) — Colored by Dominant Stage",
                labels={"color": "Dominant", "symbol": "Ambiguity"}
            )
            st.plotly_chart(fig4, use_container_width=True)
        except Exception:
            pass

    # ---------- 5) 2D scatter of embeddings (PCA→2D) ----------
    if embeddings is not None and embeddings.shape[0] >= 2:
        try:
            pca2 = PCA(n_components=2)
            xy = pca2.fit_transform(embeddings)
            fig5 = px.scatter(
                x=xy[:, 0], y=xy[:, 1],
                color=df["dominant"],
                symbol=df["ambiguous"].map({True: "ambiguous", False: "clear"}),
                hover_name=df["t"].astype(str),
                title="2D Embedding Projection (PCA-2) — Colored by Dominant Stage",
                labels={"color": "Dominant", "symbol": "Ambiguity"},
            )
            st.plotly_chart(fig5, use_container_width=True)
        except Exception:
            pass

    # ---------- 6) Sankey: aggregated transitions between consecutive dominant stages ----------
    stage_to_idx = {s: i for i, s in enumerate(STAGE_LIST)}
    trans = np.zeros((len(STAGE_LIST), len(STAGE_LIST)), dtype=int)
    dom = df["dominant"].tolist()
    for a, b in zip(dom[:-1], dom[1:]):
        trans[stage_to_idx[a], stage_to_idx[b]] += 1

    sources, targets, values = [], [], []
    for i, s in enumerate(STAGE_LIST):
        for j, t_ in enumerate(STAGE_LIST):
            if trans[i, j] > 0:
                sources.append(i)
                targets.append(len(STAGE_LIST) + j)  # split nodes into left/right partitions
                values.append(int(trans[i, j]))

    fig6 = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=12,
            label=STAGE_LIST + [f"{s}'" for s in STAGE_LIST],
        ),
        link=dict(source=sources, target=targets, value=values)
    )])
    fig6.update_layout(title_text="Dominant Stage Transitions — Sankey (t→t+1)")
    st.plotly_chart(fig6, use_container_width=True)

    # ---------- 7) 3D Surface: stage prob surface over (message index, stage index) ----------
    Z = prob_mat.T  # (5, T)
    fig7 = go.Figure(data=[go.Surface(
        z=Z,
        x=df["t"],  # columns
        y=list(range(len(STAGE_LIST))),  # rows = stage index
        colorbar=dict(title="p_t"),
    )])
    fig7.update_layout(
        title="Stage Probability Surface (Stage × Message Index)",
        scene=dict(
            xaxis_title="t (message index)",
            yaxis_title="stage index",
            zaxis_title="p",
            yaxis=dict(
                tickmode="array", tickvals=list(range(len(STAGE_LIST))), ticktext=STAGE_LIST
            ),
        ),
    )
    st.plotly_chart(fig7, use_container_width=True)

    # ---------- 8) Rolling entropy line (uncertainty over time) ----------
    df["entropy_smooth"] = df["entropy"].rolling(3, min_periods=1, center=True).mean()
    fig8 = px.line(
        df, x="t", y=["entropy", "entropy_smooth"],
        title="Prediction Uncertainty — Entropy over Time",
        labels={"value": "entropy", "variable": "series"}
    )
    st.plotly_chart(fig8, use_container_width=True)

    # ---------- 9) Message weights decomposition (stacked bars: dwell-like, tokens, event) ----------
    comp = pd.DataFrame({
        "t": df["t"],
        "dwell_like": df["dwell_like"].fillna(1.0),
        "tokens": df["tokens"].astype(float),
        "event": df["event_flag"].astype(float)
    })
    # Normalize per-component across all messages to show relative contribution
    for c in ["dwell_like", "tokens", "event"]:
        s = comp[c].sum()
        comp[c] = comp[c] / (s if s > 0 else 1.0)
    fig9 = go.Figure()
    fig9.add_bar(x=comp["t"], y=comp["dwell_like"], name="dwell")
    fig9.add_bar(x=comp["t"], y=comp["tokens"], name="tokens")
    fig9.add_bar(x=comp["t"], y=comp["event"], name="event")
    fig9.update_layout(barmode="stack", title="Message Weight Components — Normalized Stack")
    st.plotly_chart(fig9, use_container_width=True)

    # ---------- 10) Dominance margin histogram + confidence line ----------
    fig10a = px.histogram(
        df, x="margin", nbins=20,
        title="Dominance Margin Distribution (Top1 − Top2)",
        labels={"margin": "Top1−Top2 smoothed probability"}
    )
    st.plotly_chart(fig10a, use_container_width=True)

    df["max_p"] = df[[f"p[{s}]" for s in STAGE_LIST]].max(axis=1)
    fig10b = px.line(
        df, x="t", y="max_p",
        title="Maximum Smoothed Probability (Confidence) over Time",
        labels={"max_p": "max p_t", "t": "message index"}
    )
    st.plotly_chart(fig10b, use_container_width=True)


def run_streamlit():
    st.set_page_config(page_title="EDU-AI Stage Detector (Gemini)", layout="wide")
    st.title("EDU-AI Cognitive Stage Detector — Gemini")
    st.caption("Gemini Embeddings + rules, temporal smoothing, and weighted aggregation")

    with st.sidebar:
        st.subheader("Settings")
        lambda_fusion = st.slider("λ (rules weight)", 0.0, 1.0, 0.5, 0.05)
        beta_smooth = st.slider("β (temporal smoothing)", 0.0, 1.0, 0.7, 0.05)
        tau = st.slider("Temperature τ (similarity→probs)", 1.0, 20.0, 10.0, 1.0)

        st.markdown("---")
        st.write("Aggregation weights (normalized automatically)")
        g_time = st.slider("γₜ (dwell time)", 0.0, 1.0, 0.4, 0.05)
        g_len = st.slider("γₗ (token length)", 0.0, 1.0, 0.4, 0.05)
        g_event = st.slider("γₑ (event)", 0.0, 1.0, 0.2, 0.05)
        total = g_time + g_len + g_event
        if total == 0:
            g_time, g_len, g_event = 0.4, 0.4, 0.2
        else:
            g_time, g_len, g_event = (g_time/total, g_len/total, g_event/total)

        st.markdown("---")
        use_fallback = st.checkbox(
            "Use Local TF-IDF Fallback",
            value=False,
            help="If you don't have a Google API key configured"
        )

    cfg = DetectorConfig(
        lambda_fusion=lambda_fusion,
        beta_smooth=beta_smooth,
        tau=tau,
        gamma_time=g_time,
        gamma_len=g_len,
        gamma_event=g_event,
    )

    if use_fallback and StageDetectorLocal is not None:
        st.warning("Running in Local TF-IDF mode (no external API calls).")
        det = StageDetectorLocal(cfg=None)  # uses its own config class; fine for demo
    else:
        det = StageDetectorGemini(cfg=cfg)

    st.subheader("Enter Messages")
    st.write("Add learner messages in order. Optionally mark **Event** to up-weight (e.g., hint request/test).")
    if "messages" not in st.session_state:
        st.session_state.messages = []  # type: List[Message]

    with st.form("add_msg", clear_on_submit=True):
        text = st.text_area("Message text", height=90, placeholder="Type a learner message here...")
        col1, col2 = st.columns(2)
        with col1:
            has_event = st.checkbox("Event (hint/test/reflect trigger)", value=False)
        with col2:
            use_now = st.checkbox("Use current time as timestamp", value=True)
        ts = time.time() if use_now else None
        submitted = st.form_submit_button("Add message")
        if submitted and (text.strip() != ""):
            st.session_state.messages.append(Message(text=text.strip(), timestamp=ts, has_event=has_event))

    if st.session_state.messages:
        df = pd.DataFrame({
            "#": list(range(1, len(st.session_state.messages)+1)),
            "text": [m.text for m in st.session_state.messages],
            "timestamp": [m.timestamp for m in st.session_state.messages],
            "event": [m.has_event for m in st.session_state.messages],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)
        if st.button("Clear messages"):
            st.session_state.messages = []

    if st.session_state.messages:
        try:
            res = det.process(st.session_state.messages)
        except Exception as e:
            st.error(f"Error during processing: {e}")
            st.stop()

        st.subheader("Session Stage Percentages (π)")
        pi_df = pd.DataFrame({"stage": list(res.pi.keys()), "percentage": [round(v, 4) for v in res.pi.values()]})
        st.bar_chart(pi_df, x="stage", y="percentage", use_container_width=True)
        st.write(pi_df.set_index("stage"))

        st.subheader("Per-Message Inference")
        rows = []
        for i, r in enumerate(res.per_message, start=1):
            dom = max(r.p_smoothed, key=r.p_smoothed.get)
            rows.append({
                "t": i,
                "dominant": dom,
                "ambiguous": r.ambiguous,
                "weight": round(r.weight, 4),
                **{f"p[{k}]": round(r.p_smoothed[k], 4) for k in STAGES},
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("“Ambiguous” is based on prediction entropy; consider showing striped UI for those messages.")

        # ---------------- Advanced analytics section ----------------
        st.markdown("## 📊 Advanced Analytics (Plotly)")
        render_advanced_graphs(det, res, st.session_state.messages)

    else:
        st.info("Add at least one message to compute stage percentages.")


if __name__ == "__main__":
    if st is None:
        print("This app requires Streamlit. Install and run with:")
        print("  pip install streamlit google-generativeai numpy pandas scikit-learn plotly python-dotenv")
        print("  export GOOGLE_API_KEY=...")
        print("  streamlit run app_gemini.py")
    else:
        run_streamlit()