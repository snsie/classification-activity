"""Starter app for Streamlit Community Cloud: four plain sections that read results/ and show correct numbers.

Run locally:  streamlit run streamlit_app.py
Build each section up to its required minimum (see the activity website). Keep every number
coming from m11.py, and label every table and plot "Development (OOF)" or "Test".
"""

import pandas as pd
import streamlit as st

import m11

DEV = "Development (OOF)"
TEST = "Test"
SECTIONS = ["1. Beyond Accuracy", "2. A Threshold for the Job", "3. Ranking vs. Being Right", "4. A Fair Comparison"]

st.set_page_config(page_title="Judge the Model", layout="wide")


@st.cache_data
def load(name: str):
    return m11.load(name)


def starter_note(text: str) -> None:
    st.info(f"Starter display: {text} Build this section up to its required minimum.")


def recommendations() -> None:
    d = load("decisions")
    recs = (d or {}).get("recommendations", {})
    with st.container(border=True):
        st.subheader("Recommendations")
        if not any(recs.values()):
            st.caption("No recommendations yet: fill in `Recommendations` in DECISIONS.md, run "
                       "`python -m analysis export`, then commit and push.")
        for k in m11.DATASETS:
            if any(recs.values()):
                st.markdown(f"**{m11.DATASET_LABEL[k]}:** {recs.get(k) or '_not written yet_'}")


# ---- 1. Beyond Accuracy ------------------------------------------------------
def beyond_accuracy(dev: dict, model: str) -> None:
    starter_note("the four counts at threshold 0.50, and accuracy.")
    p, y = m11.oof(dev, model)
    c = m11.confusion(p, y, 0.5)
    st.caption(f"{DEV} · {m11.MODEL_LABEL[model]}, threshold 0.50")
    st.table(pd.DataFrame([{**{k.upper(): v for k, v in c.items()}, "Accuracy": m11.fmt_pct(m11.metrics(c)["accuracy"])}]))


# ---- 2. A Threshold for the Job ------------------------------------------------
def threshold(dev: dict, model: str) -> None:
    starter_note("metrics at 0.50 only.")
    p, y = m11.oof(dev, model)
    m = m11.metrics(m11.confusion(p, y, 0.5))
    st.caption(f"{DEV} · {m11.MODEL_LABEL[model]}, threshold 0.50")
    st.table(pd.DataFrame([{"Sensitivity": m11.fmt_pct(m["sensitivity"]), "Specificity": m11.fmt_pct(m["specificity"]),
                            "Precision": m11.fmt_pct(m["precision"]), "Flagged per 100": f"{m['flagged_per_100']:.1f}"}]))


# ---- 3. Ranking vs. Being Right ------------------------------------------------
def ranking(dev: dict, model: str) -> None:
    starter_note("AUROC of the pooled out-of-fold probabilities, in a table.")
    rows = [{"Model": m11.MODEL_LABEL[m], "AUROC": m11.fmt_num(m11.auroc(*m11.oof(dev, m)))} for m in m11.MODELS if m in dev["models"]]
    st.caption(f"{DEV} · AUROC (ranking, not accuracy)")
    st.table(pd.DataFrame(rows).set_index("Model"))


# ---- 4. A Fair Comparison ------------------------------------------------------
def fair_comparison(dev: dict, model: str) -> None:
    starter_note("mean ± SD of the 5 cross-validated AUROCs.")
    s = m11.fold_summary(dev)
    rows = [{"Model": m11.MODEL_LABEL[m], "CV AUROC (mean ± SD)": f"{m11.fmt_num(v['mean_cv'])} ± {m11.fmt_num(v['sd_cv'])}"}
            for m, v in s["models"].items()]
    st.caption(f"{DEV} · CV AUROC across 5 folds")
    st.table(pd.DataFrame(rows).set_index("Model"))
    test = load(f"{dev['dataset']}_test")
    st.caption(f"{TEST} · " + ("results are in results/" + dev["dataset"] + "_test.json." if test
                              else "locked until you run `python -m analysis test` (once)."))


RENDER = dict(zip(SECTIONS, [beyond_accuracy, threshold, ranking, fair_comparison]))

st.title("Judge the Model")
st.caption("Which classifier would you trust for this job, at what threshold, and with what limits?")
recommendations()
section = st.sidebar.radio("Section", SECTIONS)
key = st.sidebar.selectbox("Dataset", m11.DATASETS, format_func=m11.DATASET_LABEL.get)
dev = load(f"{key}_dev")
model = st.sidebar.selectbox("Model", m11.MODELS,
                             format_func=lambda m: m11.MODEL_LABEL[m] + ("" if m in dev["models"] else " (not trained yet)"))
st.header(section)
st.caption(f"{m11.DATASET_LABEL[key]}: {dev['n']:,} development rows, {dev['n_pos']:,} positive.")
if model in dev["models"]:
    RENDER[section](dev, model)
else:
    st.warning(f"{m11.MODEL_LABEL[model]} has no results yet: run `python -m analysis train`, then commit and push `results/`.")
