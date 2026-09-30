"""Streamlit page to try a trained run by hand: `jev-model ui`.

It only talks to `JevModel.predictor`, so every registered model works here without changes.
This file lives in its own directory because streamlit puts the script's directory on
`sys.path`, where `jev_model`'s `config.py`/`data.py` would shadow other top-level modules."""

import random
from pathlib import Path

import streamlit as st
from datasets import load_from_disk

from jev_model import models
from jev_model.base import CONFIG_FILE, RUN_FILE, read_run
from jev_model.config import dump, load_config, read_file
from jev_model.constants import DATA_DIR, PARTITIONS, RUNS_DIR
from jev_model.data import built_names, partition

LAST = "last trained"


def find_runs(runs_dir: Path) -> list[Path]:
    """Run directories of registered models under `runs_dir`, newest first."""
    runs = [p.parent for p in runs_dir.rglob(RUN_FILE) if read_run(p.parent).get("model") in models]
    return sorted(runs, key=lambda p: (p / RUN_FILE).stat().st_mtime, reverse=True)


def open_run(run_dir: Path):
    model = models[read_run(run_dir)["model"]]
    return model, load_config(model.config_class, base=read_file(run_dir / CONFIG_FILE))


@st.cache_resource(max_entries=1, show_spinner="Loading the model...")
def get_predictor(run_dir: Path, stage: str | None, device: str):
    # max_entries=1: switching runs drops the previous backbone instead of stacking them in memory
    model, config = open_run(run_dir)
    return model.predictor(config, run_dir, device, stage)


@st.cache_resource(max_entries=4, show_spinner="Loading the dataset...")
def get_partitions(data_dir: Path, name: str) -> dict[str, list[dict]]:
    # the same split the models train and evaluate on, so `test` rows are really unseen
    return partition(load_from_disk(str(data_dir / name)))


def default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def fill(sample: dict):
    st.session_state.state = sample["state"]
    st.session_state.question = sample["question"]
    st.session_state.options = "\n".join(option["text"] for option in sample["options"])
    st.session_state.expected = {"options": [o["text"] for o in sample["options"]], "label": sample["label"]}
    st.session_state.pop("result", None)


def load_sample(data_dir: Path, pick_random: bool):
    rows = get_partitions(data_dir, st.session_state.dataset)[st.session_state.partition]
    if not rows:
        st.session_state.sample_error = f"{st.session_state.dataset} has no {st.session_state.partition} rows"
        return
    if pick_random:
        st.session_state.index = random.randrange(len(rows))
    st.session_state.index = min(st.session_state.index, len(rows) - 1)
    fill(rows[st.session_state.index])


def parse_options(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


st.set_page_config(page_title="jev model playground", layout="wide")
st.title("jev model playground")

for key in ("state", "question", "options"):
    st.session_state.setdefault(key, "")

with st.sidebar:
    st.header("Run")
    runs_dir = Path(st.text_input("Runs directory", str(RUNS_DIR))).expanduser()
    runs = find_runs(runs_dir) if runs_dir.is_dir() else []
    if not runs:
        st.warning(f"No runs under `{runs_dir}`. Train one with `jev-model train MODEL`.")
        st.stop()
    run_dir = st.selectbox("Run", runs, format_func=lambda p: str(p.relative_to(runs_dir)))
    info = read_run(run_dir)
    model, config = open_run(run_dir)
    st.caption(f"**{model.name}**: {model.description}")
    stage = st.selectbox("Stage", [LAST, *info.get("stages", [])], help="which stage's weights to use")
    device = st.text_input("Device", default_device())
    with st.expander("Config"):
        st.code(dump(config), language="json")

    st.header("Sample from a dataset")
    data_dir = Path(getattr(config, "data_dir", DATA_DIR))
    names = built_names(data_dir)
    if names:
        # datasets the run was configured with first, as the rest are out of distribution for it
        trained = [name for name in getattr(config, "datasets", []) if name in names]
        st.selectbox("Dataset", trained + [name for name in names if name not in trained], key="dataset")
        st.selectbox("Partition", PARTITIONS, index=PARTITIONS.index("test"), key="partition")
        st.number_input("Row", min_value=0, step=1, key="index")
        left, right = st.columns(2)
        left.button("Load row", on_click=load_sample, args=(data_dir, False), width="stretch")
        right.button("Random", on_click=load_sample, args=(data_dir, True), width="stretch")
        if error := st.session_state.pop("sample_error", None):
            st.error(error)
    else:
        st.caption(f"No built datasets under `{data_dir}` (run `jev prepare`).")

form, output = st.columns(2)

with form:
    st.text_area("State", key="state", height=220, placeholder="context the question is about (may be empty)")
    st.text_area("Question", key="question", height=80)
    st.text_area("Options", key="options", height=160, placeholder="one per line")
    options = parse_options(st.session_state.options)
    if st.button("Predict", type="primary", disabled=not (st.session_state.question.strip() and len(options) >= 2)):
        try:
            predictor = get_predictor(run_dir, None if stage == LAST else stage, device)
            (probs,) = predictor.predict([(st.session_state.state, st.session_state.question, options)])
            st.session_state.result = {"options": options, "probs": probs}
        except Exception as error:  # a bad stage, too many options, an over-long prompt, a missing checkpoint
            st.session_state.pop("result", None)
            st.error(f"{type(error).__name__}: {error}")

with output:
    result = st.session_state.get("result")
    if result is None:
        st.info("Fill in a question and at least two options, or load a dataset row, then press Predict.")
    else:
        expected = st.session_state.get("expected")
        # the label only applies while the options are still the ones the row came with
        labels = expected["label"] if expected and expected["options"] == result["options"] else None
        best = max(range(len(result["probs"])), key=result["probs"].__getitem__)
        st.metric("Prediction", result["options"][best], f"{result['probs'][best]:.1%}", delta_color="off")
        if labels:
            target = max(range(len(labels)), key=labels.__getitem__)
            if target == best:
                st.success("Matches the label.")
            else:
                st.error(f"Label: {result['options'][target]}")
        rows = [
            {"option": option, "probability": p, **({"label": labels[i]} if labels else {})}
            for i, (option, p) in enumerate(zip(result["options"], result["probs"]))
        ]
        st.dataframe(
            rows,
            hide_index=True,
            width="stretch",
            column_config={
                "probability": st.column_config.ProgressColumn(format="%.3f", min_value=0.0, max_value=1.0),
                "label": st.column_config.NumberColumn(format="%.2f"),
            },
        )
