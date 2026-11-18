"""Streamlit demo: research topic -> evidence-grounded research gaps.

Launch from the repository root:  streamlit run streamlit_app.py

The page is a thin front end over ``src.workflow.demo``, which runs the same
LangGraph workflow as ``experiments/run_langgraph.py`` in a new, isolated run
directory. Nothing is computed here: counts, gaps, decisions and the report all
come from the saved workflow state.
"""
from __future__ import annotations

import streamlit as st

from src.workflow import demo, watcher_log_filter

# Drop only the file watcher's harmless torchvision probe tracebacks on lazy
# transformers modules; the watcher and hot reload are unchanged.
watcher_log_filter.install()

DEFAULT_TOPIC = ("Multi-agent LLM systems for scientific literature review and automated "
                 "research gap discovery")

st.set_page_config(page_title="Research Gap Discovery", layout="wide")
st.title("Multi-Agent Research Gap Discovery")
st.caption(
    "Retrieves and ranks scholarly papers, analyzes the accessible ones, and proposes "
    "candidate research gaps only when exact evidence from at least two papers supports "
    "them. Gaps, questions and hypotheses are limited to the analyzed papers."
)

with st.form("run_form"):
    topic = st.text_input("Research topic", value=DEFAULT_TOPIC)
    left, middle, right = st.columns(3)
    target = left.number_input(
        "Target papers to analyze", min_value=2, max_value=demo.MAX_TARGET_PAPERS,
        value=demo.DEMO_TARGET_PAPERS, step=1,
        help="Each analyzed paper costs one analysis request. Failed or inaccessible "
             "papers are replaced by the next relevant candidate.")
    minimum = middle.number_input(
        "Minimum papers to continue", min_value=2, max_value=demo.MAX_TARGET_PAPERS,
        value=2, step=1,
        help="The run stops honestly if fewer relevant papers can be analyzed.")
    per_source = right.number_input(
        "Results per source and query", min_value=1, max_value=demo.MAX_RESULTS_PER_SOURCE,
        value=5, step=1,
        help="More results give more candidates (and more retrieval calls).")
    estimate_clicked = st.form_submit_button("Estimate cost (no API calls)")
    run_clicked = st.form_submit_button("Run", type="primary")

settings = demo.DemoSettings(topic=topic, target_papers=int(target),
                             minimum_papers=int(minimum),
                             max_results_per_source=int(per_source))

if estimate_clicked:
    try:
        st.session_state["estimate"] = demo.estimate(settings)
    except ValueError as error:
        st.error(str(error))
if st.session_state.get("estimate"):
    with st.expander("Estimated requests and tokens (dry run, nothing sent)", expanded=True):
        st.code(st.session_state["estimate"], language=None)

if run_clicked:
    try:
        settings.validate()
    except ValueError as error:
        st.error(str(error))
    else:
        with st.status("Running the research workflow...", expanded=True) as status_box:
            def progress(stage, state):
                status_box.write(demo.describe_progress(stage, state))

            try:
                st.session_state["result"] = demo.run_demo(settings, progress=progress)
            except Exception as error:  # show the failure; never fake a result
                st.session_state["result"] = None
                status_box.update(label="The workflow failed", state="error")
                st.exception(error)
            else:
                result = st.session_state["result"]
                status_box.update(
                    label=f"Workflow {result.status}",
                    state="complete" if result.status == "COMPLETE" else "error",
                    expanded=False)

result = st.session_state.get("result")
if result is not None:
    st.subheader(f"Workflow status: {result.status}")
    if result.stop_reason:
        (st.info if result.status == "STOPPED" else st.error)(result.stop_reason)
    st.caption(f"Run directory: {result.run_dir}")

    counts = result.counts
    if counts is not None:
        cols = st.columns(5)
        cols[0].metric("Retrieved", counts.retrieved,
                       help=(f"{counts.relevant} at or above the relevance threshold"
                             if counts.relevant is not None else None))
        cols[1].metric("Downloaded", counts.downloaded)
        cols[2].metric("Analyzed", counts.analyzed, help=f"Target {result.target_papers}")
        cols[3].metric("Failed", counts.failed,
                       help=f"Download {counts.failed_download}, analysis {counts.failed_analysis}")
        cols[4].metric("Skipped", counts.skipped,
                       help=(f"Below relevance {counts.skipped_below_relevance}, "
                             f"max papers reached {counts.skipped_max_papers}, "
                             f"quota {counts.skipped_quota}"))
        st.write(
            f"Failed: {counts.failed_download} download, {counts.failed_analysis} analysis. "
            f"Skipped: {counts.skipped_below_relevance} below relevance, "
            f"{counts.skipped_max_papers} after the target was reached, "
            f"{counts.skipped_quota} because of API quota.")
        if not result.target_reached:
            st.warning(
                f"Target of {result.target_papers} analyzed papers not reached: only "
                f"{counts.analyzed} relevant paper(s) could be downloaded and analyzed.")

    st.subheader("Candidate gaps")
    if result.gaps:
        st.dataframe(result.gaps, width="stretch", hide_index=True)
        accepted = [row for row in result.gaps if row["critic"] == "ACCEPT"]
        st.write(f"{len(accepted)} of {len(result.gaps)} gap(s) accepted by the Critic. "
                 "Questions and hypotheses are generated only for accepted gaps.")
    else:
        st.write("No candidate gap was produced.")

    if result.report_markdown:
        st.download_button("Download Markdown report", data=result.report_markdown,
                           file_name=f"{result.run_dir.name}-report.md",
                           mime="text/markdown")
        with st.expander("Report", expanded=False):
            st.markdown(result.report_markdown)
