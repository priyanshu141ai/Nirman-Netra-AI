"""Map-oriented human review dashboard backed only by the versioned API."""

import streamlit as st

from nirman_netra.config import load_settings
from nirman_netra.dashboard.client import DashboardApiClient

SAFE_RESULT_LABEL = "Potential structural change — inspector review required"


def render(client: DashboardApiClient) -> None:
    st.set_page_config(page_title="NirmanNetra AI", layout="wide")
    st.title("NirmanNetra AI")
    st.caption("Municipal decision-support; model results are not legal findings")
    overview, queue, evidence, review = st.tabs(
        ["Municipality map", "Review queues", "Evidence comparison", "Inspector review"]
    )
    with overview:
        st.subheader("Municipality and ward overview")
        st.json(client.quality())
        st.caption("Parcel, approved footprint, setback and detected-change map layers")
    with queue:
        st.subheader("High-risk and assigned-case queues")
        st.dataframe(client.cases()["items"], use_container_width=True)
    with evidence:
        st.subheader("Historical, current and registered imagery")
        st.info(SAFE_RESULT_LABEL)
        st.caption(
            "Model-generated building mask · structural-change mask · added/removed polygons"
        )
        case_id = st.text_input("Case ID")
        if case_id:
            case = client.case(case_id)
            st.json(case)
            st.json(client.result(str(case["change_result_id"])))
            if case.get("parcel_id"):
                st.json(client.parcel(str(case["parcel_id"])))
    with review:
        st.subheader("Evidence timeline and human-controlled actions")
        st.warning("Accept, dismiss, or request reinspection only after evidence review")


def main() -> None:
    settings = load_settings()
    render(DashboardApiClient(settings.dashboard_api_url))


if __name__ == "__main__":
    main()
