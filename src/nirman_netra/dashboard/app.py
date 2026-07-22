"""Empty dashboard shell; business logic belongs outside presentation code."""

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="NirmanNetra AI", layout="wide")
    st.title("NirmanNetra AI")
    st.caption("Geospatial inspection decision-support platform")


if __name__ == "__main__":
    main()
