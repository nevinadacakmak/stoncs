"""
Run a simple end-to-end Stoncs demo:
1) Upload demo market and news data to Snowflake
2) Run narrative detection and persist results

Usage: set Snowflake env vars, then run `python run_demo.py`
"""
from stoncs.ingest import upload_demo_to_snowflake
from stoncs.narratives import detect_narratives


def main():
    print("Uploading demo data to Snowflake...")
    upload_demo_to_snowflake()

    print("Detecting narratives and computing trends...")
    detect_narratives()

    print("Done. You can now run the Streamlit app: `streamlit run -p 8501 stoncs/app.py`")


if __name__ == "__main__":
    main()
