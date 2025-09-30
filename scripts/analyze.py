import argparse
from core.ai_pipeline import run_ai_analysis


def main():
    parser = argparse.ArgumentParser(description="Run AI analysis on previously scraped data")
    parser.add_argument("--username", "-u", type=str, help="Instagram username", required=False)
    args = parser.parse_args()

    username = args.username or input("Enter Instagram username: ").strip()
    if not username:
        print("Username is required")
        return

    result = run_ai_analysis(username=username, data_folder="instagram_data", output_folder="instagram_data")
    if result:
        print("Done.")
    else:
        print("Analysis failed.")


if __name__ == "__main__":
    main()
