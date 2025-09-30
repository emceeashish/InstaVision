import argparse
from core.scraper import scrape_instagram_profile, save_data
from core.ai_pipeline import run_ai_analysis


def main():
    parser = argparse.ArgumentParser(description="Scrape Instagram data and run AI analysis")
    parser.add_argument("--username", "-u", type=str, help="Instagram username", required=False)
    args = parser.parse_args()

    username = args.username or input("Enter Instagram username: ").strip()
    if not username:
        print("Username is required")
        return

    profile_info, posts = scrape_instagram_profile(username)
    if profile_info and posts is not None:
        save_data(profile_info, posts, username)
        result = run_ai_analysis(username=username, data_folder="instagram_data", output_folder="instagram_data")
        if result:
            print("Scrape + Analysis complete.")
    else:
        print("Scraping failed.")


if __name__ == "__main__":
    main()
