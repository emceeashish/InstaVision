import argparse
from core.scraper import scrape_instagram_profile, save_data


def main():
    parser = argparse.ArgumentParser(description="Scrape Instagram profile and posts")
    parser.add_argument("--username", "-u", type=str, help="Instagram username", required=False)
    args = parser.parse_args()

    username = args.username or input("Enter Instagram username: ").strip()
    if not username:
        print("Username is required")
        return

    profile_info, posts = scrape_instagram_profile(username)
    if profile_info and posts is not None:
        save_data(profile_info, posts, username)
        print(f"Scraped {len(posts)} posts for @{username}")
    else:
        print("Scraping failed.")


if __name__ == "__main__":
    main()
