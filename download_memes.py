#!/usr/bin/env python3
"""
Download Meme Templates from Imgflip API
Uses the official Imgflip API to get popular meme templates
API Documentation: https://imgflip.com/api
"""

import os
import requests
from pathlib import Path

# Create output directory
TEMPLATES_DIR = Path(__file__).parent / "meme_templates"
TEMPLATES_DIR.mkdir(exist_ok=True)

# Imgflip API endpoint
IMGFLIP_API_URL = "https://api.imgflip.com/get_memes"

# Meme names we want to download (will match against API response)
# These are the memes referenced in our training data
WANTED_MEMES = [
    "Drake Hotline Bling",
    "Distracted Boyfriend",
    "One Does Not Simply",
    "Surprised Pikachu",
    "Two Buttons",
    "Woman Yelling At Cat",
    "Expanding Brain",
    "Is This A Pigeon",
    "This Is Fine",
    "Stonks",
    "Change My Mind",
    "Always Has Been",
    "Bernie I Am Once Again Asking For Your Support",
    "Left Exit 12 Off Ramp",
    "Sleeping Shaq",
    "They're The Same Picture",
    "Batman Slapping Robin",
    "Roll Safe Think About It",
    "Gru's Plan",
    "Mocking Spongebob",
    "Disaster Girl",
    "Hide the Pain Harold",
    "Success Kid",
    "First Time",
    "Tuxedo Winnie The Pooh",
    "Ancient Aliens",
    "Futurama Fry",
    "Flex Tape",
    "Confused Math Lady",
    "Monkey Puppet",
    "Boardroom Meeting Suggestion",
    "Running Away Balloon",
    "Panik Kalm Panik",
    "Trade Offer",
    "Sad Pablo Escobar",
    "Mr Incredible Becoming Uncanny",
    "Anakin Padme 4 Panel",
    "Epic Handshake",
    "UNO Draw 25 Cards",
    "Bike Fall",
]


def fetch_meme_list():
    """Fetch the list of popular memes from Imgflip API."""
    print("📡 Fetching meme list from Imgflip API...")

    try:
        response = requests.get(IMGFLIP_API_URL, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data.get("success"):
            memes = data["data"]["memes"]
            print(f"✅ Found {len(memes)} memes from API")
            return memes
        else:
            print(f"❌ API error: {data.get('error_message', 'Unknown error')}")
            return []
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return []


def normalize_name(name: str) -> str:
    """Normalize meme name for comparison and filename."""
    return name.lower().replace("'", "").replace("'", "").replace(" ", "_").replace("-", "_")


def download_image(url: str, save_path: Path) -> bool:
    """Download an image from URL and save to path."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        # Verify it's an image
        content_type = response.headers.get("content-type", "")
        if "image" not in content_type:
            print(f"  ⚠️  Response doesn't appear to be an image")
            return False

        with open(save_path, "wb") as f:
            f.write(response.content)

        return True
    except requests.RequestException as e:
        print(f"  ❌ Download failed: {e}")
        return False


def find_matching_meme(wanted_name: str, api_memes: list) -> dict | None:
    """Find a meme from API that matches the wanted name."""
    wanted_normalized = normalize_name(wanted_name)

    for meme in api_memes:
        api_normalized = normalize_name(meme["name"])

        # Exact match
        if wanted_normalized == api_normalized:
            return meme

        # Partial match (wanted name contained in API name or vice versa)
        if wanted_normalized in api_normalized or api_normalized in wanted_normalized:
            return meme

        # Word-based matching for common variations
        wanted_words = set(wanted_normalized.split("_"))
        api_words = set(api_normalized.split("_"))
        common_words = wanted_words & api_words

        # If most important words match
        if len(common_words) >= min(2, len(wanted_words)):
            return meme

    return None


def download_all_templates():
    """Download all wanted meme templates."""
    print("🎭 Meme Template Downloader (using Imgflip API)")
    print("=" * 55)

    # Fetch meme list from API
    api_memes = fetch_meme_list()

    if not api_memes:
        print("❌ Could not fetch memes from API")
        return

    print(f"\n📋 Looking for {len(WANTED_MEMES)} specific templates...\n")

    successful = 0
    failed = 0
    not_found = 0

    for wanted_name in WANTED_MEMES:
        # Find matching meme in API response
        meme = find_matching_meme(wanted_name, api_memes)

        if meme is None:
            print(f"❓ {wanted_name} - Not found in API")
            not_found += 1
            continue

        # Create filename from normalized name
        filename = normalize_name(meme["name"]) + ".jpg"
        save_path = TEMPLATES_DIR / filename

        # Skip if already exists
        if save_path.exists():
            print(f"✅ {meme['name']} - Already exists")
            successful += 1
            continue

        print(f"⬇️  Downloading {meme['name']}...")

        if download_image(meme["url"], save_path):
            print(f"   ✅ Saved as {filename}")
            successful += 1
        else:
            failed += 1

    print("\n" + "=" * 55)
    print(f"📊 Results: {successful} downloaded, {failed} failed, {not_found} not found")
    print(f"📁 Templates saved to: {TEMPLATES_DIR.absolute()}")


def list_available_templates():
    """List all downloaded templates."""
    print("\n📋 Downloaded Templates:")
    print("-" * 40)

    templates = list(TEMPLATES_DIR.glob("*.jpg")) + list(TEMPLATES_DIR.glob("*.png"))

    if not templates:
        print("No templates found. Run download first!")
        return

    for template in sorted(templates):
        size_kb = template.stat().st_size / 1024
        print(f"  • {template.stem}: {size_kb:.1f} KB")

    print(f"\nTotal: {len(templates)} templates")


def list_api_memes():
    """List all memes available from the API."""
    print("\n📋 Memes available from Imgflip API:")
    print("-" * 50)

    api_memes = fetch_meme_list()

    if not api_memes:
        return

    for i, meme in enumerate(api_memes[:50], 1):  # Show top 50
        print(f"  {i:2}. {meme['name']} (id: {meme['id']})")

    print(f"\n... and {len(api_memes) - 50} more")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download meme templates from Imgflip API")
    parser.add_argument("--list", action="store_true", help="List downloaded templates")
    parser.add_argument("--api-list", action="store_true", help="List memes available from API")

    args = parser.parse_args()

    if args.list:
        list_available_templates()
    elif args.api_list:
        list_api_memes()
    else:
        download_all_templates()
        list_available_templates()
