import argparse
import csv
import re
import time
from html import unescape
from pathlib import Path
from urllib.parse import urljoin

import requests

from olx_collect import (
    STAYSURE_COLUMNS,
    Listing,
    estimate_cleanliness,
    format_row,
    infer_amenities,
    is_room_listing,
    parse_bhk,
    parse_furnishing,
    parse_price,
    parse_size,
    redact_private_text,
)


CARD_PATTERN = re.compile(
    r'<a[^>]+href="(?P<href>/item/[^"]+)"[^>]*>(?P<body>.*?)</a>',
    re.S,
)
IMAGE_ID_PATTERN = re.compile(r"https?://apollo\.olx\.in(?::443)?/v1/files/([^/]+)/image", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect OLX rental listing cards with HTTP when browser rendering is blocked."
    )
    parser.add_argument(
        "--start-url",
        action="append",
        required=True,
        help="OLX search/category URL. Pass multiple times to merge cities.",
    )
    parser.add_argument("--max-images", type=int, default=120)
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument("--output-csv", type=Path, default=Path("Dataset/rooms_dataset_topup.csv"))
    parser.add_argument("--image-dir", type=Path, default=Path("Dataset/rooms"))
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument(
        "--room-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep listing cards that look like room/PG/bachelor rental posts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    seen_urls = set()
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    for start_url in args.start_url:
        if len(rows) >= args.max_images:
            break
        print(f"Opening {start_url}", flush=True)
        try:
            response = session.get(start_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"  skipped start URL: {exc}", flush=True)
            continue

        for listing in parse_listing_cards(response.text, start_url):
            if len(rows) >= args.max_images:
                break
            if listing.url in seen_urls:
                continue
            seen_urls.add(listing.url)
            if args.room_only and not is_room_listing(listing):
                print(f"  skipped: listing does not look room-focused ({listing.title})", flush=True)
                continue
            if listing.rent_price <= 0 or not listing.image_urls:
                print(f"  skipped: missing price/image ({listing.title})", flush=True)
                continue

            listing_id = args.start_id + len(rows)
            image_name = download_image(session, listing.image_urls[0], args.image_dir, listing_id)
            if not image_name:
                continue
            row = format_row(listing_id, listing, [image_name, "", "", "", ""], include_extra=False)
            rows.append(row)
            print(f"  collected {len(rows)}/{args.max_images}: {listing.title}", flush=True)
            time.sleep(args.delay)

    write_csv(args.output_csv, rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv}", flush=True)


def parse_listing_cards(html: str, base_url: str) -> list[Listing]:
    listings = []
    for match in CARD_PATTERN.finditer(html):
        body = match.group("body")
        title = extract_aut_text(body, "itemTitle")
        price = extract_aut_text(body, "itemPrice")
        details = extract_aut_text(body, "itemDetails")
        location = extract_aut_text(body, "item-location") or "Unknown"
        description = title
        text_blob = " ".join([title, details, location, description])
        image_urls = listing_image_urls(body)
        listing = Listing(
            url=urljoin(base_url, unescape(match.group("href"))),
            title=title,
            description=redact_private_text(description),
            location=location,
            size_sqft=parse_size(details),
            furnishing_type=parse_furnishing(text_blob),
            cleanliness_score=estimate_cleanliness(text_blob),
            amenities=infer_amenities(text_blob),
            rent_price=parse_price(price),
            bhk=parse_bhk(details),
            image_urls=image_urls,
        )
        listings.append(listing)
    return listings


def extract_aut_text(html: str, aut_id: str) -> str:
    match = re.search(
        rf'data-aut-id="{re.escape(aut_id)}"[^>]*>(?P<value>.*?)</(?:span|h3|div)>',
        html,
        re.S,
    )
    if not match:
        return ""
    value = re.sub(r"<[^>]+>", " ", match.group("value"))
    return clean_text(unescape(value))


def listing_image_urls(html: str) -> list[str]:
    urls = []
    seen = set()
    for image_id in IMAGE_ID_PATTERN.findall(html):
        if image_id in seen:
            continue
        seen.add(image_id)
        urls.append(f"https://apollo.olx.in/v1/files/{image_id}/image;s=600x1200;q=80;f=webp")
    return urls


def download_image(session: requests.Session, image_url: str, image_dir: Path, listing_id: int) -> str:
    name = f"room_{listing_id}_img1.webp"
    path = image_dir / name
    try:
        response = session.get(image_url, timeout=20)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"  image skipped: {exc}", flush=True)
        return ""
    path.write_bytes(response.content)
    return name


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=STAYSURE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


if __name__ == "__main__":
    main()
