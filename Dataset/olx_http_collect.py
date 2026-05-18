import argparse
import csv
import re
import time
from html import unescape
from pathlib import Path
from urllib.parse import urljoin

import requests

from olx_collect import (
    IMAGE_EXTENSIONS,
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
MAX_ROW_IMAGES = 5


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
    parser.add_argument("--max-images", type=int, default=120, help="Stop after this many downloaded image references.")
    parser.add_argument(
        "--images-per-listing",
        type=int,
        default=MAX_ROW_IMAGES,
        help="Maximum gallery images to keep per listing row.",
    )
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
    image_count = 0
    seen_urls = set()
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    images_per_listing = max(1, min(args.images_per_listing, MAX_ROW_IMAGES))

    for start_url in args.start_url:
        if image_count >= args.max_images:
            break
        print(f"Opening {start_url}", flush=True)
        try:
            response = session.get(start_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"  skipped start URL: {exc}", flush=True)
            continue

        for listing in parse_listing_cards(response.text, start_url):
            if image_count >= args.max_images:
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
            listing.image_urls = collect_detail_image_urls(session, listing, images_per_listing)
            remaining = args.max_images - image_count
            image_names = download_images(
                session,
                listing.image_urls[: min(images_per_listing, remaining)],
                args.image_dir,
                listing_id,
            )
            if not any(image_names):
                continue
            image_count += sum(1 for image_name in image_names if image_name)
            row = format_row(listing_id, listing, image_names, include_extra=False)
            rows.append(row)
            print(
                f"  collected {image_count}/{args.max_images} images from row {len(rows)}: {listing.title}",
                flush=True,
            )
            time.sleep(args.delay)

    write_csv(args.output_csv, rows)
    print(f"Wrote {len(rows)} rows with {image_count} image references to {args.output_csv}", flush=True)


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
        if image_id in seen or should_skip_image_id(image_id):
            continue
        seen.add(image_id)
        urls.append(f"https://apollo.olx.in/v1/files/{image_id}/image;s=600x1200;q=80;f=webp")
    return urls


def collect_detail_image_urls(session: requests.Session, listing: Listing, limit: int) -> list[str]:
    detail_urls = list(listing.image_urls)
    try:
        response = session.get(listing.url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"  detail page skipped: {exc}", flush=True)
    else:
        detail_urls.extend(listing_image_urls(response.text))
    return dedupe_urls(detail_urls)[:limit]


def should_skip_image_id(image_id: str) -> bool:
    lowered = image_id.lower()
    return lowered.startswith("alias-") or "-in" not in lowered


def dedupe_urls(urls: list[str]) -> list[str]:
    deduped = []
    seen = set()
    for url in urls:
        image_id = url.split("/v1/files/", 1)[-1].split("/image", 1)[0]
        if image_id in seen:
            continue
        seen.add(image_id)
        deduped.append(url)
    return deduped


def download_images(
    session: requests.Session,
    image_urls: list[str],
    image_dir: Path,
    listing_id: int,
) -> list[str]:
    names = ["", "", "", "", ""]
    for index, image_url in enumerate(image_urls[:MAX_ROW_IMAGES], start=1):
        suffix = image_suffix(image_url)
        name = f"room_{listing_id}_img{index}{suffix}"
        path = image_dir / name
        try:
            response = session.get(image_url, timeout=20)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"  image skipped: {exc}", flush=True)
            continue
        path.write_bytes(response.content)
        names[index - 1] = name
    return names


def image_suffix(url: str) -> str:
    suffix = Path(url.split("?", 1)[0].split(";", 1)[0]).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return suffix
    return ".webp"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=STAYSURE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


if __name__ == "__main__":
    main()
