import argparse
import csv
import hashlib
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
BAD_LISTING_PATTERNS = [
    r"\b(?:need|wanted|looking\s+for|required|requirement|chahiye|chahie|chaiye)\b",
    r"\b(?:roommate|room\s+mate|flatmate|room\s+partner|partner\s+chahiye)\b",
    r"\b(?:shop|office|godown|store\s+room|commercial|worker|supervisor|job)\b",
    r"\b(?:hour|hourly|1\s*hour|one\s*hour)\b",
    r"\b(?:sale|sell|buy|purchase)\b",
]


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
    parser.add_argument(
        "--dedupe-image-dir",
        type=Path,
        default=None,
        help="Directory of already-collected images used to skip duplicate downloads. Defaults to --image-dir.",
    )
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
    seen_image_hashes = load_existing_image_hashes(args.dedupe_image_dir or args.image_dir)

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
            if not is_valid_rental_listing(listing):
                print(f"  skipped: low-quality/non-rental listing ({listing.title})", flush=True)
                continue
            if listing.rent_price <= 0 or not listing.image_urls:
                print(f"  skipped: missing price/image ({listing.title})", flush=True)
                continue

            listing_id = args.start_id + len(rows)
            enrich_listing_from_detail(session, listing, images_per_listing)
            normalize_listing_fields(listing)
            remaining = args.max_images - image_count
            image_names = download_images(
                session,
                listing.image_urls[: min(images_per_listing, remaining)],
                args.image_dir,
                listing_id,
                seen_image_hashes,
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


def enrich_listing_from_detail(session: requests.Session, listing: Listing, limit: int) -> None:
    detail_urls = list(listing.image_urls)
    try:
        response = session.get(listing.url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"  detail page skipped: {exc}", flush=True)
    else:
        detail_text = html_to_text(response.text)
        if detail_text:
            listing.description = redact_private_text(detail_text[:1200])
            text_blob = " ".join([listing.title, listing.description, listing.location])
            listing.furnishing_type = parse_furnishing(text_blob)
            listing.cleanliness_score = estimate_cleanliness(text_blob)
            listing.amenities = infer_amenities(text_blob)
            if re.search(r"\battach(?:ed)?\s+(bathroom|washroom)\b", text_blob, re.I):
                listing.bathroom_attached = 1
        detail_urls.extend(listing_image_urls(response.text))
    listing.image_urls = dedupe_urls(detail_urls)[:limit]


def is_valid_rental_listing(listing: Listing) -> bool:
    text = clean_text(" ".join([listing.title, listing.description, listing.location])).lower()
    if any(re.search(pattern, text, re.I) for pattern in BAD_LISTING_PATTERNS):
        return False
    if not (800 <= listing.rent_price <= 80_000):
        return False
    return bool(re.search(r"\b(?:rent|rented|room|pg|bhk|rk|flat|house|available)\b", text, re.I))


def normalize_listing_fields(listing: Listing) -> None:
    text_blob = " ".join([listing.title, listing.description, listing.location])
    listing.location = normalize_location(listing.location)
    listing.bhk = normalize_bhk(listing.bhk, text_blob)
    listing.size_sqft = normalize_size_sqft(listing.size_sqft, listing.bhk)
    listing.furnishing_type = clean_text(listing.furnishing_type) or "Unknown"
    listing.amenities = sorted(set(listing.amenities)) or ["None"]
    listing.cleanliness_score = listing.cleanliness_score or 5.0


def normalize_location(value: str) -> str:
    location = clean_text(value) or "Unknown, Chhattisgarh"
    if "chhattisgarh" not in location.lower():
        location = f"{location}, Chhattisgarh"
    return location


def normalize_bhk(value: str, text: str) -> str:
    value = clean_text(value)
    if value and value.lower() != "unknown":
        return value
    lowered = text.lower()
    if re.search(r"\b(?:single|one|1)\s+(?:room|rk|bhk)\b", lowered):
        return "1"
    if re.search(r"\b(?:two|2)\s+(?:room|rk|bhk)\b", lowered):
        return "2"
    if re.search(r"\b(?:three|3)\s+(?:room|rk|bhk)\b", lowered):
        return "3"
    if re.search(r"\b(?:four|4)\s+(?:room|rk|bhk)\b", lowered):
        return "4"
    if "pg" in lowered or "hostel" in lowered:
        return "1"
    return "1"


def normalize_size_sqft(size_sqft: int, bhk: str) -> int:
    if 80 <= int(size_sqft or 0) <= 5000:
        return int(size_sqft)
    defaults = {"1": 300, "2": 650, "3": 1000, "4": 1400}
    return defaults.get(str(bhk), 300)


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
    seen_image_hashes: set[str],
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
        digest = hashlib.sha256(response.content).hexdigest()
        if digest in seen_image_hashes:
            print("  image skipped: duplicate content", flush=True)
            continue
        seen_image_hashes.add(digest)
        path.write_bytes(response.content)
        names[index - 1] = name
    return names


def load_existing_image_hashes(image_dir: Path) -> set[str]:
    if not image_dir.exists():
        return set()
    hashes = set()
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
            except OSError:
                continue
    print(f"Loaded {len(hashes)} existing image hashes for dedupe", flush=True)
    return hashes


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


def html_to_text(value: str) -> str:
    return clean_text(unescape(re.sub(r"<[^>]+>", " ", value or "")))


if __name__ == "__main__":
    main()
