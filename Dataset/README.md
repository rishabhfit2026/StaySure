# Dataset

Expected CSV columns:

```text
id,img1,img2,img3,img4,img5,location,size_sqft,furnishing_type,cleanliness_score,amenities,rent_price,bhk
```

Room images can be stored in `Dataset/rooms/` locally or provided through a Kaggle dataset path such as:

```text
/kaggle/input/staysure-room-images/rooms
```

Image columns may contain filenames, relative paths, absolute paths, or public image URLs.

## Collect Public OLX Rental Data

Use `olx_collect.py` to collect public rental listings into the same CSV layout as the
Google Sheet. One OLX listing becomes one dataset row, and the listing's own gallery
photos fill `img1` through `img5` when available. By default the collector keeps
room-focused posts and filters out OLX category icons, footer images, social icons,
and resized duplicate images. Respect OLX's terms, keep collection slow, and do not
collect phone numbers, seller names, or private contact details.

Install scraper dependencies:

```bash
pip install -r Dataset/requirements-scraper.txt
python -m playwright install chromium
```

Collect Bhilai rental listings and download up to five listing images per room:

```bash
python Dataset/olx_collect.py \
  --start-url "https://www.olx.in/bhilai_g4059463/for-rent-houses-apartments_c1723" \
  --max-listings 200 \
  --room-only \
  --download-images \
  --output-csv Dataset/rooms_dataset.csv \
  --output-xlsx Dataset/rooms_dataset.xlsx \
  --image-dir Dataset/rooms
```

If OLX blocks headless browsing, run with a visible browser:

```bash
python Dataset/olx_collect.py \
  --headful \
  --max-listings 200 \
  --room-only \
  --download-images
```

Use `--no-room-only` only when you intentionally want full-house and full-apartment
listings in the dataset too.

If browser rendering times out but normal HTTP access works, collect listing-card
data with the fallback collector. It opens each listing page over HTTP and fills up
to five gallery images from the same home/listing into `img1` through `img5`:

```bash
python Dataset/olx_http_collect.py \
  --start-url "https://www.olx.in/en-in/bilaspur_g4059465/q-room-rent" \
  --start-url "https://www.olx.in/en-in/raipur_g4059473/q-room-rent" \
  --start-url "https://www.olx.in/en-in/bhilai_g4059463/q-room-rent" \
  --max-images 111 \
  --images-per-listing 5 \
  --start-id 120 \
  --output-csv Dataset/rooms_dataset_topup.csv \
  --image-dir Dataset/rooms \
  --dedupe-image-dir Dataset/rooms
```

Append a compatible top-up CSV into the main training file:

```bash
python Dataset/merge_room_datasets.py \
  --base-csv Dataset/rooms_dataset.csv \
  --append-csv Dataset/rooms_dataset_topup.csv \
  --output-csv Dataset/rooms_dataset.csv \
  --output-xlsx Dataset/rooms_dataset.xlsx
```

Clean required metadata fields after merging so model inputs do not contain blank
`bhk`, `amenities`, `location`, or zero `size_sqft` values:

```bash
python Dataset/clean_room_dataset.py \
  --input-csv Dataset/rooms_dataset.csv \
  --output-csv Dataset/rooms_dataset.csv \
  --output-xlsx Dataset/rooms_dataset.xlsx
```

The generated CSV is ignored by git because it can become large. Import
`Dataset/rooms_dataset.csv` into Google Sheets with **File > Import > Upload**,
or upload the CSV and `Dataset/rooms/` folder to Kaggle for training.

Train with the generated local dataset:

```bash
python Model/train.py \
  --csv-path Dataset/rooms_dataset.csv \
  --image-root Dataset/rooms \
  --output-dir Model/artifacts \
  --epochs 25 \
  --batch-size 16 \
  --folds 5
```
