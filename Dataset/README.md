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
