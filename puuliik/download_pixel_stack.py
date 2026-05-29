import os
import json
import time
import requests
import geopandas as gpd
import pandas as pd
from datetime import datetime, timedelta

# =========================================================
# SISENDID
# =========================================================

AOI_FILE = r"C:\Users\liisl\Documents\Kaugseire\Puuliik\DATA\pixel_test_area.shp"

OUT_DIR = r"C:\Users\liisl\Documents\Kaugseire\Puuliik\PIXEL_TEST\rasters"

CLIENT_ID = "sh-dad62d67-d690-4377-b4b6-ce52aefedf25"
CLIENT_SECRET = "IQ9nhQLoqHAv4zPVTxzYpP9pnB47CMQv"

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# =========================================================
# SEADISTUS
# =========================================================

# Esimene test: kasuta tähtsamaid kuupäevi
DATES = [
    "2025-04-01",
    "2025-05-01",
    "2025-05-31",
    "2025-07-20",
    "2025-09-28",
    "2025-10-08",
    "2025-10-18"
]

# 10-päevane aken nagu sinu aegridade loogikas
WINDOW_DAYS = 10

# väljundi resolutsioon meetrites
RESOLUTION = 10

# kasutame EPSG:3301, sest AOI on selles süsteemis ja Eesti jaoks mugav
TARGET_EPSG = 3301

# väljundis on 14 kihti
BAND_NAMES = [
    "B2", "B3", "B4", "B5", "B6", "B7",
    "B8", "B8A", "B11", "B12",
    "NDVI", "NDRE", "NDMI", "NBR"
]

# =========================================================
# EVALSCRIPT
# =========================================================

EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL", "dataMask"],
      units: ["REFLECTANCE", "REFLECTANCE", "REFLECTANCE", "REFLECTANCE", "REFLECTANCE",
              "REFLECTANCE", "REFLECTANCE", "REFLECTANCE", "REFLECTANCE", "REFLECTANCE",
              "DN", "DN"]
    }],
    output: {
      bands: 14,
      sampleType: "FLOAT32",
      nodataValue: -9999
    }
  };
}

function evaluatePixel(sample) {
  let good = sample.dataMask == 1 && (sample.SCL == 4 || sample.SCL == 5 || sample.SCL == 6);

  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04 + 1e-6);
  let ndre = (sample.B08 - sample.B05) / (sample.B08 + sample.B05 + 1e-6);
  let ndmi = (sample.B08 - sample.B11) / (sample.B08 + sample.B11 + 1e-6);
  let nbr  = (sample.B08 - sample.B12) / (sample.B08 + sample.B12 + 1e-6);

  if (good) {
    return [
      sample.B02, sample.B03, sample.B04, sample.B05, sample.B06, sample.B07,
      sample.B08, sample.B8A, sample.B11, sample.B12,
      ndvi, ndre, ndmi, nbr
    ];
  } else {
    return [
      -9999, -9999, -9999, -9999, -9999, -9999,
      -9999, -9999, -9999, -9999,
      -9999, -9999, -9999, -9999
    ];
  }
}
"""

# =========================================================
# FUNKTSIOONID
# =========================================================

def get_token():
    r = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET
        },
        timeout=60
    )

    print("TOKEN STATUS:", r.status_code)

    if r.status_code != 200:
        print(r.text)
        raise RuntimeError("Tokeni küsimine ebaõnnestus.")

    return r.json()["access_token"]


def load_aoi_bbox():
    gdf = gpd.read_file(AOI_FILE)

    print("AOI CRS:", gdf.crs)
    print("AOI objektide arv:", len(gdf))

    if gdf.empty:
        raise RuntimeError("AOI fail on tühi.")

    # Teisenda EPSG:3301 süsteemi
    if gdf.crs is None:
        raise RuntimeError("AOI failil puudub CRS. Määra QGIS-is EPSG:3301.")

    gdf = gdf.to_crs(epsg=TARGET_EPSG)

    minx, miny, maxx, maxy = gdf.total_bounds

    width = int((maxx - minx) / RESOLUTION)
    height = int((maxy - miny) / RESOLUTION)

    if width <= 0 or height <= 0:
        raise RuntimeError("AOI bbox on vigane või liiga väike.")

    print("AOI bounds EPSG:3301:", [minx, miny, maxx, maxy])
    print("Raster size:", width, "x", height)

    return [minx, miny, maxx, maxy], width, height


def date_window(date_str):
    start = datetime.strptime(date_str, "%Y-%m-%d")
    end = start + timedelta(days=WINDOW_DAYS)

    return (
        start.strftime("%Y-%m-%dT00:00:00Z"),
        end.strftime("%Y-%m-%dT23:59:59Z")
    )


def make_process_payload(bbox, width, height, start_time, end_time):
    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {
                    "crs": f"http://www.opengis.net/def/crs/EPSG/0/{TARGET_EPSG}"
                }
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": start_time,
                            "to": end_time
                        },
                        "maxCloudCoverage": 30,
                        "mosaickingOrder": "leastCC"
                    }
                }
            ]
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [
                {
                    "identifier": "default",
                    "format": {
                        "type": "image/tiff"
                    }
                }
            ]
        },
        "evalscript": EVALSCRIPT
    }

    return payload


def download_one_date(token, date_str, bbox, width, height):
    start_time, end_time = date_window(date_str)

    out_tif = os.path.join(OUT_DIR, f"pixel_stack_{date_str}.tif")

    if os.path.exists(out_tif):
        print(f"Juba olemas, jätan vahele: {out_tif}")
        return out_tif

    payload = make_process_payload(bbox, width, height, start_time, end_time)

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "image/tiff"
    }

    print(f"\nLaen kuupäeva: {date_str}")
    print("Ajavahemik:", start_time, "kuni", end_time)

    r = requests.post(PROCESS_URL, headers=headers, json=payload, timeout=300)

    print("STATUS:", r.status_code)

    if r.status_code == 401:
        raise RuntimeError("401 Unauthorized. Token või õigused ei sobi.")

    if r.status_code == 429:
        raise RuntimeError("429 Rate limit. Proovi hiljem või lisa retry.")

    if r.status_code >= 400:
        print(r.text[:1000])
        raise RuntimeError(f"Process API viga: {r.status_code}")

    with open(out_tif, "wb") as f:
        f.write(r.content)

    print("Salvestatud:", out_tif)
    return out_tif


def save_band_metadata():
    meta_csv = os.path.join(OUT_DIR, "band_order.csv")
    df = pd.DataFrame({
        "band_index": list(range(1, len(BAND_NAMES) + 1)),
        "band_name": BAND_NAMES
    })
    df.to_csv(meta_csv, index=False, encoding="utf-8-sig")
    print("Band order salvestatud:", meta_csv)


# =========================================================
# MAIN
# =========================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    token = get_token()
    bbox, width, height = load_aoi_bbox()

    save_band_metadata()

    for date_str in DATES:
        try:
            download_one_date(token, date_str, bbox, width, height)
            time.sleep(1)
        except Exception as e:
            print(f"Viga kuupäeval {date_str}: {e}")

    print("\nValmis.")
    print("Rasterid on kaustas:", OUT_DIR)
    print("Ava GeoTIFF-id QGIS-is ja kontrolli, kas need paiknevad õigesti.")


if __name__ == "__main__":
    main()