import os
import time
import random
import requests
import geopandas as gpd
import pandas as pd

# =========================
# FAILID
# =========================

INPUT = r"C:\Users\liisl\Documents\Kaugseire\Puuliik\DATA\final_split_clean.gpkg"
OUTPUT = r"C:\Users\liisl\Documents\Kaugseire\Puuliik\DATA\Subregion2_val_timeseries1704w.csv"

# ebaõnnestunud objektide logi
FAILED_LOG = OUTPUT.replace(".csv", "_failed_ids.txt")

# =========================
# AJAPERIOOD
# =========================

START = "2025-01-01T00:00:00Z"
END = "2025-03-31T23:59:59Z"

# =========================
# FILTRID
# =========================

SELECT_SUBREGION = "Subregion2"
SELECT_SPLIT = "val"

# =========================
# DEBUG
# =========================

DEBUG = False
DEBUG_N_PER_SPLIT = 5

# =========================
# MUUD SEADISTUSED
# =========================

ID_COL = "id_left"
REQUEST_EPSG = 3301
CRS_URN = "http://www.opengis.net/def/crs/EPSG/0/3301"

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
STATS_URL = "https://sh.dataspace.copernicus.eu/statistics/v1"

SAVE_EVERY_N_STANDS = 10

# Kui True, kustutab vana CSV ja alustab nullist
# Kui False, jätkab olemasoleva CSV peale
OVERWRITE_OUTPUT = False

# Kui True, jätab juba CSV-s olemasolevad id-d vahele
SKIP_ALREADY_PROCESSED = True

# paus edukate päringute vahel
SLEEP_BETWEEN_REQUESTS = 0.8

# retry seaded
MAX_RETRIES = 12
INITIAL_BACKOFF = 3.0
MAX_BACKOFF = 90.0

# =========================
# EVALSCRIPT
# =========================

EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL", "dataMask"],
      units: ["REFLECTANCE","REFLECTANCE","REFLECTANCE","REFLECTANCE","REFLECTANCE","REFLECTANCE",
              "REFLECTANCE","REFLECTANCE","REFLECTANCE","REFLECTANCE","DN","DN"]
    }],
    output: [
      { id: "bands", bands: 10, sampleType: "FLOAT32" },
      { id: "indices", bands: 4, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1, sampleType: "UINT8" }
    ]
  };
}

function evaluatePixel(sample) {
  let good = sample.dataMask == 1 && (sample.SCL == 4 || sample.SCL == 5 || sample.SCL == 6);

  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04 + 1e-6);
  let ndre = (sample.B8A - sample.B05) / (sample.B8A + sample.B05 + 1e-6);
  let ndmi = (sample.B08 - sample.B11) / (sample.B08 + sample.B11 + 1e-6);
  let nbr  = (sample.B08 - sample.B12) / (sample.B08 + sample.B12 + 1e-6);

  if (good) {
    return {
      bands: [
        sample.B02, sample.B03, sample.B04, sample.B05, sample.B06,
        sample.B07, sample.B08, sample.B8A, sample.B11, sample.B12
      ],
      indices: [ndvi, ndre, ndmi, nbr],
      dataMask: [1]
    };
  } else {
    return {
      bands: [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
      indices: [NaN, NaN, NaN, NaN],
      dataMask: [0]
    };
  }
}
"""

# =========================
# ABIFUNKTSIOONID
# =========================

def get_access_token():
    client_id = os.getenv("SH_CLIENT_ID")
    client_secret = os.getenv("SH_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise RuntimeError("Puuduvad SH_CLIENT_ID või SH_CLIENT_SECRET")

    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }

    r = requests.post(TOKEN_URL, data=data, timeout=60)
    r.raise_for_status()
    return r.json()["access_token"]


def build_payload(geometry_geojson):
    return {
        "input": {
            "bounds": {
                "geometry": geometry_geojson,
                "properties": {"crs": CRS_URN}
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {"from": START, "to": END},
                        "maxCloudCoverage": 30
                    }
                }
            ]
        },
        "aggregation": {
            "timeRange": {"from": START, "to": END},
            "aggregationInterval": {"of": "P10D"},
            "evalscript": EVALSCRIPT,
            "resx": 10,
            "resy": 10
        },
        "calculations": {
            "default": {
                "statistics": {
                    "default": {
                        "percentiles": {"k": [50]}
                    }
                }
            }
        }
    }


def normalize_filter_to_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def apply_filters(gdf):
    subregions = normalize_filter_to_list(SELECT_SUBREGION)
    splits = normalize_filter_to_list(SELECT_SPLIT)

    if subregions is not None:
        gdf = gdf[gdf["subregion"].isin(subregions)].copy()

    if splits is not None:
        gdf = gdf[gdf["dataset_split"].isin(splits)].copy()

    return gdf


def write_results(df_part, output_path, first_write):
    if df_part.empty:
        return

    mode = "w" if first_write else "a"
    header = first_write

    df_part.to_csv(
        output_path,
        mode=mode,
        header=header,
        index=False,
        encoding="utf-8-sig"
    )


def load_processed_ids(output_csv, id_col="id"):
    if not os.path.exists(output_csv):
        return set()

    try:
        df_existing = pd.read_csv(output_csv, usecols=[id_col])
        processed_ids = set(df_existing[id_col].dropna().astype(str).unique())
        print(f"Leidsin olemasolevast CSV-st {len(processed_ids)} juba töödeldud ID-d.")
        return processed_ids
    except Exception as e:
        print(f"Ei suutnud olemasolevat CSV-d lugeda: {e}")
        return set()


def post_stats_with_retry(payload, row_id, token_state):
    delay = INITIAL_BACKOFF

    for attempt in range(1, MAX_RETRIES + 1):
        headers = {
            "Authorization": f"Bearer {token_state['token']}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        try:
            r = requests.post(STATS_URL, headers=headers, json=payload, timeout=120)
        except requests.exceptions.RequestException as e:
            wait_time = min(delay, MAX_BACKOFF) + random.uniform(0, 1)
            print(f"Võrguviga objektil {row_id}: {e}. Ootan {wait_time:.1f} s (katse {attempt}/{MAX_RETRIES})")
            time.sleep(wait_time)
            delay *= 2
            continue

        if r.status_code == 200:
            return r

        if r.status_code == 401:
            print(f"401 token aegus objektil {row_id}, uuendan tokeni")
            token_state["token"] = get_access_token()
            time.sleep(1)
            continue

        if r.status_code == 429:
            wait_time = min(delay, MAX_BACKOFF) + random.uniform(0, 1)
            print(f"429 rate limit objektil {row_id}, ootan {wait_time:.1f} s (katse {attempt}/{MAX_RETRIES})")
            time.sleep(wait_time)
            delay *= 2
            continue

        if r.status_code in (500, 502, 503, 504):
            wait_time = min(delay, MAX_BACKOFF) + random.uniform(0, 1)
            print(f"{r.status_code} ajutine serveriviga objektil {row_id}, ootan {wait_time:.1f} s (katse {attempt}/{MAX_RETRIES})")
            print(r.text[:500])
            time.sleep(wait_time)
            delay *= 2
            continue

        print(f"HTTP {r.status_code} for {row_id}")
        print(r.text[:1000])
        r.raise_for_status()

    raise RuntimeError(f"Päring ebaõnnestus pärast {MAX_RETRIES} katset: {row_id}")


# =========================
# MAIN
# =========================

def main():
    gdf = gpd.read_file(INPUT)

    print("Veerud:")
    print(gdf.columns.tolist())

    gdf = apply_filters(gdf)

    if gdf.empty:
        raise RuntimeError("Pärast filtreerimist ei jäänud ühtegi objekti.")

    print("\nValitud subregionid:")
    print(sorted(gdf["subregion"].dropna().unique().tolist()))

    print("\nValitud splitid:")
    print(sorted(gdf["dataset_split"].dropna().unique().tolist()))

    print("\nObjekte kokku:", len(gdf))
    print(gdf.groupby(["subregion", "dataset_split"]).size())

    if DEBUG:
        if "dataset_split" in gdf.columns:
            groups = []
            for _, part in gdf.groupby("dataset_split"):
                n = min(DEBUG_N_PER_SPLIT, len(part))
                groups.append(part.sample(n=n, random_state=42))
            gdf = pd.concat(groups, ignore_index=True)
        else:
            gdf = gdf.sample(min(DEBUG_N_PER_SPLIT, len(gdf)), random_state=42)

        print("\nDEBUG objekte:", len(gdf))

    if gdf.crs is None:
        raise RuntimeError("Sisendfailil puudub CRS")

    gdf = gdf.to_crs(REQUEST_EPSG)

    token_state = {"token": get_access_token()}

    processed_ids = set()

    if OVERWRITE_OUTPUT and os.path.exists(OUTPUT):
        os.remove(OUTPUT)
        processed_ids = set()
    elif SKIP_ALREADY_PROCESSED and os.path.exists(OUTPUT):
        processed_ids = load_processed_ids(OUTPUT, id_col="id")

    if os.path.exists(FAILED_LOG):
        os.remove(FAILED_LOG)

    results_buffer = []
    failed_ids = []
    first_write = not os.path.exists(OUTPUT)
    processed_count = 0
    skipped_count = 0

    total_objects = len(gdf)

    for _, row in gdf.iterrows():
        row_id = row.get(ID_COL, row.name)
        row_id_str = str(row_id)

        if SKIP_ALREADY_PROCESSED and row_id_str in processed_ids:
            skipped_count += 1
            print(f"Skip already processed ({skipped_count}): {row_id}")
            continue

        try:
            geom_json = row.geometry.__geo_interface__
            payload = build_payload(geom_json)

            r = post_stats_with_retry(payload, row_id, token_state)
            data = r.json()

            for interval in data.get("data", []):
                outputs = interval["outputs"]

                band_stats = outputs["bands"]["bands"]
                idx_stats = outputs["indices"]["bands"]

                results_buffer.append({
                    "id": row_id,
                    "date_from": interval["interval"]["from"],
                    "date_to": interval["interval"]["to"],

                    "B2_mean": band_stats["B0"]["stats"].get("mean"),
                    "B3_mean": band_stats["B1"]["stats"].get("mean"),
                    "B4_mean": band_stats["B2"]["stats"].get("mean"),
                    "B5_mean": band_stats["B3"]["stats"].get("mean"),
                    "B6_mean": band_stats["B4"]["stats"].get("mean"),
                    "B7_mean": band_stats["B5"]["stats"].get("mean"),
                    "B8_mean": band_stats["B6"]["stats"].get("mean"),
                    "B8A_mean": band_stats["B7"]["stats"].get("mean"),
                    "B11_mean": band_stats["B8"]["stats"].get("mean"),
                    "B12_mean": band_stats["B9"]["stats"].get("mean"),

                    "NDVI_mean": idx_stats["B0"]["stats"].get("mean"),
                    "NDRE_mean": idx_stats["B1"]["stats"].get("mean"),
                    "NDMI_mean": idx_stats["B2"]["stats"].get("mean"),
                    "NBR_mean": idx_stats["B3"]["stats"].get("mean"),

                    "target": row.get("target"),
                    "split": row.get("dataset_split"),
                    "subregion": row.get("subregion"),
                })

            processed_count += 1
            processed_ids.add(row_id_str)
            print(f"OK {processed_count}/{total_objects}: {row_id}")

            if processed_count % SAVE_EVERY_N_STANDS == 0 and results_buffer:
                df_part = pd.DataFrame(results_buffer)
                write_results(df_part, OUTPUT, first_write)
                first_write = False
                results_buffer = []
                print(f"Vahe-salvestus tehtud: {OUTPUT}")

            time.sleep(SLEEP_BETWEEN_REQUESTS)

        except Exception as e:
            print(f"Error {row_id}: {e}")
            failed_ids.append(str(row_id))

    if results_buffer:
        df_part = pd.DataFrame(results_buffer)
        write_results(df_part, OUTPUT, first_write)

    if failed_ids:
        with open(FAILED_LOG, "w", encoding="utf-8") as f:
            for fid in failed_ids:
                f.write(f"{fid}\n")
        print(f"Ebaõnnestunud ID-d salvestatud: {FAILED_LOG}")

    print("\nDONE")
    print(f"Processed this run: {processed_count}")
    print(f"Skipped already processed: {skipped_count}")
    print(f"Saved to: {OUTPUT}")


if __name__ == "__main__":
    main()