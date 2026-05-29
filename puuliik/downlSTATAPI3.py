import os
import time
import random
import requests
import geopandas as gpd
import pandas as pd

from sentinelhub import (
    SHConfig,
    SentinelHubStatistical,
    DataCollection,
    Geometry,
    CRS
)

# =========================================================
# SISEND / VÄLJUND
# =========================================================

INPUT = r"C:\Users\liisl\Documents\Kaugseire\Puuliik\DATA\final_split_clean.gpkg"
OUTPUT = r"C:\Users\liisl\Documents\Kaugseire\Puuliik\DATA\timeseries_ext_full_test_mean_median.csv"

ID_COL = "id_left"

# Soovi korral filtreeri
SELECT_SUBREGION = "Subregion2"   # nt "Subregion2" või None
SELECT_SPLIT = "test"            # nt "train", "val", "test" või None

# =========================================================
# AJAVAHEMIK
# =========================================================

START = "2025-01-01"
END   = "2025-12-31"

# =========================================================
# API / DOWNLOAD SEADISTUS
# =========================================================

MAX_RETRIES = 10
SLEEP_BETWEEN_REQUESTS = 0.15

DEBUG = False
DEBUG_N_PER_SPLIT = 5

# =========================================================
# AUTENTIMINE
# =========================================================

# PANE SIIA OMA UUE KONTO PÄRIS VÄÄRTUSED
DIRECT_CLIENT_ID = "sh-dad62d67-d690-4377-b4b6-ce52aefedf25"
DIRECT_CLIENT_SECRET = "IQ9nhQLoqHAv4zPVTxzYpP9pnB47CMQv"

SH_BASE_URL = "https://sh.dataspace.copernicus.eu"
SH_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_S2L2A = DataCollection.SENTINEL2_L2A.define_from(
    name="s2l2a_cdse",
    service_url=SH_BASE_URL
)

def get_config():
    """
    Ei loe config faili ega environment variable'eid.
    Kasutab ainult siin skriptis ette antud võtmeid.
    """
    config = SHConfig(use_defaults=True)
    config.sh_client_id = DIRECT_CLIENT_ID
    config.sh_client_secret = DIRECT_CLIENT_SECRET
    config.sh_base_url = SH_BASE_URL
    config.sh_token_url = SH_TOKEN_URL

    print("BASE URL:", config.sh_base_url)
    print("TOKEN URL:", config.sh_token_url)
    print("CLIENT ID olemas:", bool(config.sh_client_id))
    print("CLIENT SECRET olemas:", bool(config.sh_client_secret))
    print("CLIENT ID algus:", config.sh_client_id[:10] if config.sh_client_id else "PUUDUB")

    return config


def test_token(config):
    """
    Testib kohe alguses, kas antud client_id/client_secret töötab.
    """
    data = {
        "grant_type": "client_credentials",
        "client_id": config.sh_client_id,
        "client_secret": config.sh_client_secret
    }

    r = requests.post(config.sh_token_url, data=data, timeout=30)

    print("TOKEN TEST STATUS:", r.status_code)
    if r.status_code != 200:
        print("TOKEN TEST BODY:", r.text)
        raise RuntimeError("Token test ebaõnnestus. Kontrolli client_id / client_secret väärtusi.")

    print("Token test OK")


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
    output: [
      { id: "bands", bands: 10, sampleType: "FLOAT32" },
      { id: "indices", bands: 6, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1, sampleType: "UINT8" }
    ]
  };
}

function evaluatePixel(sample) {
  let good = sample.dataMask == 1 && (sample.SCL == 4 || sample.SCL == 5 || sample.SCL == 6);

  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04 + 1e-6);
  let ndre = (sample.B08 - sample.B05) / (sample.B08 + sample.B05 + 1e-6);
  let ndmi = (sample.B08 - sample.B11) / (sample.B08 + sample.B11 + 1e-6);
  let nbr  = (sample.B08 - sample.B12) / (sample.B08 + sample.B12 + 1e-6);

  let evi = 2.5 * ((sample.B08 - sample.B04) / (sample.B08 + 6.0 * sample.B04 - 7.5 * sample.B02 + 1.0 + 1e-6));
  let cire = (sample.B08 / (sample.B05 + 1e-6)) - 1.0;

  if (good) {
    return {
      bands: [
        sample.B02, sample.B03, sample.B04, sample.B05, sample.B06,
        sample.B07, sample.B08, sample.B8A, sample.B11, sample.B12
      ],
      indices: [ndvi, ndre, ndmi, nbr, evi, cire],
      dataMask: [1]
    };
  } else {
    return {
      bands: [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
      indices: [NaN, NaN, NaN, NaN, NaN, NaN],
      dataMask: [0]
    };
  }
}
"""

# =========================================================
# ABIFUNKTSIOONID
# =========================================================

def apply_filters(gdf):
    if SELECT_SUBREGION is not None:
        gdf = gdf[gdf["subregion"] == SELECT_SUBREGION].copy()

    if SELECT_SPLIT is not None:
        gdf = gdf[gdf["dataset_split"] == SELECT_SPLIT].copy()

    return gdf


def make_request(geom, config):
    request = SentinelHubStatistical(
        aggregation=SentinelHubStatistical.aggregation(
            evalscript=EVALSCRIPT,
            time_interval=(START, END),
            aggregation_interval="P10D",
            resolution=(10, 10)
        ),
        input_data=[
            SentinelHubStatistical.input_data(
                data_collection=CDSE_S2L2A,
                maxcc=0.3
            )
        ],
        geometry=geom,
        config=config,
        calculations={
            "default": {
                "statistics": {
                    "default": {
                        "percentiles": {
                            "k": [50]
                        }
                    }
                }
            }
        }
    )
    return request


def request_with_retry(row_id, geom, config):
    """
    Retry loogika:
    - 401 AccessToken expired -> tee config uuesti
    - 429 / 502 / 503 -> oota ja proovi uuesti
    - unauthorized_client -> ära retry lõputult, see on credentiali viga
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            request = make_request(geom, config)
            data = request.get_data()
            return data, config

        except Exception as e:
            txt = str(e)

            if "unauthorized_client" in txt or "Invalid client or Invalid client credentials" in txt:
                print(f"Error {row_id}: credentialid on valed või ei sobi token endpointiga.")
                print(txt)
                return None, config

            if "401" in txt or "AccessToken signature expired" in txt or "Unauthorized" in txt:
                print(f"401 token probleem {row_id}, uuendan konfiguratsiooni (katse {attempt}/{MAX_RETRIES})")
                config = get_config()
                wait = min(2 * attempt, 15)
                time.sleep(wait)
                continue

            if "429" in txt or "Too Many Requests" in txt or "RATE_LIMIT_EXCEEDED" in txt:
                wait = min((2 ** attempt) + random.uniform(0, 1.5), 60)
                print(f"429 rate limit for {row_id}, ootan {wait:.1f} s (katse {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                continue

            if "502" in txt or "503" in txt or "Bad Gateway" in txt or "Service Unavailable" in txt:
                wait = min((2 ** attempt) + random.uniform(0, 1.5), 60)
                print(f"Ajutine serveri viga {row_id}, ootan {wait:.1f} s (katse {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                continue

            print(f"Error {row_id}: {e}")
            return None, config

    print(f"Error {row_id}: maksimaalne retry arv täis")
    return None, config


def get_stat_mean_and_median(stats_dict):
    mean_val = stats_dict.get("mean")
    median_val = stats_dict.get("percentiles", {}).get("50.0")
    return mean_val, median_val


def parse_stats(data, row):
    out = []

    if not data or "data" not in data[0]:
        return out

    for interval in data[0]["data"]:
        outputs = interval["outputs"]

        band_stats = outputs["bands"]["bands"]
        idx_stats = outputs["indices"]["bands"]

        B2_mean, B2_median = get_stat_mean_and_median(band_stats["B0"]["stats"])
        B3_mean, B3_median = get_stat_mean_and_median(band_stats["B1"]["stats"])
        B4_mean, B4_median = get_stat_mean_and_median(band_stats["B2"]["stats"])
        B5_mean, B5_median = get_stat_mean_and_median(band_stats["B3"]["stats"])
        B6_mean, B6_median = get_stat_mean_and_median(band_stats["B4"]["stats"])
        B7_mean, B7_median = get_stat_mean_and_median(band_stats["B5"]["stats"])
        B8_mean, B8_median = get_stat_mean_and_median(band_stats["B6"]["stats"])
        B8A_mean, B8A_median = get_stat_mean_and_median(band_stats["B7"]["stats"])
        B11_mean, B11_median = get_stat_mean_and_median(band_stats["B8"]["stats"])
        B12_mean, B12_median = get_stat_mean_and_median(band_stats["B9"]["stats"])

        NDVI_mean, NDVI_median = get_stat_mean_and_median(idx_stats["B0"]["stats"])
        NDRE_mean, NDRE_median = get_stat_mean_and_median(idx_stats["B1"]["stats"])
        NDMI_mean, NDMI_median = get_stat_mean_and_median(idx_stats["B2"]["stats"])
        NBR_mean, NBR_median = get_stat_mean_and_median(idx_stats["B3"]["stats"])
        EVI_mean, EVI_median = get_stat_mean_and_median(idx_stats["B4"]["stats"])
        CIRE_mean, CIRE_median = get_stat_mean_and_median(idx_stats["B5"]["stats"])

        out.append({
            "id": row["id"],
            "date_from": interval["interval"]["from"],
            "date_to": interval["interval"]["to"],

            "B2_mean": B2_mean, "B2_median": B2_median,
            "B3_mean": B3_mean, "B3_median": B3_median,
            "B4_mean": B4_mean, "B4_median": B4_median,
            "B5_mean": B5_mean, "B5_median": B5_median,
            "B6_mean": B6_mean, "B6_median": B6_median,
            "B7_mean": B7_mean, "B7_median": B7_median,
            "B8_mean": B8_mean, "B8_median": B8_median,
            "B8A_mean": B8A_mean, "B8A_median": B8A_median,
            "B11_mean": B11_mean, "B11_median": B11_median,
            "B12_mean": B12_mean, "B12_median": B12_median,

            "NDVI_mean": NDVI_mean, "NDVI_median": NDVI_median,
            "NDRE_mean": NDRE_mean, "NDRE_median": NDRE_median,
            "NDMI_mean": NDMI_mean, "NDMI_median": NDMI_median,
            "NBR_mean": NBR_mean, "NBR_median": NBR_median,
            "EVI_mean": EVI_mean, "EVI_median": EVI_median,
            "CIRE_mean": CIRE_mean, "CIRE_median": CIRE_median,

            "target": row.get("target"),
            "split": row.get("dataset_split"),
            "subregion": row.get("subregion")
        })

    return out


# =========================================================
# MAIN
# =========================================================

def main():
    gdf = gpd.read_file(INPUT)

    print("Veerud:")
    print(gdf.columns.tolist())
    print("Sisend CRS:", gdf.crs)
    print("Bounds enne teisendust:", gdf.total_bounds)

    if ID_COL not in gdf.columns:
        raise ValueError(f"Puudub ID veerg: {ID_COL}")

    gdf = gdf.rename(columns={ID_COL: "id"})
    gdf = apply_filters(gdf)

    if gdf.empty:
        raise ValueError("Pärast filtreerimist ei jäänud ühtegi objekti.")

    if DEBUG:
        if "dataset_split" in gdf.columns:
            gdf = gdf.groupby("dataset_split", group_keys=False).sample(
                n=min(DEBUG_N_PER_SPLIT, gdf.groupby("dataset_split").size().min()),
                random_state=42
            )
        else:
            gdf = gdf.sample(n=min(20, len(gdf)), random_state=42)

    print("Objekte kokku:", len(gdf))

    # Teisendame API jaoks WGS84-ks
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs(4326)

    print("CRS pärast teisendust:", gdf.crs)
    print("Bounds pärast teisendust:", gdf.total_bounds)

    # loo config üks kord
    config = get_config()

    # testi token kohe alguses
    test_token(config)

    results = []

    for i, row in enumerate(gdf.itertuples(index=False), start=1):
        row = row._asdict()
        row_id = row["id"]

        geom = Geometry(row["geometry"], CRS.WGS84)
        data, config = request_with_retry(row_id, geom, config)

        if data is not None:
            parsed = parse_stats(data, row)
            results.extend(parsed)
            print(f"OK {i}/{len(gdf)}: {row_id}")
        else:
            print(f"FAILED {i}/{len(gdf)}: {row_id}")

        time.sleep(SLEEP_BETWEEN_REQUESTS)

        if i % 100 == 0:
            df_tmp = pd.DataFrame(results)
            df_tmp.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
            print(f"Vahepealne salvestus: {OUTPUT} | ridu: {len(df_tmp)}")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print("\nDONE")
    print(f"Rows written: {len(df)}")
    print(f"Saved to: {OUTPUT}")


if __name__ == "__main__":
    main()