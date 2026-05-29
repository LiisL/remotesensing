import os
import requests
import geopandas as gpd
import pandas as pd

INPUT = r"C:\Users\liisl\Documents\Kaugseire\Puuliik\DATA\final_split_clean.gpkg"
OUTPUT = r"C:\Users\liisl\Documents\Kaugseire\Puuliik\DATA\timeseries.csv"

START = "2020-05-01T00:00:00Z"
END = "2020-09-30T23:59:59Z"

ID_COL = "id_left"
DEBUG = True
DEBUG_N_PER_SPLIT = 5

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
STATS_URL = STATS_URL = "https://sh.dataspace.copernicus.eu/statistics/v1"

# kasutame meetrilist CRS-i
REQUEST_EPSG = 3301
CRS_URN = "http://www.opengis.net/def/crs/EPSG/0/3301"

EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02", "B03", "B04", "B08", "SCL", "dataMask"],
      units: ["REFLECTANCE", "REFLECTANCE", "REFLECTANCE", "REFLECTANCE", "DN", "DN"]
    }],
    output: [
      { id: "bands", bands: 4, sampleType: "FLOAT32" },
      { id: "ndvi", bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1, sampleType: "UINT8" }
    ]
  };
}

function evaluatePixel(sample) {
  let good = sample.dataMask == 1 && (sample.SCL == 4 || sample.SCL == 5 || sample.SCL == 6);
  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04 + 1e-6);

  if (good) {
    return {
      bands: [sample.B02, sample.B03, sample.B04, sample.B08],
      ndvi: [ndvi],
      dataMask: [1]
    };
  } else {
    return {
      bands: [NaN, NaN, NaN, NaN],
      ndvi: [NaN],
      dataMask: [0]
    };
  }
}
"""

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
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/3301"
                }
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": START,
                            "to": END
                        },
                        "maxCloudCoverage": 30
                    }
                }
            ]
        },
        "aggregation": {
            "timeRange": {
                "from": START,
                "to": END
            },
            "aggregationInterval": {
                "of": "P10D"
            },
            "evalscript": EVALSCRIPT,
            "resx": 10,
            "resy": 10
        },
        "calculations": {
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
    }

def main():
    gdf = gpd.read_file(INPUT)

    print("Veerud:")
    print(gdf.columns.tolist())

    if DEBUG:
        gdf = gdf.groupby("dataset_split", group_keys=False).sample(
            n=DEBUG_N_PER_SPLIT, random_state=42
        )
        print("Test size:", len(gdf))

    if gdf.crs is None:
        raise RuntimeError("Sisendfailil puudub CRS")

    # teisendame MEETRILISSE CRS-i, mitte WGS84
    gdf = gdf.to_crs(REQUEST_EPSG)

    token = get_access_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    results = []

    for _, row in gdf.iterrows():
        row_id = row.get(ID_COL, row.name)

        try:
            geom_json = row.geometry.__geo_interface__
            payload = build_payload(geom_json)

            r = requests.post(STATS_URL, headers=headers, json=payload, timeout=120)

            if r.status_code != 200:
                print(f"HTTP {r.status_code} for {row_id}")
                print(r.text)

            r.raise_for_status()
            data = r.json()

            for interval in data.get("data", []):
                outputs = interval["outputs"]

                band_stats = outputs["bands"]["bands"]
                ndvi_stats = outputs["ndvi"]["bands"]["B0"]["stats"]

                results.append({
                    "id": row_id,
                    "date_from": interval["interval"]["from"],
                    "date_to": interval["interval"]["to"],
                    "B2_mean": band_stats["B0"]["stats"].get("mean"),
                    "B3_mean": band_stats["B1"]["stats"].get("mean"),
                    "B4_mean": band_stats["B2"]["stats"].get("mean"),
                    "B8_mean": band_stats["B3"]["stats"].get("mean"),
                    "NDVI_mean": ndvi_stats.get("mean"),
                    "target": row.get("target"),
                    "split": row.get("dataset_split"),
                    "subregion": row.get("subregion"),
                })

            print(f"OK: {row_id}")

        except Exception as e:
            print(f"Error {row_id}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print("DONE")
    print(f"Rows written: {len(df)}")
    print(f"Saved to: {OUTPUT}")

if __name__ == "__main__":
    main()