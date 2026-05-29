import os
import re
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import joblib
import rasterio

from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.features import rasterize

# =========================================================
# SISENDID
# =========================================================

MODEL_FILE = r"C:\Users\liisl\Documents\Kaugseire\Puuliik\DATA\Subregion2_rf_model_with_aux0405.joblib"

RASTER_DIR = r"C:\Users\liisl\Documents\Kaugseire\Puuliik\PIXEL_TEST\rasters"

AOI_FILE = r"C:\Users\liisl\Documents\Kaugseire\Puuliik\DATA\pixel_test_area.shp"

ELEV_RASTER = r"C:\Users\liisl\Documents\Kaugseire\Puuliik\DATA\H_20_21_22_23.tif"
SOIL_RASTER = r"C:\Users\liisl\Documents\Kaugseire\Puuliik\DATA\soil_siffer_10m.tif"

# PANE SIIA OMA ÜLE-EESTILISE METSAMASKI TEE
FOREST_MASK_RASTER = r"C:\Users\liisl\Documents\Kaugseire\data\lageraie26\metsamask\metsamask.tif"

OUT_DIR = r"C:\Users\liisl\Documents\Kaugseire\Puuliik\PIXEL_TEST\predictions"

SPECIES_TIF = os.path.join(OUT_DIR, "species_map_forestmask.tif")
CONF_TIF = os.path.join(OUT_DIR, "confidence_map_forestmask.tif")
CLASS_CODES_CSV = os.path.join(OUT_DIR, "class_codes.csv")

# =========================================================
# SEADISTUS
# =========================================================

NODATA_CLASS = 255
NODATA_FLOAT = -9999.0

# Metsamaski väärtused.
# Kui sinu metsamaskis on mets = 1, jäta nii.
# Kui mets on mingi muu väärtus, muuda näiteks [1, 2] vms.
FOREST_VALUES = [1]

BAND_NAMES = [
    "B2", "B3", "B4", "B5", "B6", "B7",
    "B8", "B8A", "B11", "B12",
    "NDVI", "NDRE", "NDMI", "NBR"
]

AUX_NUMERIC_COLS = [
    "elev_mean", "elev_median", "elev_std", "elev_min", "elev_max"
]

SOIL_PREFIX = "soil_main_"

# =========================================================
# ABIFUNKTSIOONID
# =========================================================

def date_from_filename(path):
    name = os.path.basename(path)
    m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    if not m:
        raise ValueError(f"Kuupäeva ei leitud failinimest: {name}")
    return m.group(1)


def get_reference_raster(stack_files):
    if not stack_files:
        raise RuntimeError("Raster stack faile ei leitud.")

    ref = stack_files[0]

    with rasterio.open(ref) as src:
        profile = src.profile.copy()
        height = src.height
        width = src.width
        transform = src.transform
        crs = src.crs

    return ref, profile, height, width, transform, crs


def make_aoi_mask(aoi_file, height, width, transform, crs):
    aoi = gpd.read_file(aoi_file)

    if aoi.empty:
        raise RuntimeError("AOI fail on tühi.")

    aoi = aoi.to_crs(crs)

    shapes = [(geom, 1) for geom in aoi.geometry if geom is not None]

    mask = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8"
    )

    return mask.astype(bool)


def read_raster_aligned(path, ref_profile, resampling):
    """
    Loeb rastri ja joondab selle reference rasteri ruudustikule.
    """
    with rasterio.open(path) as src:
        with WarpedVRT(
            src,
            crs=ref_profile["crs"],
            transform=ref_profile["transform"],
            width=ref_profile["width"],
            height=ref_profile["height"],
            resampling=resampling
        ) as vrt:
            arr = vrt.read(1).astype("float32")

            nodata = vrt.nodata
            if nodata is not None:
                arr[arr == nodata] = np.nan

            return arr


def make_forest_mask(path, ref_profile):
    """
    Loeb metsamaski ja teeb sellest boolean maski.
    Eeldus: mets = FOREST_VALUES.
    """
    forest_arr = read_raster_aligned(
        path,
        ref_profile,
        Resampling.nearest
    )

    forest_mask = np.isin(forest_arr, FOREST_VALUES)

    # NaN ei ole mets
    forest_mask = forest_mask & ~np.isnan(forest_arr)

    return forest_mask


def build_feature_table(stack_data, stack_dates, elev_arr, soil_arr, feature_cols):
    h, w = elev_arr.shape
    n_pixels = h * w

    data = {}

    # Sentinel-2 rasterid
    for date_str, arr in zip(stack_dates, stack_data):
        date_key = date_str.replace("-", "_")

        for band_idx, band_name in enumerate(BAND_NAMES):
            col = f"{band_name}_mean_{date_key}"

            if col in feature_cols:
                band = arr[band_idx].astype("float32")
                band[band == NODATA_FLOAT] = np.nan
                band[band <= -9990] = np.nan
                data[col] = band.reshape(-1)

    # AUX: kõrgus
    elev_flat = elev_arr.reshape(-1)

    if "elev_mean" in feature_cols:
        data["elev_mean"] = elev_flat
    if "elev_median" in feature_cols:
        data["elev_median"] = elev_flat
    if "elev_min" in feature_cols:
        data["elev_min"] = elev_flat
    if "elev_max" in feature_cols:
        data["elev_max"] = elev_flat
    if "elev_std" in feature_cols:
        data["elev_std"] = np.zeros(n_pixels, dtype="float32")

    # AUX: mullastik one-hot
    soil_flat = soil_arr.reshape(-1)
    soil_feature_cols = [c for c in feature_cols if c.startswith(SOIL_PREFIX)]

    if soil_feature_cols:
        soil_series = (
            pd.Series(soil_flat)
            .round()
            .astype("Int64")
            .astype(str)
        )
        soil_as_str = soil_series.values

        for col in soil_feature_cols:
            soil_code = col.replace(SOIL_PREFIX, "")
            data[col] = (soil_as_str == soil_code).astype("uint8")

    # DataFrame korraga olemasolevatest veergudest
    X = pd.DataFrame(data)

    # Lisa puuduvad mudeli feature'id korraga, mitte ükshaaval
    missing_cols = [c for c in feature_cols if c not in X.columns]

    if missing_cols:
        missing_df = pd.DataFrame(
            np.nan,
            index=X.index,
            columns=missing_cols
        )
        X = pd.concat([X, missing_df], axis=1)

    # Õige järjekord nagu mudelis
    X = X[feature_cols].copy()

    return X


# =========================================================
# MAIN
# =========================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    bundle = joblib.load(MODEL_FILE)

    model = bundle["model"]
    imputer = bundle["imputer"]
    feature_cols = bundle["feature_cols"]

    stack_files = sorted(glob.glob(os.path.join(RASTER_DIR, "pixel_stack_*.tif")))

    print("Leitud raster-stack faile:", len(stack_files))
    for f in stack_files:
        print(" ", os.path.basename(f))

    ref_file, profile, height, width, transform, crs = get_reference_raster(stack_files)

    print("\nReference raster:", ref_file)
    print("Size:", width, "x", height)
    print("CRS:", crs)

    # AOI mask
    print("\nTeen AOI maski...")
    aoi_mask = make_aoi_mask(AOI_FILE, height, width, transform, crs)

    # Metsamask
    print("Loen ja joondan metsamaski...")
    forest_mask = make_forest_mask(FOREST_MASK_RASTER, profile)

    print("AOI piksleid:", int(aoi_mask.sum()))
    print("Metsapiksleid reference extentis:", int(forest_mask.sum()))
    print("AOI sees metsapiksleid:", int((aoi_mask & forest_mask).sum()))

    # Loe Sentinel-2 stackid
    stack_data = []
    stack_dates = []

    for path in stack_files:
        date_str = date_from_filename(path)
        stack_dates.append(date_str)

        with rasterio.open(path) as src:
            arr = src.read().astype("float32")

            if src.count != len(BAND_NAMES):
                raise RuntimeError(
                    f"{path}: oodatud {len(BAND_NAMES)} bandi, aga failis on {src.count}"
                )

            stack_data.append(arr)

    print("\nKuupäevad:", stack_dates)

    # AUX rasterid samale ruudustikule
    print("\nLoen ja joondan kõrgusrastri...")
    elev_arr = read_raster_aligned(
        ELEV_RASTER,
        profile,
        Resampling.bilinear
    )

    print("Loen ja joondan mullastikurastri...")
    soil_arr = read_raster_aligned(
        SOIL_RASTER,
        profile,
        Resampling.nearest
    )

    # Feature tabel
    print("\nEhitan pikslite feature tabelit...")
    X = build_feature_table(
        stack_data=stack_data,
        stack_dates=stack_dates,
        elev_arr=elev_arr,
        soil_arr=soil_arr,
        feature_cols=feature_cols
    )

    print("X shape:", X.shape)
    print("Mudeli feature arv:", len(feature_cols))

    # Mask: AOI + mets
    valid_mask_2d = aoi_mask & forest_mask
    valid_mask = valid_mask_2d.reshape(-1)

    # Ära ennusta piksleid, kus kõik Sentinel-2 tunnused puuduvad
    sentinel_cols = [
        c for c in feature_cols
        if any(c.startswith(prefix + "_mean_") for prefix in BAND_NAMES)
    ]

    if sentinel_cols:
        has_s2_data = X[sentinel_cols].notna().any(axis=1).values
        valid_mask = valid_mask & has_s2_data

    print("\nPiksleid kokku:", height * width)
    print("AOI + mets + S2 andmetega piksleid:", int(valid_mask.sum()))

    if valid_mask.sum() == 0:
        raise RuntimeError("Ühtegi ennustatavat pikslit ei leitud. Kontrolli AOI, metsamaski ja rastereid.")

    species_code_flat = np.full(height * width, NODATA_CLASS, dtype="uint8")
    conf_flat = np.full(height * width, NODATA_FLOAT, dtype="float32")

    X_valid = X.loc[valid_mask].copy()

    print("\nImpute + predict...")
    X_imp = imputer.transform(X_valid)

    pred_labels = model.predict(X_imp)
    proba = model.predict_proba(X_imp)
    conf = proba.max(axis=1)

    classes = list(model.classes_)
    class_to_code = {cls: i + 1 for i, cls in enumerate(classes)}

    pred_codes = np.array([class_to_code[p] for p in pred_labels], dtype="uint8")

    species_code_flat[valid_mask] = pred_codes
    conf_flat[valid_mask] = conf.astype("float32")

    species_map = species_code_flat.reshape(height, width)
    conf_map = conf_flat.reshape(height, width)

    # Klassikoodide tabel
    class_df = pd.DataFrame({
        "class_code": [class_to_code[c] for c in classes],
        "class_name": classes
    })

    class_df.to_csv(CLASS_CODES_CSV, index=False, encoding="utf-8-sig")
    print("Klassikoodid:", CLASS_CODES_CSV)
    print(class_df)

    # Species raster
    species_profile = profile.copy()
    species_profile.update(
        count=1,
        dtype="uint8",
        nodata=NODATA_CLASS,
        compress="lzw"
    )

    with rasterio.open(SPECIES_TIF, "w", **species_profile) as dst:
        dst.write(species_map, 1)

    print("Puuliigikaart salvestatud:", SPECIES_TIF)

    # Confidence raster
    conf_profile = profile.copy()
    conf_profile.update(
        count=1,
        dtype="float32",
        nodata=NODATA_FLOAT,
        compress="lzw"
    )

    with rasterio.open(CONF_TIF, "w", **conf_profile) as dst:
        dst.write(conf_map, 1)

    print("Confidence kaart salvestatud:", CONF_TIF)

    print("\nValmis.")
    print("Ava QGIS-is:")
    print(" ", SPECIES_TIF)
    print(" ", CONF_TIF)


if __name__ == "__main__":
    main()