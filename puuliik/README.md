# Metsamask 

Google Draivi link kaartidega: https://drive.google.com/drive/folders/1yg0egGQGLHNd248swkJUmDQV9AeU7zUI?usp=drive_link

# Google Drive Data Files

## 1. H_20_21_22_23.zip

### Contents

This archive contains forest height raster data used in the machine learning workflow.


### Potential Applications

* Elevation-based forest analysis
* Terrain influence on tree species distribution
* Machine learning auxiliary features

---

# 2. m_valjalõige.zip

### Contents

A clipped subset extracted from the forest growing stock volume map (“tagavara kaart”).

### Data Included

* Growing stock raster subset
* Forest volume information
* Local test-area extraction

### Purpose

Used for:

* Small-area testing
* Visualization
* Combining forest volume information with tree species analysis

### Potential Applications

* Forest inventory analysis
* Biomass estimation
* Forest structure assessment

---

# 3. metsamask.zip

### Contents

A raster-based forest mask representing forest land extent and boundaries.

### Data Included

* Forest / non-forest classification raster
* Forest boundary mask

### Purpose

Used to:

* Remove non-forest pixels before prediction
* Restrict classification only to forested areas
* Improve pixel-based species mapping quality

### Potential Applications

* Forest masking
* Land cover preprocessing
* Forest-only AI predictions

---

# 4. puuliik_filt_pindala.zip

### Contents

Training dataset containing RMK pure forest stand polygons with dominant tree species labels.

### Data Included

* Forest stand polygons
* Pure stand training areas
* Tree species labels
* Spatial training data

### Purpose

Used as supervised machine learning training data for tree species classification.

### Main Tree Species

* Birch
* Norway spruce
* Scots pine
* Grey alder
* Aspen
* Black alder

### Potential Applications

* Tree species classification
* Remote sensing model training
* Forest inventory mapping
* AI-based forestry workflows

---

# Overall Workflow

The datasets together create a complete forestry AI and remote sensing workflow:

1. RMK pure stand polygons used as training data
2. Sentinel-2 time series extraction
3. Vegetation index calculation
4. Elevation and forest mask integration
5. Random Forest model training
6. Pixel-based forest species prediction
7. Tree species map generation

---

# Technologies Used

* Python
* Random Forest (Scikit-learn)
* Sentinel Hub Statistical API
* Copernicus Data Space Ecosystem (CDSE)
* GeoPandas
* Rasterio
* QGIS
* Sentinel-2 imagery
* Digital Elevation Models (DEM)


