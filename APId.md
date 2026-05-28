# APIs Used in This Project

## 1. Sentinel Hub Statistical API
Used for downloading Sentinel-2 multi-temporal statistics for forest areas and pixels.

### Example Python Code

```python
from sentinelhub import (
    SHConfig,
    SentinelHubStatistical,
    DataCollection,
    Geometry,
    CRS
)

config = SHConfig()
config.sh_client_id = "YOUR_CLIENT_ID"
config.sh_client_secret = "YOUR_CLIENT_SECRET"

request = SentinelHubStatistical(
    aggregation=SentinelHubStatistical.aggregation(
        evalscript=EVALSCRIPT,
        time_interval=("2025-01-01", "2025-12-31"),
        aggregation_interval="P10D",
        resolution=(10, 10)
    ),
    input_data=[
        SentinelHubStatistical.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            maxcc=0.3
        )
    ],
    geometry=Geometry(geometry, CRS.WGS84),
    config=config
)

data = request.get_data()
```

---

# 2. Copernicus Data Space Ecosystem (CDSE)

Platform used for accessing Sentinel-2 satellite imagery and cloud services.

## Authentication Example

```python
import requests

token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

data = {
    "grant_type": "client_credentials",
    "client_id": "YOUR_CLIENT_ID",
    "client_secret": "YOUR_CLIENT_SECRET"
}

response = requests.post(token_url, data=data)

print(response.json())
```

---

# 3. OData API

Used for searching and accessing Copernicus satellite products.

## Example Query

```python
import requests

url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

response = requests.get(url)

print(response.json())
```

---

# 4. RESTO Search API

Used for searching satellite imagery metadata.

## Example Query

```python
import requests

url = "https://catalogue.dataspace.copernicus.eu/resto/api/collections/Sentinel2/search.json"

params = {
    "startDate": "2025-01-01",
    "completionDate": "2025-12-31",
    "maxRecords": 5
}

response = requests.get(url, params=params)

print(response.json())
```

---

# Documentation Links

- Copernicus Data Space Documentation
- Copernicus OData API Documentation
- Sentinel Hub Documentation
- RESTO Search API Documentation
