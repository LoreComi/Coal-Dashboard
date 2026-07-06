# Databricks notebook source
# DBTITLE 1,Sandbox Table Refresh
# MAGIC %md
# MAGIC ## Refresh Sandbox Tables for Weather Coal Desk App
# MAGIC
# MAGIC Refreshes the 4 dynamic tables in `dna_snbx_weather.coal_desk` that mirror production data.
# MAGIC - `temperature_actuals` — Current-year ERA5 actuals
# MAGIC - `temperature_forecast` — ECMWF-ENS temperature forecast
# MAGIC - `precipitation_actuals` — Current-year precip actuals
# MAGIC - `precipitation_forecast` — ECMWF-ENS precip forecast
# MAGIC
# MAGIC Climatology tables are static (ERA5 2000-2024) and do NOT need refresh.

# COMMAND ----------

# DBTITLE 1,Define spatial filters
# Spatial filters matching the app's bounding boxes + city coordinates
MAP_BOXES = """
    (latitude BETWEEN 20 AND 46.5 AND longitude BETWEEN 90 AND 146.5)
    OR (latitude BETWEEN 36 AND 72.5 AND longitude BETWEEN -13 AND 36.5)
    OR (latitude BETWEEN 28.5 AND 55.5 AND longitude BETWEEN -130 AND -70)
"""

CITY_COORDS = """
    OR (latitude = 29.0 AND longitude = 77.0)
    OR (latitude = 22.5 AND longitude = 70.0)
    OR (latitude = 22.0 AND longitude = 88.0)
    OR (latitude = 33.5 AND longitude = -7.5)
    OR (latitude = -25.5 AND longitude = -49.0)
    OR (latitude = -8.5 AND longitude = -35.0)
    OR (latitude = -24.0 AND longitude = -46.0)
    OR (latitude = 9.0 AND longitude = -79.5)
"""

# Three Gorges watershed (for precipitation)
WATERSHED_BOX = "OR (latitude BETWEEN 24.5 AND 35.9 AND longitude BETWEEN 90.5 AND 111.2)"

TEMP_FILTER = f"({MAP_BOXES}{CITY_COORDS})"
PRECIP_FILTER = f"({MAP_BOXES}{WATERSHED_BOX})"

# COMMAND ----------

# DBTITLE 1,Refresh temperature_actuals
spark.sql(f"""
CREATE OR REPLACE TABLE dna_snbx_weather.coal_desk.temperature_actuals AS
SELECT delivery_start, value, latitude, longitude
FROM dna_prod_silver.meteomatics.temperature
WHERE model = 'ecmwf-era5'
  AND curve_name = 't_mean_2m_24h_c_ecmwf_era5_p1d'
  AND YEAR(delivery_start) >= 2025
  AND {TEMP_FILTER}
""")
print("✓ temperature_actuals refreshed")

# COMMAND ----------

# DBTITLE 1,Refresh temperature_forecast
spark.sql(f"""
CREATE OR REPLACE TABLE dna_snbx_weather.coal_desk.temperature_forecast AS
SELECT delivery_start, value, latitude, longitude
FROM dna_prod_silver.meteomatics.temperature_forecast
WHERE model = 'ecmwf-ens'
  AND curve_name = 't_mean_2m_24h_c_ecmwf_ens_p1d'
  AND {TEMP_FILTER}
""")
print("✓ temperature_forecast refreshed")

# COMMAND ----------

# DBTITLE 1,Refresh precipitation_actuals
spark.sql(f"""
CREATE OR REPLACE TABLE dna_snbx_weather.coal_desk.precipitation_actuals AS
SELECT delivery_start, value, latitude, longitude, model, curve_name
FROM dna_prod_silver.meteomatics.precipitation
WHERE model = 'ecmwf-era5'
  AND curve_name = 'precip_24h_mm_ecmwf_era5_p1d'
  AND YEAR(delivery_start) >= 2025
  AND {PRECIP_FILTER}
""")
print("✓ precipitation_actuals refreshed")

# COMMAND ----------

# DBTITLE 1,Refresh precipitation_forecast
spark.sql(f"""
CREATE OR REPLACE TABLE dna_snbx_weather.coal_desk.precipitation_forecast AS
SELECT delivery_start, value, latitude, longitude
FROM dna_prod_silver.meteomatics.precipitation_forecast
WHERE model = 'ecmwf-ens'
  AND curve_name = 'precip_24h_mm_ecmwf_ens_p1d'
  AND CAST(delivery_start AS DATE) >= CURRENT_DATE() - INTERVAL 7 DAYS
  AND {PRECIP_FILTER}
""")
print("✓ precipitation_forecast refreshed")

# COMMAND ----------


