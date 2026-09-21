"""
VARUNA Satellite Data Services
===============================
Real INSAT-3D/3DR data integration via MOSDAC API.

Uses the official MOSDAC download API:
  POST https://mosdac.gov.in/download_api/gettoken
  GET  https://mosdac.gov.in/apios/datasets.json
  GET  https://mosdac.gov.in/download_api/download?id=...
"""

from app.services.satellite.mosdac_fetcher import MOSDACSatelliteFetcher, satellite_fetcher

__all__ = ["MOSDACSatelliteFetcher", "satellite_fetcher"]
