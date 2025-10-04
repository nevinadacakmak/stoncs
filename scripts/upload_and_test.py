"""Upload synthetic demo data to Snowflake and run a quick recommendation check.

This script:
- Loads .env into environment
- Imports local `ingest` module by path and calls upload_demo_to_snowflake(use_live=False)
- Imports `optimizer` and runs compute_asset_metrics -> combine_with_narratives -> cluster_risk_levels -> recommend_portfolio
- Prints a concise summary of rows inserted and sample recommendations
"""
from pathlib import Path
import importlib.util
import os
ROOT = Path(__file__).resolve().parent.parent
# Load .env
dotenv = ROOT / '.env'
if dotenv.exists():
    for line in dotenv.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        k = k.strip(); v = v.strip()
        if k and k not in os.environ:
            os.environ[k] = v

# Ensure snowflake_api_client is importable by other modules (import by path and insert into sys.modules)
spec_sf = importlib.util.spec_from_file_location('snowflake_api_client', str(ROOT / 'snowflake_api_client.py'))
sf = importlib.util.module_from_spec(spec_sf)
spec_sf.loader.exec_module(sf)
import sys
sys.modules['snowflake_api_client'] = sf
sys.modules['stoncs.snowflake_api_client'] = sf

# Import ingest by path after client is available so its module-level bindings are correct
spec = importlib.util.spec_from_file_location('ingest', str(ROOT / 'ingest.py'))
ingest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingest)

print('Uploading demo data to Snowflake (this may take a moment)')
try:
    ingest.upload_demo_to_snowflake(use_live=False)
    print('Upload complete')
except Exception as e:
    print('Upload failed:', e)
    raise

# Run optimizer pipeline
spec_opt = importlib.util.spec_from_file_location('optimizer', str(ROOT / 'optimizer.py'))
opt = importlib.util.module_from_spec(spec_opt)
spec_opt.loader.exec_module(opt)

print('Computing metrics from Snowflake...')
df = opt.compute_asset_metrics()
print('metrics rows:', len(df))
if len(df) > 0:
    df = opt.cluster_risk_levels(df)
    df = opt.combine_with_narratives(df)
    rec = opt.recommend_portfolio(10000, 0.5, df)
    print('recommendations rows:', len(rec))
    print(rec.head().to_dict())
else:
    print('No market rows found after upload - something went wrong')
