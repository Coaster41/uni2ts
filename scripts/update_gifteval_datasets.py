from pathlib import Path
from uni2ts.common.env import env

storage = env.LOTSA_V1_PATH
available = {p.name for p in Path(storage).iterdir() if p.is_dir()}

# Check against each builder's dataset_list
from uni2ts.data.builder.lotsa_v1.gluonts import GluonTSDatasetBuilder
from uni2ts.data.builder.lotsa_v1.buildings_bench import Buildings900KDatasetBuilder
from uni2ts.data.builder.lotsa_v1.chronos_synth import ChronosSyntheticDatasetBuilder
from uni2ts.data.builder.lotsa_v1.cloudops_tsf import CloudOpsTSFDatasetBuilder
from uni2ts.data.builder.lotsa_v1.cmip6 import CMIP6DatasetBuilder
from uni2ts.data.builder.lotsa_v1.era5 import ERA5DatasetBuilder
from uni2ts.data.builder.lotsa_v1.largest import LargeSTDatasetBuilder
from uni2ts.data.builder.lotsa_v1.lib_city import LibCityDatasetBuilder
from uni2ts.data.builder.lotsa_v1.others import OthersLOTSADatasetBuilder
from uni2ts.data.builder.lotsa_v1.proenfo import ProEnFoDatasetBuilder


for builder_cls in [GluonTSDatasetBuilder,
                    Buildings900KDatasetBuilder,
                    ChronosSyntheticDatasetBuilder,
                    CloudOpsTSFDatasetBuilder,
                    CMIP6DatasetBuilder,
                    ERA5DatasetBuilder,
                    LargeSTDatasetBuilder,
                    LibCityDatasetBuilder,
                    OthersLOTSADatasetBuilder,
                    ProEnFoDatasetBuilder]:  
    expected = set(builder_cls.dataset_list)
    missing = expected - available
    present = expected & available
    print(f"\n{builder_cls.__name__}:")
    print(f"  Present: {sorted(present)}")
    print(f"  MISSING: {sorted(missing)}")