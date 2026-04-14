# scripts/build_chronos_synth.py
from uni2ts.common.env import env
from uni2ts.data.builder.lotsa_v1.chronos_synth import ChronosSyntheticDatasetBuilder

LOCAL_DOWNLOAD_DIR = "/srv/disk00/ctadler/uni2ts/datasets"  # where huggingface-cli downloaded to

builder = ChronosSyntheticDatasetBuilder(
    datasets=["chronos_tsmixup_10m", "chronos_kernel_synth_1m"],
    storage_path=env.GIFT_EVAL_PRETRAIN_PATH,
)

for ds_name in builder.datasets:
    builder.build_dataset(
        ds_name,
        source_path=LOCAL_DOWNLOAD_DIR,  # reads local parquet, no re-download
        num_proc=8,
    )