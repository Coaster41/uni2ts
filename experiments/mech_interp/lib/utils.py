import torch


def _load_module(ckpt_path: str | None, model_name: str, device: str | torch.device = "cpu"):
    """Load a model module from a checkpoint path, or build a tiny model if None."""
    from uni2ts.model.moiraic.module import MoiraicModule
    from uni2ts.model.moiraie.module import MoiraieModule
    from experiments.mech_interp.block1_probing.train_probes import PATCH_SIZE, PRED_PATCHES

    if ckpt_path is None:
        print(f"  No checkpoint for {model_name} — using tiny in-memory model.")
        tiny = dict(d_model=64, d_ff=128, num_layers=2, patch_size=PATCH_SIZE,
                    max_seq_len=64, attn_dropout_p=0.0, dropout_p=0.0)
        if model_name == "moiraie":
            module = MoiraieModule(**tiny, num_predict_token=1)
        else:
            module = MoiraicModule(**tiny, num_predict_token=PRED_PATCHES)
        return module.eval().to(device)

    print(f"  Loading {model_name} from {ckpt_path} (device={device})")
    if model_name == "moiraie":
        module = MoiraieModule.from_pretrained(ckpt_path)
    else:
        module = MoiraicModule.from_pretrained(ckpt_path)

    return module.eval().to(device)