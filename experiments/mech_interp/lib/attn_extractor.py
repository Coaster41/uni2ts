import torch

from uni2ts.model.moiraic.module import MoiraicModule


class AttentionExtractor:
    def __init__(self, module):
        self.module = module

    def run(self, batch) -> dict[int, torch.Tensor]:
        kwargs = dict(**batch, training_mode=False, return_attn_weights=True)
        if isinstance(self.module, MoiraicModule):
            kwargs.update(past_cache=None, return_cache=False)
        with torch.no_grad():
            _, all_attn_weights = self.module(**kwargs)
        return {i: w.detach().cpu() for i, w in enumerate(all_attn_weights)}
