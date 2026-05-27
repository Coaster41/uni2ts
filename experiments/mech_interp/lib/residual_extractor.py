import torch


class ResidualExtractor:
    """Context manager that captures per-layer post-FFN residual stream activations."""

    def __init__(self, module):
        self._module = module
        self._handles = []
        self._activations = {}
        self._active = False

    def __enter__(self):
        self._activations = {}
        self._handles = []
        for i, layer in enumerate(self._module.encoder.layers):
            def hook(mod, inp, out, idx=i):
                x = out if isinstance(out, torch.Tensor) else out[0]
                self._activations[idx] = x.detach().cpu()
            self._handles.append(layer.register_forward_hook(hook))
        # Capture post-projection, pre-attention activation under key -1
        def in_proj_hook(mod, inp, out, _idx=-1):
            x = out if isinstance(out, torch.Tensor) else out[0]
            self._activations[-1] = x.detach().cpu()
        self._handles.append(self._module.in_proj.register_forward_hook(in_proj_hook))
        self._active = True
        return self

    def __exit__(self, *_):
        for h in self._handles:
            h.remove()
        self._handles = []
        self._active = False

    def run(self, batch) -> dict:
        if not self._active:
            raise RuntimeError("ResidualExtractor must be used as a context manager")
        self._activations = {}
        is_moiraic = type(self._module).__name__.startswith("Moiraic")
        with torch.no_grad():
            if is_moiraic:
                self._module(**batch, training_mode=False, past_cache=None, return_cache=False)
            else:
                self._module(**batch, training_mode=False)
        return dict(self._activations)
