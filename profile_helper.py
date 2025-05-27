import torch, torch.nn.functional as F, torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

def profile_call(fn, data, name="model_inference", n_iters=10,
                 sort_by="cpu_time_total", row_limit=20, **kwargs):
    """
    Profile an arbitrary callable with torch.profiler.

    Parameters
    ----------
    fn : callable
        The function / nn.Module / lambda you want to time.
    *args :
        Positional inputs to `fn`.
    device : str | torch.device
        'cpu' (default) or 'cuda'.
    n_iters : int
        How many forward passes to profile.
    use_cuda : bool | None
        If None → auto-detect CUDA and device=='cuda'.
    sort_by : str
        Column name for table sorting (`cpu_time_total`, `self_cuda_time_total`, ...).
    row_limit : int
        Max rows to display.
    **kwargs :
        Keyword arguments forwarded to `fn`.

    Returns
    -------
    result :
        Output of `fn(*args, **kwargs)` from the last iteration.
    prof : torch.profiler.profile
        The profiler object for deeper inspection.
    """

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function(name):
            for _ in range(n_iters):
                outputs = fn(data)
    print(prof.key_averages(group_by_input_shape=True).table(sort_by=sort_by, row_limit=row_limit))
if __name__ == "__main__":
    # CustomMish = torch.compile(CustomMish, mode='reduce-overhead', backend="aot_eager")  # Compile the model for better performance
    class CustomMish(nn.Module):
        """Same math, JIT-compiled (lets torch.compile fuse)"""
        def forward(self, x):
            return x * torch.tanh(F.softplus(x))
    x = torch.randn(1_000_000_000)  # sample workload
    CustomMishCompiled = CustomMish().to("mps")  
    CustomMishCompiled = torch.compile(CustomMishCompiled, mode="reduce-overhead", backend="aot_eager")  # Compile the model for better performance
    CustomMish = CustomMish().to("mps")  
    profile_call(
        CustomMishCompiled,
        x,
        name="CustomMishCompiled",
        n_iters=10,
        sort_by="cpu_time_total",
        row_limit=10
    )
    profile_call(
        CustomMish,
        x,
        name="CustomMish",
        n_iters=10,
        sort_by="cpu_time_total",
        row_limit=10
    )
# -------------------------------------------------------------------------
# EXAMPLE USAGE
# -------------------------------------------------------------------------
