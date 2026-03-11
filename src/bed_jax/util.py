"""Shared utility functions for the JAX backend."""

try:
    import jax
except Exception as exc:  # pragma: no cover - import-time dependency guard
    raise ImportError(
        "bed_jax requires JAX. Install with `pip install -e '.[jax-cpu]'` "
        "or `pip install -e '.[jax]'."
    ) from exc


def resolve_device(device):
    """Return a concrete JAX device from None/string/device input."""
    err = 'device must be None, "cpu", "gpu", or a jax.Device instance'

    if device is None:
        for platform in ("cpu", "gpu", None):
            try:
                devices = (
                    jax.devices(platform)
                    if platform is not None
                    else jax.devices()
                )
            except RuntimeError:
                devices = []
            if devices:
                return devices[0]
        raise RuntimeError("No JAX devices are available.")

    if hasattr(device, "platform"):
        return device

    if not isinstance(device, str):
        raise ValueError(err)

    requested = device.strip().lower()
    if requested not in ("cpu", "gpu"):
        raise ValueError(err)

    try:
        devices = jax.devices(requested)
    except RuntimeError:
        devices = []
    if not devices:
        if requested == "gpu":
            raise RuntimeError(
                "GPU device requested but no JAX GPU backend is available."
            )
        raise RuntimeError(
            "CPU device requested but no JAX CPU backend is available."
        )

    return devices[0]
