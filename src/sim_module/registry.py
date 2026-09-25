"""Registry and factory for perturbations."""

from typing import Callable, Dict, List, Type

PERTURBATION_FACTORY: Dict[str, type] = {}


def register_perturbation(name: str) -> Callable[[type], type]:
    """Class decorator that registers a perturbation under the catalog's `perturbation.type` name."""

    def decorator(cls: type) -> type:
        if name in PERTURBATION_FACTORY:
            raise ValueError(f"perturbation {name!r} is already registered")
        cls.name = name  # type: ignore[attr-defined]
        PERTURBATION_FACTORY[name] = cls
        return cls

    return decorator


def PerturbationFactory(name: str) -> Type:  # noqa: N802 - factory name follows the project's registry convention
    """Return the perturbation class for a catalog type.

    Unlike a dataset factory, an unknown name raises: a silent default would let a scenario run unperturbed.
    """
    try:
        return PERTURBATION_FACTORY[name]
    except KeyError:
        raise KeyError(f"no perturbation registered for {name!r}; registered: {sorted(PERTURBATION_FACTORY)}") from None


def registered_perturbations() -> List[str]:
    """Names of all registered perturbations."""
    return sorted(PERTURBATION_FACTORY)
