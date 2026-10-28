"""Species strategy factory.

Usage:
    from src.species import get_handler
    handler = get_handler("broiler")
    print(handler.display_name)
"""

from src.species.base import SpeciesHandler
from src.species.broiler import BroilerHandler
from src.species.layer import LayerHandler

_HANDLERS = {
    "broiler": BroilerHandler(),
    "layer": LayerHandler(),
}


def get_handler(species: str = None) -> SpeciesHandler:
    """Get the species handler.

    Args:
        species: species identifier ('broiler' or 'layer').
                 Read from Config when None.

    Returns:
        A SpeciesHandler instance.
    """
    if species is None:
        from src.config import Config
        species = Config.SPECIES
    handler = _HANDLERS.get(species)
    if handler is None:
        raise ValueError(f"Unknown species: {species}. Available: {list(_HANDLERS)}")
    return handler
