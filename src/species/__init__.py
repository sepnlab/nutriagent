"""品种策略工厂。

用法：
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
    """获取品种处理器。

    Args:
        species: 品种标识符（'broiler' 或 'layer'）。
                 为 None 时从 Config 读取。

    Returns:
        SpeciesHandler 实例。
    """
    if species is None:
        from src.config import Config
        species = Config.SPECIES
    handler = _HANDLERS.get(species)
    if handler is None:
        raise ValueError(f"Unknown species: {species}. Available: {list(_HANDLERS)}")
    return handler
