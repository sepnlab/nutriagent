"""Species handler abstract base — 集中管理品种相关的元数据、列名、字段描述。"""

from abc import ABC, abstractmethod


class SpeciesHandler(ABC):
    """品种策略基类。新增品种时继承此类并注册到 __init__.py 即可。"""

    @property
    @abstractmethod
    def key(self) -> str:
        """品种标识符，如 'broiler' / 'layer'。"""
        ...

    # ── 元数据 ──────────────────────────────────────────

    @property
    @abstractmethod
    def display_name(self) -> str:
        """显示名称，如 'Broiler (meat-type chicken)'。"""
        ...

    @property
    @abstractmethod
    def breeds(self) -> str:
        """可用品种列表（供 system prompt 使用）。"""
        ...

    @property
    @abstractmethod
    def phases(self) -> str:
        """生产阶段描述（供 system prompt 使用）。"""
        ...

    @property
    @abstractmethod
    def metrics(self) -> str:
        """关键性能指标（供 system prompt 使用）。"""
        ...

    # ── 性能管理工具 ──────────────────────────────────────

    @property
    @abstractmethod
    def performance_breed_column(self) -> str:
        """CSV 中的品种列名。"""
        ...

    @property
    @abstractmethod
    def performance_age_column(self) -> str:
        """CSV 中的日龄/周龄列名。"""
        ...

    @property
    @abstractmethod
    def performance_age_field(self) -> str:
        """Pydantic schema 中的 age 参数字段名。"""
        ...

    @property
    @abstractmethod
    def performance_age_desc(self) -> str:
        """age 参数的描述文本。"""
        ...

    @abstractmethod
    def performance_tool_description(self) -> str:
        """工具的完整 description 文本。"""
        ...

    @property
    @abstractmethod
    def performance_params(self) -> dict:
        """{参数名: 描述} 映射，用于 schema 和输出格式化。"""
        ...

    @property
    @abstractmethod
    def processing_params(self) -> list:
        """加工性能参数列表（仅肉鸡有）。"""
        ...

    # ── 营养需要量工具 ────────────────────────────────────

    @property
    @abstractmethod
    def nutrient_breed_column(self) -> str:
        """CSV 中的品种列名。"""
        ...

    @property
    @abstractmethod
    def nutrient_breed_label(self) -> str:
        """品种标签，如 'Broiler genetic lines'。"""
        ...

    @property
    @abstractmethod
    def nutrient_available_breeds(self) -> str:
        """可用品种，如 'Ross 308, Ross 708, Cobb 500'。"""
        ...

    @property
    @abstractmethod
    def nutrient_phase_help(self) -> str:
        """阶段参数的 help text，含示例阶段名。"""
        ...

    @abstractmethod
    def nutrient_tool_description(self) -> str:
        """工具的完整 description 文本。"""
        ...
