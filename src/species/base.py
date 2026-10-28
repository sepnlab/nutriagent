"""Species handler abstract base — centralizes species-specific metadata, column names, and field descriptions."""

from abc import ABC, abstractmethod


class SpeciesHandler(ABC):
    """Species strategy base class. To add a species, subclass this and register it in __init__.py."""

    @property
    @abstractmethod
    def key(self) -> str:
        """Species identifier, e.g. 'broiler' / 'layer'."""
        ...

    # ── Metadata ────────────────────────────────────────

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Display name, e.g. 'Broiler (meat-type chicken)'."""
        ...

    @property
    @abstractmethod
    def breeds(self) -> str:
        """Available breed list (used in the system prompt)."""
        ...

    @property
    @abstractmethod
    def phases(self) -> str:
        """Production phase description (used in the system prompt)."""
        ...

    @property
    @abstractmethod
    def metrics(self) -> str:
        """Key performance metrics (used in the system prompt)."""
        ...

    # ── Performance management tool ─────────────────────

    @property
    @abstractmethod
    def performance_breed_column(self) -> str:
        """Breed column name in the CSV."""
        ...

    @property
    @abstractmethod
    def performance_age_column(self) -> str:
        """Day/week age column name in the CSV."""
        ...

    @property
    @abstractmethod
    def performance_age_field(self) -> str:
        """Name of the age parameter field in the Pydantic schema."""
        ...

    @property
    @abstractmethod
    def performance_age_desc(self) -> str:
        """Description text for the age parameter."""
        ...

    @abstractmethod
    def performance_tool_description(self) -> str:
        """Full description text for the tool."""
        ...

    @property
    @abstractmethod
    def performance_params(self) -> dict:
        """{parameter name: description} mapping, used for the schema and output formatting."""
        ...

    @property
    @abstractmethod
    def processing_params(self) -> list:
        """List of processing-performance parameters (broilers only)."""
        ...

    # ── Nutrient requirements tool ──────────────────────

    @property
    @abstractmethod
    def nutrient_breed_column(self) -> str:
        """Breed column name in the CSV."""
        ...

    @property
    @abstractmethod
    def nutrient_breed_label(self) -> str:
        """Breed label, e.g. 'Broiler genetic lines'."""
        ...

    @property
    @abstractmethod
    def nutrient_available_breeds(self) -> str:
        """Available breeds, e.g. 'Ross 308, Ross 708, Cobb 500'."""
        ...

    @property
    @abstractmethod
    def nutrient_phase_help(self) -> str:
        """Help text for the phase parameter, including example phase names."""
        ...

    @abstractmethod
    def nutrient_tool_description(self) -> str:
        """Full description text for the tool."""
        ...
