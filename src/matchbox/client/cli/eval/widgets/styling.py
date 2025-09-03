"""Styling utilities for entity resolution evaluation UI."""


class GroupStyler:
    """Generate consistent colours and symbols with cycling to minimise duplicates."""

    # High contrast colours distributed to avoid similar adjacents
    COLOURS = [
        "red",
        "blue",
        "green",
        "yellow",
        "magenta",
        "cyan",
        "bright_red",
        "bright_green",
        "bright_blue",
        "bright_yellow",
        "bright_magenta",
        "bright_cyan",
        "white",
        "bright_white",
    ]

    # Distinct Unicode symbols for visual differentiation
    SYMBOLS = [
        "■",
        "●",
        "▲",
        "◆",
        "★",
        "⬢",
        "♦",
        "▼",
        "○",
        "△",
        "◇",
        "☆",
        "⬡",
        "✦",
        "✧",
        "⟐",
    ]

    # Class-level tracking for consistent assignments
    _group_styles = {}  # group_name -> (colour, symbol)
    _used_colours = set()
    _used_symbols = set()
    _colour_index = 0
    _symbol_index = 0

    @classmethod
    def get_style(cls, group_name: str) -> tuple[str, str]:
        """Get consistent colour and symbol for a group name using cycling."""
        # Return cached style if already assigned
        if group_name in cls._group_styles:
            return cls._group_styles[group_name]

        # Assign next available colour
        colour = cls._get_next_colour()
        symbol = cls._get_next_symbol()

        # Cache the assignment
        cls._group_styles[group_name] = (colour, symbol)
        return colour, symbol

    @classmethod
    def _get_next_colour(cls) -> str:
        """Get the next colour in cycle, avoiding duplicates when possible."""
        # If we haven't used all colours yet, find an unused one
        if len(cls._used_colours) < len(cls.COLOURS):
            while cls.COLOURS[cls._colour_index] in cls._used_colours:
                cls._colour_index = (cls._colour_index + 1) % len(cls.COLOURS)

        colour = cls.COLOURS[cls._colour_index]
        cls._used_colours.add(colour)
        cls._colour_index = (cls._colour_index + 1) % len(cls.COLOURS)

        return colour

    @classmethod
    def _get_next_symbol(cls) -> str:
        """Get the next symbol in cycle, avoiding duplicates when possible."""
        # If we haven't used all symbols yet, find an unused one
        if len(cls._used_symbols) < len(cls.SYMBOLS):
            while cls.SYMBOLS[cls._symbol_index] in cls._used_symbols:
                cls._symbol_index = (cls._symbol_index + 1) % len(cls.SYMBOLS)

        symbol = cls.SYMBOLS[cls._symbol_index]
        cls._used_symbols.add(symbol)
        cls._symbol_index = (cls._symbol_index + 1) % len(cls.SYMBOLS)

        return symbol

    @classmethod
    def get_display_text(cls, group_name: str, count: int) -> tuple[str, str]:
        """Get formatted display text with colour and symbol."""
        colour, symbol = cls.get_style(group_name)
        text = f"{symbol} {group_name.upper()} ({count})"
        return text, colour

    @classmethod
    def reset(cls):
        """Reset all assignments (useful for testing)."""
        cls._group_styles.clear()
        cls._used_colours.clear()
        cls._used_symbols.clear()
        cls._colour_index = 0
        cls._symbol_index = 0
