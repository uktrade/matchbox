"""Styling utilities for entity resolution evaluation UI."""

GROUP_STYLES = {
    # Row: q w e r t y u i o p
    "q": ("■", "#e53935"),  # red
    "w": ("●", "#43a047"),  # green
    "e": ("▲", "#1e88e5"),  # blue
    "r": ("◆", "#fdd835"),  # yellow
    "t": ("★", "#d81b60"),  # magenta
    "y": ("⬢", "#00acc1"),  # cyan
    "u": ("♦", "#fb8c00"),  # orange
    "i": ("▼", "#00897b"),  # teal
    "o": ("○", "#8e24aa"),  # purple
    "p": ("△", "#6d4c41"),  # brown
    # Row: a s d f g h j k l
    "a": ("◇", "#d81b60"),  # magenta
    "s": ("☆", "#00acc1"),  # cyan
    "d": ("⬡", "#e53935"),  # red
    "f": ("✦", "#43a047"),  # green
    "g": ("✧", "#1e88e5"),  # blue
    "h": ("⟐", "#fdd835"),  # yellow
    "j": ("✚", "#d81b60"),  # magenta
    "k": ("✖", "#3949ab"),  # indigo (keeps distance from teal above-left)
    "l": ("□", "#e53935"),  # red
    # Row: z x c v b n m
    "z": ("▽", "#fb8c00"),  # orange
    "x": ("◯", "#8e24aa"),  # purple (far from cyan/red above)
    "c": ("◻", "#fdd835"),  # yellow
    "v": ("◼", "#6d4c41"),  # brown
    "b": ("⬤", "#e53935"),  # red
    "n": ("☑", "#43a047"),  # green
    "m": ("✤", "#1e88e5"),  # blue
}


def get_group_style(group: str) -> tuple[str, str]:
    """Get symbol and colour for a group letter.

    Args:
        group: Single letter group identifier (a-z)

    Returns:
        Tuple of (symbol, colour)
    """
    return GROUP_STYLES.get(group.lower(), ("■", "white"))


def get_display_text(group: str, count: int) -> tuple[str, str]:
    """Get formatted display text with colour and symbol.

    Args:
        group: Single letter group identifier (a-z) or "unassigned"
        count: Number of items in this group

    Returns:
        Tuple of (formatted_text, colour)
    """
    if group == "unassigned":
        return f"UNASSIGNED ({count})", "dim"

    symbol, colour = get_group_style(group)
    text = f"{symbol} {group.upper()} ({count})"
    return text, colour


def generate_css_classes() -> str:
    """Generate CSS classes for all groups.

    Returns:
        CSS string with all group styling classes
    """
    lines = []
    for group, (_symbol, colour) in GROUP_STYLES.items():
        lines.append(f"ComparisonTable .group-{group} {{ background: {colour}; }}")
    return "\n".join(lines)
