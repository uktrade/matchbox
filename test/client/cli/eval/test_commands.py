"""Tests for keyboard shortcut functionality."""

import polars as pl

from matchbox.client.cli.eval.ui import EvaluationState, GroupStyler
from matchbox.client.cli.eval.utils import create_evaluation_item
from matchbox.common.dtos import DataTypes
from matchbox.common.sources import (
    RelationalDBLocation,
    SourceConfig,
    SourceField,
)


class TestKeyboardShortcuts:
    """Test keyboard shortcut functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.state = EvaluationState()

    def test_single_letter_group(self):
        """Test single letter group selection."""
        self.state.set_group_selection("a")
        assert self.state.current_group_selection == "a"

        self.state.clear_group_selection()
        assert self.state.current_group_selection == ""

    def test_group_switching(self):
        """Test switching between different groups."""
        # Set group A
        self.state.set_group_selection("a")
        assert self.state.current_group_selection == "a"

        # Switch to group S
        self.state.set_group_selection("s")
        assert self.state.current_group_selection == "s"

        # Switch to group D
        self.state.set_group_selection("d")
        assert self.state.current_group_selection == "d"

        # Clear group
        self.state.clear_group_selection()
        assert self.state.current_group_selection == ""

    def test_set_group_overrides(self):
        """Test that setting a group overrides the previous group."""
        self.state.set_group_selection("a")
        assert self.state.current_group_selection == "a"

        # Setting a new group replaces the old one
        self.state.set_group_selection("b")
        assert self.state.current_group_selection == "b"

        # Setting same group again should still work
        self.state.set_group_selection("b")
        assert self.state.current_group_selection == "b"

    def test_case_insensitive_keys(self):
        """Test that uppercase keys are converted to lowercase."""
        self.state.set_group_selection("A")
        assert self.state.current_group_selection == "a"

        self.state.set_group_selection("S")
        assert self.state.current_group_selection == "s"

    def test_number_key_parsing(self):
        """Test number key to column number conversion."""
        assert self.state.parse_number_key("1") == 1
        assert self.state.parse_number_key("5") == 5
        assert self.state.parse_number_key("9") == 9
        assert self.state.parse_number_key("0") == 10  # 0 maps to column 10
        assert self.state.parse_number_key("a") is None  # Non-number returns None

    def test_various_single_letter_groups(self):
        """Test various single letter groups work."""
        # Test different letters
        test_letters = ["q", "w", "e", "r", "t", "y", "a", "s", "d", "f"]

        for letter in test_letters:
            self.state.set_group_selection(letter)
            assert self.state.current_group_selection == letter

    def test_non_alphabetic_keys_ignored(self):
        """Test that non-alphabetic keys are ignored."""
        # These should not change the current group selection
        original_group = self.state.current_group_selection
        self.state.set_group_selection("1")  # Number
        self.state.set_group_selection("!")  # Special character
        self.state.set_group_selection("space")  # Word key
        assert self.state.current_group_selection == original_group


class TestGroupStyler:
    """Test the GroupStyler class for consistent colour/symbol assignment."""

    def setup_method(self):
        """Reset GroupStyler state before each test."""
        GroupStyler.reset()

    def test_style_consistency(self):
        """Test that same group name always gets same colour/symbol."""
        test_groups = ["qwerty", "asdf", "xyz", "hello", "world"]

        for group in test_groups:
            style1 = GroupStyler.get_style(group)
            style2 = GroupStyler.get_style(group)
            assert style1 == style2, f"Inconsistent style for group: {group}"

    def test_different_groups_different_styles(self):
        """Test that different group names get different styles (usually)."""
        # While hash collisions are possible, these specific groups should differ
        style_a = GroupStyler.get_style("a")
        style_qwerty = GroupStyler.get_style("qwerty")

        # At least one component (colour or symbol) should be different
        assert style_a != style_qwerty

    def test_display_text_format(self):
        """Test display text formatting."""
        text, colour = GroupStyler.get_display_text("qwerty", 5)

        assert "QWERTY" in text
        assert "(5)" in text
        assert colour in GroupStyler.COLOURS

    def test_colour_symbol_within_bounds(self):
        """Test that generated colours and symbols are within defined ranges."""
        for group in ["a", "qwerty", "xyz", "hello", "test123"]:
            colour, symbol = GroupStyler.get_style(group)

            assert colour in GroupStyler.COLOURS
            assert symbol in GroupStyler.SYMBOLS

    def test_colour_cycling_minimises_duplicates(self):
        """Test that colours cycle through all available options before repeating."""
        # Get styles for groups equal to the number of available colours
        num_colours = len(GroupStyler.COLOURS)
        groups = [f"group_{i}" for i in range(num_colours)]

        styles = [GroupStyler.get_style(group) for group in groups]
        colours = [style[0] for style in styles]
        symbols = [style[1] for style in styles]

        # All colours should be unique (no duplicates)
        assert len(set(colours)) == num_colours, (
            f"Expected {num_colours} unique colours, got {len(set(colours))}"
        )

        # All symbols should be unique (no duplicates) up to available symbols
        num_symbols = len(GroupStyler.SYMBOLS)
        expected_unique_symbols = min(num_colours, num_symbols)
        assert len(set(symbols)) == expected_unique_symbols

        # When we exceed available colours, then we should get repeats
        extra_group = GroupStyler.get_style("extra_group")
        extra_colour = extra_group[0]
        # This colour should now be a repeat
        assert extra_colour in colours

    def test_assignment_persistence(self):
        """Test that once assigned, groups keep their colours consistently."""
        # Assign some groups
        style_a = GroupStyler.get_style("a")
        style_b = GroupStyler.get_style("b")
        style_c = GroupStyler.get_style("c")

        # Getting them again should return the same styles
        assert GroupStyler.get_style("a") == style_a
        assert GroupStyler.get_style("b") == style_b
        assert GroupStyler.get_style("c") == style_c

        # Even if we request them in different order
        assert GroupStyler.get_style("c") == style_c
        assert GroupStyler.get_style("a") == style_a
        assert GroupStyler.get_style("b") == style_b


class TestFieldProcessing:
    """Test the SourceConfig-based field processing functionality."""

    def test_create_processed_comparison_data(self):
        """Test that field processing works with SourceConfig data."""

        # Create mock SourceConfig objects
        location = RelationalDBLocation(name="test_db")

        source_a = SourceConfig(
            location=location,
            name="source_a",
            extract_transform="SELECT * FROM table_a",
            key_field=SourceField(name="id", type=DataTypes.STRING),
            index_fields=(
                SourceField(name="company_name", type=DataTypes.STRING),
                SourceField(name="registration_id", type=DataTypes.STRING),
            ),
        )

        source_b = SourceConfig(
            location=location,
            name="source_b",
            extract_transform="SELECT * FROM table_b",
            key_field=SourceField(name="id", type=DataTypes.STRING),
            index_fields=(
                SourceField(name="company_name", type=DataTypes.STRING),
                SourceField(name="address", type=DataTypes.STRING),
            ),
        )

        # Create test DataFrame with qualified field names
        df = pl.DataFrame(
            {
                "leaf": [1, 2, 3],
                "source_a_company_name": ["Company A", "Company B", "Company C"],
                "source_a_registration_id": ["REG001", "REG002", "REG003"],
                "source_b_company_name": [
                    "Company A Ltd",
                    "Company B Inc",
                    "Company C Corp",
                ],
                "source_b_address": ["123 Main St", "456 Oak Ave", "789 Pine Rd"],
            }
        )

        # Create evaluation item with new paradigm
        evaluation_item = create_evaluation_item(df, [source_a, source_b], 123)

        # Verify structure
        assert isinstance(evaluation_item.display_dataframe, pl.DataFrame)
        assert isinstance(evaluation_item.duplicate_groups, list)
        assert isinstance(evaluation_item.display_columns, list)
        assert isinstance(evaluation_item.leaf_to_display_mapping, dict)
        assert evaluation_item.cluster_id == 123

        # Check expected number of display columns (should be 3 - no duplicates)
        assert len(evaluation_item.display_columns) == 3
        assert len(evaluation_item.duplicate_groups) == 3

        # Check that all leaf IDs are accounted for in duplicate groups
        all_leaf_ids = []
        for group in evaluation_item.duplicate_groups:
            all_leaf_ids.extend(group)
        assert sorted(all_leaf_ids) == [1, 2, 3]

        # Check display dataframe has expected fields
        field_names = evaluation_item.display_dataframe["field_name"].unique().to_list()
        assert "company_name" in field_names
        assert "registration_id" in field_names or "address" in field_names
