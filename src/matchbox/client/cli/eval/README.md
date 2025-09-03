# Entity resolution evaluation tool

## Overview

The entity resolution evaluation tool is a terminal-based data labelling application built with [Textual](https://textual.textualize.io/). It provides an interactive interface for human evaluation of entity resolution results, allowing users to group records that represent the same real-world entity.

This tool is part of the broader Matchbox ecosystem, enabling collaborative, measurable, and iterative entity matching workflows. The evaluation interface bridges the gap between automated matching algorithms and human judgement, providing ground truth data for model improvement and performance measurement.

### Key capabilities

- **Interactive record grouping**: Side-by-side comparison of entity records with keyboard-driven grouping
- **Real-time precision-recall analysis**: Live plotting of model performance as judgements are submitted  
- **Collaborative evaluation**: Multi-user support with user attribution for all judgements
- **Flexible data sources**: Works with any entity resolution pipeline in the Matchbox ecosystem
- **Performance optimised**: Handles large datasets efficiently with smart queuing and caching

## Architecture overview

The evaluation tool follows a clean modular architecture designed for maintainability, testability, and performance. The design emphasises separation of concerns, with clear boundaries between state management, UI rendering, input handling, and business logic.

### Core architectural principles

1. **Single responsibility**: Each module has one clear purpose
2. **Observer pattern**: State changes propagate automatically through the UI
3. **Dependency injection**: Components receive their dependencies explicitly
4. **Model-view separation**: Business logic is separate from presentation logic
5. **Testability**: Each component can be tested in isolation

### Dependency flow

```
app.py (orchestration)
├── state.py (single source of truth)
├── handlers.py (input/actions) → plot/data.py (validation)
├── widgets/ (UI components)
├── modals.py (modal screens)
└── plot/ (plotting system)
```

## Directory structure

### Core modules

```
src/matchbox/client/cli/eval/
├── app.py              # Main application orchestration
├── state.py            # Centralised state management
├── handlers.py         # Input handling and actions
├── modals.py           # Modal screen definitions
├── utils.py            # Utility functions and data models
└── styles.css          # Textual styling
```

### UI components

```
widgets/
├── __init__.py
├── table.py            # Record comparison table
├── status.py           # Status bar components
└── styling.py          # Visual styling utilities
```

### Plot system

```
plot/
├── __init__.py
├── data.py             # Plot data validation and preparation
└── widget.py          # Plot rendering widget
```

### Testing

```
test/client/cli/eval/
├── test_app_integration.py  # Integration tests
├── test_state.py           # State management tests
├── test_handlers.py        # Input handling tests
├── test_plot_data.py       # Plot validation tests (critical)
└── test_widgets.py         # UI component tests
```

## Core design patterns

### Observer pattern for state management

The `EvaluationState` class implements the observer pattern, allowing UI components to automatically update when state changes:

```python
# State notifies all registered observers
state.add_listener(self._on_state_change)

# UI components refresh automatically
def _on_state_change(self):
    self.refresh()
```

This ensures the UI stays in sync without manual coordination between components.

### Single source of truth

All application state lives in the `EvaluationState` class, preventing data inconsistencies and race conditions. Components read from state but don't maintain their own copies of data.

### Dependency injection

Components receive their dependencies through constructor injection rather than importing globally:

```python
class ComparisonDisplayTable(Widget):
    def __init__(self, state: EvaluationState, **kwargs):
        super().__init__(**kwargs)
        self.state = state  # Injected dependency
```

This makes testing easier and reduces coupling between modules.

## State management system

### EvaluationState: Single source of truth

The `EvaluationState` class (`state.py`) serves as the single source of truth for all application state. It manages:

- **Queue state**: Current entity, position, painted items
- **UI state**: View mode, group selection, display data
- **Plot state**: Evaluation data, loading status, errors
- **User state**: Authentication, resolution settings
- **Status state**: Messages, colours, timers

### EvaluationQueue: Entity management

The `EvaluationQueue` class provides a deque-based queue that maintains position illusion while allowing efficient rotation:

```python
# Rotation maintains user's sense of position
queue.move_next()      # Rotate forward
queue.move_previous()  # Rotate backward

# Painted items are tracked separately
painted = queue.painted_items
queue.submit_all_painted()  # Remove submitted items
```

### State flow lifecycle

1. **Initialisation**: App creates state with default values
2. **Authentication**: User credentials stored in state
3. **Data loading**: Samples and evaluation data loaded asynchronously
4. **Interactive phase**: User interactions update state via handlers
5. **Submission**: Painted items submitted and removed from queue
6. **Cleanup**: State cleared on exit

### Observer pattern implementation

```python
class EvaluationState:
    def _notify_listeners(self):
        """Notify all listeners of state changes."""
        for callback in self.listeners:
            try:
                callback()
            except Exception:
                # Don't let listener errors crash the UI
                pass
```

Error handling in observers ensures that UI component failures don't cascade through the system.

## Input handling and actions

### Key routing strategy

The `EvaluationHandlers` class (`handlers.py`) implements a layered key routing strategy:

1. **Navigation keys**: Passed through to Textual's binding system
2. **Dynamic keys**: Handled directly (letters, numbers, slash)
3. **Action keys**: Delegated to specific action methods

```python
async def handle_key_input(self, event):
    # Navigation: let bindings handle
    if key in ["left", "right", "enter", "space"]:
        return
    
    # Dynamic: handle immediately
    if key.isalpha():
        self.state.set_group_selection(key)
        event.prevent_default()
```

### Dynamic vs static key bindings

**Dynamic keys** change behaviour based on application state:
- Letter keys: Set group selection (a-z)
- Number keys: Assign columns to selected group (1-9, 0)
- Slash key: Plot toggle (with state validation)

**Static keys** have fixed behaviour defined in `BINDINGS`:
- Arrow keys: Entity navigation
- Space: Submit and fetch
- Escape: Clear assignments

### Action delegation pattern

The main app delegates all actions to handlers, maintaining separation of concerns:

```python
# app.py
async def action_next_entity(self):
    await self.handlers.action_next_entity()

# handlers.py  
async def action_next_entity(self):
    # Actual implementation with state management
```

### Event flow

```
User input → app.on_key() → handlers.handle_key_input() → state updates → UI refresh
```

## UI component architecture

### Widget composition

UI components are composed hierarchically with clear data dependencies:

```
EntityResolutionApp
├── Header (Textual built-in)
├── StatusBar
│   ├── StatusBarLeft (entity progress, groups)
│   └── StatusBarRight (status messages)
├── ComparisonDisplayTable (record comparison)
└── Footer (Textual built-in)
```

### Table rendering system

The `ComparisonDisplayTable` (`widgets/table.py`) renders entity records in two modes:

**Compact view**: One row per field, deduplicated columns
```
Field     | 1    | 2 (×3) | 3
name      | ACME | ACME   | Acme Corp
address   | 123  | 123    | 123 Main
```

**Detailed view**: Source attribution with field grouping
```
Field                | 1    | 2 (×3) | 3
name (CRN)          | ACME |        |
name (DUNS)         |      | ACME   |
name (CDMS)         |      |        | Acme Corp
```

### Group styling system

The `GroupStyler` class (`widgets/styling.py`) provides consistent visual differentiation:

- **Colour cycling**: High-contrast colours distributed to avoid adjacency
- **Symbol assignment**: Unicode symbols for additional differentiation  
- **Consistency**: Same group always gets same colour/symbol combination
- **Conflict avoidance**: Avoids duplicates until all options are exhausted

### Status bar design

**Left status bar**: Entity progress and group assignments
- Current position (e.g., "Entity 3/10")
- Painted count (items ready for submission)
- Group counts with visual indicators

**Right status bar**: System status with strict length limits
- 12 character maximum with validation
- Emoji/symbol prefixes for quick recognition
- Auto-clearing timers for transient messages

## Plot system design

### Plot data validation (critical component)

The `plot/data.py` module provides the single source of truth for plot readiness validation:

```python
def can_show_plot(state) -> tuple[bool, str]:
    """Single source of truth for plot display readiness."""
    # Priority-ordered checks
    if state.is_loading_eval_data:
        return False, "⏳ Loading"
    if state.eval_data_error:
        return False, "⚠ Error" 
    if state.eval_data is None:
        return False, "⚠ No data"
    
    # Data sufficiency check (no try/except for control flow)
    pr_data = state.eval_data.precision_recall()
    if pr_data is None or len(pr_data) < 2:
        return False, "∅ Sparse"
    
    return True, ""
```

This function is critical for preventing the slash key modal bug by ensuring the modal only shows when data is ready.

### Widget vs data separation

**Clear separation of concerns**:
- `plot/data.py`: Validates data availability and readiness
- `plot/widget.py`: Renders plots based on validated data
- App-level: Decides whether to show modal based on validation

This prevents the dual-system conflicts that caused the original slash key bug.

### Error handling strategy

**No try/except for control flow**: Error handling uses explicit state checking rather than exception catching for business logic decisions.

**Graceful degradation**: Plot widget handles missing data gracefully with informative messages rather than crashing.

### Modal integration

Plot modals are created on-demand only after validation succeeds:

```python
# handlers.py
if can_show:
    success = await refresh_judgements_for_plot(state)
    if success:
        self.app.push_screen(PlotModal(state))
```

## Testing architecture

### Test file organisation

Tests are organised by architectural layer to match the modular code structure:

- **Unit tests**: Test individual modules in isolation
- **Integration tests**: Test cross-module interactions
- **Component tests**: Test UI components with mock state
- **Regression tests**: Specific tests for known issues (e.g., slash key bug)

### Testing patterns

**State testing**: Mock-free testing of state management logic
```python
def test_group_selection(self, state):
    state.set_group_selection("A")
    assert state.current_group_selection == "a"
```

**Handler testing**: Mock external dependencies, test logic
```python
@patch('handlers.can_show_plot')
async def test_plot_toggle(self, mock_can_show, handlers):
    mock_can_show.return_value = (False, "⏳ Loading")
    await handlers.handle_plot_toggle()
```

**Widget testing**: Test rendering with controlled state
```python
def test_render_compact_view(self, mock_state, mock_current_item):
    table = ComparisonDisplayTable(mock_state)
    result = table.render()
    assert isinstance(result, Table)
```

### Critical regression tests

The slash key modal bug has dedicated regression tests in `test_plot_data.py`:

```python
def test_slash_key_never_shows_modal_when_not_ready(self):
    """Critical test: slash key should NEVER show modal when plot is not ready."""
```

These tests verify all scenarios that previously caused incorrect modal display.

## Development guidelines

### Code style and conventions

- **Modern Python**: Use PEP 604 (union) and PEP 585 (generics) syntax
- **Type hints**: All functions and methods have complete type annotations
- **Docstrings**: All public methods documented with purpose and parameters
- **No comments**: Code should be self-documenting; comments only for complex algorithms

### Adding new features

1. **State first**: Add any new state to `EvaluationState`
2. **Handlers second**: Add input handling to `EvaluationHandlers`
3. **UI last**: Update widgets to reflect new state
4. **Tests throughout**: Write tests for each layer

### Error handling strategy

**Explicit over implicit**: Use explicit state checking rather than try/except for control flow.

**Graceful degradation**: UI should handle missing or invalid data gracefully.

**User feedback**: Always provide clear status messages for error conditions.

**Logging**: Log detailed errors for debugging but show simplified messages to users.

### Performance considerations

**State updates**: Batch state updates where possible to minimise observer notifications.

**Table rendering**: Large tables use Polars for efficient data processing.

**Queue operations**: Deque-based queue provides O(1) rotation operations.

**Lazy loading**: Plot data and widgets are created on-demand.

## Key design decisions and rationales

**Maintainability vs simplicity**: The modular architecture requires more files but makes each component easier to understand and modify.

**Performance vs clarity**: Some operations could be more efficient with tighter coupling, but the clarity benefits outweigh performance costs for this use case.

**Testability vs directness**: Dependency injection adds indirection but makes comprehensive testing possible.

### Observer pattern choice

**Alternative considered**: Manual UI updates after each state change.

**Choice rationale**: Observer pattern ensures UI consistency automatically and reduces coupling between state and UI components.

**Trade-off**: Slight performance overhead for automatic updates vs risk of UI inconsistency with manual updates.

### Input handling architecture

**Alternative considered**: All keys handled through Textual's binding system.

**Choice rationale**: Dynamic keys (letters, numbers) need contextual behaviour that static bindings can't provide efficiently.

**Trade-off**: Mixed input handling approaches vs consistent but inflexible binding system.

---

This architecture serves as the foundation for a maintainable, testable, and extensible entity resolution evaluation tool. The modular design ensures that future enhancements can be made without compromising the system's stability or clarity.