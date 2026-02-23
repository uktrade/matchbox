# Summary of Changes for Issue #487

**Issue:** `n` in `Source.sample()` not working as intended  
**URL:** https://github.com/uktrade/matchbox/issues/487

## Problem

`Source.sample(n)` was not respecting the `n` parameter and returned all rows regardless of the value passed. It also accepted invalid types like strings.

## Root Cause

The `sample()` method was using `batch_size` instead of a proper SQL `LIMIT` clause, so all data was fetched and only batch chunking was applied.

## Changes Made

### 1. `src/matchbox/client/locations.py` - Added `limit` parameter to `execute()`

- Added `limit: int | None = None` to the abstract `Location.execute()` method signature and docstring
- Added `limit` parameter to all three `@overload` signatures for `RelationalDBLocation.execute()`
- Updated implementation to wrap the SQL in a `LIMIT` subquery when `limit` is specified

### 2. `src/matchbox/client/sources.py` - Updated `fetch()` and `sample()`

- Added `limit: int | None = None` parameter to all `fetch()` overloads and implementation
- Passes `limit` through to `location.execute()`
- Updated `sample()` to:
  - Validate that `n` is an integer (raises `TypeError` otherwise)
  - Validate that `n` is positive (raises `ValueError` otherwise)
  - Use `limit=n` instead of `batch_size=n`
  - Fixed return type annotation from `None` to `QueryReturnClass`

### 3. `test/client/test_locations.py` - Added test for `limit`

- Added assertion to verify `limit=3` returns exactly 3 rows

### 4. `test/client/test_sources.py` - Added `test_source_sample_validation`

- Tests that string `n` raises `TypeError`
- Tests that negative/zero `n` raises `ValueError`
- Tests that float `n` raises `TypeError`
- Tests that valid positive integer works correctly
