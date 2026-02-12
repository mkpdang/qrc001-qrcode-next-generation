# Refactor — Clean Up, Restructure & Audit

Clean up the specified module or file: improve code organisation, readability, correctness, and internal consistency.

## Scope

When the user specifies a file, module, or directory:

1. **Read every file** in the target scope.
2. Apply the checks below.
3. Make edits directly — do not just report findings.
4. Preserve all existing functionality (refactoring is behaviour-preserving).
5. Run the existing test/verify commands afterwards to confirm nothing broke.

## Checks to Apply

### 1. Structure & Organisation
- Functions longer than ~40 lines → extract helpers with clear names.
- Related functions scattered across the file → group them under section comments (`# --- Section ---`).
- Imports: sort into stdlib / third-party / local blocks, remove unused imports.
- Dead code: remove commented-out code, unused variables, unreachable branches.

### 2. Naming & Readability
- Variable names: rename single-letter or cryptic names (`r`, `c`, `s`) to descriptive ones **only in new code or clearly ambiguous spots**. Don't rename well-understood loop variables like `i`, `r`, `c` in tight numeric loops.
- Function names: should be verb-first (`compute_X`, `render_Y`, `verify_Z`), not noun phrases.
- Magic numbers → named constants with a comment explaining the value.
- Consistent naming style throughout the module (snake_case for functions/vars, PascalCase for classes).

### 3. Type Hints & Signatures
- Add type hints to all public function signatures (args + return).
- Use `X | None` instead of `Optional[X]`.
- Use concrete types (`list[int]`) not abstract (`Sequence[int]`) unless the function genuinely needs the abstraction.

### 4. Error Handling
- Replace bare `except:` with specific exception types.
- Add guard clauses (early returns) for invalid input instead of deep nesting.
- Ensure error messages include the failing value, not just "invalid input".

### 5. Logic Simplification
- Nested if/else chains → early returns, dictionary dispatch, or match/case.
- Repeated code blocks → extract into a helper.
- `for` loops building a list → list comprehensions where it improves readability (not when it hurts it).
- Boolean expressions: simplify `if x == True:` → `if x:`, `if len(lst) > 0:` → `if lst:`.

### 6. Performance (obvious wins only)
- Set lookups instead of `x in list` when the list is iterated multiple times.
- Avoid re-computing values inside loops that could be computed once outside.
- Use `dict.get(key, default)` instead of `if key in dict: ... else: ...`.

### 7. Audit Trail
After all edits, produce a short summary at the end:

```
## Refactor Summary
- Files modified: N
- Functions extracted: X
- Dead code removed: Y lines
- Type hints added: Z functions
- Key changes: [bullet list of significant restructurings]
- Tests: PASS / FAIL
```

## What NOT to do
- **Do not change public API signatures** (function names, parameter names/order) without explicit approval.
- **Do not add new features** — this is clean-up only.
- **Do not add comments to obvious code** — only comment non-obvious logic.
- **Do not rewrite working code** just for style — only touch code with a clear improvement.
- **Do not add docstrings** to private helper functions unless the logic is non-obvious.
- **Do not change logging behaviour** — preserve all `@trace` and `audit()` calls.

## Applying to target

When the user specifies a file or module:
1. Read the file(s)
2. Apply all checks above
3. Edit in place
4. Run verification to confirm no regressions
5. Output the refactor summary

ARGUMENTS: $ARGUMENTS
