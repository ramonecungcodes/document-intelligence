"""A row with nothing in it is not a row."""
from core.normalize import is_blank
from core.rules import RULES

# Fields that make a row worth keeping. A row missing all of them carries no
# information at all -- it is padding, usually from a ruled but unfilled table.
SUBSTANTIVE = ("description", "amount", "quantity", "unit_price", "company", "title")


@RULES.register(
    "drop_empty_rows",
    help="Remove repeating-group entries where every substantive field is blank.",
)
def drop_empty_rows(record: dict) -> int:
    removed = 0

    def clean(container):
        nonlocal removed
        for group in ("line_items", "work_history"):
            rows = container.get(group)
            if not isinstance(rows, list):
                continue
            kept = [r for r in rows
                    if not isinstance(r, dict)
                    or any(not is_blank(r.get(f)) for f in SUBSTANTIVE)]
            removed += len(rows) - len(kept)
            container[group] = kept
        for nested in container.get("sections") or []:
            if isinstance(nested, dict):
                clean(nested)

    clean(record)
    return removed
