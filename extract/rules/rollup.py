"""A totals row is not a line item."""
from core.normalize import parse_money
from core.rules import RULES

# Labels a summary row is printed under. Matching the name alone is never enough.
ROLLUP_LABELS = {
    "subtotal", "sub total", "tax", "total", "due", "amount due", "balance",
    "balance due", "service total", "total due", "sum", "grand total",
    "subtotal / tax", "subtotal and tax",
}
TOLERANCE = 0.01


def _is_rollup(row: dict, container: dict) -> bool:
    """Named like a roll-up AND equal to one.

    The value test is what makes this safe. Real invoices bill tax as an ordinary line:
    "Tax" for $42.75 among items totalling $3,000 is a line item and stays one. Only a
    row named "Tax" whose amount *is* the container's tax figure is the summary line
    printed at the foot of the same table.
    """
    label = str(row.get("description") or "").strip().lower().rstrip(":")
    if label not in ROLLUP_LABELS:
        return False
    amount = parse_money(row.get("amount"))
    if amount is None:
        return True                     # named like a roll-up, carries nothing
    for key in ("subtotal", "tax", "total"):
        reference = parse_money(container.get(key))
        if reference is not None and abs(amount - reference) <= TOLERANCE:
            return True
    return False


@RULES.register(
    "drop_rollup_rows",
    help="Remove Subtotal/Tax/Total rows the model listed among the line items, "
         "but only when the amount matches the container's own figure.",
)
def drop_rollup_rows(record: dict) -> int:
    removed = 0

    def clean(container):
        nonlocal removed
        rows = container.get("line_items")
        if isinstance(rows, list):
            kept = [r for r in rows if not (isinstance(r, dict) and _is_rollup(r, container))]
            removed += len(rows) - len(kept)
            container["line_items"] = kept
        for nested in container.get("sections") or []:
            if isinstance(nested, dict):
                clean(nested)

    clean(record)
    return removed
