"""Page designs for invoices, purchase orders and multi-bill invoices.

Ten apiece, and the count is the point rather than decoration.

A classifier trained on a generated corpus can score perfectly by learning the
generator instead of the document. Measured directly: DiT scored 1.000 on faxes when
held out by document, and 0.958 when held out by *page design* -- the gap is what it
had memorised about these templates. With three invoice designs there was no way to
reserve one for testing without spending a third of the training data, so the question
went unasked for invoices and purchase orders entirely.

Ten makes it askable. Reserve two designs, keep eight, and the test set is drawn from
pages the model has genuinely never seen.

So the designs have to differ structurally, not cosmetically. Recolouring one template
ten times would inflate the count and teach nothing: the model would still see one
arrangement of blocks. These vary where the header sits, whether there is a sidebar,
how the table is ruled, where the totals land, and how dense the type is -- the things
a vision model actually keys on.

What they may not vary is the content contract. Every design prints the same fields
with the same labels, because the labels file is shared: a design that omitted a field,
or printed one the ground truth says is absent, would be grading the extractor against
a page that disagrees with its own answer key.
"""
from __future__ import annotations

import datetime
import html


# ------------------------------------------------------------------ helpers
def esc(s):
    return html.escape(str(s))


def money(x):
    return "${:,.2f}".format(x)


def d(dt):
    return dt.strftime("%b %d, %Y")


def addr_str(a):
    return f"{a['line1']}, {a['city']}, {a['state']} {a['zip']}"


def _rows(items, cls=""):
    c = f" class='{cls}'" if cls else ""
    return "".join(
        f"<tr{c}><td>{esc(i['description'])}</td><td class='r'>{i['quantity']}</td>"
        f"<td class='r'>{money(i['unit_price'])}</td><td class='r'>{money(i['amount'])}</td></tr>"
        for i in items)


SIDEBAR_CSS = "html,body{height:100%}"   # so a full-height rail has something to fill


def _css(font, size="13px", pad="40px", extra=""):
    return (f"<style>*{{box-sizing:border-box}}body{{font-family:{font};color:#222;margin:0;"
            f"padding:{pad};font-size:{size};background:#fff}}.r{{text-align:right}}"
            f"table{{width:100%;border-collapse:collapse}}th,td{{padding:7px 9px}}"
            f".muted{{color:#666}}.lbl{{font-size:10px;text-transform:uppercase;"
            f"letter-spacing:.06em;color:#777}}{extra}</style>")


def _mono(name):
    return "".join(w[0] for w in name.split()[:2]).upper()


def _thead(cols, style=""):
    head = "".join(f"<th class='r'>{c}</th>" if i else f"<th style='text-align:left'>{c}</th>"
                   for i, c in enumerate(cols))
    return f"<thead><tr style='{style}'>{head}</tr></thead>"


ITEM_COLS = ("Description", "Qty", "Unit", "Amount")


# ================================================================== invoices
def inv_0(comp, x):
    """Split header over a heavy rule; filled table head; totals block right."""
    color, accent, rows = comp["color"], comp["accent"], _rows(x["line_items"])
    return _css(x["_font"]) + f"""
<div style="display:flex;justify-content:space-between;border-bottom:3px solid {color};padding-bottom:14px">
<div><div style="font-size:26px;font-weight:700;color:{color}">{esc(comp['name'])}</div>
<div class="muted">{esc(comp['tag'])}</div><div class="muted">{esc(addr_str(x['_vendor_addr']))}</div></div>
<div style="text-align:right"><div style="font-size:30px;color:{accent};font-weight:700">INVOICE</div>
<div><b>#{esc(x['invoice_number'])}</b></div><div class="muted">Date: {esc(d(x['_idate']))}</div>
<div class="muted">Due: {esc(d(x['_ddate']))}</div></div></div>
<div style="display:flex;justify-content:space-between;margin-top:18px"><div><div class="muted">BILL TO</div>
<b>{esc(x['bill_to'])}</b><br><span class="muted">{esc(addr_str(x['_bill_addr']))}</span></div>
<div><div class="muted">PO NUMBER</div><b>{esc(x['po_number'])}</b><br>
<span class="muted">Terms: {esc(x['terms'])}</span></div></div>
<table style="margin-top:18px">{_thead(ITEM_COLS, f"background:{color};color:#fff")}<tbody>{rows}</tbody></table>
<table style="width:300px;margin-left:auto;margin-top:10px"><tr><td class="muted">Subtotal</td>
<td class="r">{money(x['subtotal'])}</td></tr><tr><td class="muted">Tax ({int(x['_taxrate']*100)}%)</td>
<td class="r">{money(x['tax'])}</td></tr><tr><td style="border-top:2px solid {color};font-weight:700">Total</td>
<td class="r" style="border-top:2px solid {color};font-weight:700;color:{accent}">{money(x['total'])}</td></tr></table>
<p class="muted" style="margin-top:30px">Remit to {esc(comp['name'])}. Thank you for your business.</p>"""


def inv_1(comp, x):
    """Rounded colour banner with a monogram tile; borderless table."""
    color, accent, rows = comp["color"], comp["accent"], _rows(x["line_items"])
    return _css(x["_font"]) + f"""
<div style="background:{color};color:#fff;padding:22px 24px;border-radius:8px;display:flex;
justify-content:space-between;align-items:center">
<div style="display:flex;align-items:center;gap:14px"><div style="width:52px;height:52px;border-radius:10px;
background:{accent};display:flex;align-items:center;justify-content:center;font-weight:800;font-size:20px">
{_mono(comp['name'])}</div><div><div style="font-size:20px;font-weight:700">{esc(comp['name'])}</div>
<div style="opacity:.85;font-size:12px">{esc(comp['tag'])}</div></div></div>
<div style="text-align:right"><div style="font-size:22px;letter-spacing:3px">INVOICE</div>
<div style="opacity:.9">{esc(x['invoice_number'])}</div></div></div>
<div style="display:flex;justify-content:space-between;margin-top:20px"><div><div class="muted">Billed To</div>
<b>{esc(x['bill_to'])}</b><br><span class="muted">{esc(addr_str(x['_bill_addr']))}</span></div>
<div style="text-align:right"><div class="muted">Issued {esc(d(x['_idate']))}</div>
<div class="muted">Due {esc(d(x['_ddate']))}</div>
<div class="muted">PO {esc(x['po_number'])} &middot; {esc(x['terms'])}</div></div></div>
<table style="margin-top:18px">{_thead(ITEM_COLS, f"border-bottom:2px solid {accent}")}<tbody>{rows}</tbody></table>
<div style="display:flex;justify-content:flex-end;margin-top:14px"><div style="width:280px">
<div style="display:flex;justify-content:space-between;padding:4px 0" class="muted">Subtotal
<span>{money(x['subtotal'])}</span></div>
<div style="display:flex;justify-content:space-between;padding:4px 0" class="muted">Tax ({int(x['_taxrate']*100)}%)
<span>{money(x['tax'])}</span></div>
<div style="display:flex;justify-content:space-between;padding:10px 0;border-top:2px solid {color};
font-weight:800;color:{color}">Total Due<span>{money(x['total'])}</span></div></div></div>"""


def inv_2(comp, x):
    """Left accent bar, three-column meta grid, hairline table."""
    color, accent, rows = comp["color"], comp["accent"], _rows(x["line_items"])
    return _css(x["_font"]) + f"""
<div style="border-left:6px solid {accent};padding-left:16px">
<div style="font-size:22px;font-weight:700">{esc(comp['name'])}</div>
<div class="muted">{esc(comp['tag'])} &middot; {esc(addr_str(x['_vendor_addr']))}</div></div>
<div style="margin-top:24px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px">
<div><div class="lbl">Invoice</div><b>{esc(x['invoice_number'])}</b></div>
<div><div class="lbl">Date / Due</div>{esc(d(x['_idate']))} &rarr; {esc(d(x['_ddate']))}</div>
<div><div class="lbl">PO / Terms</div>{esc(x['po_number'])} &middot; {esc(x['terms'])}</div></div>
<div style="margin-top:14px"><span class="muted">Bill to:</span> <b>{esc(x['bill_to'])}</b>,
{esc(addr_str(x['_bill_addr']))}</div>
<table style="margin-top:16px">{_thead(ITEM_COLS, "border-bottom:1px solid #333")}<tbody>{rows}</tbody></table>
<div style="margin-top:12px;text-align:right"><div class="muted">Subtotal {money(x['subtotal'])} |
Tax {money(x['tax'])}</div><div style="font-size:18px;font-weight:800;color:{accent};margin-top:6px">
TOTAL {money(x['total'])}</div></div>"""


def inv_3(comp, x):
    """Centred formal: everything on the axis, double rules, no fills."""
    color, rows = comp["color"], _rows(x["line_items"])
    return _css(x["_font"], extra="td,th{border-bottom:1px solid #ddd}") + f"""
<div style="text-align:center;border-bottom:3px double {color};padding-bottom:16px">
<div style="font-size:27px;font-weight:700;letter-spacing:.04em">{esc(comp['name'])}</div>
<div class="muted">{esc(comp['tag'])}</div>
<div class="muted" style="font-size:11.5px">{esc(addr_str(x['_vendor_addr']))}</div></div>
<div style="text-align:center;margin:22px 0 6px"><div style="font-size:19px;letter-spacing:.42em;
text-transform:uppercase;color:{color}">Invoice</div>
<div class="muted" style="margin-top:4px">No. {esc(x['invoice_number'])} &middot;
Issued {esc(d(x['_idate']))} &middot; Due {esc(d(x['_ddate']))}</div></div>
<div style="display:flex;justify-content:center;gap:60px;margin:20px 0;text-align:center">
<div><div class="lbl">Bill to</div><b>{esc(x['bill_to'])}</b><br>
<span class="muted">{esc(addr_str(x['_bill_addr']))}</span></div>
<div><div class="lbl">Purchase order</div><b>{esc(x['po_number'])}</b><br>
<span class="muted">{esc(x['terms'])}</span></div></div>
<table>{_thead(ITEM_COLS, f"border-top:2px solid {color};border-bottom:2px solid {color}")}
<tbody>{rows}</tbody></table>
<div style="margin:14px auto 0;width:340px">
<div style="display:flex;justify-content:space-between" class="muted">Subtotal<span>{money(x['subtotal'])}</span></div>
<div style="display:flex;justify-content:space-between" class="muted">Tax<span>{money(x['tax'])}</span></div>
<div style="display:flex;justify-content:space-between;border-top:3px double {color};margin-top:6px;
padding-top:6px;font-weight:700">Total<span>{money(x['total'])}</span></div></div>"""


def inv_4(comp, x):
    """Dark left sidebar carrying vendor and meta; items to the right."""
    color, accent, rows = comp["color"], comp["accent"], _rows(x["line_items"])
    return _css(x["_font"], pad="0", extra=SIDEBAR_CSS) + f"""
<div style="display:flex;min-height:100vh">
<div style="width:210px;background:{color};color:#fff;padding:34px 20px;flex:none">
<div style="font-size:18px;font-weight:800;line-height:1.25">{esc(comp['name'])}</div>
<div style="opacity:.8;font-size:11px;margin-top:4px">{esc(comp['tag'])}</div>
<div style="opacity:.75;font-size:11px;margin-top:14px">{esc(addr_str(x['_vendor_addr']))}</div>
<div style="height:1px;background:rgba(255,255,255,.3);margin:20px 0"></div>
<div style="font-size:11px;opacity:.75;text-transform:uppercase;letter-spacing:.06em">Invoice</div>
<div style="font-weight:700;margin-bottom:12px">{esc(x['invoice_number'])}</div>
<div style="font-size:11px;opacity:.75;text-transform:uppercase;letter-spacing:.06em">Issued</div>
<div style="margin-bottom:12px">{esc(d(x['_idate']))}</div>
<div style="font-size:11px;opacity:.75;text-transform:uppercase;letter-spacing:.06em">Due</div>
<div style="margin-bottom:12px">{esc(d(x['_ddate']))}</div>
<div style="font-size:11px;opacity:.75;text-transform:uppercase;letter-spacing:.06em">PO</div>
<div>{esc(x['po_number'])}</div></div>
<div style="flex:1;padding:34px 30px">
<div class="lbl">Bill to</div><b style="font-size:15px">{esc(x['bill_to'])}</b>
<div class="muted">{esc(addr_str(x['_bill_addr']))}</div>
<div class="muted" style="margin-top:4px">Terms: {esc(x['terms'])}</div>
<table style="margin-top:22px">{_thead(ITEM_COLS, "border-bottom:1.5px solid #444")}<tbody>{rows}</tbody></table>
<div style="margin-top:16px;text-align:right"><span class="muted">Subtotal {money(x['subtotal'])}
&nbsp;&middot;&nbsp; Tax {money(x['tax'])}</span>
<div style="font-size:20px;font-weight:800;color:{accent};margin-top:6px">{money(x['total'])}</div>
<div class="lbl">Total due</div></div></div></div>"""


def inv_5(comp, x):
    """Typewriter statement: monospace, no colour, dotted leaders."""
    rows = "".join(
        f"<tr><td>{esc(i['description'])}</td><td class='r'>{i['quantity']}</td>"
        f"<td class='r'>{money(i['unit_price'])}</td><td class='r'>{money(i['amount'])}</td></tr>"
        for i in x["line_items"])
    return _css("'Courier New', Courier, monospace", size="12px",
                extra="td,th{padding:3px 6px}hr{border:0;border-top:1px dashed #999}") + f"""
<div style="text-align:center;font-weight:700;letter-spacing:.2em">I N V O I C E</div>
<hr>
<pre style="margin:10px 0;font:inherit;white-space:pre-wrap">{esc(comp['name'])}
{esc(addr_str(x['_vendor_addr']))}
{esc(comp['tag'])}</pre>
<hr>
<pre style="margin:10px 0;font:inherit;white-space:pre-wrap">INVOICE NO. : {esc(x['invoice_number'])}
DATE        : {esc(d(x['_idate']))}
DUE         : {esc(d(x['_ddate']))}
PO NUMBER   : {esc(x['po_number'])}
TERMS       : {esc(x['terms'])}
BILL TO     : {esc(x['bill_to'])}
              {esc(addr_str(x['_bill_addr']))}</pre>
<hr>
<table>{_thead(("DESCRIPTION", "QTY", "UNIT", "AMOUNT"), "border-bottom:1px solid #000")}
<tbody>{rows}</tbody></table>
<hr>
<div style="margin-left:auto;width:300px">
<div style="display:flex;justify-content:space-between">SUBTOTAL<span>{money(x['subtotal'])}</span></div>
<div style="display:flex;justify-content:space-between">TAX<span>{money(x['tax'])}</span></div>
<div style="display:flex;justify-content:space-between;font-weight:700;border-top:1px solid #000;
margin-top:4px;padding-top:4px">TOTAL DUE<span>{money(x['total'])}</span></div></div>"""


def inv_6(comp, x):
    """Thin top strip, zebra-striped ruled grid, full-width totals bar."""
    color, accent = comp["color"], comp["accent"]
    rows = "".join(
        f"<tr style='background:{'#f4f4f6' if n % 2 else '#fff'}'><td>{esc(i['description'])}</td>"
        f"<td class='r'>{i['quantity']}</td><td class='r'>{money(i['unit_price'])}</td>"
        f"<td class='r'>{money(i['amount'])}</td></tr>" for n, i in enumerate(x["line_items"]))
    return _css(x["_font"], pad="0", extra="td,th{border:1px solid #d8d8de}") + f"""
<div style="height:9px;background:{accent}"></div>
<div style="padding:28px 40px 40px">
<div style="display:flex;justify-content:space-between;align-items:flex-end">
<div><div style="font-size:23px;font-weight:800;color:{color}">{esc(comp['name'])}</div>
<div class="muted">{esc(comp['tag'])} &middot; {esc(addr_str(x['_vendor_addr']))}</div></div>
<div style="text-align:right"><div class="lbl">Invoice</div>
<div style="font-size:17px;font-weight:700">{esc(x['invoice_number'])}</div></div></div>
<table style="margin-top:20px;font-size:12px"><tbody>
<tr><td class="lbl" style="background:#f4f4f6;width:110px">Bill to</td><td>{esc(x['bill_to'])}</td>
<td class="lbl" style="background:#f4f4f6;width:110px">Issued</td><td>{esc(d(x['_idate']))}</td></tr>
<tr><td class="lbl" style="background:#f4f4f6">Address</td><td>{esc(addr_str(x['_bill_addr']))}</td>
<td class="lbl" style="background:#f4f4f6">Due</td><td>{esc(d(x['_ddate']))}</td></tr>
<tr><td class="lbl" style="background:#f4f4f6">PO number</td><td>{esc(x['po_number'])}</td>
<td class="lbl" style="background:#f4f4f6">Terms</td><td>{esc(x['terms'])}</td></tr></tbody></table>
<table style="margin-top:16px">{_thead(ITEM_COLS, f"background:{color};color:#fff")}<tbody>{rows}</tbody></table>
<div style="margin-top:14px;background:{color};color:#fff;padding:11px 14px;display:flex;
justify-content:space-between;align-items:center">
<span style="opacity:.85">Subtotal {money(x['subtotal'])} &nbsp; Tax {money(x['tax'])}</span>
<span style="font-size:17px;font-weight:800">TOTAL {money(x['total'])}</span></div></div>"""


def inv_7(comp, x):
    """Every block in its own bordered box, arranged on a grid."""
    color, accent, rows = comp["color"], comp["accent"], _rows(x["line_items"])
    box = "border:1.5px solid #333;padding:11px 13px"
    return _css(x["_font"]) + f"""
<div style="display:grid;grid-template-columns:1.4fr 1fr;gap:12px">
<div style="{box}"><div style="font-size:20px;font-weight:800;color:{color}">{esc(comp['name'])}</div>
<div class="muted">{esc(comp['tag'])}</div><div class="muted">{esc(addr_str(x['_vendor_addr']))}</div></div>
<div style="{box};text-align:center;display:flex;flex-direction:column;justify-content:center">
<div style="letter-spacing:.28em;font-size:15px">INVOICE</div>
<div style="font-size:19px;font-weight:800;color:{accent}">{esc(x['invoice_number'])}</div></div></div>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-top:12px">
<div style="{box}"><div class="lbl">Bill to</div><b>{esc(x['bill_to'])}</b>
<div class="muted">{esc(addr_str(x['_bill_addr']))}</div></div>
<div style="{box}"><div class="lbl">Dates</div>Issued {esc(d(x['_idate']))}<br>Due {esc(d(x['_ddate']))}</div>
<div style="{box}"><div class="lbl">Reference</div>PO {esc(x['po_number'])}<br>{esc(x['terms'])}</div></div>
<div style="{box};margin-top:12px;padding:0">
<table>{_thead(ITEM_COLS, "background:#eee")}<tbody>{rows}</tbody></table></div>
<div style="display:grid;grid-template-columns:1fr 300px;gap:12px;margin-top:12px">
<div style="{box};color:#666">Remit to {esc(comp['name'])}, {esc(addr_str(x['_vendor_addr']))}.</div>
<div style="{box}"><div style="display:flex;justify-content:space-between" class="muted">Subtotal
<span>{money(x['subtotal'])}</span></div>
<div style="display:flex;justify-content:space-between" class="muted">Tax<span>{money(x['tax'])}</span></div>
<div style="display:flex;justify-content:space-between;font-weight:800;color:{accent};
border-top:1.5px solid #333;margin-top:5px;padding-top:5px">Total<span>{money(x['total'])}</span></div></div></div>"""


def inv_8(comp, x):
    """Dense: one meta strip across the top, small type, wide table."""
    color, accent, rows = comp["color"], comp["accent"], _rows(x["line_items"])
    return _css(x["_font"], size="11px", pad="30px",
                extra="td,th{padding:4px 7px}") + f"""
<div style="display:flex;justify-content:space-between;align-items:baseline">
<span style="font-size:16px;font-weight:800;color:{color}">{esc(comp['name'])}</span>
<span class="muted">{esc(comp['tag'])} &middot; {esc(addr_str(x['_vendor_addr']))}</span></div>
<div style="border-top:1px solid {color};border-bottom:1px solid {color};margin-top:8px;padding:6px 0;
display:flex;gap:26px;flex-wrap:wrap">
<span><span class="lbl">Invoice</span> <b>{esc(x['invoice_number'])}</b></span>
<span><span class="lbl">Issued</span> {esc(d(x['_idate']))}</span>
<span><span class="lbl">Due</span> {esc(d(x['_ddate']))}</span>
<span><span class="lbl">PO</span> {esc(x['po_number'])}</span>
<span><span class="lbl">Terms</span> {esc(x['terms'])}</span>
<span><span class="lbl">Bill to</span> <b>{esc(x['bill_to'])}</b>, {esc(addr_str(x['_bill_addr']))}</span></div>
<table style="margin-top:10px">{_thead(ITEM_COLS, "border-bottom:1px solid #999")}<tbody>{rows}</tbody></table>
<div style="border-top:1px solid {color};margin-top:8px;padding-top:6px;text-align:right">
<span class="muted">Subtotal {money(x['subtotal'])} &nbsp; Tax {money(x['tax'])} &nbsp;</span>
<b style="color:{accent};font-size:14px">Total {money(x['total'])}</b></div>"""


def inv_9(comp, x):
    """Amount-due hero: the figure first, the paperwork after."""
    color, accent, rows = comp["color"], comp["accent"], _rows(x["line_items"])
    return _css(x["_font"], pad="44px") + f"""
<div style="display:flex;justify-content:space-between;align-items:flex-start">
<div><div style="font-size:15px;font-weight:700">{esc(comp['name'])}</div>
<div class="muted" style="font-size:11px">{esc(addr_str(x['_vendor_addr']))}</div></div>
<div style="text-align:right"><div class="lbl">Amount due</div>
<div style="font-size:40px;font-weight:800;color:{accent};line-height:1">{money(x['total'])}</div>
<div class="muted" style="margin-top:4px">by {esc(d(x['_ddate']))}</div></div></div>
<div style="height:2px;background:{color};margin:24px 0"></div>
<div style="display:flex;gap:44px">
<div><div class="lbl">Invoice</div><b>{esc(x['invoice_number'])}</b></div>
<div><div class="lbl">Issued</div>{esc(d(x['_idate']))}</div>
<div><div class="lbl">PO</div>{esc(x['po_number'])}</div>
<div><div class="lbl">Terms</div>{esc(x['terms'])}</div>
<div><div class="lbl">Bill to</div><b>{esc(x['bill_to'])}</b></div></div>
<table style="margin-top:24px">{_thead(ITEM_COLS, "border-bottom:1px solid #ccc")}<tbody>{rows}</tbody></table>
<div style="display:flex;justify-content:flex-end;gap:26px;margin-top:12px" class="muted">
<span>Subtotal {money(x['subtotal'])}</span><span>Tax {money(x['tax'])}</span>
<span style="color:{color};font-weight:800">Total {money(x['total'])}</span></div>"""


INVOICE = [inv_0, inv_1, inv_2, inv_3, inv_4, inv_5, inv_6, inv_7, inv_8, inv_9]


# ================================================================== purchase orders
def po_0(buyer, x):
    """Split header over a rule; filled table head; totals right."""
    color, accent, rows = buyer["color"], buyer["accent"], _rows(x["line_items"])
    return _css(x["_font"]) + f"""
<div style="display:flex;justify-content:space-between;border-bottom:2px solid {color};padding-bottom:12px">
<div><div style="font-size:24px;font-weight:700;color:{color}">{esc(buyer['name'])}</div>
<div class="muted">{esc(addr_str(x['_buyer_addr']))}</div></div>
<div style="text-align:right"><div style="font-size:26px;font-weight:800;color:{accent}">PURCHASE ORDER</div>
<div><b>PO #{esc(x['po_number'])}</b></div><div class="muted">{esc(d(x['_pdate']))}</div></div></div>
<div style="display:flex;justify-content:space-between;margin-top:16px">
<div><div class="muted">VENDOR</div><b>{esc(x['vendor'])}</b><br>
<span class="muted">{esc(addr_str(x['_vendor_addr']))}</span></div>
<div><div class="muted">SHIP TO</div><b>{esc(buyer['name'])}</b><br>
<span class="muted">{esc(addr_str(x['_ship_addr']))}</span></div>
<div><div class="muted">DELIVER BY</div><b>{esc(d(x['_deliver']))}</b><br>
<span class="muted">{esc(x['terms'])}</span></div></div>
<table style="margin-top:16px">{_thead(ITEM_COLS, f"background:{color};color:#fff")}<tbody>{rows}</tbody></table>
<table style="width:300px;margin-left:auto;margin-top:10px"><tr><td class="muted">Subtotal</td>
<td class="r">{money(x['subtotal'])}</td></tr><tr><td class="muted">Tax</td>
<td class="r">{money(x['tax'])}</td></tr><tr><td style="font-weight:800;border-top:2px solid {color}">Total</td>
<td class="r" style="font-weight:800;border-top:2px solid {color}">{money(x['total'])}</td></tr></table>
<p class="muted" style="margin-top:24px">Authorized by Procurement &middot; {esc(buyer['name'])}</p>"""


def po_1(buyer, x):
    """Colour banner, borderless table, totals inline right."""
    color, accent, rows = buyer["color"], buyer["accent"], _rows(x["line_items"])
    return _css(x["_font"]) + f"""
<div style="background:{color};color:#fff;padding:18px 22px;display:flex;justify-content:space-between;
align-items:center;border-radius:6px"><div style="font-weight:700;font-size:20px">{esc(buyer['name'])}</div>
<div style="text-align:right"><div style="letter-spacing:2px">PURCHASE ORDER</div>
<div>{esc(x['po_number'])} &middot; {esc(d(x['_pdate']))}</div></div></div>
<div style="display:flex;justify-content:space-between;margin-top:18px">
<div><div class="muted">Vendor</div><b>{esc(x['vendor'])}</b><br>
<span class="muted">{esc(addr_str(x['_vendor_addr']))}</span></div>
<div style="text-align:right"><div class="muted">Deliver by {esc(d(x['_deliver']))}</div>
<div class="muted">Terms {esc(x['terms'])}</div></div></div>
<table style="margin-top:16px">{_thead(ITEM_COLS, f"border-bottom:2px solid {accent}")}<tbody>{rows}</tbody></table>
<div style="text-align:right;margin-top:12px"><span class="muted">Subtotal {money(x['subtotal'])} |
Tax {money(x['tax'])}</span><div style="font-weight:800;color:{color};font-size:18px">
TOTAL {money(x['total'])}</div></div>"""


def po_2(buyer, x):
    """Three hard-bordered header boxes: vendor, ship-to, delivery."""
    color, accent, rows = buyer["color"], buyer["accent"], _rows(x["line_items"])
    box = "border:2px solid #222;padding:10px 12px"
    return _css(x["_font"]) + f"""
<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:12px">
<div style="font-size:22px;font-weight:800">{esc(buyer['name'])}</div>
<div style="font-size:20px;font-weight:800;letter-spacing:.1em;color:{accent}">PURCHASE ORDER</div></div>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0;border:2px solid #222">
<div style="padding:10px 12px;border-right:2px solid #222"><div class="lbl">Vendor</div>
<b>{esc(x['vendor'])}</b><div class="muted">{esc(addr_str(x['_vendor_addr']))}</div></div>
<div style="padding:10px 12px;border-right:2px solid #222"><div class="lbl">Ship to</div>
<b>{esc(buyer['name'])}</b><div class="muted">{esc(addr_str(x['_ship_addr']))}</div></div>
<div style="padding:10px 12px"><div class="lbl">PO number</div><b>{esc(x['po_number'])}</b>
<div class="muted">Ordered {esc(d(x['_pdate']))}</div>
<div class="muted">Deliver by {esc(d(x['_deliver']))}</div>
<div class="muted">{esc(x['terms'])}</div></div></div>
<table style="margin-top:14px;border:2px solid #222">
{_thead(ITEM_COLS, f"background:{color};color:#fff")}<tbody>{rows}</tbody></table>
<div style="{box};width:320px;margin-left:auto;margin-top:12px">
<div style="display:flex;justify-content:space-between" class="muted">Subtotal<span>{money(x['subtotal'])}</span></div>
<div style="display:flex;justify-content:space-between" class="muted">Tax<span>{money(x['tax'])}</span></div>
<div style="display:flex;justify-content:space-between;font-weight:800;border-top:2px solid #222;
margin-top:5px;padding-top:5px">Total<span>{money(x['total'])}</span></div></div>"""


def po_3(buyer, x):
    """Centred formal order, double rules, signature line."""
    color, rows = buyer["color"], _rows(x["line_items"])
    return _css(x["_font"], extra="td,th{border-bottom:1px solid #e0e0e0}") + f"""
<div style="text-align:center;border-bottom:3px double {color};padding-bottom:14px">
<div style="font-size:25px;font-weight:700">{esc(buyer['name'])}</div>
<div class="muted" style="font-size:11.5px">{esc(addr_str(x['_buyer_addr']))}</div></div>
<div style="text-align:center;margin:20px 0"><div style="font-size:18px;letter-spacing:.38em;
text-transform:uppercase;color:{color}">Purchase Order</div>
<div class="muted" style="margin-top:4px">No. {esc(x['po_number'])} &middot; {esc(d(x['_pdate']))}</div></div>
<div style="display:flex;justify-content:center;gap:52px;text-align:center;margin-bottom:18px">
<div><div class="lbl">Vendor</div><b>{esc(x['vendor'])}</b><br>
<span class="muted">{esc(addr_str(x['_vendor_addr']))}</span></div>
<div><div class="lbl">Ship to</div><b>{esc(buyer['name'])}</b><br>
<span class="muted">{esc(addr_str(x['_ship_addr']))}</span></div>
<div><div class="lbl">Deliver by</div><b>{esc(d(x['_deliver']))}</b><br>
<span class="muted">{esc(x['terms'])}</span></div></div>
<table>{_thead(ITEM_COLS, f"border-top:2px solid {color};border-bottom:2px solid {color}")}
<tbody>{rows}</tbody></table>
<div style="margin:14px auto 0;width:330px">
<div style="display:flex;justify-content:space-between" class="muted">Subtotal<span>{money(x['subtotal'])}</span></div>
<div style="display:flex;justify-content:space-between" class="muted">Tax<span>{money(x['tax'])}</span></div>
<div style="display:flex;justify-content:space-between;border-top:3px double {color};margin-top:6px;
padding-top:6px;font-weight:700">Total<span>{money(x['total'])}</span></div></div>
<div style="margin-top:44px;border-top:1px solid #999;width:260px;padding-top:5px" class="muted">
Authorized signature &middot; Procurement</div>"""


def po_4(buyer, x):
    """Dark sidebar carrying the order meta; items to the right."""
    color, accent, rows = buyer["color"], buyer["accent"], _rows(x["line_items"])
    return _css(x["_font"], pad="0", extra=SIDEBAR_CSS) + f"""
<div style="display:flex;min-height:100vh">
<div style="width:205px;background:{color};color:#fff;padding:32px 20px;flex:none">
<div style="letter-spacing:.16em;font-size:12px;opacity:.85">PURCHASE ORDER</div>
<div style="font-size:21px;font-weight:800;margin-top:6px">{esc(x['po_number'])}</div>
<div style="height:1px;background:rgba(255,255,255,.3);margin:18px 0"></div>
<div style="font-size:11px;opacity:.75">ORDERED</div><div style="margin-bottom:10px">{esc(d(x['_pdate']))}</div>
<div style="font-size:11px;opacity:.75">DELIVER BY</div><div style="margin-bottom:10px">{esc(d(x['_deliver']))}</div>
<div style="font-size:11px;opacity:.75">TERMS</div><div style="margin-bottom:10px">{esc(x['terms'])}</div>
<div style="font-size:11px;opacity:.75">SHIP TO</div>
<div style="font-size:11.5px">{esc(buyer['name'])}<br>{esc(addr_str(x['_ship_addr']))}</div></div>
<div style="flex:1;padding:32px 30px">
<div style="font-size:21px;font-weight:800;color:{color}">{esc(buyer['name'])}</div>
<div class="muted">{esc(addr_str(x['_buyer_addr']))}</div>
<div style="margin-top:18px"><div class="lbl">Vendor</div><b style="font-size:15px">{esc(x['vendor'])}</b>
<div class="muted">{esc(addr_str(x['_vendor_addr']))}</div></div>
<table style="margin-top:20px">{_thead(ITEM_COLS, "border-bottom:1.5px solid #444")}<tbody>{rows}</tbody></table>
<div style="margin-top:14px;text-align:right"><span class="muted">Subtotal {money(x['subtotal'])}
&nbsp;&middot;&nbsp; Tax {money(x['tax'])}</span>
<div style="font-size:19px;font-weight:800;color:{accent}">{money(x['total'])}</div>
<div class="lbl">Order total</div></div></div></div>"""


def po_5(buyer, x):
    """Monospace requisition, all-caps labels, boxed."""
    rows = _rows(x["line_items"])
    return _css("'Courier New', Courier, monospace", size="12px",
                extra="td,th{padding:3px 6px}hr{border:0;border-top:1px dashed #999}") + f"""
<div style="border:2px solid #000;padding:12px">
<div style="text-align:center;font-weight:700;letter-spacing:.2em">PURCHASE ORDER</div>
<hr>
<pre style="margin:8px 0;font:inherit;white-space:pre-wrap">PO NUMBER   : {esc(x['po_number'])}
ORDER DATE  : {esc(d(x['_pdate']))}
DELIVER BY  : {esc(d(x['_deliver']))}
TERMS       : {esc(x['terms'])}</pre>
<hr>
<pre style="margin:8px 0;font:inherit;white-space:pre-wrap">BUYER       : {esc(buyer['name'])}
              {esc(addr_str(x['_buyer_addr']))}
VENDOR      : {esc(x['vendor'])}
              {esc(addr_str(x['_vendor_addr']))}
SHIP TO     : {esc(addr_str(x['_ship_addr']))}</pre>
</div>
<table style="margin-top:12px">{_thead(("DESCRIPTION", "QTY", "UNIT", "AMOUNT"), "border-bottom:1px solid #000")}
<tbody>{rows}</tbody></table>
<hr>
<div style="margin-left:auto;width:290px">
<div style="display:flex;justify-content:space-between">SUBTOTAL<span>{money(x['subtotal'])}</span></div>
<div style="display:flex;justify-content:space-between">TAX<span>{money(x['tax'])}</span></div>
<div style="display:flex;justify-content:space-between;font-weight:700;border-top:1px solid #000;
margin-top:4px;padding-top:4px">ORDER TOTAL<span>{money(x['total'])}</span></div></div>"""


def po_6(buyer, x):
    """Top strip, key-value grid, zebra table, full-width totals bar."""
    color, accent = buyer["color"], buyer["accent"]
    rows = "".join(
        f"<tr style='background:{'#f5f5f7' if n % 2 else '#fff'}'><td>{esc(i['description'])}</td>"
        f"<td class='r'>{i['quantity']}</td><td class='r'>{money(i['unit_price'])}</td>"
        f"<td class='r'>{money(i['amount'])}</td></tr>" for n, i in enumerate(x["line_items"]))
    return _css(x["_font"], pad="0", extra="td,th{border:1px solid #dadae0}") + f"""
<div style="height:9px;background:{accent}"></div>
<div style="padding:26px 40px 40px">
<div style="display:flex;justify-content:space-between;align-items:flex-end">
<div style="font-size:22px;font-weight:800;color:{color}">{esc(buyer['name'])}</div>
<div style="text-align:right"><div class="lbl">Purchase order</div>
<div style="font-size:17px;font-weight:700">{esc(x['po_number'])}</div></div></div>
<table style="margin-top:18px;font-size:12px"><tbody>
<tr><td class="lbl" style="background:#f5f5f7;width:104px">Vendor</td><td>{esc(x['vendor'])}</td>
<td class="lbl" style="background:#f5f5f7;width:104px">Ordered</td><td>{esc(d(x['_pdate']))}</td></tr>
<tr><td class="lbl" style="background:#f5f5f7">Ship to</td><td>{esc(addr_str(x['_ship_addr']))}</td>
<td class="lbl" style="background:#f5f5f7">Deliver by</td><td>{esc(d(x['_deliver']))}</td></tr>
<tr><td class="lbl" style="background:#f5f5f7">Buyer</td><td>{esc(addr_str(x['_buyer_addr']))}</td>
<td class="lbl" style="background:#f5f5f7">Terms</td><td>{esc(x['terms'])}</td></tr></tbody></table>
<table style="margin-top:14px">{_thead(ITEM_COLS, f"background:{color};color:#fff")}<tbody>{rows}</tbody></table>
<div style="margin-top:14px;background:{color};color:#fff;padding:11px 14px;display:flex;
justify-content:space-between"><span style="opacity:.85">Subtotal {money(x['subtotal'])} &nbsp;
Tax {money(x['tax'])}</span><span style="font-size:17px;font-weight:800">TOTAL {money(x['total'])}</span></div></div>"""


def po_7(buyer, x):
    """Bordered panels on a grid."""
    color, accent, rows = buyer["color"], buyer["accent"], _rows(x["line_items"])
    box = "border:1.5px solid #333;padding:11px 13px"
    return _css(x["_font"]) + f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
<div style="{box}"><div style="font-size:19px;font-weight:800;color:{color}">{esc(buyer['name'])}</div>
<div class="muted">{esc(addr_str(x['_buyer_addr']))}</div></div>
<div style="{box};text-align:center"><div style="letter-spacing:.22em">PURCHASE ORDER</div>
<div style="font-size:19px;font-weight:800;color:{accent}">{esc(x['po_number'])}</div>
<div class="muted">{esc(d(x['_pdate']))}</div></div></div>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-top:12px">
<div style="{box}"><div class="lbl">Vendor</div><b>{esc(x['vendor'])}</b>
<div class="muted">{esc(addr_str(x['_vendor_addr']))}</div></div>
<div style="{box}"><div class="lbl">Ship to</div><b>{esc(buyer['name'])}</b>
<div class="muted">{esc(addr_str(x['_ship_addr']))}</div></div>
<div style="{box}"><div class="lbl">Delivery</div><b>{esc(d(x['_deliver']))}</b>
<div class="muted">{esc(x['terms'])}</div></div></div>
<div style="{box};margin-top:12px;padding:0">
<table>{_thead(ITEM_COLS, "background:#eee")}<tbody>{rows}</tbody></table></div>
<div style="{box};width:310px;margin-left:auto;margin-top:12px">
<div style="display:flex;justify-content:space-between" class="muted">Subtotal<span>{money(x['subtotal'])}</span></div>
<div style="display:flex;justify-content:space-between" class="muted">Tax<span>{money(x['tax'])}</span></div>
<div style="display:flex;justify-content:space-between;font-weight:800;color:{accent};
border-top:1.5px solid #333;margin-top:5px;padding-top:5px">Total<span>{money(x['total'])}</span></div></div>"""


def po_8(buyer, x):
    """Dense single-strip meta, small type."""
    color, accent, rows = buyer["color"], buyer["accent"], _rows(x["line_items"])
    return _css(x["_font"], size="11px", pad="30px", extra="td,th{padding:4px 7px}") + f"""
<div style="display:flex;justify-content:space-between;align-items:baseline">
<span style="font-size:16px;font-weight:800;color:{color}">{esc(buyer['name'])}</span>
<span style="letter-spacing:.14em;font-weight:700;color:{accent}">PURCHASE ORDER</span></div>
<div style="border-top:1px solid {color};border-bottom:1px solid {color};margin-top:8px;padding:6px 0;
display:flex;gap:24px;flex-wrap:wrap">
<span><span class="lbl">PO</span> <b>{esc(x['po_number'])}</b></span>
<span><span class="lbl">Ordered</span> {esc(d(x['_pdate']))}</span>
<span><span class="lbl">Deliver by</span> {esc(d(x['_deliver']))}</span>
<span><span class="lbl">Terms</span> {esc(x['terms'])}</span>
<span><span class="lbl">Vendor</span> <b>{esc(x['vendor'])}</b>, {esc(addr_str(x['_vendor_addr']))}</span>
<span><span class="lbl">Ship to</span> {esc(addr_str(x['_ship_addr']))}</span></div>
<table style="margin-top:10px">{_thead(ITEM_COLS, "border-bottom:1px solid #999")}<tbody>{rows}</tbody></table>
<div style="border-top:1px solid {color};margin-top:8px;padding-top:6px;text-align:right">
<span class="muted">Subtotal {money(x['subtotal'])} &nbsp; Tax {money(x['tax'])} &nbsp;</span>
<b style="color:{accent};font-size:14px">Total {money(x['total'])}</b></div>"""


def po_9(buyer, x):
    """Order-total hero, then the paperwork."""
    color, accent, rows = buyer["color"], buyer["accent"], _rows(x["line_items"])
    return _css(x["_font"], pad="44px") + f"""
<div style="display:flex;justify-content:space-between;align-items:flex-start">
<div><div style="font-size:15px;font-weight:700">{esc(buyer['name'])}</div>
<div class="muted" style="font-size:11px">{esc(addr_str(x['_buyer_addr']))}</div></div>
<div style="text-align:right"><div class="lbl">Order total</div>
<div style="font-size:38px;font-weight:800;color:{accent};line-height:1">{money(x['total'])}</div>
<div class="muted" style="margin-top:4px">deliver by {esc(d(x['_deliver']))}</div></div></div>
<div style="height:2px;background:{color};margin:24px 0"></div>
<div style="display:flex;gap:40px;flex-wrap:wrap">
<div><div class="lbl">PO number</div><b>{esc(x['po_number'])}</b></div>
<div><div class="lbl">Ordered</div>{esc(d(x['_pdate']))}</div>
<div><div class="lbl">Terms</div>{esc(x['terms'])}</div>
<div><div class="lbl">Vendor</div><b>{esc(x['vendor'])}</b></div>
<div><div class="lbl">Ship to</div>{esc(addr_str(x['_ship_addr']))}</div></div>
<table style="margin-top:24px">{_thead(ITEM_COLS, "border-bottom:1px solid #ccc")}<tbody>{rows}</tbody></table>
<div style="display:flex;justify-content:flex-end;gap:24px;margin-top:12px" class="muted">
<span>Subtotal {money(x['subtotal'])}</span><span>Tax {money(x['tax'])}</span>
<span style="color:{color};font-weight:800">Total {money(x['total'])}</span></div>"""


PURCHASE_ORDER = [po_0, po_1, po_2, po_3, po_4, po_5, po_6, po_7, po_8, po_9]


# ================================================================== multi-bill
def _mb_period(sec):
    if not sec["service_period_start"] and not sec["service_period_end"]:
        return "&mdash;"
    a = d(datetime.date.fromisoformat(sec["service_period_start"])) if sec["service_period_start"] else "?"
    b = d(datetime.date.fromisoformat(sec["service_period_end"])) if sec["service_period_end"] else "?"
    return f"{esc(a)} &ndash; {esc(b)}"


def _mb_site(sec, prefix=" &middot; "):
    """The service address behind a SITE label, or nothing at all.

    Most sections have no per-service address and must render none: the extractor is
    asked to return null for those, and a corpus that prints something anyway would be
    grading the model against a page that disagrees with its own labels.
    """
    if not sec.get("service_location"):
        return ""
    return prefix + "<span class='lbl'>Site</span> " + esc(sec["service_location"])


def _mb_ident(sec, sep="&nbsp;&nbsp;"):
    """The section's identifiers, each behind its own label.

    A field the reader has to infer from column position is a field the extractor has
    to infer too. Labelling them is what a real bill does, and it is the difference
    between a hard document and an ambiguous one.
    """
    bits = [f"<span class='lbl'>Code</span> {esc(sec['service_code'])}",
            f"<span class='lbl'>Account</span> {esc(sec['account_number'])}"]
    if sec.get("reference_number"):
        bits.append(f"<span class='lbl'>{esc(sec['reference_label'])}</span> "
                    f"{esc(sec['reference_number'])}")
    if sec.get("cost_center"):
        bits.append(f"<span class='lbl'>Cost centre</span> {esc(sec['cost_center'])}")
    return sep.join(bits)


def _mb_rows(sec, color):
    return (f"<tr><td colspan='4' style='padding-top:12px;font-weight:700;color:{color}'>"
            f"{esc(sec['service_type'])} &middot; {esc(sec['account_number'])}</td></tr>"
            + _rows(sec["line_items"]) +
            f"<tr><td colspan='3' class='r muted'>Subtotal / Tax</td><td class='r'>"
            f"{money(sec['subtotal'])} / {money(sec['tax'])}</td></tr>"
            f"<tr><td colspan='3' class='r' style='font-weight:700'>{esc(sec['service_code'])} total</td>"
            f"<td class='r' style='font-weight:700'>{money(sec['total'])}</td></tr>")


def _mb_base(x):
    return _css(x["_font"], size="12.5px", pad="38px")


def _mb_head(vendor, x, color, accent):
    return (f"<div style='display:flex;justify-content:space-between;border-bottom:3px solid {color};"
            f"padding-bottom:12px'><div><div style='font-size:23px;font-weight:700;color:{color}'>"
            f"{esc(vendor['name'])}</div><div class='muted'>{esc(vendor['tag'])}</div>"
            f"<div class='muted'>{esc(addr_str(x['_vendor_addr']))}</div></div>"
            f"<div style='text-align:right'><div style='font-size:26px;color:{accent};font-weight:700'>"
            f"INVOICE</div><div><b>#{esc(x['invoice_number'])}</b></div>"
            f"<div class='muted'>Date: {esc(d(x['_idate']))}</div>"
            f"<div class='muted'>Due: {esc(d(x['_ddate']))}</div></div></div>"
            f"<div style='display:flex;justify-content:space-between;margin-top:14px'>"
            f"<div><div class='lbl'>Bill to</div><b>{esc(x['bill_to'])}</b><br>"
            f"<span class='muted'>{esc(addr_str(x['_bill_addr']))}</span></div>"
            f"<div><div class='lbl'>Master account</div><b>{esc(x['master_account'])}</b><br>"
            f"<span class='muted'>Terms: {esc(x['terms'])}</span></div>"
            f"<div style='text-align:right'><div class='lbl'>Services billed</div>"
            f"<b style='font-size:17px;color:{accent}'>{x['section_count']}</b>"
            f"<div class='muted'>pay each separately</div></div></div>")


def _mb_note(vendor, x):
    return (f"<p class='muted' style='margin-top:22px'>Each service above is billed to its own account and "
            f"may be remitted separately. Reference the account number shown for that service when paying. "
            f"Remit to {esc(vendor['name'])}, {esc(x['remit_to'])}.</p>")


def _mb_totals(x, color, accent, label="Total due"):
    return (f"<table style='width:320px;margin-left:auto;margin-top:16px'>"
            f"<tr><td class='muted'>Invoice subtotal</td><td class='r'>{money(x['subtotal'])}</td></tr>"
            f"<tr><td class='muted'>Invoice tax</td><td class='r'>{money(x['tax'])}</td></tr>"
            f"<tr><td style='border-top:2px solid {color};font-weight:700'>{label}</td>"
            f"<td class='r' style='border-top:2px solid {color};font-weight:700;color:{accent}'>"
            f"{money(x['total'])}</td></tr></table>")


def mb_0(vendor, x):
    """Summary table with a column per identifier, then a detail block per service."""
    color, accent = vendor["color"], vendor["accent"]
    summ = "".join(
        f"<tr><td><b>{esc(s['service_type'])}</b></td><td>{esc(s['service_code'])}</td>"
        f"<td>{esc(s['account_number'])}</td>"
        f"<td>{esc(s['reference_label'])} {esc(s['reference_number'])}</td>"
        f"<td>{_mb_period(s)}</td><td>{esc(s['cost_center'])}</td>"
        f"<td class='r' style='font-weight:700'>{money(s['total'])}</td></tr>" for s in x["sections"])
    detail = "".join(
        f"<div style='margin-top:16px;border-left:4px solid {accent};padding-left:12px'>"
        f"<div style='font-weight:700;color:{color}'>{esc(s['service_type'])}</div>"
        f"<div class='muted' style='margin:2px 0'>{_mb_ident(s)}</div>"
        f"<div class='muted'>Service period {_mb_period(s)}{_mb_site(s)}</div>"
        f"<table>{_thead(ITEM_COLS, 'border-bottom:1px solid #999')}"
        f"<tbody>{_rows(s['line_items'])}</tbody></table>"
        f"<div class='r' style='margin-top:4px'>Subtotal {money(s['subtotal'])} &middot; "
        f"Tax {money(s['tax'])} &middot; <b style='color:{accent}'>Service total {money(s['total'])}"
        f"</b></div></div>" for s in x["sections"])
    return _mb_base(x) + _mb_head(vendor, x, color, accent) + (
        f"<table style='margin-top:18px'><thead><tr style='background:{color};color:#fff'>"
        f"<th style='text-align:left'>Service</th><th style='text-align:left'>Code</th>"
        f"<th style='text-align:left'>Account</th><th style='text-align:left'>Reference</th>"
        f"<th style='text-align:left'>Service period</th><th style='text-align:left'>Cost centre</th>"
        f"<th class='r'>Amount due</th></tr></thead><tbody>{summ}</tbody></table>{detail}"
    ) + _mb_totals(x, color, accent) + _mb_note(vendor, x)


def mb_1(vendor, x):
    """Boxed per-service statements, one labelled line per identifier."""
    color, accent = vendor["color"], vendor["accent"]
    cards = "".join(
        f"<div style='border:1px solid #ccc;border-top:5px solid {color};margin-top:14px;padding:12px 14px'>"
        f"<div style='font-size:15px;font-weight:700;color:{color}'>{esc(s['service_type'])}</div>"
        f"<table style='width:auto;margin-top:6px'>"
        f"<tr><td class='lbl'>Service code</td><td><b>{esc(s['service_code'])}</b></td>"
        f"<td class='lbl' style='padding-left:22px'>Account</td><td><b>{esc(s['account_number'])}</b></td></tr>"
        f"<tr><td class='lbl'>{esc(s['reference_label'])}</td><td><b>{esc(s['reference_number'])}</b></td>"
        f"<td class='lbl' style='padding-left:22px'>Cost centre</td><td>{esc(s['cost_center'])}</td></tr>"
        f"<tr><td class='lbl'>Service period</td><td colspan='3'>{_mb_period(s)}{_mb_site(s)}</td></tr></table>"
        f"<table style='margin-top:8px'>{_thead(ITEM_COLS, 'border-bottom:1px solid #bbb')}"
        f"<tbody>{_rows(s['line_items'])}</tbody></table>"
        f"<div style='display:flex;justify-content:flex-end;gap:18px;margin-top:6px'>"
        f"<span class='muted'>Subtotal {money(s['subtotal'])}</span>"
        f"<span class='muted'>Tax {money(s['tax'])}</span>"
        f"<span style='font-weight:800;color:{accent}'>Due {money(s['total'])}</span></div></div>"
        for s in x["sections"])
    return _mb_base(x) + _mb_head(vendor, x, color, accent) + cards + (
        f"<div style='display:flex;justify-content:flex-end;margin-top:18px'><div style='width:300px'>"
        f"<div style='display:flex;justify-content:space-between' class='muted'>Sum of services"
        f"<span>{money(x['subtotal'])}</span></div>"
        f"<div style='display:flex;justify-content:space-between' class='muted'>Tax"
        f"<span>{money(x['tax'])}</span></div>"
        f"<div style='display:flex;justify-content:space-between;border-top:2px solid {color};"
        f"padding-top:8px;font-weight:800;color:{color}'>Total due<span>{money(x['total'])}</span></div>"
        f"</div></div>") + _mb_note(vendor, x)


def mb_2(vendor, x):
    """A labelled legend of services, then one continuous ledger."""
    color, accent = vendor["color"], vendor["accent"]
    legend = "".join(
        f"<div style='border-bottom:1px dotted #bbb;padding:5px 0'>"
        f"<b>{esc(s['service_type'])}</b><br><span class='muted'>{_mb_ident(s)}</span><br>"
        f"<span class='muted'><span class='lbl'>Period</span> {_mb_period(s)}"
        f"{_mb_site(s, ' &nbsp; ')}</span></div>" for s in x["sections"])
    body = "".join(_mb_rows(s, color) for s in x["sections"])
    return _mb_base(x) + _mb_head(vendor, x, color, accent) + (
        f"<div style='margin-top:16px;border:1px solid #ddd;padding:10px 12px'>"
        f"<div class='lbl' style='margin-bottom:4px'>Separately payable services</div>{legend}</div>"
        f"<table style='margin-top:10px'>{_thead(ITEM_COLS, f'border-bottom:2px solid {color}')}"
        f"<tbody>{body}</tbody></table>") + _mb_totals(x, color, accent) + _mb_note(vendor, x)


def mb_3(vendor, x):
    """Full per-service statements, each with its own banner, stacked."""
    color, accent = vendor["color"], vendor["accent"]
    blocks = "".join(
        f"<div style='margin-top:18px'>"
        f"<div style='background:{color};color:#fff;padding:8px 12px;display:flex;"
        f"justify-content:space-between'><b>{esc(s['service_type'])}</b>"
        f"<span>{esc(s['service_code'])} &middot; {esc(s['account_number'])}</span></div>"
        f"<div style='border:1px solid #ddd;border-top:0;padding:10px 12px'>"
        f"<div class='muted'><span class='lbl'>{esc(s['reference_label'])}</span> "
        f"{esc(s['reference_number'])} &nbsp; <span class='lbl'>Cost centre</span> "
        f"{esc(s['cost_center'])} &nbsp; <span class='lbl'>Period</span> {_mb_period(s)}"
        f"{_mb_site(s, ' &nbsp; ')}</div>"
        f"<table style='margin-top:8px'>{_thead(ITEM_COLS, 'border-bottom:1px solid #ccc')}"
        f"<tbody>{_rows(s['line_items'])}</tbody></table>"
        f"<div class='r' style='margin-top:6px'><span class='muted'>Subtotal {money(s['subtotal'])} "
        f"&middot; Tax {money(s['tax'])}</span> &nbsp; <b style='color:{accent}'>"
        f"{money(s['total'])}</b></div></div></div>" for s in x["sections"])
    return (_mb_base(x) + _mb_head(vendor, x, color, accent) + blocks
            + _mb_totals(x, color, accent) + _mb_note(vendor, x))


def mb_4(vendor, x):
    """Two-column card grid: services side by side."""
    color, accent = vendor["color"], vendor["accent"]
    cards = "".join(
        f"<div style='border:1px solid #ccc;padding:10px 12px'>"
        f"<div style='font-weight:700;color:{color};border-bottom:2px solid {accent};"
        f"padding-bottom:4px'>{esc(s['service_type'])}</div>"
        f"<div class='muted' style='margin:6px 0;font-size:11.5px'>{_mb_ident(s, '<br>')}</div>"
        f"<div class='muted' style='font-size:11.5px'><span class='lbl'>Period</span> "
        f"{_mb_period(s)}{_mb_site(s, '<br>')}</div>"
        f"<table style='margin-top:6px;font-size:11.5px'>{_thead(ITEM_COLS, 'border-bottom:1px solid #ddd')}"
        f"<tbody>{_rows(s['line_items'])}</tbody></table>"
        f"<div class='r' style='margin-top:5px;font-weight:700;color:{accent}'>"
        f"Due {money(s['total'])}</div></div>" for s in x["sections"])
    return _mb_base(x) + _mb_head(vendor, x, color, accent) + (
        f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:16px'>{cards}</div>"
    ) + _mb_totals(x, color, accent) + _mb_note(vendor, x)


def mb_5(vendor, x):
    """Monospace ledger, services as indented headings."""
    color, accent = vendor["color"], vendor["accent"]
    body = "".join(
        f"<pre style='margin:14px 0 4px;font:inherit;white-space:pre-wrap;font-weight:700'>"
        f"{esc(s['service_type'])}</pre>"
        f"<pre style='margin:0 0 4px;font:inherit;white-space:pre-wrap'>"
        f"  CODE     : {esc(s['service_code'])}\n"
        f"  ACCOUNT  : {esc(s['account_number'])}\n"
        f"  {esc(s['reference_label']).upper():<9}: {esc(s['reference_number'])}\n"
        f"  COST CTR : {esc(s['cost_center'])}\n"
        f"  PERIOD   : {_mb_period(s)}"
        f"{chr(10) + '  SITE     : ' + esc(s['service_location']) if s.get('service_location') else ''}</pre>"
        f"<table>{_thead(('DESCRIPTION', 'QTY', 'UNIT', 'AMOUNT'), 'border-bottom:1px solid #000')}"
        f"<tbody>{_rows(s['line_items'])}</tbody></table>"
        f"<div class='r' style='margin-top:3px'>SERVICE TOTAL {money(s['total'])}</div>"
        for s in x["sections"])
    return _css("'Courier New', Courier, monospace", size="11.5px", pad="34px",
                extra="td,th{padding:2px 6px}hr{border:0;border-top:1px dashed #999}") + f"""
<div style="text-align:center;font-weight:700;letter-spacing:.18em">CONSOLIDATED INVOICE</div><hr>
<pre style="margin:8px 0;font:inherit;white-space:pre-wrap">{esc(vendor['name'])}
{esc(addr_str(x['_vendor_addr']))}

INVOICE NO.    : {esc(x['invoice_number'])}
DATE / DUE     : {esc(d(x['_idate']))} / {esc(d(x['_ddate']))}
MASTER ACCOUNT : {esc(x['master_account'])}
BILL TO        : {esc(x['bill_to'])}
TERMS          : {esc(x['terms'])}
SERVICES       : {x['section_count']} (each payable separately)</pre><hr>
{body}<hr>
<div style="margin-left:auto;width:310px">
<div style="display:flex;justify-content:space-between">SUBTOTAL<span>{money(x['subtotal'])}</span></div>
<div style="display:flex;justify-content:space-between">TAX<span>{money(x['tax'])}</span></div>
<div style="display:flex;justify-content:space-between;font-weight:700;border-top:1px solid #000;
margin-top:4px;padding-top:4px">TOTAL DUE<span>{money(x['total'])}</span></div></div>""" + _mb_note(vendor, x)


def mb_6(vendor, x):
    """Summary first and totals immediately after; line-item appendix at the end."""
    color, accent = vendor["color"], vendor["accent"]
    summ = "".join(
        f"<tr style='background:{'#f5f5f7' if n % 2 else '#fff'}'>"
        f"<td><b>{esc(s['service_type'])}</b><div class='muted' style='font-size:11px'>"
        f"{_mb_ident(s)}</div></td><td>{_mb_period(s)}</td>"
        f"<td class='r'>{money(s['subtotal'])}</td><td class='r'>{money(s['tax'])}</td>"
        f"<td class='r' style='font-weight:700;color:{accent}'>{money(s['total'])}</td></tr>"
        for n, s in enumerate(x["sections"]))
    appendix = "".join(
        f"<div style='margin-top:12px'><div class='lbl'>{esc(s['service_type'])} &middot; "
        f"{esc(s['account_number'])}{_mb_site(s)}</div>"
        f"<table>{_thead(ITEM_COLS, 'border-bottom:1px solid #ddd')}"
        f"<tbody>{_rows(s['line_items'])}</tbody></table></div>" for s in x["sections"])
    return _mb_base(x) + _mb_head(vendor, x, color, accent) + (
        f"<table style='margin-top:18px'><thead><tr style='background:{color};color:#fff'>"
        f"<th style='text-align:left'>Service &amp; identifiers</th>"
        f"<th style='text-align:left'>Service period</th><th class='r'>Subtotal</th>"
        f"<th class='r'>Tax</th><th class='r'>Amount due</th></tr></thead>"
        f"<tbody>{summ}</tbody></table>") + _mb_totals(x, color, accent) + (
        f"<div style='margin-top:26px;border-top:2px solid {color};padding-top:10px'>"
        f"<div class='lbl'>Appendix &mdash; line items by service</div>{appendix}</div>"
    ) + _mb_note(vendor, x)


def mb_7(vendor, x):
    """Left index rail of services, detail in the right column."""
    color, accent = vendor["color"], vendor["accent"]
    index = "".join(
        f"<div style='padding:7px 0;border-bottom:1px solid rgba(255,255,255,.25)'>"
        f"<div style='font-weight:700'>{esc(s['service_type'])}</div>"
        f"<div style='opacity:.8;font-size:10.5px'>{esc(s['service_code'])} &middot; "
        f"{esc(s['account_number'])}</div>"
        f"<div style='opacity:.9;margin-top:2px'>{money(s['total'])}</div></div>"
        for s in x["sections"])
    detail = "".join(
        f"<div style='margin-bottom:16px'>"
        f"<div style='font-weight:700;color:{color};border-bottom:1px solid #ddd;padding-bottom:3px'>"
        f"{esc(s['service_type'])}</div>"
        f"<div class='muted' style='margin:5px 0;font-size:11.5px'>{_mb_ident(s)}<br>"
        f"<span class='lbl'>Period</span> {_mb_period(s)}{_mb_site(s, ' &nbsp; ')}</div>"
        f"<table>{_thead(ITEM_COLS, 'border-bottom:1px solid #e2e2e2')}"
        f"<tbody>{_rows(s['line_items'])}</tbody></table>"
        f"<div class='r' style='margin-top:4px'><span class='muted'>Subtotal {money(s['subtotal'])} "
        f"&middot; Tax {money(s['tax'])}</span> <b style='color:{accent}'>{money(s['total'])}</b>"
        f"</div></div>" for s in x["sections"])
    return _css(x["_font"], size="12px", pad="0", extra=SIDEBAR_CSS) + f"""
<div style="display:flex;min-height:100vh">
<div style="width:215px;background:{color};color:#fff;padding:30px 18px;flex:none">
<div style="font-size:17px;font-weight:800;line-height:1.25">{esc(vendor['name'])}</div>
<div style="opacity:.8;font-size:11px">{esc(vendor['tag'])}</div>
<div style="height:1px;background:rgba(255,255,255,.3);margin:16px 0"></div>
<div style="font-size:10.5px;opacity:.8;text-transform:uppercase;letter-spacing:.06em">Invoice</div>
<div style="font-weight:700;margin-bottom:8px">{esc(x['invoice_number'])}</div>
<div style="font-size:10.5px;opacity:.8;text-transform:uppercase;letter-spacing:.06em">Master account</div>
<div style="font-weight:700;margin-bottom:8px">{esc(x['master_account'])}</div>
<div style="font-size:10.5px;opacity:.8;text-transform:uppercase;letter-spacing:.06em">Due</div>
<div style="margin-bottom:14px">{esc(d(x['_ddate']))}</div>
<div style="font-size:10.5px;opacity:.8;text-transform:uppercase;letter-spacing:.06em">
{x['section_count']} services, paid separately</div>{index}</div>
<div style="flex:1;padding:30px 28px">
<div style="display:flex;justify-content:space-between;border-bottom:2px solid {color};padding-bottom:8px">
<div><div class="lbl">Bill to</div><b>{esc(x['bill_to'])}</b>
<div class="muted">{esc(addr_str(x['_bill_addr']))}</div></div>
<div style="text-align:right"><div class="lbl">Issued</div>{esc(d(x['_idate']))}
<div class="muted">Terms {esc(x['terms'])}</div></div></div>
<div style="margin-top:16px">{detail}</div>
{_mb_totals(x, color, accent)}{_mb_note(vendor, x)}</div></div>"""


def mb_8(vendor, x):
    """Dense zebra table per service, minimal chrome."""
    color, accent = vendor["color"], vendor["accent"]
    blocks = "".join(
        f"<tr style='background:{color};color:#fff'><td colspan='4'><b>{esc(s['service_type'])}</b>"
        f"&nbsp; {esc(s['service_code'])} &middot; {esc(s['account_number'])} &middot; "
        f"{esc(s['reference_label'])} {esc(s['reference_number'])} &middot; "
        f"{esc(s['cost_center'])} &middot; {_mb_period(s)}"
        f"{(' &middot; ' + esc(s['service_location'])) if s.get('service_location') else ''}</td></tr>"
        + "".join(
            f"<tr style='background:{'#f6f6f8' if n % 2 else '#fff'}'><td>{esc(i['description'])}</td>"
            f"<td class='r'>{i['quantity']}</td><td class='r'>{money(i['unit_price'])}</td>"
            f"<td class='r'>{money(i['amount'])}</td></tr>" for n, i in enumerate(s["line_items"]))
        + f"<tr><td colspan='3' class='r muted'>Subtotal {money(s['subtotal'])} &middot; "
          f"Tax {money(s['tax'])}</td><td class='r' style='font-weight:800;color:{accent}'>"
          f"{money(s['total'])}</td></tr>" for s in x["sections"])
    return _css(x["_font"], size="11.5px", pad="30px",
                extra="td,th{padding:4px 7px;border:1px solid #dcdce2}") + (
        f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
        f"<span style='font-size:17px;font-weight:800;color:{color}'>{esc(vendor['name'])}</span>"
        f"<span class='muted'>{esc(vendor['tag'])}</span></div>"
        f"<div style='border-top:1px solid {color};border-bottom:1px solid {color};margin-top:7px;"
        f"padding:5px 0;display:flex;gap:20px;flex-wrap:wrap'>"
        f"<span><span class='lbl'>Invoice</span> <b>{esc(x['invoice_number'])}</b></span>"
        f"<span><span class='lbl'>Master account</span> <b>{esc(x['master_account'])}</b></span>"
        f"<span><span class='lbl'>Issued</span> {esc(d(x['_idate']))}</span>"
        f"<span><span class='lbl'>Due</span> {esc(d(x['_ddate']))}</span>"
        f"<span><span class='lbl'>Terms</span> {esc(x['terms'])}</span>"
        f"<span><span class='lbl'>Bill to</span> <b>{esc(x['bill_to'])}</b></span>"
        f"<span><span class='lbl'>Services</span> {x['section_count']}, paid separately</span></div>"
        f"<table style='margin-top:12px'>{_thead(ITEM_COLS, 'background:#e9e9ee')}"
        f"<tbody>{blocks}</tbody></table>") + _mb_totals(x, color, accent) + _mb_note(vendor, x)


def mb_9(vendor, x):
    """Remittance coupons: one detachable stub per service."""
    color, accent = vendor["color"], vendor["accent"]
    coupons = "".join(
        f"<div style='border:2px dashed #999;margin-top:14px;padding:0;display:flex'>"
        f"<div style='flex:1;padding:10px 12px'>"
        f"<div style='font-weight:700;color:{color};font-size:14px'>{esc(s['service_type'])}</div>"
        f"<div class='muted' style='margin:4px 0;font-size:11.5px'>{_mb_ident(s, '<br>')}</div>"
        f"<div class='muted' style='font-size:11.5px'><span class='lbl'>Period</span> {_mb_period(s)}"
        f"{_mb_site(s, '<br>')}</div>"
        f"<table style='margin-top:7px;font-size:11.5px'>"
        f"{_thead(ITEM_COLS, 'border-bottom:1px solid #ddd')}"
        f"<tbody>{_rows(s['line_items'])}</tbody></table></div>"
        f"<div style='width:170px;border-left:2px dashed #999;padding:10px 12px;background:#fafafa;"
        f"display:flex;flex-direction:column;justify-content:center;text-align:center'>"
        f"<div class='lbl'>Pay this service</div>"
        f"<div style='font-size:20px;font-weight:800;color:{accent}'>{money(s['total'])}</div>"
        f"<div class='muted' style='font-size:10.5px;margin-top:4px'>Account<br>"
        f"<b>{esc(s['account_number'])}</b></div>"
        f"<div class='muted' style='font-size:10.5px;margin-top:3px'>Subtotal {money(s['subtotal'])}<br>"
        f"Tax {money(s['tax'])}</div></div></div>" for s in x["sections"])
    return (_mb_base(x) + _mb_head(vendor, x, color, accent) + coupons
            + _mb_totals(x, color, accent) + _mb_note(vendor, x))


MULTI_BILL = [mb_0, mb_1, mb_2, mb_3, mb_4, mb_5, mb_6, mb_7, mb_8, mb_9]


# ------------------------------------------------------------------ dispatch
def inv_html(layout, comp, x):
    return INVOICE[layout % len(INVOICE)](comp, x)


def po_html(layout, buyer, x):
    return PURCHASE_ORDER[layout % len(PURCHASE_ORDER)](buyer, x)


def mb_html(layout, vendor, x):
    return MULTI_BILL[layout % len(MULTI_BILL)](vendor, x)
