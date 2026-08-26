#!/usr/bin/env python3
"""Synthetic business-document generator.

Produces invoices, purchase orders, and resumes as standalone HTML with
varied layouts, plus ground-truth JSON labels for evaluation.

Usage:
  python generate.py                          # default set (seed 42) -> ../  (the dataset root)
  python generate.py --seed 99 --out ./heldout   # a fresh set the model has NOT seen
  python generate.py --seed 7 --resumes 60 --invoices-per-company 5 --out ./round2

Re-run with a DIFFERENT --seed to get brand-new documents/data for held-out
testing. Same seed + same args = identical output (reproducible).

After generating HTML, render PDFs with render_pdfs.py (uses headless Chrome).
"""
import os, json, random, html, datetime, argparse

# ------------------------------------------------------------------ helpers
def esc(s): return html.escape(str(s))
def money(x): return "${:,.2f}".format(x)
def d(dt): return dt.strftime("%b %d, %Y")
def iso(dt): return dt.strftime("%Y-%m-%d")

FONTS = ["Georgia, 'Times New Roman', serif", "'Segoe UI', Arial, sans-serif",
         "'Helvetica Neue', Helvetica, Arial, sans-serif", "'Trebuchet MS', Verdana, sans-serif",
         "Cambria, serif", "'Courier New', monospace", "Calibri, 'Segoe UI', sans-serif"]
CITIES = [("Clearwater","FL","33755"),("Austin","TX","78701"),("Columbus","OH","43215"),
          ("Denver","CO","80202"),("Raleigh","NC","27601"),("Portland","OR","97204"),
          ("Tampa","FL","33602"),("Madison","WI","53703"),("Boise","ID","83702"),
          ("Richmond","VA","23219"),("Mesa","AZ","85201"),("Omaha","NE","68102")]
STREETS = ["Maple Ave","Industrial Pkwy","Commerce Dr","Oak Street","Harbor Blvd","Lakeview Rd",
           "Franklin St","Sunset Way","Riverside Dr","Cedar Ln","Market St","Pinecrest Ct"]
TERMS = ["Net 15","Net 30","Net 45","Due on receipt","2/10 Net 30"]
CUSTOMERS = ["Cascade Analytics","Harbor Point Medical","Sterling & Rowe LLP","Peak Ridge Manufacturing",
             "Bright Harbor Academy","Copperline Retail","Elmwood Property Group","Summit Fabrication",
             "Riverstone Financial","Blue Sage Hospitality","Fairfield Utilities","Onyx Software Labs",
             "Greenfield Foods","Atlas Construction","Nimbus Media","Keystone Insurance"]

def addr():
    c = random.choice(CITIES)
    return {"line1": f"{random.randint(100,9899)} {random.choice(STREETS)}", "city": c[0], "state": c[1], "zip": c[2]}
def addr_str(a): return f"{a['line1']}, {a['city']}, {a['state']} {a['zip']}"
def phone(): return f"({random.randint(200,989)}) {random.randint(200,989)}-{random.randint(1000,9999)}"

COMPANIES = [
    dict(name="Northwind Components LLC", tag="Industrial Parts & Fasteners", color="#1f3a5f", accent="#c9832b",
         items=[("Hex bolt M8x40 (box/100)",12,48),("Stainless washer kit",6,26),("Bearing assembly 6204",22,60),
                ("Steel bracket, powder-coated",14,55),("Hydraulic fitting 1/2 in",9,34),("Cable gland IP68",5,19)]),
    dict(name="Meridian Office Supply Co.", tag="Office Supplies & Furniture", color="#0f766e", accent="#e07a3f",
         items=[("Copy paper, 10-ream case",39,52),("Ergonomic task chair",149,320),("Whiteboard 4x3 ft",64,120),
                ("Ballpoint pens (dozen)",4,9),("Standing desk converter",180,240),("Toner cartridge, black",78,110)]),
    dict(name="BluePeak Cloud Services", tag="Managed IT & Hosting", color="#3730a3", accent="#f59e0b",
         items=[("Managed hosting, per node/mo",45,45),("Backup storage, per TB/mo",8,8),("Incident response, 10 hrs",95,950),
                ("SSL certificate, wildcard/yr",120,120),("VPN seat, per user/mo",6,6),("Monitoring, per endpoint/mo",3,3)]),
    dict(name="Ironleaf Landscaping Inc.", tag="Commercial Grounds Care", color="#166534", accent="#a16207",
         items=[("Lawn maintenance, per visit",85,85),("Mulch install, per cu yd",42,42),("Irrigation repair, per hr",65,65),
                ("Seasonal cleanup",240,240),("Tree trimming, per tree",110,110),("Fertilizer treatment",130,130)]),
    dict(name="Solstice Catering Group", tag="Corporate Catering & Events", color="#b45309", accent="#0e7490",
         items=[("Boxed lunch, per person",14,14),("Coffee service, per 20",48,48),("Hot buffet, per person",26,26),
                ("Dessert platter",55,55),("Event staff, per hr",32,32),("Beverage package, per person",7,7)]),
    dict(name="Vanguard Logistics Partners", tag="Freight & Distribution", color="#334155", accent="#dc2626",
         items=[("LTL freight, per shipment",210,210),("Pallet handling, per pallet",18,18),("Warehouse storage, per mo",340,340),
                ("Expedited delivery surcharge",95,95),("Fuel surcharge",47,47),("Liftgate service",40,40)]),
    dict(name="Lumen Print & Design", tag="Commercial Printing", color="#a21caf", accent="#0d9488",
         items=[("Business cards, per 500",39,39),("Brochure, tri-fold per 250",180,180),("Banner 3x6 ft",85,85),
                ("Booklet, 16pp per 100",240,240),("Design hour",75,75),("Foam board sign",28,28)]),
]

def line_items(pool, nmin=2, nmax=6):
    picks = random.sample(pool, random.randint(nmin, min(nmax, len(pool))))
    out=[]
    for desc, lo, hi in picks:
        qty=random.randint(1,25); unit=round(random.uniform(lo,hi),2)
        out.append(dict(description=desc, quantity=qty, unit_price=unit, amount=round(qty*unit,2)))
    return out
def totals(items, tr):
    sub=round(sum(i["amount"] for i in items),2); tax=round(sub*tr,2); return sub,tax,round(sub+tax,2)

# ------------------------------------------------------------------ invoice / PO templates
def _rows(items):
    return "".join(f"<tr><td>{esc(i['description'])}</td><td class='r'>{i['quantity']}</td>"
                   f"<td class='r'>{money(i['unit_price'])}</td><td class='r'>{money(i['amount'])}</td></tr>" for i in items)

def inv_html(layout, comp, x):
    font=x["_font"]; color=comp["color"]; accent=comp["accent"]; rows=_rows(x["line_items"])
    mono="".join(w[0] for w in comp["name"].split()[:2]).upper()
    base=f"<style>*{{box-sizing:border-box}}body{{font-family:{font};color:#222;margin:0;padding:40px;font-size:13px;background:#fff}}.r{{text-align:right}}table{{width:100%;border-collapse:collapse;margin-top:18px}}th,td{{padding:8px 10px}}.muted{{color:#666}}</style>"
    if layout==0:
        return base+f"""<div style="display:flex;justify-content:space-between;border-bottom:3px solid {color};padding-bottom:14px">
<div><div style="font-size:26px;font-weight:700;color:{color}">{esc(comp['name'])}</div><div class="muted">{esc(comp['tag'])}</div><div class="muted">{esc(addr_str(x['_vendor_addr']))}</div></div>
<div style="text-align:right"><div style="font-size:30px;color:{accent};font-weight:700">INVOICE</div><div><b>#{esc(x['invoice_number'])}</b></div><div class="muted">Date: {esc(d(x['_idate']))}</div><div class="muted">Due: {esc(d(x['_ddate']))}</div></div></div>
<div style="display:flex;justify-content:space-between;margin-top:18px"><div><div class="muted">BILL TO</div><b>{esc(x['bill_to'])}</b><br><span class="muted">{esc(addr_str(x['_bill_addr']))}</span></div><div><div class="muted">PO NUMBER</div><b>{esc(x['po_number'])}</b><br><span class="muted">Terms: {esc(x['terms'])}</span></div></div>
<table><thead><tr style="background:{color};color:#fff"><th style="text-align:left">Description</th><th class="r">Qty</th><th class="r">Unit</th><th class="r">Amount</th></tr></thead><tbody>{rows}</tbody></table>
<table style="width:300px;margin-left:auto;margin-top:10px"><tr><td class="muted">Subtotal</td><td class="r">{money(x['subtotal'])}</td></tr><tr><td class="muted">Tax ({int(x['_taxrate']*100)}%)</td><td class="r">{money(x['tax'])}</td></tr><tr><td style="border-top:2px solid {color};font-weight:700">Total</td><td class="r" style="border-top:2px solid {color};font-weight:700;color:{accent}">{money(x['total'])}</td></tr></table>
<p class="muted" style="margin-top:30px">Remit to {esc(comp['name'])}. Thank you for your business.</p>"""
    if layout==1:
        return base+f"""<div style="background:{color};color:#fff;padding:22px 24px;border-radius:8px;display:flex;justify-content:space-between;align-items:center">
<div style="display:flex;align-items:center;gap:14px"><div style="width:52px;height:52px;border-radius:10px;background:{accent};display:flex;align-items:center;justify-content:center;font-weight:800;font-size:20px">{mono}</div><div><div style="font-size:20px;font-weight:700">{esc(comp['name'])}</div><div style="opacity:.85;font-size:12px">{esc(comp['tag'])}</div></div></div>
<div style="text-align:right"><div style="font-size:22px;letter-spacing:3px">INVOICE</div><div style="opacity:.9">{esc(x['invoice_number'])}</div></div></div>
<div style="display:flex;justify-content:space-between;margin-top:20px"><div><div class="muted">Billed To</div><b>{esc(x['bill_to'])}</b><br><span class="muted">{esc(addr_str(x['_bill_addr']))}</span></div><div style="text-align:right"><div class="muted">Issued {esc(d(x['_idate']))}</div><div class="muted">Due {esc(d(x['_ddate']))}</div><div class="muted">PO {esc(x['po_number'])} &middot; {esc(x['terms'])}</div></div></div>
<table><thead><tr style="border-bottom:2px solid {accent}"><th style="text-align:left">Item</th><th class="r">Qty</th><th class="r">Rate</th><th class="r">Amount</th></tr></thead><tbody>{rows}</tbody></table>
<div style="display:flex;justify-content:flex-end;margin-top:14px"><div style="width:280px"><div style="display:flex;justify-content:space-between;padding:4px 0" class="muted">Subtotal<span>{money(x['subtotal'])}</span></div><div style="display:flex;justify-content:space-between;padding:4px 0" class="muted">Tax ({int(x['_taxrate']*100)}%)<span>{money(x['tax'])}</span></div><div style="display:flex;justify-content:space-between;padding:10px 0;border-top:2px solid {color};font-weight:800;color:{color}">Total Due<span>{money(x['total'])}</span></div></div></div>"""
    return base+f"""<div style="border-left:6px solid {accent};padding-left:16px"><div style="font-size:22px;font-weight:700">{esc(comp['name'])}</div><div class="muted">{esc(comp['tag'])} &middot; {esc(addr_str(x['_vendor_addr']))}</div></div>
<div style="margin-top:24px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px"><div><div class="muted" style="font-size:11px;text-transform:uppercase">Invoice</div><b>{esc(x['invoice_number'])}</b></div><div><div class="muted" style="font-size:11px;text-transform:uppercase">Date / Due</div>{esc(d(x['_idate']))} &rarr; {esc(d(x['_ddate']))}</div><div><div class="muted" style="font-size:11px;text-transform:uppercase">PO / Terms</div>{esc(x['po_number'])} &middot; {esc(x['terms'])}</div></div>
<div style="margin-top:14px"><span class="muted">Bill to:</span> <b>{esc(x['bill_to'])}</b>, {esc(addr_str(x['_bill_addr']))}</div>
<table><thead><tr style="border-bottom:1px solid #333"><th style="text-align:left">Description</th><th class="r">Qty</th><th class="r">Unit</th><th class="r">Amount</th></tr></thead><tbody>{rows}</tbody></table>
<div style="margin-top:12px;text-align:right"><div class="muted">Subtotal {money(x['subtotal'])} | Tax {money(x['tax'])}</div><div style="font-size:18px;font-weight:800;color:{accent};margin-top:6px">TOTAL {money(x['total'])}</div></div>"""

def po_html(layout, buyer, x):
    font=x["_font"]; color=buyer["color"]; accent=buyer["accent"]; rows=_rows(x["line_items"])
    base=f"<style>*{{box-sizing:border-box}}body{{font-family:{font};color:#222;margin:0;padding:40px;font-size:13px}}.r{{text-align:right}}table{{width:100%;border-collapse:collapse;margin-top:16px}}th,td{{padding:8px 10px}}.muted{{color:#666}}</style>"
    if layout==0:
        return base+f"""<div style="display:flex;justify-content:space-between;border-bottom:2px solid {color};padding-bottom:12px"><div><div style="font-size:24px;font-weight:700;color:{color}">{esc(buyer['name'])}</div><div class="muted">{esc(addr_str(x['_buyer_addr']))}</div></div><div style="text-align:right"><div style="font-size:26px;font-weight:800;color:{accent}">PURCHASE ORDER</div><div><b>PO #{esc(x['po_number'])}</b></div><div class="muted">{esc(d(x['_pdate']))}</div></div></div>
<div style="display:flex;justify-content:space-between;margin-top:16px"><div><div class="muted">VENDOR</div><b>{esc(x['vendor'])}</b><br><span class="muted">{esc(addr_str(x['_vendor_addr']))}</span></div><div><div class="muted">SHIP TO</div><b>{esc(buyer['name'])}</b><br><span class="muted">{esc(addr_str(x['_ship_addr']))}</span></div><div><div class="muted">DELIVER BY</div><b>{esc(d(x['_deliver']))}</b><br><span class="muted">{esc(x['terms'])}</span></div></div>
<table><thead><tr style="background:{color};color:#fff"><th style="text-align:left">Description</th><th class="r">Qty</th><th class="r">Unit</th><th class="r">Amount</th></tr></thead><tbody>{rows}</tbody></table>
<table style="width:300px;margin-left:auto;margin-top:10px"><tr><td class="muted">Subtotal</td><td class="r">{money(x['subtotal'])}</td></tr><tr><td class="muted">Tax</td><td class="r">{money(x['tax'])}</td></tr><tr><td style="font-weight:800;border-top:2px solid {color}">Total</td><td class="r" style="font-weight:800;border-top:2px solid {color}">{money(x['total'])}</td></tr></table>
<p class="muted" style="margin-top:24px">Authorized by Procurement &middot; {esc(buyer['name'])}</p>"""
    return base+f"""<div style="background:{color};color:#fff;padding:18px 22px;display:flex;justify-content:space-between;align-items:center;border-radius:6px"><div style="font-weight:700;font-size:20px">{esc(buyer['name'])}</div><div style="text-align:right"><div style="letter-spacing:2px">PURCHASE ORDER</div><div>{esc(x['po_number'])} &middot; {esc(d(x['_pdate']))}</div></div></div>
<div style="display:flex;justify-content:space-between;margin-top:18px"><div><div class="muted">Vendor</div><b>{esc(x['vendor'])}</b><br><span class="muted">{esc(addr_str(x['_vendor_addr']))}</span></div><div style="text-align:right"><div class="muted">Deliver by {esc(d(x['_deliver']))}</div><div class="muted">Terms {esc(x['terms'])}</div></div></div>
<table><thead><tr style="border-bottom:2px solid {accent}"><th style="text-align:left">Item</th><th class="r">Qty</th><th class="r">Unit</th><th class="r">Amount</th></tr></thead><tbody>{rows}</tbody></table>
<div style="text-align:right;margin-top:12px"><span class="muted">Subtotal {money(x['subtotal'])} | Tax {money(x['tax'])}</span><div style="font-weight:800;color:{color};font-size:18px">TOTAL {money(x['total'])}</div></div>"""

# ------------------------------------------------------------------ resume data
FIRST=["James","Maria","David","Aisha","Carlos","Emily","Wei","Priya","Michael","Sofia","Andre","Nia",
       "Ethan","Olivia","Raj","Hannah","Marcus","Lena","Diego","Grace","Tyler","Fatima","Noah","Chloe",
       "Ibrahim","Zoe","Ryan","Amara","Kevin","Isabella","Omar","Yuki","Brandon","Leah","Samuel","Rosa",
       "Derek","Naomi","Victor","Elena"]
LAST=["Reyes","Thompson","Nguyen","Patel","Okafor","Sullivan","Kim","Alvarez","Bennett","Rossi","Cohen",
      "Mwangi","Fischer","Delgado","Walsh","Haddad","Brooks","Ivanova","Santos","Park","Carter","Ali",
      "Murphy","Chen","Dubois","Silva","Foster","Ahmed","Wright","Romano","Khan","Tanaka","Green","Weber",
      "Morales","Jackson","Novak","Osei","Petrov","Vega"]
SCHOOLS=["Penn State University","University of Florida","Ohio State University","Arizona State University",
         "University of Texas","Georgia Tech","Purdue University","University of Washington",
         "Rutgers University","Michigan State University"]
DEGREES=["B.S. Computer Science","B.S. Information Technology","B.A. Business Administration",
         "B.S. Human Resource Management","B.S. Information Systems","B.B.A. Management","M.S. Data Science",
         "B.S. Management Information Systems"]

ROLE_DATA = {
 "developer": dict(label="Software Developer",
   titles=["Software Engineer","Senior Software Engineer","Backend Developer","Full-Stack Developer","Software Engineer II"],
   companies=["Onyx Software Labs","Nimbus Media","Cascade Analytics","Riverstone Financial","BluePeak Cloud Services","Copperline Retail"],
   skills=["Python","JavaScript","TypeScript","React","Node.js","PostgreSQL","Docker","Kubernetes","AWS","REST APIs","Git","CI/CD","Redis","GraphQL","pytest","Microservices"],
   certs=["AWS Certified Developer – Associate","Certified Kubernetes Application Developer"],
   bullets=["Built and shipped {f} microservices handling {n}M requests/month.","Reduced API p95 latency by {p}% by refactoring the {f} data layer.",
            "Led migration of the {f} monolith to containerized services on AWS.","Designed REST APIs consumed by {n} internal teams.",
            "Introduced automated testing raising coverage to {p}%.","Mentored {n} junior engineers and ran code reviews."]),
 "management": dict(label="Manager",
   titles=["Engineering Manager","Director of Operations","Program Manager","IT Manager","Senior Project Manager"],
   companies=["Atlas Construction","Keystone Insurance","Peak Ridge Manufacturing","Elmwood Property Group","Fairfield Utilities","Summit Fabrication"],
   skills=["Team Leadership","Agile/Scrum","Budgeting","Roadmapping","Stakeholder Management","Hiring","OKRs","Vendor Management","P&L","Risk Management","Jira","Change Management"],
   certs=["PMP","Certified ScrumMaster (CSM)","ITIL Foundation"],
   bullets=["Managed a team of {n} across {f} product lines, delivering on schedule and under budget.","Owned a ${n}M budget and improved margin by {p}%.",
            "Introduced OKRs and cut delivery cycle time by {p}%.","Grew the department from {n} to {m} staff over two years.",
            "Led {f} cross-functional initiatives with executive stakeholders.","Reduced turnover by {p}% through mentoring and career pathing."]),
 "hr": dict(label="HR Professional",
   titles=["HR Business Partner","Talent Acquisition Specialist","HR Generalist","People Operations Manager","Recruiter"],
   companies=["Bright Harbor Academy","Blue Sage Hospitality","Greenfield Foods","Harbor Point Medical","Sterling & Rowe LLP","Keystone Insurance"],
   skills=["Recruiting","Onboarding","Employee Relations","HRIS (Workday)","Benefits Administration","Performance Management","FMLA/ADA Compliance","Compensation","DEI Programs","Conflict Resolution","ATS","Payroll"],
   certs=["SHRM-CP","PHR (Professional in Human Resources)","Workday HCM Certification"],
   bullets=["Filled {n} requisitions per quarter with a {p}-day average time-to-hire.","Rolled out an onboarding program improving 90-day retention by {p}%.",
            "Managed employee relations for {n} staff across {f} sites.","Administered benefits and open enrollment for {n} employees.",
            "Launched a DEI initiative increasing diverse hires by {p}%.","Implemented an HRIS migration to Workday for {n} employees."]),
 "rpa": dict(label="RPA Developer",
   titles=["RPA Developer","Senior RPA Developer","Automation Engineer","UiPath Developer","RPA Consultant"],
   companies=["Keystone Insurance","Riverstone Financial","Fairfield Utilities","Peak Ridge Manufacturing","Harbor Point Medical","Atlas Construction"],
   skills=["UiPath Studio","UiPath Orchestrator","REFramework","Document Understanding","VB.NET","C#","Python","SQL","REST APIs","Attended/Unattended Bots","Queues & Assets","Process Discovery","Git","Automation Anywhere"],
   certs=["UiPath Certified Advanced RPA Developer (UiARD)","UiPath Certified RPA Associate","Automation Anywhere Advanced"],
   bullets=["Developed {n} production UiPath automations, eliminating {p} hours/week of manual work.","Built Document Understanding workflows for classification and extraction across {n} document types.",
            "Designed REFramework-based bots with Orchestrator queues, retries, and monitoring.","Automated {f} finance processes across legacy systems without APIs.",
            "Stood up an RPA Center of Excellence: standards, reusable components, and governance.","Reduced processing errors by {p}% through exception handling and validation."]),
}

def bullet(b):
    return b.replace("{n}",str(random.randint(3,40))).replace("{m}",str(random.randint(20,80))) \
            .replace("{p}",str(random.randint(15,60))).replace("{f}",random.choice(["four","several","key","five","core","multiple"]))

def make_person(role):
    rd=ROLE_DATA[role]
    fn=random.choice(FIRST); ln=random.choice(LAST)
    name=f"{fn} {ln}"
    yrs=random.randint(3,18)
    ncompanies=random.randint(2,4)
    end=2026; hist=[]
    titles=random.sample(rd["titles"], min(ncompanies,len(rd["titles"])))
    comps=random.sample(rd["companies"], min(ncompanies,len(rd["companies"])))
    for i in range(ncompanies):
        span=random.randint(1,5); start=end-span
        hist.append(dict(company=comps[i], title=titles[i], start_year=start, end_year=("Present" if i==0 else end),
                         bullets=[bullet(x) for x in random.sample(rd["bullets"], random.randint(2,3))]))
        end=start
    c=random.choice(CITIES)
    return dict(name=name, first=fn, last=ln, target_role=rd["label"],
        email=f"{fn.lower()}.{ln.lower()}@email.com", phone=phone(),
        location=f"{c[0]}, {c[1]}", years_experience=yrs,
        current_title=hist[0]["title"], skills=random.sample(rd["skills"], random.randint(7,10)),
        work_history=hist, education=dict(degree=random.choice(DEGREES), school=random.choice(SCHOOLS), year=2026-yrs-random.randint(0,3)),
        certifications=random.sample(rd["certs"], random.randint(1,min(2,len(rd["certs"])))),
        summary=f"{rd['label']} with {yrs}+ years of experience delivering results across {random.choice(['enterprise','fast-growth','regulated','mid-market'])} environments.")

def _skills_html(p): return "".join(f"<li>{esc(s)}</li>" for s in p["skills"])
def _exp_html(p, bullet_tag="li"):
    out=""
    for h in p["work_history"]:
        bl="".join(f"<{bullet_tag}>{esc(b)}</{bullet_tag}>" for b in h["bullets"])
        out+=f"<div class='job'><div class='jtop'><b>{esc(h['title'])}</b> &middot; {esc(h['company'])}<span class='dates'>{h['start_year']} &ndash; {h['end_year']}</span></div><ul>{bl}</ul></div>"
    return out

def resume_html(layout, p, font, color):
    contact=f"{esc(p['email'])} &middot; {esc(p['phone'])} &middot; {esc(p['location'])}"
    edu=f"{esc(p['education']['degree'])}, {esc(p['education']['school'])} ({p['education']['year']})"
    certs=", ".join(esc(c) for c in p["certifications"])
    skills=_skills_html(p); exp=_exp_html(p)
    B=f"<style>*{{box-sizing:border-box}}body{{font-family:{font};color:#1a1a1a;margin:0;padding:44px;font-size:13px;line-height:1.5}}h1{{margin:0}}ul{{margin:6px 0}}.dates{{float:right;color:#666;font-weight:400}}.job{{margin-bottom:12px}}.jtop{{border-bottom:1px solid #eee;padding-bottom:2px}} a{{color:inherit}}</style>"
    if layout==0:  # classic centered serif
        return B+f"""<div style="text-align:center;border-bottom:2px solid {color};padding-bottom:10px"><h1 style="font-size:28px;letter-spacing:1px">{esc(p['name'])}</h1><div style="color:{color};font-weight:600">{esc(p['current_title'])}</div><div style="color:#555;font-size:12px;margin-top:4px">{contact}</div></div>
<h3 style="color:{color}">Summary</h3><p>{esc(p['summary'])}</p><h3 style="color:{color}">Experience</h3>{exp}<h3 style="color:{color}">Skills</h3><ul style="columns:2">{skills}</ul><h3 style="color:{color}">Education & Certifications</h3><p>{edu}<br>{certs}</p>"""
    if layout==1:  # two-column left sidebar
        return B+f"""<div style="display:flex;gap:24px"><aside style="width:230px;background:{color};color:#fff;padding:22px;border-radius:8px"><h1 style="font-size:22px">{esc(p['name'])}</h1><div style="opacity:.9">{esc(p['current_title'])}</div><hr style="border-color:rgba(255,255,255,.3)"><div style="font-size:12px;line-height:1.9">{esc(p['email'])}<br>{esc(p['phone'])}<br>{esc(p['location'])}</div><h4>Skills</h4><ul style="padding-left:16px">{skills}</ul><h4>Certifications</h4><div style="font-size:12px">{certs}</div></aside>
<main style="flex:1"><h3 style="color:{color}">Profile</h3><p>{esc(p['summary'])}</p><h3 style="color:{color}">Experience</h3>{exp}<h3 style="color:{color}">Education</h3><p>{edu}</p></main></div>"""
    if layout==2:  # modern header band
        return B+f"""<div style="background:{color};color:#fff;padding:24px;border-radius:8px;margin:-44px -44px 20px"><h1 style="font-size:26px">{esc(p['name'])}</h1><div style="opacity:.9">{esc(p['current_title'])} &middot; {esc(p['location'])}</div><div style="opacity:.8;font-size:12px;margin-top:4px">{esc(p['email'])} &middot; {esc(p['phone'])}</div></div>
<p>{esc(p['summary'])}</p><h3 style="border-left:4px solid {color};padding-left:8px">Experience</h3>{exp}<div style="display:flex;gap:30px"><div style="flex:1"><h3 style="border-left:4px solid {color};padding-left:8px">Skills</h3><ul>{skills}</ul></div><div style="flex:1"><h3 style="border-left:4px solid {color};padding-left:8px">Education</h3><p>{edu}</p><p>{certs}</p></div></div>"""
    if layout==3:  # minimalist
        return B+f"""<h1 style="font-size:30px;font-weight:300">{esc(p['name'])}</h1><div style="letter-spacing:3px;text-transform:uppercase;color:#888;font-size:12px">{esc(p['current_title'])}</div><div style="color:#666;font-size:12px;margin:6px 0 20px">{contact}</div>
<p style="color:#444">{esc(p['summary'])}</p><div style="border-top:1px solid #ddd;margin:16px 0"></div><h4 style="letter-spacing:2px;text-transform:uppercase;font-size:12px;color:#888">Experience</h4>{exp}<h4 style="letter-spacing:2px;text-transform:uppercase;font-size:12px;color:#888">Skills</h4><p>{', '.join(esc(s) for s in p['skills'])}</p><h4 style="letter-spacing:2px;text-transform:uppercase;font-size:12px;color:#888">Education</h4><p>{edu} &middot; {certs}</p>"""
    if layout==4:  # right sidebar
        return B+f"""<div style="display:flex;gap:24px"><main style="flex:1"><h1 style="color:{color};font-size:26px">{esc(p['name'])}</h1><div style="font-weight:600">{esc(p['current_title'])}</div><p style="margin-top:10px">{esc(p['summary'])}</p><h3 style="color:{color}">Experience</h3>{exp}</main>
<aside style="width:210px;border-left:3px solid {color};padding-left:18px"><h4>Contact</h4><div style="font-size:12px;line-height:1.9">{esc(p['email'])}<br>{esc(p['phone'])}<br>{esc(p['location'])}</div><h4>Skills</h4><ul style="padding-left:16px">{skills}</ul><h4>Education</h4><div style="font-size:12px">{edu}</div><h4>Certifications</h4><div style="font-size:12px">{certs}</div></aside></div>"""
    if layout==5:  # compact dense
        return B.replace("font-size:13px","font-size:11.5px")+f"""<div style="display:flex;justify-content:space-between;align-items:baseline"><h1 style="font-size:22px">{esc(p['name'])}</h1><span style="color:#666;font-size:11px">{contact}</span></div><div style="color:{color};font-weight:700">{esc(p['current_title'])}</div><p style="margin:6px 0">{esc(p['summary'])}</p>
<b style="color:{color}">EXPERIENCE</b>{exp}<b style="color:{color}">SKILLS</b> {', '.join(esc(s) for s in p['skills'])}<br><b style="color:{color}">EDUCATION</b> {edu} | {certs}"""
    if layout==6:  # timeline / date rail
        jobs=""
        for h in p["work_history"]:
            bl="".join(f"<li>{esc(b)}</li>" for b in h["bullets"])
            jobs+=f"<div style='display:flex;gap:16px;margin-bottom:12px'><div style='width:90px;color:{color};font-weight:700;font-size:12px'>{h['start_year']}&ndash;{h['end_year']}</div><div style='flex:1'><b>{esc(h['title'])}</b>, {esc(h['company'])}<ul>{bl}</ul></div></div>"
        return B+f"""<h1 style="font-size:26px">{esc(p['name'])}</h1><div style="color:{color};font-weight:600">{esc(p['current_title'])}</div><div style="color:#666;font-size:12px">{contact}</div><p style="margin-top:12px">{esc(p['summary'])}</p><h3 style="color:{color}">Career Timeline</h3>{jobs}<h3 style="color:{color}">Skills & Education</h3><p>{', '.join(esc(s) for s in p['skills'])}</p><p>{edu} &middot; {certs}</p>"""
    # layout 7: boxed cards
    return B+f"""<div style="background:#f6f6f6;padding:20px;border-radius:8px;border-left:6px solid {color}"><h1 style="font-size:26px;margin:0">{esc(p['name'])}</h1><div style="color:{color};font-weight:700">{esc(p['current_title'])}</div><div style="color:#666;font-size:12px">{contact}</div></div>
<div style="margin-top:14px;padding:14px;border:1px solid #eee;border-radius:8px"><b style="color:{color}">Summary</b><p>{esc(p['summary'])}</p></div>
<div style="margin-top:12px;padding:14px;border:1px solid #eee;border-radius:8px"><b style="color:{color}">Experience</b>{exp}</div>
<div style="display:flex;gap:12px;margin-top:12px"><div style="flex:1;padding:14px;border:1px solid #eee;border-radius:8px"><b style="color:{color}">Skills</b><ul>{skills}</ul></div><div style="flex:1;padding:14px;border:1px solid #eee;border-radius:8px"><b style="color:{color}">Education</b><p>{edu}</p><b style="color:{color}">Certifications</b><p>{certs}</p></div></div>"""

RESUME_COLORS=["#1f3a5f","#0f766e","#3730a3","#166534","#b45309","#a21caf","#334155","#0e7490","#7c2d12","#4338ca"]

# ------------------------------------------------------------------ forms
EMPLOYERS=["Cascade Analytics","Peak Ridge Manufacturing","Harbor Point Medical","Keystone Insurance",
           "Atlas Construction","Riverstone Financial","Greenfield Foods","Nimbus Media","Onyx Software Labs","Fairfield Utilities"]
DEPARTMENTS=["Finance","Operations","Engineering","Human Resources","Sales","Customer Support","Logistics","IT","Marketing","Legal"]
POSITIONS=["Accountant","Operations Analyst","Software Engineer","HR Generalist","Account Executive",
           "Support Specialist","Warehouse Lead","Systems Administrator","Marketing Coordinator","Paralegal"]
BANKS=["First National Bank","Coastal Credit Union","Summit Savings Bank","Meridian Bank","Cornerstone Financial","Harborview Credit Union"]
CLAIM_TYPES=["Auto","Property","Health","General Liability","Workers Compensation"]
CLAIM_STATUS=["Open","Under Review","Approved","Denied","Pending Documentation"]
LOAN_TYPES=["Mortgage","Auto","Personal","Business","Student","Home Equity"]
LOAN_PURPOSE=["Home purchase","Vehicle purchase","Debt consolidation","Working capital","Tuition","Home improvement","Medical expenses"]
TAX_CLASS=["Individual/sole proprietor","C Corporation","S Corporation","Partnership","Trust/estate","Limited liability company"]
FILING=["Single","Married filing jointly","Head of household"]
EMPTYPE=["Full-time","Part-time","Contract","Temporary"]
PAYFREQ=["Weekly","Bi-weekly","Semi-monthly","Monthly"]

def ssn(): return f"{random.randint(100,899):03d}-{random.randint(10,99):02d}-{random.randint(1000,9999)}"
def ein(): return f"{random.randint(10,99)}-{random.randint(1000000,9999999)}"
def routing(): return str(random.randint(100000000,999999999))
def acct(): return str(random.randint(1000000,999999999999))
def bdate(minage=22,maxage=62):
    y=2026-random.randint(minage,maxage); return datetime.date(y,random.randint(1,12),random.randint(1,28))
def usd(x): return "${:,.2f}".format(x)
def ck(v): return ("&#9745; " if v else "&#9744; ")  # checked / empty box

def render_form(meta, sections, variant, font, color):
    """sections: list of (section_title, [(label, value_html), ...])"""
    head=(f"<div style='display:flex;justify-content:space-between;align-items:flex-start;"
          f"border-bottom:3px solid {color};padding-bottom:10px;margin-bottom:16px'>"
          f"<div><div style='font-size:12px;color:#666'>{esc(meta['org'])}</div>"
          f"<div style='font-size:22px;font-weight:800;color:{color}'>{esc(meta['title'])}</div>"
          f"<div style='font-size:11px;color:#888'>{esc(meta.get('subtitle',''))}</div></div>"
          f"<div style='text-align:right;font-size:11px'><div style='border:1px solid {color};padding:4px 8px;border-radius:4px'>"
          f"<b>Form {esc(meta['form_no'])}</b></div></div></div>")
    css=(f"<style>*{{box-sizing:border-box}}body{{font-family:{font};color:#1a1a1a;margin:0;padding:40px;font-size:12.5px}}"
         f".sec{{margin-bottom:14px}}.st{{font-weight:700;color:{color};text-transform:uppercase;font-size:11px;"
         f"letter-spacing:.5px;border-bottom:1px solid #ddd;padding-bottom:3px;margin:14px 0 8px}}"
         f".fl{{color:#555}}.fv{{font-weight:600}}</style>")
    body=""
    for stitle, fields in sections:
        body+=f"<div class='sec'><div class='st'>{esc(stitle)}</div>"
        if variant==0:  # single column, label : value rows
            for lab,val in fields:
                body+=(f"<div style='display:flex;padding:4px 0;border-bottom:1px dotted #e5e5e5'>"
                       f"<span class='fl' style='width:230px'>{esc(lab)}</span><span class='fv'>{val}</span></div>")
        elif variant==1:  # two-column boxed cells
            body+="<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px'>"
            for lab,val in fields:
                body+=(f"<div style='border:1px solid #ddd;border-radius:5px;padding:6px 8px'>"
                       f"<div class='fl' style='font-size:10px;text-transform:uppercase'>{esc(lab)}</div>"
                       f"<div class='fv'>{val}</div></div>")
            body+="</div>"
        else:  # official bordered rows
            body+="<table style='width:100%;border-collapse:collapse'>"
            for lab,val in fields:
                body+=(f"<tr><td style='border:1px solid #999;padding:6px 8px;width:240px;background:#f3f3f3;color:#333'>{esc(lab)}</td>"
                       f"<td style='border:1px solid #999;padding:6px 8px;font-weight:600'>{val}</td></tr>")
            body+="</table>"
        body+="</div>"
    sign=(f"<div style='margin-top:22px;display:flex;justify-content:space-between;border-top:1px solid #ccc;padding-top:12px'>"
          f"<div>Signature: <b>{esc(meta.get('signature',''))}</b></div><div>Date: <b>{esc(meta.get('sign_date',''))}</b></div></div>"
          if meta.get('signature') else "")
    return f"<!doctype html><html><head><meta charset='utf-8'>{css}</head><body>{head}{body}{sign}</body></html>"

def opts(all_opts, chosen):  # render a checkbox option list, one chosen
    return " &nbsp; ".join(ck(o==chosen)+esc(o) for o in all_opts)

def make_onboarding(i):
    fn=random.choice(FIRST); ln=random.choice(LAST); a=addr(); dobd=bdate(); start=datetime.date(2026,random.randint(1,12),random.randint(1,28))
    L=dict(form_type="onboarding", employee_name=f"{fn} {ln}", ssn=ssn(), date_of_birth=iso(dobd),
        home_address=addr_str(a), personal_email=f"{fn.lower()}.{ln.lower()}@email.com", phone=phone(),
        job_title=random.choice(POSITIONS), department=random.choice(DEPARTMENTS),
        manager=f"{random.choice(FIRST)} {random.choice(LAST)}", start_date=iso(start),
        employment_type=random.choice(EMPTYPE), pay_rate=round(random.uniform(45000,145000),2),
        pay_frequency=random.choice(PAYFREQ), bank_name=random.choice(BANKS),
        bank_routing=routing(), bank_account=acct(), w4_filing_status=random.choice(FILING),
        allowances=random.randint(0,4), emergency_contact_name=f"{random.choice(FIRST)} {random.choice(LAST)}",
        emergency_contact_phone=phone(), i9_verified=random.random()>0.2, handbook_acknowledged=random.random()>0.15)
    meta=dict(org=random.choice(EMPLOYERS), title="New Hire Onboarding Form", subtitle="Confidential — Human Resources",
              form_no=f"HR-ONB-{1000+i}", signature=L["employee_name"], sign_date=start.strftime("%m/%d/%Y"))
    sections=[("Employee Information",[("Full Legal Name",esc(L["employee_name"])),("Social Security No.",esc(L["ssn"])),
                ("Date of Birth",dobd.strftime("%m/%d/%Y")),("Home Address",esc(L["home_address"])),
                ("Personal Email",esc(L["personal_email"])),("Phone",esc(L["phone"]))]),
        ("Position",[("Job Title",esc(L["job_title"])),("Department",esc(L["department"])),("Reporting Manager",esc(L["manager"])),
                ("Start Date",start.strftime("%m/%d/%Y")),("Employment Type",opts(EMPTYPE,L["employment_type"])),
                ("Annual Pay Rate",usd(L["pay_rate"])),("Pay Frequency",esc(L["pay_frequency"]))]),
        ("Direct Deposit",[("Bank Name",esc(L["bank_name"])),("Routing Number",esc(L["bank_routing"])),("Account Number",esc(L["bank_account"]))]),
        ("Federal Withholding (W-4)",[("Filing Status",opts(FILING,L["w4_filing_status"])),("Dependents / Allowances",str(L["allowances"]))]),
        ("Emergency Contact",[("Name",esc(L["emergency_contact_name"])),("Phone",esc(L["emergency_contact_phone"]))]),
        ("Acknowledgements",[("I-9 Employment Eligibility Verified",ck(L["i9_verified"])+("Yes" if L["i9_verified"] else "No")),
                ("Employee Handbook Received",ck(L["handbook_acknowledged"])+("Yes" if L["handbook_acknowledged"] else "No"))])]
    return meta, sections, L

def make_claim(i):
    fn=random.choice(FIRST); ln=random.choice(LAST); ctype=random.choice(CLAIM_TYPES)
    loss=datetime.date(2026,random.randint(1,10),random.randint(1,28)); rep=loss+datetime.timedelta(days=random.randint(0,20))
    amt=round(random.uniform(500,45000),2); ded=round(random.choice([250,500,1000,2500]),2)
    L=dict(form_type="claim", claim_number=f"CLM-{random.randint(200000,299999)}", policy_number=f"POL-{random.randint(1000000,9999999)}",
        claimant_name=f"{fn} {ln}", claim_type=ctype, date_of_loss=iso(loss), date_reported=iso(rep),
        incident_location=addr_str(addr()), loss_description=random.choice(
            ["Rear-end collision at intersection","Water damage from burst pipe","Theft of equipment from premises",
             "Slip and fall in parking lot","Storm damage to roof","Fire damage to inventory","Vehicle vandalism"]),
        claim_amount=amt, deductible=ded, adjuster_name=f"{random.choice(FIRST)} {random.choice(LAST)}",
        status=random.choice(CLAIM_STATUS), contact_phone=phone())
    meta=dict(org=random.choice(["Keystone Insurance","Riverstone Financial","Harbor Point Medical"]),
              title=f"{ctype} Insurance Claim Form", subtitle="Claims Department", form_no=f"CL-{2000+i}",
              signature=L["claimant_name"], sign_date=rep.strftime("%m/%d/%Y"))
    sections=[("Claim Details",[("Claim Number",esc(L["claim_number"])),("Policy Number",esc(L["policy_number"])),
                ("Claim Type",opts(CLAIM_TYPES,ctype)),("Status",opts(CLAIM_STATUS,L["status"]))]),
        ("Claimant",[("Name",esc(L["claimant_name"])),("Contact Phone",esc(L["contact_phone"])),("Incident Location",esc(L["incident_location"]))]),
        ("Loss Information",[("Date of Loss",loss.strftime("%m/%d/%Y")),("Date Reported",rep.strftime("%m/%d/%Y")),
                ("Description",esc(L["loss_description"]))]),
        ("Financials",[("Amount Claimed",usd(amt)),("Deductible",usd(ded)),("Assigned Adjuster",esc(L["adjuster_name"]))])]
    return meta, sections, L

def make_w9(i):
    indiv=random.random()>0.5; fn=random.choice(FIRST); ln=random.choice(LAST); a=addr()
    name=f"{fn} {ln}"; biz=("" if indiv else random.choice(["","Blue Sage Hospitality LLC","Summit Fabrication Inc","Copperline Retail Co"]))
    tc=random.choice(TAX_CLASS); use_ssn=indiv or random.random()>0.5
    L=dict(form_type="w9", name=name, business_name=biz, tax_classification=tc,
        address=a["line1"], city_state_zip=f"{a['city']}, {a['state']} {a['zip']}",
        tin_type=("SSN" if use_ssn else "EIN"), ssn=(ssn() if use_ssn else ""), ein=("" if use_ssn else ein()),
        requester=random.choice(EMPLOYERS))
    signd=datetime.date(2026,random.randint(1,12),random.randint(1,28))
    meta=dict(org="Internal Revenue Service", title="Form W-9", subtitle="Request for Taxpayer Identification Number and Certification",
              form_no="W-9", signature=name, sign_date=signd.strftime("%m/%d/%Y"))
    tin_row = ("Social Security Number", esc(L["ssn"])) if use_ssn else ("Employer Identification Number (EIN)", esc(L["ein"]))
    sections=[("Identification",[("Name (as shown on your income tax return)",esc(name)),
                ("Business name/disregarded entity name",esc(biz) or "&mdash;".replace("&mdash;","-")),
                ("Federal tax classification",opts(TAX_CLASS,tc))]),
        ("Address",[("Address (number, street)",esc(L["address"])),("City, state, ZIP",esc(L["city_state_zip"])),
                ("Requester's name",esc(L["requester"]))]),
        ("Part I — Taxpayer Identification Number (TIN)",[("TIN Type",opts(["SSN","EIN"],L["tin_type"])), tin_row])]
    return meta, sections, L

def make_w4(i):
    fn=random.choice(FIRST); ln=random.choice(LAST); a=addr(); fs=random.choice(FILING)
    L=dict(form_type="w4", name=f"{fn} {ln}", ssn=ssn(), address=addr_str(a), filing_status=fs,
        multiple_jobs=random.random()>0.6, dependents_amount=random.choice([0,2000,4000,6000]),
        other_income=random.choice([0,0,1500,5000]), deductions=random.choice([0,0,3000,12000]),
        extra_withholding=random.choice([0,0,50,150]))
    signd=datetime.date(2026,random.randint(1,12),random.randint(1,28))
    meta=dict(org="Internal Revenue Service", title="Form W-4", subtitle="Employee's Withholding Certificate",
              form_no="W-4", signature=L["name"], sign_date=signd.strftime("%m/%d/%Y"))
    sections=[("Step 1 — Personal Information",[("Name",esc(L["name"])),("Social Security Number",esc(L["ssn"])),
                ("Address",esc(L["address"])),("Filing Status",opts(FILING,fs))]),
        ("Step 2",[("Multiple Jobs / Spouse Works",ck(L["multiple_jobs"])+("Yes" if L["multiple_jobs"] else "No"))]),
        ("Steps 3-4 — Adjustments",[("Claim Dependents ($)",usd(L["dependents_amount"])),("Other Income ($)",usd(L["other_income"])),
                ("Deductions ($)",usd(L["deductions"])),("Extra Withholding ($)",usd(L["extra_withholding"]))])]
    return meta, sections, L

def make_loan(i):
    fn=random.choice(FIRST); ln=random.choice(LAST); a=addr(); lt=random.choice(LOAN_TYPES); dobd=bdate(25,60)
    amt=round(random.uniform(5000,650000),2); term=random.choice([12,24,36,48,60,120,180,360])
    L=dict(form_type="loan", application_number=f"LN-{random.randint(500000,599999)}", applicant_name=f"{fn} {ln}",
        ssn=ssn(), date_of_birth=iso(dobd), address=addr_str(a), phone=phone(), email=f"{fn.lower()}.{ln.lower()}@email.com",
        employer=random.choice(EMPLOYERS), job_title=random.choice(POSITIONS), years_employed=random.randint(1,20),
        annual_income=round(random.uniform(38000,220000),2), loan_type=lt, loan_amount=amt, loan_term_months=term,
        loan_purpose=random.choice(LOAN_PURPOSE), down_payment=round(amt*random.choice([0,0.05,0.1,0.2]),2),
        monthly_debt=round(random.uniform(200,4200),2), credit_score=random.randint(580,820),
        co_applicant_name=(f"{random.choice(FIRST)} {random.choice(LAST)}" if random.random()>0.6 else ""))
    meta=dict(org=random.choice(["Cornerstone Financial","Meridian Bank","First National Bank"]),
              title=f"{lt} Loan Application", subtitle="Consumer Lending", form_no=f"LA-{5000+i}",
              signature=L["applicant_name"], sign_date=datetime.date(2026,random.randint(1,12),random.randint(1,28)).strftime("%m/%d/%Y"))
    sections=[("Applicant",[("Full Name",esc(L["applicant_name"])),("Social Security No.",esc(L["ssn"])),
                ("Date of Birth",dobd.strftime("%m/%d/%Y")),("Address",esc(L["address"])),("Phone",esc(L["phone"])),("Email",esc(L["email"]))]),
        ("Employment & Income",[("Employer",esc(L["employer"])),("Job Title",esc(L["job_title"])),
                ("Years Employed",str(L["years_employed"])),("Annual Income",usd(L["annual_income"])),
                ("Monthly Debt Payments",usd(L["monthly_debt"])),("Credit Score",str(L["credit_score"]))]),
        ("Loan Request",[("Application No.",esc(L["application_number"])),("Loan Type",opts(LOAN_TYPES,lt)),
                ("Amount Requested",usd(L["loan_amount"])),("Term (months)",str(L["loan_term_months"])),
                ("Purpose",esc(L["loan_purpose"])),("Down Payment",usd(L["down_payment"]))]),
        ("Co-Applicant",[("Name",esc(L["co_applicant_name"]) or "None")])]
    return meta, sections, L

FORM_KINDS=[("onboarding",make_onboarding),("claim",make_claim),("w9",make_w9),("w4",make_w4),("loan",make_loan)]
FORM_COLORS=["#1f3a5f","#7c2d12","#0f766e","#3730a3","#166534","#334155","#a21caf","#0e7490"]

# ------------------------------------------------------------------ irregularities (defect injection)
# Each defect mutates the document's DATA (so the rendered PDF and the label stay
# consistent) and returns a tag recorded under the label's "irregularities" list,
# so you can score whether your agent catches the problem.

def defect_invoice(x):
    irr=[]
    for defect in random.sample(["total_mismatch","subtotal_mismatch","tax_miscalculated",
            "line_item_math_error","missing_invoice_number","missing_bill_to","negative_line_amount"],
            random.randint(1,2)):
        if defect=="total_mismatch": x["total"]=round(x["total"]+random.choice([-1,1])*random.uniform(8,260),2)
        elif defect=="subtotal_mismatch": x["subtotal"]=round(x["subtotal"]+random.uniform(12,190),2)
        elif defect=="tax_miscalculated": x["tax"]=round(x["tax"]+random.uniform(6,95),2)
        elif defect=="line_item_math_error":
            it=random.choice(x["line_items"]); it["amount"]=round(it["amount"]+random.uniform(9,140),2)
        elif defect=="missing_invoice_number": x["invoice_number"]=""
        elif defect=="missing_bill_to": x["bill_to"]=""
        elif defect=="negative_line_amount":
            it=random.choice(x["line_items"]); it["amount"]=-abs(it["amount"])
        irr.append(defect)
    return sorted(set(irr))

def defect_po(x):
    irr=[]
    for defect in random.sample(["total_mismatch","subtotal_mismatch","line_item_math_error",
            "missing_po_number","missing_vendor","negative_line_amount"], random.randint(1,2)):
        if defect=="total_mismatch": x["total"]=round(x["total"]+random.choice([-1,1])*random.uniform(8,260),2)
        elif defect=="subtotal_mismatch": x["subtotal"]=round(x["subtotal"]+random.uniform(12,190),2)
        elif defect=="line_item_math_error":
            it=random.choice(x["line_items"]); it["amount"]=round(it["amount"]+random.uniform(9,140),2)
        elif defect=="missing_po_number": x["po_number"]=""
        elif defect=="missing_vendor": x["vendor"]=""
        elif defect=="negative_line_amount":
            it=random.choice(x["line_items"]); it["amount"]=-abs(it["amount"])
        irr.append(defect)
    return sorted(set(irr))

def defect_resume(p):
    irr=[]
    for defect in random.sample(["missing_email","missing_phone","no_skills_listed",
            "missing_employment_dates","impossible_employment_dates"], random.randint(1,2)):
        if defect=="missing_email": p["email"]=""
        elif defect=="missing_phone": p["phone"]=""
        elif defect=="no_skills_listed": p["skills"]=[]
        elif defect=="missing_employment_dates":
            h=random.choice(p["work_history"]); h["start_year"]=""; h["end_year"]=""
        elif defect=="impossible_employment_dates":
            h=random.choice(p["work_history"])
            ey=h["end_year"] if isinstance(h["end_year"],int) else 2024
            h["start_year"]=ey+random.randint(1,4)   # start AFTER it ended
        irr.append(defect)
    return sorted(set(irr))

def _set_field(sections, sub, display):
    for _st, fields in sections:
        for i in range(len(fields)):
            if sub.lower() in fields[i][0].lower():
                fields[i]=(fields[i][0], display); return True
    return False
def _bad_ssn(): return random.choice(["000-00-0000",
        f"{random.randint(10,99)}-{random.randint(100,999)}-{random.randint(10,99)}","123-45-678"])
def _uncheck(all_opts): return " &nbsp; ".join(ck(False)+esc(o) for o in all_opts)

def defect_form(ftype, meta, sections, L):
    pool=["missing_signature","missing_sign_date"]
    if ftype in ("onboarding","w4","loan"): pool.append("invalid_ssn_format")
    if ftype=="onboarding": pool+=["invalid_routing_number","missing_bank_account","no_filing_status_selected"]
    if ftype=="claim": pool+=["date_reported_before_loss","negative_claim_amount","missing_adjuster"]
    if ftype=="w9": pool+=["no_tax_classification_selected","missing_tin","no_tin_type_selected"]
    if ftype=="w4": pool.append("no_filing_status_selected")
    if ftype=="loan": pool+=["credit_score_out_of_range","negative_income","down_payment_exceeds_loan"]
    irr=[]
    for defect in random.sample(pool, random.randint(1,2)):
        if defect=="missing_signature": meta["signature"]=""
        elif defect=="missing_sign_date": meta["sign_date"]=""
        elif defect=="invalid_ssn_format":
            b=_bad_ssn(); _set_field(sections,"Social Security",esc(b)); L["ssn"]=b
        elif defect=="invalid_routing_number":
            b=str(random.randint(100,99999)); _set_field(sections,"Routing",esc(b)); L["bank_routing"]=b
        elif defect=="missing_bank_account":
            _set_field(sections,"Account Number",""); L["bank_account"]=""
        elif defect=="no_filing_status_selected":
            _set_field(sections,"Filing Status",_uncheck(FILING))
            L["w4_filing_status"]=""; L["filing_status"]=""
        elif defect=="date_reported_before_loss":
            y,m,dd=[int(z) for z in L.get("date_of_loss","2026-06-01").split("-")]
            e=datetime.date(y,m,dd)-datetime.timedelta(days=random.randint(5,40))
            _set_field(sections,"Date Reported",e.strftime("%m/%d/%Y")); L["date_reported"]=e.isoformat()
        elif defect=="negative_claim_amount":
            b=-abs(L["claim_amount"]); _set_field(sections,"Amount Claimed",usd(b)); L["claim_amount"]=b
        elif defect=="missing_adjuster":
            _set_field(sections,"Adjuster",""); L["adjuster_name"]=""
        elif defect=="no_tax_classification_selected":
            _set_field(sections,"Federal tax classification",_uncheck(TAX_CLASS)); L["tax_classification"]=""
        elif defect=="missing_tin":
            _set_field(sections,"Social Security Number",""); _set_field(sections,"Employer Identification","")
            L["ssn"]=""; L["ein"]=""
        elif defect=="no_tin_type_selected":
            _set_field(sections,"TIN Type",_uncheck(["SSN","EIN"])); L["tin_type"]=""
        elif defect=="credit_score_out_of_range":
            b=random.choice([905,1000,120]); _set_field(sections,"Credit Score",str(b)); L["credit_score"]=b
        elif defect=="negative_income":
            b=-abs(L["annual_income"]); _set_field(sections,"Annual Income",usd(b)); L["annual_income"]=b
        elif defect=="down_payment_exceeds_loan":
            b=round(L["loan_amount"]+random.uniform(1000,50000),2); _set_field(sections,"Down Payment",usd(b)); L["down_payment"]=b
        irr.append(defect)
    return sorted(set(irr))

# ------------------------------------------------------------------ multi-bill invoices
# One invoice, several separately-payable sections: a utility billing water and gas
# together, a carrier billing three shipments, a landlord billing four sites. AP has to
# raise a separate internal invoice per section, so each one needs its own account
# reference and its own total -- and the section totals have to roll up to the invoice.
COST_CENTERS = ["CC-1010 Facilities","CC-2040 Operations","CC-3300 Engineering","CC-4120 Logistics",
                "CC-5005 Administration","CC-6180 Manufacturing","CC-7250 Retail Ops","CC-8090 IT"]

MULTIBILL_VENDORS = [
    dict(name="Cascade Municipal Utilities", tag="Water &middot; Sewer &middot; Gas &middot; Stormwater",
         color="#0e7490", accent="#b45309", inv_prefix="CMU", acct_prefix="UTL", ref_label="Meter",
         multisite=False, services=[
        dict(name="Water Service", code="WTR", items=[("Water consumption, per 1,000 gal",3.10,4.90),
             ("Base service charge",18.00,32.00),("Cross-connection control fee",4.50,9.00)]),
        dict(name="Sewer Service", code="SWR", items=[("Sewer volume charge, per 1,000 gal",4.20,6.60),
             ("Sewer base charge",22.00,38.00),("Treatment surcharge",11.00,26.00)]),
        dict(name="Natural Gas", code="GAS", items=[("Natural gas, per therm",0.62,1.35),
             ("Distribution charge",14.00,29.00),("Meter service charge",7.50,12.00)]),
        dict(name="Electric Service", code="ELC", items=[("Electricity, per kWh",0.09,0.19),
             ("Demand charge, per kW",9.00,17.00),("Grid access fee",12.00,24.00)]),
        dict(name="Stormwater", code="STM", items=[("Impervious surface fee",6.00,14.00),
             ("Watershed maintenance",8.00,19.00)])]),
    dict(name="Meridian Telecom Partners", tag="Voice &middot; Data &middot; Mobile &middot; Equipment",
         color="#3730a3", accent="#f59e0b", inv_prefix="MTP", acct_prefix="BTN", ref_label="Circuit",
         multisite=False, services=[
        dict(name="Business Voice Lines", code="VOX", items=[("Business line, per line/mo",28.00,44.00),
             ("Long distance, per minute",0.03,0.09),("Voicemail seat, per user/mo",3.00,7.00)]),
        dict(name="Dedicated Data Circuit", code="DAT", items=[("Fiber 500 Mbps, per month",420.00,780.00),
             ("Static IP block /29",25.00,45.00),("Circuit port fee",60.00,120.00)]),
        dict(name="Mobile Fleet", code="MOB", items=[("Mobile line, per line/mo",31.00,52.00),
             ("Data overage, per GB",8.00,15.00),("Device protection, per line/mo",5.00,11.00)]),
        dict(name="Equipment Lease", code="EQP", items=[("Managed router lease",45.00,95.00),
             ("Handset lease, per unit",6.00,14.00),("On-site maintenance, per hr",95.00,165.00)])]),
    dict(name="Ironleaf Facility Services", tag="Grounds &middot; Janitorial &middot; Waste, by site",
         color="#166534", accent="#a16207", inv_prefix="IFS", acct_prefix="SITE", ref_label="Route",
         multisite=True, services=[
        dict(name="Refuse &amp; Recycling", code="WST", items=[("Front-load pickup, per haul",88.00,145.00),
             ("Recycling stream, per haul",62.00,110.00),("Container rental, per month",45.00,85.00)]),
        dict(name="Janitorial", code="JAN", items=[("Nightly cleaning, per visit",120.00,240.00),
             ("Floor care, per sq ft",0.06,0.14),("Consumables restock",55.00,130.00)]),
        dict(name="Grounds Maintenance", code="GRD", items=[("Lawn maintenance, per visit",85.00,150.00),
             ("Seasonal cleanup",180.00,320.00),("Irrigation check, per hr",58.00,92.00)]),
        dict(name="Snow &amp; Ice", code="SNW", items=[("Plow event, per push",210.00,395.00),
             ("Ice melt application",95.00,180.00)])]),
    dict(name="Vanguard Freight Systems", tag="Freight billing by shipment",
         color="#334155", accent="#0e7490", inv_prefix="VFS", acct_prefix="SHP", ref_label="BOL",
         multisite=True, services=[
        dict(name="LTL Shipment", code="LTL", items=[("Linehaul charge",340.00,880.00),
             ("Fuel surcharge",48.00,140.00),("Liftgate service",65.00,95.00),("Detention, per hr",55.00,85.00)]),
        dict(name="Truckload Shipment", code="TL", items=[("Linehaul, per mile",2.10,3.40),
             ("Fuel surcharge",95.00,260.00),("Layover",250.00,400.00)]),
        dict(name="Expedited Shipment", code="EXP", items=[("Expedited linehaul",620.00,1450.00),
             ("After-hours delivery",120.00,240.00),("Fuel surcharge",70.00,190.00)]),
        dict(name="Warehousing", code="WHS", items=[("Pallet storage, per pallet/mo",14.00,26.00),
             ("Inbound handling, per pallet",9.00,18.00),("Pick and pack, per order",2.50,6.00)])]),
    dict(name="Summit Workforce Group", tag="Contract labour by cost centre",
         color="#7c2d12", accent="#0f766e", inv_prefix="SWG", acct_prefix="CTR", ref_label="Contract",
         multisite=True, services=[
        dict(name="Engineering Contractors", code="ENG", items=[("Regular hours, per hr",78.00,125.00),
             ("Overtime hours, per hr",112.00,180.00),("Agency fee",240.00,520.00)]),
        dict(name="Warehouse Temps", code="WHT", items=[("Regular hours, per hr",21.00,32.00),
             ("Overtime hours, per hr",31.00,47.00),("Background screening, per hire",38.00,72.00)]),
        dict(name="Administrative Support", code="ADM", items=[("Regular hours, per hr",26.00,41.00),
             ("Temp-to-hire conversion fee",900.00,2200.00)]),
        dict(name="IT Support Contractors", code="ITS", items=[("Regular hours, per hr",68.00,110.00),
             ("On-call premium, per week",180.00,340.00)])]),
]

def _mb_ref(vendor, svc):
    if vendor["ref_label"] == "Meter":   return f"M{random.randint(1000000,9999999)}"
    if vendor["ref_label"] == "Circuit": return f"{svc['code']}/{random.randint(10000,99999)}/DS1"
    if vendor["ref_label"] == "BOL":     return f"BOL-{random.randint(2000000,2999999)}"
    if vendor["ref_label"] == "Route":   return f"R-{random.randint(100,999)}"
    return f"C-{random.randint(10000,99999)}"

def mb_sections(vendor, idate, n):
    """Each section is a separately-payable unit: own account, own period, own total."""
    picks = random.sample(vendor["services"], min(n, len(vendor["services"])))
    secs = []
    for k, svc in enumerate(picks):
        # meters get read on different days, shipments land on different days
        pe = idate - datetime.timedelta(days=random.randint(2, 11))
        ps = pe - datetime.timedelta(days=random.randint(27, 32))
        items = line_items(svc["items"], 2, 4)
        tr = random.choice([0.0, 0.0, 0.04, 0.06, 0.07, 0.0825])
        sub, tax, tot = totals(items, tr)
        secs.append(dict(section_index=k, service_type=svc["name"].replace("&amp;", "&"),
            service_code=svc["code"],
            account_number=f"{vendor['acct_prefix']}-{random.randint(100000,999999)}",
            reference_label=vendor["ref_label"], reference_number=_mb_ref(vendor, svc),
            service_location=(addr_str(addr()) if vendor["multisite"] else None),
            cost_center=random.choice(COST_CENTERS),
            service_period_start=iso(ps), service_period_end=iso(pe),
            line_items=items, subtotal=sub, tax=tax, total=tot, _taxrate=tr))
    return secs

def _mb_roll(x):
    """Invoice totals are the roll-up of the sections. Kept as a function so a defect
    can knock the roll-up out of agreement without touching the sections."""
    s = x["sections"]
    x["subtotal"] = round(sum(v["subtotal"] for v in s), 2)
    x["tax"] = round(sum(v["tax"] for v in s), 2)
    x["total"] = round(sum(v["total"] for v in s), 2)
    x["section_count"] = len(s)

def make_multibill(vendor, idx, base_date):
    idate = base_date + datetime.timedelta(days=random.randint(0, 150))
    term = random.choice(TERMS)
    dd = {"Net 15": 15, "Net 30": 30, "Net 45": 45, "Due on receipt": 0, "2/10 Net 30": 30}[term]
    x = dict(invoice_number=f"{vendor['inv_prefix']}-2026{7000+idx}",
             bill_to=random.choice(CUSTOMERS), terms=term,
             master_account=f"{vendor['acct_prefix']}-MASTER-{random.randint(10000,99999)}",
             remit_to=addr_str(addr()),
             sections=mb_sections(vendor, idate, random.randint(2, 4)),
             _idate=idate, _ddate=idate + datetime.timedelta(days=dd),
             _font=FONTS[idx % len(FONTS)], _vendor_addr=addr(), _bill_addr=addr())
    _mb_roll(x)
    return x

def _mb_rows(sec, color):
    return (f"<tr><td colspan='4' style='padding-top:12px;font-weight:700;color:{color}'>"
            f"{sec['service_type']} &middot; {esc(sec['account_number'])}</td></tr>" + _rows(sec["line_items"]) +
            f"<tr><td colspan='3' class='r muted'>Subtotal / Tax</td><td class='r'>"
            f"{money(sec['subtotal'])} / {money(sec['tax'])}</td></tr>"
            f"<tr><td colspan='3' class='r' style='font-weight:700'>{esc(sec['service_code'])} total</td>"
            f"<td class='r' style='font-weight:700'>{money(sec['total'])}</td></tr>")

def _mb_period(sec):
    if not sec["service_period_start"] and not sec["service_period_end"]:
        return "&mdash;"
    a = d(datetime.date.fromisoformat(sec["service_period_start"])) if sec["service_period_start"] else "?"
    b = d(datetime.date.fromisoformat(sec["service_period_end"])) if sec["service_period_end"] else "?"
    return f"{esc(a)} &ndash; {esc(b)}"

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


def mb_html(layout, vendor, x):
    font = x["_font"]; color = vendor["color"]; accent = vendor["accent"]
    base = (f"<style>*{{box-sizing:border-box}}body{{font-family:{font};color:#222;margin:0;padding:38px;"
            f"font-size:12.5px;background:#fff}}.r{{text-align:right}}table{{width:100%;border-collapse:collapse}}"
            f"th,td{{padding:6px 9px}}.muted{{color:#666}}.lbl{{font-size:10px;text-transform:uppercase;"
            f"letter-spacing:.06em;color:#777}}</style>")
    head = (f"<div style='display:flex;justify-content:space-between;border-bottom:3px solid {color};padding-bottom:12px'>"
            f"<div><div style='font-size:23px;font-weight:700;color:{color}'>{vendor['name']}</div>"
            f"<div class='muted'>{vendor['tag']}</div><div class='muted'>{esc(addr_str(x['_vendor_addr']))}</div></div>"
            f"<div style='text-align:right'><div style='font-size:26px;color:{accent};font-weight:700'>INVOICE</div>"
            f"<div><b>#{esc(x['invoice_number'])}</b></div><div class='muted'>Date: {esc(d(x['_idate']))}</div>"
            f"<div class='muted'>Due: {esc(d(x['_ddate']))}</div></div></div>"
            f"<div style='display:flex;justify-content:space-between;margin-top:14px'>"
            f"<div><div class='lbl'>Bill to</div><b>{esc(x['bill_to'])}</b><br>"
            f"<span class='muted'>{esc(addr_str(x['_bill_addr']))}</span></div>"
            f"<div><div class='lbl'>Master account</div><b>{esc(x['master_account'])}</b><br>"
            f"<span class='muted'>Terms: {esc(x['terms'])}</span></div>"
            f"<div style='text-align:right'><div class='lbl'>Services billed</div>"
            f"<b style='font-size:17px;color:{accent}'>{x['section_count']}</b>"
            f"<div class='muted'>pay each separately</div></div></div>")
    note = (f"<p class='muted' style='margin-top:22px'>Each service above is billed to its own account and "
            f"may be remitted separately. Reference the account number shown for that service when paying. "
            f"Remit to {vendor['name']}, {esc(x['remit_to'])}.</p>")

    if layout == 0:      # summary table with a column per identifier, then detail
        summ = "".join(
            f"<tr><td><b>{esc(s['service_type'])}</b></td><td>{esc(s['service_code'])}</td>"
            f"<td>{esc(s['account_number'])}</td>"
            f"<td>{esc(s['reference_label'])} {esc(s['reference_number'])}</td>"
            f"<td>{_mb_period(s)}</td><td>{esc(s['cost_center'])}</td>"
            f"<td class='r' style='font-weight:700'>{money(s['total'])}</td></tr>"
            for s in x["sections"])
        detail = "".join(
            f"<div style='margin-top:16px;border-left:4px solid {accent};padding-left:12px'>"
            f"<div style='font-weight:700;color:{color}'>{esc(s['service_type'])}</div>"
            f"<div class='muted' style='margin:2px 0'>{_mb_ident(s)}</div>"
            f"<div class='muted'>Service period {_mb_period(s)}"
            f"{(' &middot; ' + esc(s['service_location'])) if s['service_location'] else ''}</div>"
            f"<table><thead><tr style='border-bottom:1px solid #999'><th style='text-align:left'>Description</th>"
            f"<th class='r'>Qty</th><th class='r'>Unit</th><th class='r'>Amount</th></tr></thead>"
            f"<tbody>{_rows(s['line_items'])}</tbody></table>"
            f"<div class='r' style='margin-top:4px'>Subtotal {money(s['subtotal'])} &middot; Tax {money(s['tax'])} &middot; "
            f"<b style='color:{accent}'>Service total {money(s['total'])}</b></div></div>" for s in x["sections"])
        return base + head + (
            f"<table style='margin-top:18px'><thead><tr style='background:{color};color:#fff'>"
            f"<th style='text-align:left'>Service</th><th style='text-align:left'>Code</th>"
            f"<th style='text-align:left'>Account</th><th style='text-align:left'>Reference</th>"
            f"<th style='text-align:left'>Service period</th><th style='text-align:left'>Cost centre</th>"
            f"<th class='r'>Amount due</th></tr></thead><tbody>{summ}</tbody></table>"
            f"{detail}"
            f"<table style='width:320px;margin-left:auto;margin-top:16px'>"
            f"<tr><td class='muted'>Invoice subtotal</td><td class='r'>{money(x['subtotal'])}</td></tr>"
            f"<tr><td class='muted'>Invoice tax</td><td class='r'>{money(x['tax'])}</td></tr>"
            f"<tr><td style='border-top:2px solid {color};font-weight:700'>Total due</td>"
            f"<td class='r' style='border-top:2px solid {color};font-weight:700;color:{accent}'>"
            f"{money(x['total'])}</td></tr></table>" + note)

    if layout == 1:      # boxed per-service statements, one labelled line per identifier
        cards = "".join(
            f"<div style='border:1px solid #ccc;border-top:5px solid {color};margin-top:14px;padding:12px 14px'>"
            f"<div style='font-size:15px;font-weight:700;color:{color}'>{esc(s['service_type'])}</div>"
            f"<table style='width:auto;margin-top:6px'>"
            f"<tr><td class='lbl'>Service code</td><td><b>{esc(s['service_code'])}</b></td>"
            f"<td class='lbl' style='padding-left:22px'>Account</td><td><b>{esc(s['account_number'])}</b></td></tr>"
            f"<tr><td class='lbl'>{esc(s['reference_label'])}</td><td><b>{esc(s['reference_number'])}</b></td>"
            f"<td class='lbl' style='padding-left:22px'>Cost centre</td><td>{esc(s['cost_center'])}</td></tr>"
            f"<tr><td class='lbl'>Service period</td><td colspan='3'>{_mb_period(s)}"
            f"{(' &middot; ' + esc(s['service_location'])) if s['service_location'] else ''}</td></tr>"
            f"</table>"
            f"<table style='margin-top:8px'><thead><tr style='border-bottom:1px solid #bbb'>"
            f"<th style='text-align:left'>Description</th><th class='r'>Qty</th><th class='r'>Unit</th>"
            f"<th class='r'>Amount</th></tr></thead><tbody>{_rows(s['line_items'])}</tbody></table>"
            f"<div style='display:flex;justify-content:flex-end;gap:18px;margin-top:6px'>"
            f"<span class='muted'>Subtotal {money(s['subtotal'])}</span>"
            f"<span class='muted'>Tax {money(s['tax'])}</span>"
            f"<span style='font-weight:800;color:{accent}'>Due {money(s['total'])}</span></div></div>"
            for s in x["sections"])
        return base + head + cards + (
            f"<div style='display:flex;justify-content:flex-end;margin-top:18px'><div style='width:300px'>"
            f"<div style='display:flex;justify-content:space-between' class='muted'>Sum of services"
            f"<span>{money(x['subtotal'])}</span></div>"
            f"<div style='display:flex;justify-content:space-between' class='muted'>Tax"
            f"<span>{money(x['tax'])}</span></div>"
            f"<div style='display:flex;justify-content:space-between;border-top:2px solid {color};"
            f"padding-top:8px;font-weight:800;color:{color}'>Total due<span>{money(x['total'])}</span></div>"
            f"</div></div>" + note)

    # layout 2: a labelled block per service, then one continuous ledger
    legend = "".join(
        f"<div style='border-bottom:1px dotted #bbb;padding:5px 0'>"
        f"<b>{esc(s['service_type'])}</b><br><span class='muted'>{_mb_ident(s)}</span><br>"
        f"<span class='muted'><span class='lbl'>Period</span> {_mb_period(s)}"
        f"{(' &nbsp; <span class=\'lbl\'>Site</span> ' + esc(s['service_location'])) if s['service_location'] else ''}"
        f"</span></div>"
        for s in x["sections"])
    body = "".join(_mb_rows(s, color) for s in x["sections"])
    return base + head + (
        f"<div style='margin-top:16px;border:1px solid #ddd;padding:10px 12px'>"
        f"<div class='lbl' style='margin-bottom:4px'>Separately payable services</div>{legend}</div>"
        f"<table style='margin-top:10px'><thead><tr style='border-bottom:2px solid {color}'>"
        f"<th style='text-align:left'>Description</th><th class='r'>Qty</th><th class='r'>Unit</th>"
        f"<th class='r'>Amount</th></tr></thead><tbody>{body}</tbody></table>"
        f"<div style='margin-top:12px;text-align:right'>"
        f"<div class='muted'>Invoice subtotal {money(x['subtotal'])} | Tax {money(x['tax'])}</div>"
        f"<div style='font-size:17px;font-weight:800;color:{accent};margin-top:5px'>"
        f"TOTAL DUE {money(x['total'])}</div></div>" + note)


def defect_multibill(x):
    """Defects specific to split billing: the roll-up disagrees, or a section cannot be
    routed to its own internal invoice because its identifier is missing or duplicated."""
    pool = ["section_total_mismatch", "invoice_total_not_sum_of_sections", "duplicate_section_account",
            "missing_section_account", "missing_section_period", "overlapping_service_periods",
            "negative_section_total", "section_count_mismatch", "section_line_item_math_error",
            "missing_invoice_number", "missing_bill_to"]
    if len(x["sections"]) < 2:
        pool = [p for p in pool if p not in ("duplicate_section_account", "overlapping_service_periods")]
    # Order matters. Anything that mutates a section re-rolls the invoice totals, which
    # would quietly repair a header defect applied earlier -- so the two defects that
    # exist purely as a disagreement in the header are always applied last.
    order = ["section_line_item_math_error", "section_total_mismatch", "negative_section_total",
             "missing_section_account", "duplicate_section_account", "overlapping_service_periods",
             "missing_section_period", "missing_invoice_number", "missing_bill_to",
             "invoice_total_not_sum_of_sections", "section_count_mismatch"]
    irr = []
    locked = set()          # sections an earlier defect needs left intact
    for defect in sorted(random.sample(pool, random.randint(1, 2)), key=order.index):
        secs = x["sections"]
        if defect == "section_total_mismatch":
            s = random.choice(secs)
            s["total"] = round(s["total"] + random.choice([-1, 1]) * random.uniform(9, 180), 2)
            _mb_roll(x)                      # the invoice still foots to the printed sections
        elif defect == "invoice_total_not_sum_of_sections":
            x["total"] = round(x["total"] + random.choice([-1, 1]) * random.uniform(15, 320), 2)
        elif defect == "duplicate_section_account":
            have = [i for i in range(len(secs)) if secs[i]["account_number"]]
            if len(have) < 2:
                continue
            a, b = random.sample(have, 2)
            secs[b]["account_number"] = secs[a]["account_number"]
        elif defect == "missing_section_account":
            secs[random.choice([i for i in range(len(secs))])]["account_number"] = ""
        elif defect == "missing_section_period":
            free = [i for i in range(len(secs)) if i not in locked]
            if not free:                 # an overlap defect owns every section; skip rather
                continue                 # than record a tag the document does not show
            s = secs[random.choice(free)]
            s["service_period_start"] = ""; s["service_period_end"] = ""
        elif defect == "overlapping_service_periods":
            have = [i for i in range(len(secs)) if secs[i]["service_period_start"]]
            if len(have) < 2:
                continue
            a, b = random.sample(have, 2)
            secs[b]["service_period_start"] = secs[a]["service_period_start"]
            secs[b]["service_period_end"] = secs[a]["service_period_end"]
            locked |= {a, b}
        elif defect == "negative_section_total":
            s = random.choice(secs); s["total"] = -abs(s["total"]); _mb_roll(x)
        elif defect == "section_count_mismatch":
            x["section_count"] = len(secs) + random.choice([-1, 1, 1])
        elif defect == "section_line_item_math_error":
            s = random.choice(secs); it = random.choice(s["line_items"])
            it["amount"] = round(it["amount"] + random.uniform(8, 130), 2)
        elif defect == "missing_invoice_number":
            x["invoice_number"] = ""
        elif defect == "missing_bill_to":
            x["bill_to"] = ""
        irr.append(defect)
    return sorted(set(irr))

# ------------------------------------------------------------------ main
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="change this for a fresh, unseen set")
    ap.add_argument("--out", default=None,
                    help="output dataset root (default: dataset root, or <root>/irregular with --irregular)")
    ap.add_argument("--irregular", action="store_true",
                    help="inject defects (unsigned, total mismatch, invalid SSN, impossible dates, etc.)")
    ap.add_argument("--invoices-per-company", type=int, default=8)
    ap.add_argument("--pos-per-company", type=int, default=8)
    ap.add_argument("--resumes", type=int, default=40)
    ap.add_argument("--multibill", type=int, default=40,
                    help="invoices carrying several separately-payable services on one document")
    ap.add_argument("--forms-per-type", type=int, default=40,
                    help="forms per type: onboarding, claim, government (W-9/W-4 mix), loan")
    a=ap.parse_args()
    random.seed(a.seed)
    base_root=os.environ.get("DI_DATASET_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
    OUT=a.out or (os.path.join(base_root,"irregular") if a.irregular else base_root)
    for sub in ["source_html/invoices","source_html/purchase_orders","source_html/resumes","source_html/forms",
                "source_html/multi_bill_invoices","labels",
                "invoices","purchase_orders","resumes","forms","multi_bill_invoices"]:
        os.makedirs(os.path.join(OUT,sub), exist_ok=True)

    base_date=datetime.date(2026,3,1)
    inv_labels=[]; po_labels=[]; res_labels=[]; mb_labels=[]
    ic=1000
    for ci,comp in enumerate(COMPANIES):
        slug=comp["name"].split()[0].lower()
        for n in range(a.invoices_per_company):
            idate=base_date+datetime.timedelta(days=random.randint(0,150)); term=random.choice(TERMS)
            dd={"Net 15":15,"Net 30":30,"Net 45":45,"Due on receipt":0,"2/10 Net 30":30}[term]
            items=line_items(comp["items"]); tr=random.choice([0.0,0.06,0.07,0.0825]); sub,tax,tot=totals(items,tr)
            invno=f"INV-2026{ic}"; ic+=1
            x=dict(invoice_number=invno,bill_to=random.choice(CUSTOMERS),po_number=f"PO-{random.randint(40000,49999)}",
                   terms=term,line_items=items,subtotal=sub,tax=tax,total=tot,_idate=idate,
                   _ddate=idate+datetime.timedelta(days=dd),_taxrate=tr,_font=FONTS[ci%len(FONTS)],
                   _vendor_addr=addr(),_bill_addr=addr())
            irr=defect_invoice(x) if a.irregular else []
            fn=f"{slug}_{invno}"
            open(os.path.join(OUT,"source_html/invoices",fn+".html"),"w",encoding="utf-8").write(
                f"<!doctype html><html><head><meta charset='utf-8'></head><body>{inv_html(n%3,comp,x)}</body></html>")
            inv_labels.append(dict(file=f"invoices/{fn}.pdf",doc_type="invoice",vendor_name=comp["name"],
                invoice_number=x["invoice_number"],invoice_date=iso(x["_idate"]),due_date=iso(x["_ddate"]),po_number=x["po_number"],
                bill_to=x["bill_to"],terms=term,currency="USD",subtotal=x["subtotal"],tax=x["tax"],total=x["total"],
                line_items=[{k:i[k] for k in("description","quantity","unit_price","amount")} for i in x["line_items"]],
                irregularities=irr))
    pc=70000
    for ci,buyer in enumerate(COMPANIES):
        slug=buyer["name"].split()[0].lower()
        for n in range(a.pos_per_company):
            pdate=base_date+datetime.timedelta(days=random.randint(0,150))
            supplier=random.choice([c for c in COMPANIES if c is not buyer]); items=line_items(supplier["items"])
            tr=random.choice([0.0,0.06,0.07]); sub,tax,tot=totals(items,tr); pono=f"PO-{pc}"; pc+=1; term=random.choice(TERMS)
            x=dict(po_number=pono,vendor=supplier["name"],terms=term,line_items=items,subtotal=sub,tax=tax,total=tot,
                   _pdate=pdate,_deliver=pdate+datetime.timedelta(days=random.randint(7,45)),_font=FONTS[(ci+3)%len(FONTS)],
                   _buyer_addr=addr(),_vendor_addr=addr(),_ship_addr=addr())
            irr=defect_po(x) if a.irregular else []
            fn=f"{slug}_{pono}"
            open(os.path.join(OUT,"source_html/purchase_orders",fn+".html"),"w",encoding="utf-8").write(
                f"<!doctype html><html><head><meta charset='utf-8'></head><body>{po_html(n%2,buyer,x)}</body></html>")
            po_labels.append(dict(file=f"purchase_orders/{fn}.pdf",doc_type="purchase_order",buyer=buyer["name"],
                vendor=x["vendor"],po_number=x["po_number"],po_date=iso(pdate),delivery_date=iso(x["_deliver"]),terms=term,
                currency="USD",subtotal=x["subtotal"],tax=x["tax"],total=x["total"],
                line_items=[{k:i[k] for k in("description","quantity","unit_price","amount")} for i in x["line_items"]],
                irregularities=irr))
    roles=["management","developer","hr","rpa"]
    for i in range(a.resumes):
        role=roles[i%len(roles)]; p=make_person(role)
        irr=defect_resume(p) if a.irregular else []
        layout=i%8; font=FONTS[i%len(FONTS)]; color=RESUME_COLORS[i%len(RESUME_COLORS)]
        fn=f"{p['first'].lower()}_{p['last'].lower()}_{role}_{i:02d}"
        open(os.path.join(OUT,"source_html/resumes",fn+".html"),"w",encoding="utf-8").write(
            f"<!doctype html><html><head><meta charset='utf-8'></head><body>{resume_html(layout,p,font,color)}</body></html>")
        lab={k:p[k] for k in("name","email","phone","location","target_role","current_title","years_experience","skills","certifications")}
        lab.update(file=f"resumes/{fn}.pdf",doc_type="resume",layout=layout,
                   education=p["education"],
                   work_history=[{k:h[k] for k in("company","title","start_year","end_year")} for h in p["work_history"]],
                   irregularities=irr)
        res_labels.append(lab)

    # forms: 4 requested types; government = W-9 / W-4 mix
    N=a.forms_per_type
    plan=[make_onboarding]*N + [make_claim]*N \
         + [(make_w9 if k%2==0 else make_w4) for k in range(N)] + [make_loan]*N
    form_labels=[]
    for fi,maker in enumerate(plan):
        meta,sections,L=maker(fi)
        kind=L["form_type"]
        irr=defect_form(kind,meta,sections,L) if a.irregular else []
        variant=fi%3; font=FONTS[fi%len(FONTS)]; color=FORM_COLORS[fi%len(FORM_COLORS)]
        doc=render_form(meta,sections,variant,font,color)
        fn=f"{kind}_{5000+fi:04d}"
        open(os.path.join(OUT,"source_html/forms",fn+".html"),"w",encoding="utf-8").write(doc)
        rec=dict(file=f"forms/{fn}.pdf", doc_type="form", form_type=kind, layout=variant)
        rec.update({k:v for k,v in L.items() if k!="form_type"})
        rec["irregularities"]=irr
        form_labels.append(rec)

    # multi-bill invoices: one document, several separately-payable services
    for i in range(a.multibill):
        vendor = MULTIBILL_VENDORS[i % len(MULTIBILL_VENDORS)]
        x = make_multibill(vendor, i, base_date)
        irr = defect_multibill(x) if a.irregular else []
        layout = i % 3
        slug = vendor["name"].split()[0].lower()
        fn = f"{slug}_{x['invoice_number'] or 'NOINV-%04d' % i}"
        open(os.path.join(OUT,"source_html/multi_bill_invoices",fn+".html"),"w",encoding="utf-8").write(
            f"<!doctype html><html><head><meta charset='utf-8'></head><body>{mb_html(layout,vendor,x)}</body></html>")
        mb_labels.append(dict(file=f"multi_bill_invoices/{fn}.pdf", doc_type="multi_bill_invoice",
            layout=layout, vendor_name=vendor["name"], invoice_number=x["invoice_number"],
            invoice_date=iso(x["_idate"]), due_date=iso(x["_ddate"]), bill_to=x["bill_to"],
            master_account=x["master_account"], terms=x["terms"], currency="USD",
            section_count=x["section_count"],
            sections=[{k:v for k,v in sec.items() if not k.startswith("_")} for sec in x["sections"]],
            subtotal=x["subtotal"], tax=x["tax"], total=x["total"], irregularities=irr))

    json.dump(inv_labels,open(os.path.join(OUT,"labels","invoices.json"),"w"),indent=2)
    json.dump(po_labels,open(os.path.join(OUT,"labels","purchase_orders.json"),"w"),indent=2)
    json.dump(res_labels,open(os.path.join(OUT,"labels","resumes.json"),"w"),indent=2)
    json.dump(form_labels,open(os.path.join(OUT,"labels","forms.json"),"w"),indent=2)
    json.dump(mb_labels,open(os.path.join(OUT,"labels","multi_bill_invoices.json"),"w"),indent=2)
    from collections import Counter
    fc=Counter(r["form_type"] for r in form_labels)
    mode="IRREGULAR (defects injected)" if a.irregular else "clean"
    n_irr=sum(1 for L in (inv_labels+po_labels+res_labels+form_labels+mb_labels) if L.get("irregularities"))
    print(f"mode={mode}  seed={a.seed}  out={OUT}")
    print(f"invoices={len(inv_labels)} purchase_orders={len(po_labels)} resumes={len(res_labels)} forms={len(form_labels)} {dict(fc)}")
    n_sec=sum(L["section_count"] for L in mb_labels)
    print(f"multi_bill_invoices={len(mb_labels)} (sections billed: {n_sec})")
    if a.irregular: print(f"documents with >=1 injected defect: {n_irr}/{len(inv_labels)+len(po_labels)+len(res_labels)+len(form_labels)+len(mb_labels)}")
    print(f"HTML written. Render with: python render_pdfs.py --out {OUT}")

if __name__=="__main__":
    main()
