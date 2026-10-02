"""
frontend/components/theme.py
============================
Design system for Retent Engram: one CSS block + small HTML helpers so every
page looks like part of the same app.

Usage (in any page):
    from frontend.components.theme import page_header, stat_card, pill, ...
"""
import streamlit as st

# ---------------------------------------------------------------------------
# Palette (kept in one place — charts.py imports these too)
# ---------------------------------------------------------------------------
INK      = "#0F1F3D"
MUTED    = "#64748B"
LINE     = "#E3E9F3"
NAVY     = "#0B2A5B"
PRIMARY  = "#2563EB"
GREEN    = "#16A34A"
AMBER    = "#F59E0B"
RED      = "#EF4444"
PURPLE   = "#7C3AED"
TEAL     = "#0E9F8E"

TONES = {
    #          text       soft bg     border
    "green":  ("#15803D", "#E8F8EE", "#BFE9CE"),
    "amber":  ("#B45309", "#FFF5DD", "#F6DFA4"),
    "red":    ("#DC2626", "#FDECEC", "#F7C6C6"),
    "blue":   ("#1D4ED8", "#EAF1FF", "#C7D9FB"),
    "purple": ("#6D28D9", "#F1EAFE", "#DDCDFA"),
    "teal":   ("#0F766E", "#E2F6F3", "#B4E5DE"),
    "gray":   ("#475569", "#F1F5F9", "#DDE4EE"),
}

# ---------------------------------------------------------------------------
# Icons (lucide-style, 24x24, stroke based). Inline SVG => no external files.
# ---------------------------------------------------------------------------
_ICONS = {
    "trending-up": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "calendar":    '<rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/><path d="m9 16 2 2 4-4"/>',
    "target":      '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "file-text":   '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M16 13H8M16 17H8M10 9H8"/>',
    "layers":      '<path d="m12 2 10 5-10 5L2 7z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>',
    "code":        '<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>',
    "monitor":     '<rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/>',
    "cpu":         '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>',
    "database":    '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.7 4 3 9 3s9-1.3 9-3V5"/><path d="M3 12c0 1.7 4 3 9 3s9-1.3 9-3"/>',
    "braces":      '<path d="M8 3H7a2 2 0 0 0-2 2v5a2 2 0 0 1-2 2 2 2 0 0 1 2 2v5a2 2 0 0 0 2 2h1M16 3h1a2 2 0 0 1 2 2v5a2 2 0 0 0 2 2 2 2 0 0 0-2 2v5a2 2 0 0 1-2 2h-1"/>',
    "network":     '<rect x="16" y="16" width="6" height="6" rx="1"/><rect x="2" y="16" width="6" height="6" rx="1"/><rect x="9" y="2" width="6" height="6" rx="1"/><path d="M5 16v-3a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3M12 12V8"/>',
    "tree":        '<circle cx="12" cy="5" r="2.5"/><circle cx="6" cy="19" r="2.5"/><circle cx="18" cy="19" r="2.5"/><path d="M12 7.5V12M12 12 6 16.5M12 12l6 4.5"/>',
    "search":      '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "repeat":      '<path d="m17 2 4 4-4 4"/><path d="M3 11v-1a4 4 0 0 1 4-4h14M7 22l-4-4 4-4"/><path d="M21 13v1a4 4 0 0 1-4 4H3"/>',
    "hard-drive":  '<path d="M22 12H2"/><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/><path d="M6 16h.01M10 16h.01"/>',
    "sparkles":    '<path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3z"/>',
    "flame":       '<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.07-2.14-.22-4.05 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.15.43-2.29 1-3a2.5 2.5 0 0 0 2.5 2.5z"/>',
    "clock":       '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "book":        '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
    "check":       '<path d="M20 6 9 17l-5-5"/>',
    "inbox":       '<polyline points="22 12 16 12 14 15 10 15 8 12 2 12"/><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/>',
    "upload":      '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>',
    "alert":       '<path d="m21.7 18-8-14a2 2 0 0 0-3.4 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.7-3z"/><path d="M12 9v4M12 17h.01"/>',
    "zap":         '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "award":       '<circle cx="12" cy="8" r="6"/><path d="M15.5 13.1 17 22l-5-3-5 3 1.5-8.9"/>',
}

def icon(name: str, size: int = 22, color: str = "currentColor", stroke: float = 2.0) -> str:
    body = _ICONS.get(name, _ICONS["file-text"])
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" '
            f'stroke-linecap="round" stroke-linejoin="round">{body}</svg>')

# concept_id -> (icon, tone) so each subject gets a recognisable tile
CONCEPT_ICONS = {
    "os": ("monitor", "blue"), "dbms": ("database", "amber"), "cn": ("network", "teal"),
    "dsa": ("tree", "purple"), "python_oop": ("braces", "green"),
    "process_mgmt": ("cpu", "blue"), "memory_mgmt": ("hard-drive", "purple"),
    "sql": ("database", "teal"), "recursion": ("repeat", "amber"), "binary_search": ("search", "red"),
}
def concept_icon(cid: str):
    return CONCEPT_ICONS.get(cid, ("book", "gray"))

# ---------------------------------------------------------------------------
# Logo
# ---------------------------------------------------------------------------
LOGO_MARK = """
<svg width="38" height="38" viewBox="0 0 38 38" xmlns="http://www.w3.org/2000/svg" aria-label="Retent Engram">
  <defs><linearGradient id="rtg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#0B2A5B"/><stop offset="1" stop-color="#2F6FEB"/></linearGradient></defs>
  <rect width="38" height="38" rx="11" fill="url(#rtg)"/>
  <g stroke="#BFD6FF" stroke-width="1.6" stroke-linecap="round" opacity=".9">
    <path d="M12 14.5 20 11.5 27 17.5 18.5 26.5 12 14.5"/><path d="M20 11.5 18.5 26.5"/></g>
  <g fill="#fff"><circle cx="12" cy="14.5" r="3"/><circle cx="20" cy="11.5" r="3"/>
    <circle cx="27" cy="17.5" r="3"/><circle cx="18.5" cy="26.5" r="3"/></g>
  <circle cx="27" cy="17.5" r="1.4" fill="#2F6FEB"/>
</svg>"""

# ---------------------------------------------------------------------------
# HTML helper.
#
# NOTE: we deliberately use st.markdown(..., unsafe_allow_html=True) here and
# NOT st.html(). st.html() runs its own DOMPurify pass that strips <svg> (and
# its children) entirely, which silently blanks every icon in this file. The
# markdown path's sanitizer allows svg, and because every snippet below starts
# with a recognised block tag (<div>, <span>, ...), CommonMark treats the
# whole multi-line, indented string as one raw HTML block rather than
# re-parsing the indentation as a code block — so this is safe even though
# the f-strings inherit Python's own indentation.
# ---------------------------------------------------------------------------
def html(s: str):
    st.markdown(s, unsafe_allow_html=True)

def card(key: str, **kw):
    """A white rounded card. Keys must be unique per page."""
    return st.container(key=f"card_{key}", **kw)

# ---------------------------------------------------------------------------
# Reusable blocks
# ---------------------------------------------------------------------------
def pill(text: str, tone: str = "gray", dot: bool = False) -> str:
    fg, bg, bd = TONES[tone]
    d = f'<i style="background:{fg}"></i>' if dot else ""
    return f'<span class="rt-pill" style="color:{fg};background:{bg};border-color:{bd}">{d}{text}</span>'

def page_header(title: str, subtitle: str = "", right: str = ""):
    sub = f'<div class="rt-sub">{subtitle}</div>' if subtitle else ""
    html(f'<div class="rt-head"><div><h1 class="rt-title">{title}</h1>{sub}</div>'
         f'<div class="rt-head-right">{right}</div></div>')

def section_title(text: str, sub: str = "", icon_name: str = ""):
    ic = f'<span class="rt-sec-ic">{icon(icon_name, 18)}</span>' if icon_name else ""
    s = f'<div class="rt-sec-sub">{sub}</div>' if sub else ""
    html(f'<div class="rt-sec"><div class="rt-sec-t">{ic}{text}</div>{s}</div>')

def stat_card(title: str, value: str, label: str, icon_name: str, tone: str = "blue") -> str:
    """Pastel summary card as in the reference dashboard."""
    fg, bg, bd = TONES[tone]
    return f"""
    <div class="rt-stat" style="background:{bg};border-color:{bd}">
      <div class="rt-stat-title" style="color:{fg}">{title}</div>
      <div class="rt-stat-body">
        <div class="rt-stat-ico" style="background:{bd};color:{fg}">{icon(icon_name, 30, stroke=2.2)}</div>
        <div><div class="rt-stat-val" style="color:{INK}">{value}</div>
             <div class="rt-stat-lbl">{label}</div></div>
      </div>
    </div>"""

def mini_stat(label: str, value: str, icon_name: str, tone: str = "blue", sub: str = "") -> str:
    fg, bg, bd = TONES[tone]
    s = f'<div class="rt-mini-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="rt-mini">
      <div class="rt-mini-ico" style="background:{bg};color:{fg}">{icon(icon_name, 20)}</div>
      <div><div class="rt-mini-lbl">{label}</div><div class="rt-mini-val">{value}</div>{s}</div>
    </div>"""

def ring(pct: float, size: int = 96, stroke: int = 10, color: str = PRIMARY, center: str = "") -> str:
    pct = max(0.0, min(100.0, float(pct)))
    r = (size - stroke) / 2
    c = 2 * 3.14159265 * r
    off = c * (1 - pct / 100)
    return f"""
    <div class="rt-ring" style="width:{size}px;height:{size}px">
      <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" style="transform:rotate(-90deg)">
        <circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="#E6ECF6" stroke-width="{stroke}"/>
        <circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke}"
                stroke-linecap="round" stroke-dasharray="{c:.1f}" stroke-dashoffset="{off:.1f}"/>
      </svg><div class="rt-ring-c">{center}</div></div>"""

def bar(pct: float, color: str = PRIMARY, height: int = 6) -> str:
    pct = max(0.0, min(100.0, float(pct)))
    return (f'<div class="rt-bar" style="height:{height}px"><div style="width:{pct:.0f}%;background:{color}"></div></div>')

def empty_state(icon_name: str, title: str, body: str = ""):
    b = f'<div class="rt-empty-b">{body}</div>' if body else ""
    html(f'<div class="rt-empty"><div class="rt-empty-i">{icon(icon_name, 30)}</div>'
         f'<div class="rt-empty-t">{title}</div>{b}</div>')

def recall_color(score: float) -> str:
    return RED if score < 40 else (AMBER if score < 65 else GREEN)

def recall_tone(score: float) -> str:
    return "red" if score < 40 else ("amber" if score < 65 else "green")

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{
  --bg:#F4F7FC; --surface:#fff; --ink:#0F1F3D; --ink2:#334155; --muted:#64748B; --line:#E3E9F3;
  --navy:#0B2A5B; --primary:#2563EB; --primary-soft:#EAF1FF;
  --shadow:0 1px 2px rgba(15,31,61,.04),0 8px 24px rgba(15,31,61,.06);
}
/* ---- base ---- */
.stApp{background:var(--bg)}
html,body,.stApp,.stApp p,.stApp li,.stApp label,.stApp button,.stApp input,.stApp textarea,
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp td,.stApp th,[data-testid="stMarkdownContainer"],
[data-baseweb="select"] div,[data-baseweb="tab"]{font-family:'Inter','Segoe UI',system-ui,-apple-system,Roboto,'Helvetica Neue',Arial,sans-serif}
.block-container{max-width:1200px;padding:2.4rem 2.6rem 5rem}
[data-testid="stToolbar"],[data-testid="stDecoration"],#MainMenu,footer{display:none!important}
[data-testid="stHeader"]{background:transparent}
[data-testid="stElementContainer"]:has(.rt-css-marker){display:none}
.stHeading a,[data-testid="stHeaderActionElements"]{display:none!important}
h1,h2,h3{color:var(--ink);letter-spacing:-.015em}
hr{border-color:var(--line)!important}

/* ---- sidebar ---- */
section[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line)}
[data-testid="stSidebar"][aria-expanded="true"]{min-width:272px;max-width:272px;width:272px}
[data-testid="stSidebarContent"]{padding-top:.4rem}
[data-testid="stSidebarUserContent"]{padding:1.1rem 1rem 1rem}
.rt-brand{display:flex;align-items:center;gap:.65rem;padding:.2rem .35rem .9rem}
.rt-brand-n{font-weight:800;font-size:1.02rem;color:var(--ink);letter-spacing:-.02em;line-height:1.15;white-space:nowrap}
.rt-brand-s{font-size:.72rem;color:var(--muted);font-weight:500;margin-top:2px}
.rt-user{display:flex;align-items:center;gap:.7rem;padding:.65rem .75rem;border:1px solid var(--line);
  border-radius:14px;background:#F8FAFE;margin-bottom:1.1rem}
.rt-avatar{width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,#0B2A5B,#2F6FEB);
  color:#fff;display:grid;place-items:center;font-weight:700;font-size:.95rem;flex:none}
.rt-user-n{font-weight:700;font-size:.92rem;color:var(--ink);line-height:1.15}
.rt-user-i{font-size:.75rem;color:var(--muted)}
.rt-navlabel{font-size:.68rem;font-weight:700;letter-spacing:.09em;color:#94A3B8;text-transform:uppercase;
  padding:.9rem .6rem .35rem}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]{border-radius:11px;padding:.5rem .7rem;margin:1px 0;
  color:var(--ink2);font-weight:500;transition:background .15s}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover{background:#F1F5FD}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"]{background:var(--primary-soft);
  color:var(--primary);font-weight:700}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] p{font-size:.93rem}
[data-testid="stSidebar"] .stButton>button{width:100%;justify-content:flex-start;background:transparent;border:none;
  color:var(--muted);box-shadow:none;font-weight:500}
[data-testid="stSidebar"] .stButton>button:hover{background:#FEF2F2;color:#DC2626}

/* ---- page header ---- */
.rt-head{display:flex;justify-content:space-between;align-items:flex-end;gap:1rem;margin:0 0 1.5rem}
.rt-title{font-size:1.95rem;font-weight:800;color:var(--ink);letter-spacing:-.025em;margin:0;line-height:1.15}
.rt-sub{color:var(--muted);font-size:.98rem;margin-top:.4rem}
.rt-head-right{display:flex;gap:.6rem;align-items:center}
.rt-sec{margin:1.7rem 0 .8rem}
.rt-sec-t{display:flex;align-items:center;gap:.5rem;font-size:1.12rem;font-weight:700;color:var(--ink)}
.rt-sec-ic{display:grid;place-items:center;width:30px;height:30px;border-radius:9px;background:var(--primary-soft);color:var(--primary)}
.rt-sec-sub{font-size:.86rem;color:var(--muted);margin-top:.25rem}

/* ---- pills / chips ---- */
.rt-pill{display:inline-flex;align-items:center;gap:.4rem;padding:.22rem .7rem;border-radius:999px;
  font-size:.78rem;font-weight:600;border:1px solid;white-space:nowrap}
.rt-pill i{width:7px;height:7px;border-radius:50%;display:inline-block}

/* ---- pastel stat cards (reference look) ---- */
.rt-stat{border:1px solid;border-radius:18px;padding:1.05rem 1.3rem 1.2rem;height:100%}
.rt-stat-title{font-weight:700;font-size:.98rem;margin-bottom:.85rem}
.rt-stat-body{display:flex;align-items:center;gap:1rem}
.rt-stat-ico{width:62px;height:62px;border-radius:50%;display:grid;place-items:center;flex:none;opacity:.95}
.rt-stat-val{font-size:2.35rem;font-weight:800;line-height:1;letter-spacing:-.03em}
.rt-stat-lbl{font-size:.86rem;color:var(--ink2);margin-top:.35rem;font-weight:500}

/* ---- mini stats ---- */
.rt-mini{display:flex;gap:.85rem;align-items:center;background:#fff;border:1px solid var(--line);
  border-radius:16px;padding:.95rem 1.1rem;box-shadow:var(--shadow);height:100%}
.rt-mini-ico{width:42px;height:42px;border-radius:12px;display:grid;place-items:center;flex:none}
.rt-mini-lbl{font-size:.76rem;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.05em}
.rt-mini-val{font-size:1.45rem;font-weight:800;color:var(--ink);line-height:1.15}
.rt-mini-sub{font-size:.76rem;color:var(--muted)}

/* ---- cards (st.container(key="card_*")) ---- */
[class*="st-key-card_"]{background:#fff;border:1px solid var(--line);border-radius:18px;
  padding:1.2rem 1.35rem;box-shadow:var(--shadow)}
[class*="st-key-panel_"]{background:#F3F8FF;border:1px solid #CFE0FB;border-radius:20px;padding:1.3rem 1.4rem}

/* ---- concept table (reference look) ---- */
[class*="st-key-thead"]{background:var(--navy);border-radius:14px 14px 0 0;padding:.85rem 1.1rem;margin-bottom:0;
  align-items:center}
.rt-th{color:#fff;font-weight:700;font-size:.9rem;letter-spacing:.01em}
[class*="st-key-trow_"]{background:#fff;border:1px solid var(--line);border-top:none;padding:.7rem 1.1rem;
  align-items:center}
[class*="st-key-trow_"]:last-of-type{border-radius:0 0 14px 14px}
[class*="st-key-trow_"]:hover{background:#FAFCFF}
[class*="st-key-tbl"]{gap:0!important;box-shadow:var(--shadow);border-radius:14px}
.rt-tc{display:flex;align-items:center;gap:.8rem}
.rt-tile{width:42px;height:42px;border-radius:11px;display:grid;place-items:center;flex:none}
.rt-tn{font-weight:700;color:var(--ink);font-size:.97rem;line-height:1.2}
.rt-ts{font-size:.76rem;color:var(--muted)}
.rt-recall{font-weight:800;font-size:1.1rem}
.rt-bar{background:#E9EEF7;border-radius:99px;overflow:hidden;margin-top:6px;max-width:120px}
.rt-bar>div{height:100%;border-radius:99px}

/* action buttons inside the table */
[class*="st-key-act_"] button{border-radius:10px;font-weight:600;font-size:.86rem;padding:.4rem .9rem;background:#fff;
  min-height:0;box-shadow:none}
[class*="st-key-act_high"] button{color:#DC2626;border:1.5px solid #F2A9A9}
[class*="st-key-act_high"] button:hover{background:#FDECEC;border-color:#DC2626;color:#B91C1C}
[class*="st-key-act_medium"] button{color:#1D4ED8;border:1.5px solid #A9C4F5}
[class*="st-key-act_medium"] button:hover{background:#EAF1FF;border-color:#2563EB}
[class*="st-key-act_low"] button{color:#94A3B8;border:1.5px solid #E1E7F0;background:#F8FAFC}

/* ---- generated review panel ---- */
.rt-gp-title{display:flex;align-items:center;gap:.6rem;font-size:1.15rem;font-weight:700;color:var(--navy);margin-bottom:1rem}
.rt-gp-title svg{color:var(--primary)}
.rt-gen{display:flex;background:#fff;border:1px solid var(--line);border-radius:16px;overflow:hidden;
  margin-bottom:.85rem;box-shadow:0 1px 2px rgba(15,31,61,.03)}
.rt-gen-t{width:76px;flex:none;display:grid;place-items:center;color:#fff}
.rt-gen-b{padding:1rem 1.2rem;flex:1;min-width:0}
.rt-gen-h{font-weight:700;font-size:1.02rem;margin-bottom:.4rem}
.rt-gen-p{color:var(--ink2);font-size:.93rem;line-height:1.55}
.rt-qa{border:1px solid var(--line);border-radius:12px;padding:.7rem .95rem;background:#FBFCFF;margin-top:.2rem}
.rt-qa div{font-size:.93rem;line-height:1.5;color:var(--ink2)}
.rt-qa b{color:var(--ink)}
.rt-gen-empty{color:var(--muted);font-style:normal;font-size:.9rem}

/* ---- queue cards ---- */
.rt-q-top{display:flex;justify-content:space-between;align-items:flex-start;gap:1rem}
.rt-q-name{font-size:1.15rem;font-weight:700;color:var(--ink);line-height:1.2}
.rt-q-meta{display:flex;flex-wrap:wrap;gap:.5rem;margin-top:.7rem}
.rt-chip{display:inline-flex;align-items:center;gap:.35rem;font-size:.8rem;color:var(--ink2);background:#F4F7FC;
  border:1px solid var(--line);padding:.25rem .65rem;border-radius:9px;font-weight:500}
.rt-chip svg{color:var(--muted)}
.rt-dots{display:inline-flex;gap:3px;vertical-align:middle}
.rt-dots i{width:7px;height:7px;border-radius:50%;background:#D5DDEA}
.rt-dots i.on{background:#F59E0B}
.rt-ring{position:relative;display:inline-block}
.rt-ring-c{position:absolute;inset:0;display:grid;place-items:center;font-weight:800;color:var(--ink);text-align:center;line-height:1.05}
.rt-recall-box{text-align:right;min-width:88px}
.rt-recall-box .n{font-size:1.9rem;font-weight:800;line-height:1}
.rt-recall-box .l{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;font-weight:600}

/* ---- flashcard / study ---- */
.rt-flash{background:linear-gradient(160deg,#fff,#F5F8FF);border:1px solid #D4E1F8;border-radius:22px;
  padding:2.2rem 2rem;text-align:center;box-shadow:var(--shadow)}
.rt-flash-l{font-size:.72rem;font-weight:700;letter-spacing:.12em;color:var(--primary);text-transform:uppercase;margin-bottom:.8rem}
.rt-flash-q{font-size:1.4rem;font-weight:700;color:var(--ink);line-height:1.4;letter-spacing:-.01em}
.rt-flash-a{background:#E8F8EE;border:1px solid #BFE9CE;border-radius:18px;padding:1.4rem 1.6rem;margin-top:1rem;text-align:center}
.rt-flash-a .rt-flash-l{color:#15803D}
.rt-flash-a .t{font-size:1.1rem;color:#14532D;line-height:1.55;font-weight:500}
.rt-qcard{display:flex;gap:.8rem;align-items:flex-start;margin-bottom:.4rem}
.rt-qnum{width:30px;height:30px;border-radius:9px;background:var(--primary-soft);color:var(--primary);
  display:grid;place-items:center;font-weight:800;font-size:.9rem;flex:none}
.rt-qtext{font-weight:700;color:var(--ink);font-size:1.02rem;line-height:1.45;padding-top:3px}
.rt-result{border-radius:12px;padding:.75rem 1rem;font-size:.92rem;margin:.4rem 0 .2rem;border:1px solid;line-height:1.5}
.rt-code-label{font-weight:700;color:var(--ink);font-size:.9rem;margin:.9rem 0 .35rem;display:flex;align-items:center;gap:.4rem}

/* ---- list rows (recent sessions etc.) ---- */
.rt-row{display:flex;align-items:center;gap:.85rem;padding:.7rem 0;border-bottom:1px solid #EEF2F8}
.rt-row:last-child{border-bottom:none}
.rt-row-n{font-weight:600;color:var(--ink);font-size:.93rem;line-height:1.2}
.rt-row-s{font-size:.78rem;color:var(--muted)}
.rt-row-r{margin-left:auto;text-align:right}
.rt-row-sc{font-weight:800;font-size:1rem}

/* ---- empty ---- */
.rt-empty{text-align:center;padding:2.6rem 1rem;background:#fff;border:1.5px dashed #CBD7EA;border-radius:18px}
.rt-empty-i{width:60px;height:60px;border-radius:50%;background:var(--primary-soft);color:var(--primary);
  display:grid;place-items:center;margin:0 auto .9rem}
.rt-empty-t{font-weight:700;font-size:1.05rem;color:var(--ink)}
.rt-empty-b{color:var(--muted);font-size:.9rem;margin-top:.3rem}

/* ---- login ---- */
.rt-hero{text-align:center;padding:2.2rem 0 1.4rem}
.rt-hero h1{font-size:2.5rem;font-weight:800;letter-spacing:-.035em;margin:1rem 0 .3rem;color:var(--ink)}
.rt-hero p{color:var(--muted);font-size:1.05rem;margin:0}
.rt-feat{display:flex;gap:.8rem;align-items:flex-start;padding:.9rem 0}
.rt-feat b{color:var(--ink);font-size:.95rem}
.rt-feat span{color:var(--muted);font-size:.87rem;display:block;line-height:1.45}

/* ---- streamlit widgets ---- */
.stButton>button,.stDownloadButton>button,[data-testid="stFormSubmitButton"]>button{border-radius:12px;font-weight:600;
  border:1px solid var(--line);background:#fff;color:var(--ink);padding:.5rem 1.1rem;
  transition:all .15s ease;box-shadow:0 1px 2px rgba(15,31,61,.04)}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--primary);color:var(--primary);background:#F7FAFF}
.stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"],[data-testid="stFormSubmitButton"]>button[kind="primaryFormSubmit"],
.stButton>button[data-testid="stBaseButton-primary"],[data-testid="stBaseButton-primaryFormSubmit"]{
  background:linear-gradient(135deg,#1E40AF,#2563EB);color:#fff;border:none;
  box-shadow:0 6px 16px rgba(37,99,235,.28)}
.stButton>button[data-testid="stBaseButton-primary"]:hover,[data-testid="stBaseButton-primaryFormSubmit"]:hover{
  background:linear-gradient(135deg,#1E3A8A,#1D4ED8);color:#fff;transform:translateY(-1px)}
.stButton>button:disabled{opacity:.5}
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,textarea,[data-baseweb="select"]>div{
  border-radius:12px!important;border-color:var(--line)!important;background:#fff!important}
[data-testid="stTextInput"] input:focus,textarea:focus{border-color:var(--primary)!important;box-shadow:0 0 0 3px rgba(37,99,235,.15)!important}
[data-testid="stWidgetLabel"] p{font-weight:600;color:var(--ink2);font-size:.88rem}
[data-testid="stAlert"]{border-radius:14px;border:1px solid var(--line)}
[data-testid="stExpander"]{background:#fff;border:1px solid var(--line)!important;border-radius:14px;box-shadow:none}
[data-testid="stExpander"] summary{font-weight:600}
[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);border-radius:16px;padding:.9rem 1.1rem;box-shadow:var(--shadow)}
[data-testid="stMetricLabel"] p{color:var(--muted);font-weight:600;font-size:.8rem}
[data-testid="stMetricValue"]{font-weight:800;color:var(--ink)}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:14px;overflow:hidden}
[data-baseweb="tab-list"]{gap:.4rem;border-bottom:1px solid var(--line)}
[data-baseweb="tab"]{font-weight:600;padding:.6rem 1rem;color:var(--muted)}
[data-baseweb="tab"][aria-selected="true"]{color:var(--primary)}
[data-baseweb="tab-highlight"]{background:var(--primary)!important;height:3px;border-radius:3px}
[data-testid="stProgress"] div[role="progressbar"]>div{background:linear-gradient(90deg,#2563EB,#60A5FA)}
[data-testid="stSegmentedControl"] button{font-weight:600}
[data-testid="stFileUploaderDropzone"]{border-radius:16px;border:1.5px dashed #B9C9E6;background:#F8FAFF}
[data-testid="stToast"]{border-radius:14px;box-shadow:var(--shadow)}
.stCodeBlock,pre{border-radius:12px!important}
textarea{font-family:ui-monospace,'JetBrains Mono',Menlo,Consolas,monospace!important;font-size:.88rem!important}

@media (max-width: 820px){
  .block-container{padding:1.4rem 1rem 4rem}
  .rt-title{font-size:1.5rem}
  .rt-stat-val{font-size:1.9rem}
}
</style>
<span class="rt-css-marker"></span>
"""

def inject_css():
    """Call once per run from main.py (styles persist across page switches)."""
    st.markdown(_CSS, unsafe_allow_html=True)
