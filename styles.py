"""Estilos de la interfaz original (oscuro, verde, compacto), con refinamientos."""
CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root { color-scheme: dark; }
html, body, [class*="css"], .stApp {font-family:'Inter',sans-serif !important;}
.stApp {background: #0d1117 !important; color:#e6edf3 !important;}
.block-container {max-width: 830px !important; margin:auto !important; padding:0 1.3rem 4rem !important;}
#MainMenu, footer, header, .stDeployButton {visibility:hidden !important;}
h1,h2,h3,h4,p,label {color:#e6edf3;}
.app-header {display:flex;gap:0.6rem;justify-content:space-between;align-items:center;background:#161b22;
 border-bottom:1px solid #30363d;padding:0.9rem 1.3rem;margin:0 -1.3rem 1.8rem;}
.app-header-title {font-size:1.05rem;font-weight:700;color:#e6edf3;}
.app-header-sub {font-size:0.73rem;color:#8b949e;}
.intro {font-size:.88rem;color:#8b949e;line-height:1.8;margin-bottom:.7rem;}
.stTabs [data-baseweb="tab-list"] {background:#161b22;border-radius:10px;border:1px solid #30363d;padding:4px;gap:4px;}
.stTabs [data-baseweb="tab"] {border-radius:8px;color:#aab5c0;background:transparent;
 padding:.50rem 1.15rem;font-weight:600;border:0;}
.stTabs [data-baseweb="tab"][aria-selected="true"] {background:#21262d;color:#e6edf3;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display:none !important;}
.stTextInput input, .stTextArea textarea,
.stSelectbox [data-baseweb="select"] > div {background:#21262d !important;border:1px solid #30363d !important;
 color:#e6edf3 !important;border-radius:9px !important;}
.stTextInput input:focus, .stTextArea textarea:focus {border-color:#2ea043 !important;
 box-shadow:0 0 0 2px rgba(46,160,67,.2) !important;}
.stRadio > div {background:#161b22;border:1px solid #30363d;border-radius:9px;
 padding:.65rem .85rem;gap:1.5rem;}
.stRadio label, .stSelectbox label, .stTextInput label {color:#b5bec8 !important;font-size:.86rem !important;}
/* Resaltar suavemente el modo seleccionado; el diseño oscuro/verde se conserva. */
.stRadio [role="radiogroup"] label:has(input:checked) {
  background:rgba(46,160,67,.12) !important;
  box-shadow:inset 0 0 0 1px rgba(46,160,67,.35);
  border-radius:7px;
}
.stRadio [role="radiogroup"] label:has(input:checked) p {color:#e6edf3 !important;}
.stButton button {border-radius:9px !important;font-weight:650 !important;transition:.12s !important;}
.stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"] {
 background:#2ea043 !important;border:0 !important;color:white !important;width:100% !important;
 min-height:3rem;box-shadow:0 2px 8px rgba(46,160,67,.24) !important;}
.stButton button[kind="primary"]:hover, .stFormSubmitButton button[kind="primary"]:hover {background:#238636 !important;}
.stButton button[kind="secondary"] {background:#21262d !important;border:1px solid #30363d !important;color:#d2dae4 !important;}
[data-testid="stMetric"] {background:#161b22;border:1px solid #30363d;border-radius:10px;padding:.8rem 1rem;}
[data-testid="stMetricValue"] {color:#2ea043 !important;font-weight:700;}
[data-testid="stMetricLabel"] {color:#8b949e !important;font-size:.74rem;}
hr {border-color:#30363d !important;margin:1.2rem 0 !important;}
.msg-user,.msg-bot,.lang-card,.about-sec {border-radius:10px;padding:.88rem 1.1rem;margin:.7rem 0;}
.msg-user {background:rgba(31,111,235,.1);border:1px solid rgba(56,139,253,.3);}
.msg-bot {background:#161b22;border:1px solid #30363d;border-left:3px solid #2ea043;}
.msg-user .lbl,.msg-bot .lbl {font-size:.68rem;font-weight:700;letter-spacing:.6px;
 text-transform:uppercase;margin-bottom:.55rem;}
.msg-user .lbl {color:#58a6ff;}.msg-bot .lbl {color:#2ea043;}
.msg-user .txt {color:#e6edf3;font-size:.9rem;line-height:1.65;}
.msg-bot .txt {color:#c3ccd6;font-size:.9rem;line-height:1.75;}
.method {color:#8b949e;font-size:.72rem;margin:.4rem 0 0;}
.lang-card,.about-sec {background:#161b22;border:1px solid #30363d;}
.lang-card .name {color:#2ea043;font-weight:700;font-size:.86rem;margin:0 0 .25rem;}
.lang-card .meta {font-size:.73rem;color:#8b949e;margin-bottom:.5rem;}
.lang-card .desc {font-size:.82rem;color:#adbac7;line-height:1.65;}
.about-sec h4 {color:#2ea043;font-size:.77rem;text-transform:uppercase;letter-spacing:.5px;margin:.1rem 0 .6rem;}
.about-sec p {font-size:.85rem;color:#bdc7d2;line-height:1.8;}
.chips {display:flex;flex-wrap:wrap;gap:.35rem;}
.chip {font-size:.70rem;padding:.20rem .53rem;border-radius:6px;background:rgba(255,255,255,.05);
 color:#b7c1cb;border:1px solid #30363d;}
.chip.green {color:#3fb950;background:rgba(46,160,67,.1);border-color:rgba(46,160,67,.3);}
.evidence {font-size:.8rem;line-height:1.7;color:#adbac7;margin:.4rem 0;}
.evidence a {color:#79c0ff;text-decoration:none;}
.evidence a:hover {text-decoration:underline;}
.warning {border-left:3px solid #d29922;padding:.5rem .75rem;margin:.6rem 0;background:#211e16;
 color:#d7c6a4;font-size:.77rem;}
.tfooter {text-align:center;padding:1.4rem 0;border-top:1px solid #30363d;margin-top:2rem;}
.tfooter p {font-size:.72rem;color:#8b949e;margin:.15rem 0;}
[data-testid="stExpander"] {background:#161b22;border:1px solid #30363d;border-radius:9px;}
[data-testid="stExpander"] details {border:0 !important;}
[data-testid="stAlert"] {border-radius:9px !important;}
@media(max-width:650px) {.app-header-sub{display:none}.block-container{padding:0 .8rem 2rem!important;}
 .app-header{margin:0 -.8rem 1rem;padding:.8rem;}.stTabs [data-baseweb="tab"]{padding:.45rem .7rem;font-size:.85rem;}}
</style>
"""
