import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# Page setup + CSS tightening
# -----------------------------
st.set_page_config(page_title="Coal Procurement Optimizer", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 1.0rem;}
      .stMetric {padding: 6px 10px;}
      div[data-testid="stVerticalBlockBorderWrapper"] {padding: 10px;}
      .tight-card {padding: 12px 14px; border-radius: 14px; border: 1px solid rgba(49,51,63,0.15);}
      .muted {opacity: 0.75;}
      .small {font-size: 0.92rem;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🔥 Coal Procurement Optimizer")
st.caption("Upload coal-source data → compute cost-per-calorie → optimize company-wise procurement allocation → GenBI quick queries (offline demo).")

REQUIRED_COLS = [
    "company",
    "gcv_kcal_per_kg",
    "base_price_inr_per_tonne",
    "logistics_cost_inr_per_tonne",
    "max_supply_tonnes",
]

OPTIONAL_COLS = ["mine_or_source","grade","route_mode","distance_km","moisture_pct","ash_pct","sulfur_pct","contract_type","lead_time_days"]

# -----------------------------
# Sidebar inputs
# -----------------------------
with st.sidebar:
    st.header("Controls")

    st.subheader("Upload")
    up = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx"])

    st.divider()
    st.subheader("Demand calculator (optional)")
    mode = st.radio("Demand input mode", ["Tonnes", "MW × Hours"], index=0)

    demand_tonnes = None
    if mode == "Tonnes":
        demand_tonnes = st.number_input("Demand (tonnes)", min_value=0.0, value=120000.0, step=5000.0)
    else:
        mw = st.number_input("Net load (MW)", min_value=0.0, value=500.0, step=10.0)
        hours = st.number_input("Horizon (hours)", min_value=0.0, value=720.0, step=24.0)
        eff = st.slider("Net efficiency (demo)", 0.25, 0.45, 0.35, 0.01)
        # Approx: 1 kWh = 860 kcal. Energy out = MW*hours*1000 kWh.
        kcal_out = mw * hours * 1000.0 * 860.0
        # Energy in = kcal_out / eff.
        kcal_in = kcal_out / max(eff, 1e-6)
        st.caption(f"Thermal input required ≈ {kcal_in/1e9:.3f} billion kcal")
        # Convert to tonnes using blended GCV later; placeholder uses 4000 kcal/kg
        demand_tonnes = kcal_in / (4000.0 * 1000.0)

    st.divider()
    st.subheader("Quality / policy filters")
    min_gcv = st.slider("Min GCV (kcal/kg)", 2500, 6000, 3600, 50)
    max_moist = st.slider("Max Moisture (%)", 0.0, 30.0, 18.0, 0.5)
    max_ash = st.slider("Max Ash (%)", 0.0, 60.0, 45.0, 0.5)
    max_sulfur = st.slider("Max Sulfur (%)", 0.0, 2.0, 1.2, 0.05)

    st.divider()
    st.subheader("Blending (demo)")
    target_blend_gcv = st.slider("Target blended GCV (kcal/kg)", 2500, 6000, 4000, 50)

# -----------------------------
# Load data
# -----------------------------
def read_any(upload):
    if upload is None:
        return None
    if upload.name.lower().endswith(".csv"):
        return pd.read_csv(upload)
    return pd.read_excel(upload)

df = read_any(up)

if df is None:
    st.info("Upload a CSV/XLSX to begin. You can use the provided sample template.")
    st.stop()

missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Coerce numeric columns
for c in ["gcv_kcal_per_kg","base_price_inr_per_tonne","logistics_cost_inr_per_tonne","max_supply_tonnes",
          "moisture_pct","ash_pct","sulfur_pct"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["company","gcv_kcal_per_kg","base_price_inr_per_tonne","logistics_cost_inr_per_tonne","max_supply_tonnes"]).copy()

# Derived metrics
df["delivered_cost_inr_per_tonne"] = df["base_price_inr_per_tonne"] + df["logistics_cost_inr_per_tonne"]
df["delivered_cost_inr_per_MMkcal"] = (df["delivered_cost_inr_per_tonne"] * 1000.0 / df["gcv_kcal_per_kg"]).round(2)
df["cost_per_kcal_inr"] = (df["delivered_cost_inr_per_tonne"] / (df["gcv_kcal_per_kg"] * 1000.0)).round(8)

# Apply filters (only if columns exist)
f = df.copy()
f = f[f["gcv_kcal_per_kg"] >= min_gcv]
if "moisture_pct" in f.columns:
    f = f[f["moisture_pct"].fillna(0) <= max_moist]
if "ash_pct" in f.columns:
    f = f[f["ash_pct"].fillna(0) <= max_ash]
if "sulfur_pct" in f.columns:
    f = f[f["sulfur_pct"].fillna(0) <= max_sulfur]

if f.empty:
    st.warning("No rows remain after filters. Relax constraints to proceed.")
    st.stop()

# -----------------------------
# Optimizer: allocate demand greedily by cheapest delivered ₹/MMkcal
# -----------------------------
def allocate_greedy(data: pd.DataFrame, demand_t: float) -> pd.DataFrame:
    d = data.sort_values("delivered_cost_inr_per_MMkcal", ascending=True).copy()
    remaining = float(max(demand_t, 0.0))
    alloc = []
    for _, r in d.iterrows():
        if remaining <= 1e-6:
            break
        take = min(float(r["max_supply_tonnes"]), remaining)
        alloc.append({
            "company": r["company"],
            "mine_or_source": r.get("mine_or_source", ""),
            "gcv_kcal_per_kg": float(r["gcv_kcal_per_kg"]),
            "delivered_cost_inr_per_tonne": float(r["delivered_cost_inr_per_tonne"]),
            "delivered_cost_inr_per_MMkcal": float(r["delivered_cost_inr_per_MMkcal"]),
            "allocated_tonnes": take
        })
        remaining -= take

    out = pd.DataFrame(alloc)
    out["allocation_pct"] = (out["allocated_tonnes"] / max(demand_t, 1e-6) * 100.0).round(2)

    # blended metrics
    if not out.empty:
        blended_gcv = np.average(out["gcv_kcal_per_kg"], weights=out["allocated_tonnes"])
        blended_cost_t = np.average(out["delivered_cost_inr_per_tonne"], weights=out["allocated_tonnes"])
        out.attrs["blended_gcv"] = float(blended_gcv)
        out.attrs["blended_cost_per_tonne"] = float(blended_cost_t)
        out.attrs["unmet_tonnes"] = float(max(0.0, remaining))
    else:
        out.attrs["blended_gcv"] = 0.0
        out.attrs["blended_cost_per_tonne"] = 0.0
        out.attrs["unmet_tonnes"] = float(demand_t)

    return out

alloc = allocate_greedy(f, float(demand_tonnes))

blended_gcv = alloc.attrs.get("blended_gcv", 0.0)
blended_cost_t = alloc.attrs.get("blended_cost_per_tonne", 0.0)
unmet = alloc.attrs.get("unmet_tonnes", 0.0)

# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1.15, 1.0], gap="large")

with left:
    st.markdown('<div class="tight-card">', unsafe_allow_html=True)
    st.markdown("### 📥 Uploaded dataset (post-filters)")
    st.dataframe(f.sort_values("delivered_cost_inr_per_MMkcal").reset_index(drop=True), use_container_width=True, height=320)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="tight-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Cost–Quality landscape")
    fig_scatter = px.scatter(
        f,
        x="gcv_kcal_per_kg",
        y="delivered_cost_inr_per_MMkcal",
        size="max_supply_tonnes",
        hover_data=["company"] + [c for c in OPTIONAL_COLS if c in f.columns],
        title=None,
    )
    fig_scatter.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_title="GCV (kcal/kg)",
                              yaxis_title="Delivered cost (₹/MMkcal)")
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="tight-card">', unsafe_allow_html=True)
    st.markdown("### ✅ Optimized procurement plan")
    c1, c2, c3 = st.columns(3)
    c1.metric("Demand", f"{demand_tonnes:,.0f} t")
    c2.metric("Blended GCV", f"{blended_gcv:,.0f} kcal/kg")
    c3.metric("Blended Delivered Cost", f"₹{blended_cost_t:,.0f}/t")

    if unmet > 0:
        st.warning(f"Unmet demand due to max_supply limits: **{unmet:,.0f} t** (demo constraint).")

    st.dataframe(alloc, use_container_width=True, height=260)
    st.markdown("</div>", unsafe_allow_html=True)

    tabs = st.tabs(["📈 Rankings", "🧾 Cost breakdown", "💬 GenBI Quick Query"])

    with tabs[0]:
        topn = min(12, len(f))
        rnk = f.sort_values("delivered_cost_inr_per_MMkcal").head(topn).copy()
        fig_bar = px.bar(
            rnk,
            x="company",
            y="delivered_cost_inr_per_MMkcal",
            title=None,
        )
        fig_bar.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_title="", yaxis_title="₹/MMkcal")
        st.plotly_chart(fig_bar, use_container_width=True)

        if not alloc.empty:
            fig_alloc = px.pie(alloc, names="company", values="allocated_tonnes", title=None)
            fig_alloc.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_alloc, use_container_width=True)

    with tabs[1]:
        # Choose one source for a crisp "calculator-like" breakdown
        pick = st.selectbox("Pick a source for cost decomposition", f.sort_values("delivered_cost_inr_per_MMkcal")["company"].unique())
        row = f[f["company"] == pick].iloc[0]
        base = float(row["base_price_inr_per_tonne"])
        logi = float(row["logistics_cost_inr_per_tonne"])
        delivered = float(row["delivered_cost_inr_per_tonne"])

        wf = go.Figure(go.Waterfall(
            name="₹/t",
            orientation="v",
            measure=["relative", "relative", "total"],
            x=["Base price", "Logistics", "Delivered"],
            y=[base, logi, delivered],
        ))
        wf.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="₹/tonne")
        st.plotly_chart(wf, use_container_width=True)

        st.markdown(
            f"<span class='muted small'>Delivered cost per MMkcal = ₹{row['delivered_cost_inr_per_MMkcal']:.2f} "
            f"(GCV {row['gcv_kcal_per_kg']:.0f} kcal/kg)</span>",
            unsafe_allow_html=True
        )

    with tabs[2]:
        st.markdown("### 💬 GenBI Quick Query (offline)")
        st.caption("Demo intents: company-wise quantity, top cheapest, blended GCV, unmet demand, show scatter/bar.")
        q = st.text_input("Ask a question", placeholder="e.g., 'quantity company wise' or 'top 5 cheapest'")

        def genbi(q: str):
            ql = q.strip().lower()
            if not ql:
                return None, None

            if "quantity" in ql and ("company" in ql or "wise" in ql):
                ans = alloc.groupby("company", as_index=False)["allocated_tonnes"].sum()
                ans["allocation_pct"] = (ans["allocated_tonnes"] / max(demand_tonnes, 1e-6) * 100).round(2)
                return "Company-wise procurement quantities (optimizer output):", ans

            if "top" in ql and ("cheap" in ql or "lowest" in ql):
                m = re.search(r"top\s+(\d+)", ql)
                n = int(m.group(1)) if m else 5
                n = int(np.clip(n, 3, 15))
                ans = f.sort_values("delivered_cost_inr_per_MMkcal").head(n)[
                    ["company","mine_or_source","gcv_kcal_per_kg","delivered_cost_inr_per_MMkcal","max_supply_tonnes"]
                ]
                return f"Top {n} cheapest sources by delivered ₹/MMkcal:", ans

            if "blend" in ql and "gcv" in ql:
                return f"Blended GCV (by allocated tonnes) is **{blended_gcv:,.0f} kcal/kg**.", None

            if "unmet" in ql or "shortfall" in ql:
                return f"Unmet demand due to supply caps is **{unmet:,.0f} tonnes**.", None

            if "scatter" in ql:
                return "Showing the cost–quality scatter.", fig_scatter

            if "bar" in ql or "rank" in ql:
                return "Showing ranking bar chart.", fig_bar

            return "Try: 'quantity company wise', 'top 5 cheapest', 'blended gcv', 'unmet demand', 'show scatter', 'show bar'.", None

        ans, payload = genbi(q)
        if ans:
            st.info(ans)
        if isinstance(payload, pd.DataFrame):
            st.dataframe(payload, use_container_width=True)
        if isinstance(payload, go.Figure):
            st.plotly_chart(payload, use_container_width=True)
