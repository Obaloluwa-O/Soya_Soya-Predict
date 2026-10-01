import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Soybean Harvest Predictor", page_icon="🌱")

# ---- Edit these labels to match your survey units --------------------------
HARVEST_UNIT = "harvest units"          # unit of Soya_harvest2025
FERT_UNIT = "units"                     # unit of Fertilizer_Soybean2025
# -----------------------------------------------------------------------------

# Intercrop choice -> dummy column that gets set to 1 (None = all dummies 0)
INTERCROP = {
    "None / other combination": None,
    "Maize + Soybean": "Intercrop_Maize_Soybean",
    "Paddy rice + Soybean": "Intercrop_Paddy_rice_Soybean",
    "Sorghum + Soybean": "Intercrop_Sorghum_Soybean",
}


@st.cache_resource
def load_model():
    return joblib.load("soya_ols_model.joblib")


m = load_model()
FARM = m["farm_col"]
R = m["ranges"]

st.title("🌱 Soybean Harvest Predictor")
st.caption(
    f"Log-linear OLS (farm size logged) trained on {m['n_rows']:,} farms. "
    f"Held-out R² on the log scale: {m['test_r2_log']:.2f}."
)

# ------------------------------- Inputs --------------------------------------
# Everything inside the form waits until the button is clicked
st.subheader("Farm details")
with st.form("farm_form"):
    c1, c2 = st.columns(2)

    with c1:
        farm_size = st.number_input("Farm size cultivated for soybean (ha)", min_value=0.1, value=1.0, step=0.1)
        household = st.number_input("Household size", min_value=1, value=5, step=1)
        machinery = st.radio("Machinery used for planting?", ["No", "Yes"], horizontal=True)
        intercrop = st.selectbox("Crops grown together", list(INTERCROP.keys()))

    with c2:
        fert = st.number_input(f"Fertilizer ({FERT_UNIT})", min_value=0.0, value=0.0, step=1.0)
        herb = st.number_input("Herbicide (bottles)", min_value=0.0, value=0.0, step=1.0)
        pest = st.number_input("Pesticide", min_value=0.0, value=0.0, step=1.0)
        opv = st.number_input("OPV soybean seed (kg)", min_value=0.0, value=0.0, step=1.0)
        hybrid = st.number_input("Hybrid soybean seed (kg)", min_value=0.0, value=0.0, step=1.0)

    submitted = st.form_submit_button("Predict harvest", type="primary")

# Remember the submitted values so the result stays on screen
# (for example when the expander below is opened)
if submitted:
    st.session_state["inputs"] = {
        "farm_size": farm_size, "household": household, "machinery": machinery,
        "intercrop": intercrop, "fert": fert, "herb": herb, "pest": pest,
        "opv": opv, "hybrid": hybrid,
    }

if "inputs" not in st.session_state:
    st.info("Enter the farm details above and click **Predict harvest**.")
    st.stop()

i = st.session_state["inputs"]

# ----------------------------- Prediction ------------------------------------
raw = {
    "What is your Household Size?": i["household"],
    FARM: i["farm_size"],
    "Fertilizer_Soybean2025": i["fert"],
    "OPV_soya2025(kg)": i["opv"],
    "HybridSoy2025(kg)": i["hybrid"],
    "HerbSoy2025_bottles": i["herb"],
    "pesticide_Soybean2025": i["pest"],
}

x = {f: 0.0 for f in m["features"]}
x.update({k: float(v) for k, v in raw.items() if k in x})
x[FARM] = float(np.log(i["farm_size"]))               # the model uses log farm size
if "Machinery_Yes" in x:
    x["Machinery_Yes"] = 1.0 if i["machinery"] == "Yes" else 0.0
dummy = INTERCROP[i["intercrop"]]
if dummy in x:
    x[dummy] = 1.0

pred_log = m["intercept"] + sum(m["coefs"][f] * x[f] for f in m["features"])
est = float(np.exp(pred_log))
lo = float(np.exp(pred_log - 1.96 * m["sigma"]))
hi = float(np.exp(pred_log + 1.96 * m["sigma"]))

st.divider()
st.subheader("Predicted harvest")
a, b = st.columns(2)
a.metric(f"Typical harvest ({HARVEST_UNIT})", f"{est:,.1f}")
b.metric("Per hectare", f"{est / i['farm_size']:,.1f}")
st.write(f"**Likely range (about 95% of farms like this):** {lo:,.1f} to {hi:,.1f} {HARVEST_UNIT}")
st.caption(
    "The range is wide because the model explains about a third of the variation in harvest. "
    "Treat the estimate as a typical value for a farm like this, not a guarantee."
)

# Warn when an input is outside what the model saw in training
out_of_range = []
for name, val in raw.items():
    if name in R:
        lo_r, hi_r = R[name]
        if val < lo_r or val > hi_r:
            out_of_range.append(f"{name}: {val:g} (training range {lo_r:g} to {hi_r:g})")
if out_of_range:
    st.warning("Some inputs are outside the range the model was trained on, so the prediction is less reliable:\n\n- "
               + "\n- ".join(out_of_range))

# ----------------------------- Model details ---------------------------------
with st.expander("How the model reads each input"):
    rows = []
    for f in m["features"]:
        coef = m["coefs"][f]
        if f == FARM:
            effect = f"1% more farm size, about {coef:.2f}% more harvest"
        else:
            effect = f"{(np.exp(coef) - 1) * 100:+.2f}% harvest per unit"
        rows.append({"Input": "Farm size (log)" if f == FARM else f, "Coefficient": round(coef, 4), "Effect": effect})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")