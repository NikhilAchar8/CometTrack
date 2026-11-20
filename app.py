import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="CometTrack", layout="wide")

@st.cache_data
def load():
    orbits = pd.read_csv("comet_orbits.csv")
    try:
        risk = pd.read_csv("comet_risk.csv")
    except FileNotFoundError:
        risk = None
    return orbits, risk

df, risk = load()

def solve_kepler(M, e, tol=1e-10):
    E = M if e < 0.8 else np.pi
    for _ in range(100):
        dE = (E - e*np.sin(E) - M) / (1 - e*np.cos(E))
        E -= dE
        if abs(dE) < tol:
            break
    return E

def comet_position(t, T, q, e):
    a = q / (1 - e)
    n = 0.01720209895 / a**1.5
    M = (n * (t - T)) % (2*np.pi)
    E = solve_kepler(M, e)
    nu = 2*np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
    r = a * (1 - e*np.cos(E))
    return r*np.cos(nu), r*np.sin(nu)

def to_3d(x, y, inc, om, w):
    i, O, w = np.radians([inc, om, w])
    cO, sO, cw, sw, ci, si = np.cos(O), np.sin(O), np.cos(w), np.sin(w), np.cos(i), np.sin(i)
    X = (cO*cw - sO*sw*ci)*x + (-cO*sw - sO*cw*ci)*y
    Y = (sO*cw + cO*sw*ci)*x + (-sO*sw + cO*cw*ci)*y
    Z = (sw*si)*x + (cw*si)*y
    return X, Y, Z

def circle(r, n=200):
    th = np.linspace(0, 2*np.pi, n)
    return r*np.cos(th), r*np.sin(th), np.zeros(n)

def jd_to_date(jd):
    return (datetime(2000, 1, 1, 12) + timedelta(days=jd - 2451545.0)).strftime("%d %b %Y")

def orbit_3d(c, n_frames=80):
    days = c["tp"] + np.linspace(0, c["per"], 600)
    xy = np.array([comet_position(t, c["tp"], c["q"], c["e"]) for t in days])
    X, Y, Z = to_3d(xy[:, 0], xy[:, 1], c["i"], c["om"], c["w"])
    ex, ey, ez = circle(1.0)
    jx, jy, jz = circle(5.2)
    idx = np.linspace(0, len(X)-1, n_frames).astype(int)
    fig = go.Figure(
        data=[
            go.Scatter3d(x=X, y=Y, z=Z, mode="lines", name=c["comet"], line=dict(color="deepskyblue", width=3)),
            go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", name="Sun", marker=dict(size=7, color="gold")),
            go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", name="Earth orbit", line=dict(color="royalblue", width=2)),
            go.Scatter3d(x=jx, y=jy, z=jz, mode="lines", name="Jupiter orbit", line=dict(color="orange", width=2)),
            go.Scatter3d(x=[X[0]], y=[Y[0]], z=[Z[0]], mode="markers", name="Comet", marker=dict(size=6, color="red")),
        ],
        frames=[go.Frame(data=[go.Scatter3d(x=[X[k]], y=[Y[k]], z=[Z[k]], mode="markers",
                                            marker=dict(size=6, color="red"))],
                         traces=[4], name=str(n)) for n, k in enumerate(idx)],
    )
    fig.update_layout(
        scene=dict(aspectmode="data", xaxis_title="x (AU)", yaxis_title="y (AU)", zaxis_title="z (AU)"),
        height=620, margin=dict(l=0, r=0, t=30, b=0),
        updatemenus=[dict(type="buttons", showactive=False, x=0.05, y=0,
                          buttons=[dict(label="Play", method="animate",
                                        args=[None, dict(frame=dict(duration=60, redraw=True), fromcurrent=True)]),
                                   dict(label="Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0), mode="immediate")])])],
        sliders=[dict(steps=[dict(method="animate", args=[[str(n)], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                                  label=str(n)) for n in range(len(idx))],
                      currentvalue=dict(prefix="Frame: "))],
    )
    return fig

# ---------------- Page ----------------
st.title("CometTrack")
st.caption("Comet orbit prediction and close-approach forecasting using Python | Data: NASA JPL")

pick = st.sidebar.selectbox("Choose a comet", df["comet"])
c = df[df["comet"] == pick].iloc[0]

now = datetime.now()
today_jd = 2451545.0 + (now - datetime(2000, 1, 1, 12)).total_seconds() / 86400
k = np.ceil((today_jd - c["tp"]) / c["per"])
next_peri = c["tp"] + k * c["per"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Orbital period", f"{c['per']/365.25:.1f} years")
m2.metric("Eccentricity", f"{c['e']:.3f}")
m3.metric("Closest to Sun", f"{c['q']:.3f} AU")
m4.metric("Next perihelion (Kepler estimate)", jd_to_date(next_peri),
          f"{int(next_peri - today_jd):,} days away", delta_color="off")

st.plotly_chart(orbit_3d(c), use_container_width=True)

st.subheader("Close-approach risk, 2026 to 2036")
if risk is not None:
    mine = risk[risk["comet"] == pick]
    if len(mine):
        r = mine.iloc[0]
        st.info(f"{pick}: closest approach to Earth in this window is {r['distance_AU']} AU "
                f"({r['distance_million_km']} million km) on {r['closest_date']}. Label: {r['risk']}.")
    st.dataframe(risk, use_container_width=True)
    st.caption("Risk labels are project-defined distance thresholds, not an impact forecast.")
else:
    st.warning("comet_risk.csv not found. Run Cell 8 first.")

with st.expander("Model comparison: Halley's Comet 2061 perihelion"):
    st.table(pd.DataFrame({
        "Model": ["Kepler only", "N-body (REBOUND, 4 giant planets)"],
        "Error vs reference (days)": [164.5, -23.5],
    }))
