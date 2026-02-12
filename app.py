import math
import ast
import re
import json
from io import BytesIO
from fractions import Fraction

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4


# ==============================
# Page
# ==============================
st.set_page_config(page_title="EE + Automation Toolkit", page_icon="⚡", layout="centered")
st.title("⚡ EE + Automation Toolkit")
st.caption(
    "Power/Ohm • 3-Phase • RC/RL • PWM • Graphs • PID • Bode • Motor • "
    "Od’s Calculator • Matrices (≤6×6) • Save/Load • PDF"
)

# ==============================
# Helpers
# ==============================
def to_base(value: float, unit: str) -> float:
    factors = {
        "V": 1.0, "kV": 1e3,
        "A": 1.0, "mA": 1e-3,
        "Ω": 1.0, "kΩ": 1e3, "Ohm": 1.0, "kOhm": 1e3,
        "W": 1.0, "kW": 1e3,
        "VA": 1.0, "kVA": 1e3,
        "var": 1.0, "kvar": 1e3,
        "Hz": 1.0, "kHz": 1e3,
        "s": 1.0, "ms": 1e-3,
        "F": 1.0, "µF": 1e-6, "uF": 1e-6,
        "H": 1.0, "mH": 1e-3,
        "%": 0.01,
        "rpm": 1.0,
        "Nm": 1.0,
    }
    return float(value) * factors[unit]


def fmt_si(x: float, base_unit: str) -> str:
    if x is None:
        return "—"
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return "—"
    prefixes = [(1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n")]
    ax = abs(x)
    for f, p in prefixes:
        if ax >= f:
            return f"{x / f:.6g} {p}{base_unit}"
    return f"{x:.6g} {base_unit}"


def eng_format(x: float, unit: str = "") -> str:
    if x is None:
        return "—"
    x = float(x)
    if x == 0:
        return f"0 {unit}".strip()
    if math.isnan(x) or math.isinf(x):
        return "—"
    exp3 = int(math.floor(math.log10(abs(x)) / 3) * 3)
    exp3 = max(min(exp3, 12), -12)
    scaled = x / (10 ** exp3)
    suffix = {12: "T", 9: "G", 6: "M", 3: "k", 0: "", -3: "m", -6: "µ", -9: "n", -12: "p"}.get(exp3, f"e{exp3}")
    return f"{scaled:.6g}{suffix}{unit}"


def casio_format(x, digits=10, zero_eps=1e-12) -> str:
    """
    Casio-like display:
    - snap tiny values to 0
    - round to ~10 significant digits
    - show integers without .0
    - remove trailing zeros
    """
    try:
        xf = float(x)
    except Exception:
        return str(x)

    if math.isnan(xf) or math.isinf(xf):
        return "—"

    if abs(xf) < zero_eps:
        xf = 0.0

    if xf != 0.0:
        power = int(math.floor(math.log10(abs(xf))))
        decimals = max(0, digits - 1 - power)
        xf = round(xf, decimals)
        if abs(xf) < zero_eps:
            xf = 0.0

    if abs(xf - round(xf)) < zero_eps:
        return str(int(round(xf)))

    s = f"{xf:.{digits}g}"
    return s


# ==============================
# Unit-aware expression parsing
# ==============================
PREFIX = {"G": 1e9, "M": 1e6, "k": 1e3, "": 1.0, "m": 1e-3, "u": 1e-6, "µ": 1e-6, "n": 1e-9}
UNIT_MAP = {"V": 1.0, "A": 1.0, "W": 1.0, "Hz": 1.0, "F": 1.0, "H": 1.0, "s": 1.0, "Ohm": 1.0, "Ω": 1.0}


def replace_unit_literals(expr: str) -> str:
    expr = expr.replace(",", ".")
    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    pattern = r'(\d+(\.\d+)?)\s*([GMkmunµ]?)\s*(Hz|Ohm|Ω|V|A|W|F|H|s)\b'

    def repl(m):
        num = float(m.group(1))
        pref = m.group(3) or ""
        unit = m.group(4)
        return str(num * PREFIX.get(pref, 1.0) * UNIT_MAP.get(unit, 1.0))

    return re.sub(pattern, repl, expr)


# ==============================
# Safe math functions (Calculator)
# ==============================
def _to_int(x):
    if abs(x - round(x)) > 1e-9:
        raise ValueError("Expected integer.")
    return int(round(x))


def fact(x):
    n = _to_int(float(x))
    if n < 0 or n > 170:
        raise ValueError("factorial domain error (0..170)")
    return math.factorial(n)


def nCr(n, r):
    n = _to_int(float(n))
    r = _to_int(float(r))
    if r < 0 or n < 0 or r > n:
        raise ValueError("nCr domain error")
    return math.comb(n, r)


def nPr(n, r):
    n = _to_int(float(n))
    r = _to_int(float(r))
    if r < 0 or n < 0 or r > n:
        raise ValueError("nPr domain error")
    return math.perm(n, r)


def logb(x, base):
    return np.log(x) / np.log(base)


ALLOWED_FUNCS = {
    "sqrt": np.sqrt,
    "abs": np.abs,
    "round": np.round,
    "floor": np.floor,
    "ceil": np.ceil,
    "exp": np.exp,
    "ln": np.log,
    "log": np.log10,
    "log10": np.log10,
    "logb": logb,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "fact": fact, "nCr": nCr, "nPr": nPr,
    "pi": np.pi, "e": np.e,
}

ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Call,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd, ast.Load
)


def safe_eval(expr: str, variables: dict):
    expr = expr.strip()
    if not expr:
        return None
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError("Expression contains unsupported syntax.")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCS:
                raise ValueError("Only approved functions are allowed.")
        if isinstance(node, ast.Name):
            if node.id not in variables and node.id not in ALLOWED_FUNCS:
                raise ValueError(f"Unknown name: {node.id}")
    code = compile(tree, "<expr>", "eval")
    scope = dict(ALLOWED_FUNCS)
    scope.update(variables)
    return eval(code, {"__builtins__": {}}, scope)


def solve_root(expr: str, x0: float, variables: dict, max_iter=60, tol=1e-10):
    x = float(x0)

    def f(xx):
        vv = dict(variables)
        vv["x"] = xx
        return float(safe_eval(expr, vv))

    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x
        h = 1e-6 * (1 + abs(x))
        d = (f(x + h) - f(x - h)) / (2 * h)
        if abs(d) < 1e-12:
            x = x - np.sign(fx) * 0.1 * (1 + abs(x))
        else:
            x = x - fx / d
    raise ValueError("SOLVE did not converge. Try another x0.")


# ==============================
# Fractions display
# ==============================
def to_fraction_string(x, max_den=100000):
    try:
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return None
        fr = Fraction(xf).limit_denominator(max_den)
        if abs(float(fr) - xf) < 1e-12:
            return f"{fr.numerator}/{fr.denominator}" if fr.denominator != 1 else f"{fr.numerator}"
    except Exception:
        return None
    return None


# ==============================
# Save/Load + PDF
# ==============================
def snapshot_state():
    data = {}
    for k, v in st.session_state.items():
        if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
            data[k] = v
    return data


def load_state(data: dict):
    for k, v in data.items():
        st.session_state[k] = v


def make_pdf_report(title: str, lines: list[str]) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    _, height = A4
    y = height - 60
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, title)
    y -= 30
    c.setFont("Helvetica", 11)

    for line in lines:
        if y < 60:
            c.showPage()
            y = height - 60
            c.setFont("Helvetica", 11)
        c.drawString(50, y, str(line)[:110])
        y -= 16

    c.showPage()
    c.save()
    return buffer.getvalue()


# ==============================
# Tabs
# ==============================
tab_power, tab_rc, tab_pwm, tab_graph, tab_pid, tab_bode, tab_motor, tab_calc, tab_matrix, tab_tools = st.tabs(
    ["⚡ Power & Ohm", "⏱ RC/RL", "〰 PWM", "📈 Graphs", "🎛 PID", "📉 Bode", "⚙ Motor", "🧮 Od’s Calculator", "🔢 Matrix (≤6×6)", "📦 Save/Load + PDF"]
)

# ==============================
# Power & Ohm
# ==============================
with tab_power:
    colA, colB = st.columns([1, 1])
    with colA:
        system = st.selectbox("System", ["Single-phase", "Three-phase (balanced)"], key="p_system")
    with colB:
        mode = st.selectbox("Mode", ["Forward (compute)", "Goal-seeking (solve missing)"], key="p_mode")

    st.divider()

    if system == "Single-phase":
        if mode == "Forward (compute)":
            c1, c2 = st.columns(2)
            with c1:
                U_val = st.number_input("Voltage", value=230.0, min_value=0.0, key="sp_U_val")
                U_unit = st.selectbox("Unit", ["V", "kV"], index=0, key="sp_U_unit")
            with c2:
                R_val = st.number_input("Resistance", value=10.0, min_value=1e-9, key="sp_R_val")
                R_unit = st.selectbox("Unit", ["Ω", "kΩ"], index=0, key="sp_R_unit")

            U = to_base(U_val, U_unit)
            R = to_base(R_val, R_unit)
            I = U / R
            P = U * I

            st.success(f"Current: {fmt_si(I, 'A')}")
            st.info(f"Power: {fmt_si(P, 'W')}")
            if I > 32:
                st.warning("Current > 32 A: check protection & realism.")
            if P > 5000:
                st.warning("Power > 5 kW: check ratings.")
            if R < 1:
                st.warning("Very low resistance: high current risk.")

        else:
            target = st.selectbox("Solve for", ["R", "U", "I", "P"], key="sp_target")
            c1, c2 = st.columns(2)
            with c1:
                U = st.number_input("Voltage (known)", value=230.0, min_value=0.0, key="spk_U_val")
                I = st.number_input("Current (known)", value=5.0, min_value=0.0, key="spk_I_val")
            with c2:
                R = st.number_input("Resistance (known)", value=10.0, min_value=0.0, key="spk_R_val")
                P = st.number_input("Power (known)", value=1000.0, min_value=0.0, key="spk_P_val")

            if target == "R":
                if I > 0:
                    st.success(f"R = {U / I:.6g} Ω")
                elif P > 0 and U > 0:
                    st.success(f"R = {(U * U) / P:.6g} Ω")
                else:
                    st.error("Provide (U and I) OR (U and P).")

            elif target == "U":
                if R > 0 and I > 0:
                    st.success(f"U = {I * R:.6g} V")
                elif P > 0 and I > 0:
                    st.success(f"U = {P / I:.6g} V")
                else:
                    st.error("Provide (R and I) OR (P and I).")

            elif target == "I":
                if R > 0 and U > 0:
                    st.success(f"I = {U / R:.6g} A")
                elif P > 0 and U > 0:
                    st.success(f"I = {P / U:.6g} A")
                else:
                    st.error("Provide (U and R) OR (P and U).")

            elif target == "P":
                if U > 0 and I > 0:
                    st.success(f"P = {U * I:.6g} W")
                elif U > 0 and R > 0:
                    st.success(f"P = {(U * U) / R:.6g} W")
                else:
                    st.error("Provide (U and I) OR (U and R).")

    else:
        c1, c2 = st.columns(2)
        with c1:
            U = st.number_input("Line voltage U_LL (V)", value=400.0, min_value=0.0, key="tp_U")
            I = st.number_input("Line current I (A)", value=5.0, min_value=0.0, key="tp_I")
        with c2:
            pf = st.number_input("Power factor cos(φ)", value=0.85, min_value=0.0, max_value=1.0, key="tp_pf")

        S = math.sqrt(3) * U * I
        P = S * pf
        Q = math.sqrt(max(S * S - P * P, 0.0))

        st.success(f"Active power P: {fmt_si(P, 'W')}")
        st.info(f"Apparent power S: {fmt_si(S, 'VA')}")
        st.warning(f"Reactive power Q: {fmt_si(Q, 'var')}")


# ==============================
# RC/RL
# ==============================
with tab_rc:
    kind = st.radio("Circuit", ["RC (low-pass)", "RL"], horizontal=True, key="rc_kind")
    if kind.startswith("RC"):
        c1, c2 = st.columns(2)
        with c1:
            Rv = st.number_input("R", value=1000.0, min_value=0.0, key="rc_Rv")
            Ru = st.selectbox("Unit", ["Ω", "kΩ"], key="rc_Ru")
        with c2:
            Cv = st.number_input("C", value=10.0, min_value=0.0, key="rc_Cv")
            Cu = st.selectbox("Unit", ["µF", "F"], key="rc_Cu")
        R = to_base(Rv, Ru)
        C = to_base(Cv, Cu)
        tau = R * C
        st.success(f"τ = {fmt_si(tau, 's')}")
        st.info(f"~2% settling time ≈ 4τ = {fmt_si(4 * tau, 's')}")
    else:
        c1, c2 = st.columns(2)
        with c1:
            Lv = st.number_input("L", value=50.0, min_value=0.0, key="rl_Lv")
            Lu = st.selectbox("Unit", ["mH", "H"], key="rl_Lu")
        with c2:
            Rv = st.number_input("R", value=10.0, min_value=1e-9, key="rl_Rv")
            Ru = st.selectbox("Unit", ["Ω", "kΩ"], key="rl_Ru")
        L = to_base(Lv, Lu)
        R = to_base(Rv, Ru)
        tau = L / R
        st.success(f"τ = {fmt_si(tau, 's')}")
        st.info(f"~2% settling time ≈ 4τ = {fmt_si(4 * tau, 's')}")


# ==============================
# PWM
# ==============================
with tab_pwm:
    c1, c2 = st.columns(2)
    with c1:
        f = st.number_input("Frequency (Hz)", value=1000.0, min_value=0.0, key="pwm_f")
    with c2:
        duty = st.number_input("Duty (%)", value=50.0, min_value=0.0, max_value=100.0, key="pwm_duty") / 100.0

    if f > 0:
        T = 1 / f
        Ton = duty * T
        Toff = (1 - duty) * T
        st.success(f"Period T = {fmt_si(T, 's')}")
        st.info(f"Ton = {fmt_si(Ton, 's')}")
        st.warning(f"Toff = {fmt_si(Toff, 's')}")
    else:
        st.error("Frequency must be > 0.")


# ==============================
# Graphs
# ==============================
with tab_graph:
    graph_type = st.selectbox("Graph", ["RC step response", "PWM waveform", "3-phase P vs PF"], key="g_type")
    if graph_type == "RC step response":
        V = st.number_input("Step voltage (V)", value=5.0, min_value=0.0, key="g_rc_V")
        R = st.number_input("R (Ohm)", value=1000.0, min_value=1e-9, key="g_rc_R")
        C_uF = st.number_input("C (µF)", value=10.0, min_value=0.0, key="g_rc_C")
        C = C_uF * 1e-6
        tau = R * C
        t_end = st.number_input("Plot duration (s)", value=max(5 * tau, 0.05), min_value=0.001, key="g_rc_t")
        t = np.linspace(0, t_end, 600)
        vc = V * (1 - np.exp(-t / tau)) if tau > 0 else np.zeros_like(t)
        fig = plt.figure()
        plt.plot(t, vc)
        plt.xlabel("Time (s)")
        plt.ylabel("Vc (V)")
        plt.title(f"RC Step Response (tau={tau:.4g}s)")
        st.pyplot(fig)

    elif graph_type == "PWM waveform":
        f = st.number_input("Frequency (Hz)", value=1000.0, min_value=1.0, key="g_pwm_f")
        duty = st.number_input("Duty (%)", value=50.0, min_value=0.0, max_value=100.0, key="g_pwm_d") / 100.0
        periods = st.slider("Periods", 1, 20, 5, key="g_pwm_p")
        T = 1 / f
        t = np.linspace(0, periods * T, 2000)
        phase = (t % T) / T
        y = (phase < duty).astype(float)
        fig = plt.figure()
        plt.plot(t, y)
        plt.ylim(-0.2, 1.2)
        plt.xlabel("Time (s)")
        plt.ylabel("PWM (0/1)")
        plt.title("PWM Waveform")
        st.pyplot(fig)

    else:
        U = st.number_input("U_LL (V)", value=400.0, min_value=0.0, key="g_tp_U")
        I = st.number_input("I (A)", value=5.0, min_value=0.0, key="g_tp_I")
        pf = np.linspace(0.1, 1.0, 100)
        P = math.sqrt(3) * U * I * pf
        fig = plt.figure()
        plt.plot(pf, P)
        plt.xlabel("Power factor cos(φ)")
        plt.ylabel("Active power P (W)")
        plt.title("3-phase Active Power vs Power Factor")
        st.pyplot(fig)


# ==============================
# PID
# ==============================
with tab_pid:
    st.caption("Discrete PID on a first-order plant: y' = (-y + u)/tau")
    c1, c2, c3 = st.columns(3)
    with c1:
        Kp = st.number_input("Kp", value=1.0, min_value=0.0, key="pid_Kp")
        Ki = st.number_input("Ki", value=0.2, min_value=0.0, key="pid_Ki")
    with c2:
        Kd = st.number_input("Kd", value=0.0, min_value=0.0, key="pid_Kd")
        tau = st.number_input("Plant tau (s)", value=1.0, min_value=1e-6, key="pid_tau")
    with c3:
        dt = st.number_input("dt (s)", value=0.01, min_value=1e-4, key="pid_dt")
        Tsim = st.number_input("Sim time (s)", value=8.0, min_value=0.1, key="pid_Tsim")

    r = st.number_input("Setpoint", value=1.0, key="pid_r")
    u_min, u_max = st.slider("Actuator limits", -10.0, 10.0, (-10.0, 10.0), key="pid_lim")

    n = int(Tsim / dt)
    t = np.arange(n) * dt
    y = np.zeros(n)
    u = np.zeros(n)
    e_prev = 0.0
    integ = 0.0

    for k in range(1, n):
        e = r - y[k - 1]
        integ += e * dt
        deriv = (e - e_prev) / dt
        u_raw = Kp * e + Ki * integ + Kd * deriv
        u[k] = np.clip(u_raw, u_min, u_max)
        y[k] = y[k - 1] + dt * ((-y[k - 1] + u[k]) / tau)
        e_prev = e

    fig = plt.figure()
    plt.plot(t, y, label="y(t)")
    plt.plot(t, r * np.ones_like(t), "--", label="setpoint")
    plt.xlabel("Time (s)")
    plt.ylabel("Output")
    plt.title("PID Step Response")
    plt.legend()
    st.pyplot(fig)


# ==============================
# Bode
# ==============================
with tab_bode:
    kind = st.selectbox("Model", ["RC low-pass (Vout/Vin)", "RL (I/V)"], key="b_kind")
    f_min = st.number_input("f_min (Hz)", value=1.0, min_value=0.001, key="b_fmin")
    f_max = st.number_input("f_max (Hz)", value=1e5, min_value=1.0, key="b_fmax")
    points = st.slider("Points", 100, 2000, 400, key="b_pts")
    w = 2 * np.pi * np.logspace(np.log10(f_min), np.log10(f_max), points)

    if kind.startswith("RC"):
        R = st.number_input("R (Ohm)", value=1000.0, min_value=1e-9, key="b_R")
        C_uF = st.number_input("C (µF)", value=1.0, min_value=0.0, key="b_C")
        C = C_uF * 1e-6
        H = 1 / (1 + 1j * w * R * C)
        fc = 1 / (2 * np.pi * R * C) if R * C > 0 else float("inf")
        st.info(f"Cutoff fc ≈ {fc:.4g} Hz")
    else:
        L_mH = st.number_input("L (mH)", value=10.0, min_value=0.0, key="b_L")
        R = st.number_input("R (Ohm)", value=10.0, min_value=1e-9, key="b_R2")
        L = L_mH * 1e-3
        H = 1 / (R + 1j * w * L)

    mag_db = 20 * np.log10(np.abs(H) + 1e-18)
    phase_deg = np.angle(H, deg=True)
    f = w / (2 * np.pi)

    fig1 = plt.figure()
    plt.semilogx(f, mag_db)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Bode Magnitude")
    st.pyplot(fig1)

    fig2 = plt.figure()
    plt.semilogx(f, phase_deg)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (deg)")
    plt.title("Bode Phase")
    st.pyplot(fig2)


# ==============================
# Motor
# ==============================
with tab_motor:
    st.caption("Simplified induction motor curve (educational)")
    n_sync = st.number_input("Synchronous speed (rpm)", value=1500.0, min_value=1.0, key="m_n")
    T_max = st.number_input("Max torque (Nm)", value=120.0, min_value=0.0, key="m_T")
    s_peak = st.number_input("Slip at max torque", value=0.2, min_value=0.01, max_value=1.0, key="m_s")

    n = np.linspace(0.0, n_sync * 0.999, 400)
    s = (n_sync - n) / n_sync
    T = T_max * (2 * s * s_peak) / (s * s + s_peak * s_peak + 1e-18)

    fig = plt.figure()
    plt.plot(n, T)
    plt.xlabel("Speed (rpm)")
    plt.ylabel("Torque (Nm)")
    plt.title("Torque-Speed Curve")
    st.pyplot(fig)


# ==============================
# Od’s Calculator (Casio-like exact display)
# ==============================
with tab_calc:
    st.subheader("🧮 Od’s Calculator")

    if "calc_expr" not in st.session_state:
        st.session_state.calc_expr = ""
    if "Ans" not in st.session_state:
        st.session_state.Ans = 0.0
    if "display_mode" not in st.session_state:
        st.session_state.display_mode = "Decimal"
    if "angle_mode" not in st.session_state:
        st.session_state.angle_mode = "Radians"

    def on_expr_change():
        st.session_state.calc_expr = st.session_state.expr_input

    def insert_text(txt: str):
        st.session_state.calc_expr += txt
        st.rerun()

    def backspace():
        st.session_state.calc_expr = st.session_state.calc_expr[:-1]
        st.rerun()

    def clear_all():
        st.session_state.calc_expr = ""
        st.session_state.Ans = 0.0
        st.rerun()

    def toggle_sd():
        st.session_state.display_mode = "Fraction" if st.session_state.display_mode == "Decimal" else "Decimal"
        st.rerun()

    top1, top2, top3 = st.columns([1, 1, 1])
    with top1:
        st.session_state.angle_mode = st.radio("Angle", ["Radians", "Degrees"], horizontal=True, key="od_angle_mode")
    with top2:
        st.write("Display")
        st.write(f"**{st.session_state.display_mode}**")
    with top3:
        if st.button("S↔D", key="od_sd"):
            toggle_sd()

    st.text_input(
        "Expression",
        value=st.session_state.calc_expr,
        key="expr_input",
        on_change=on_expr_change
    )

    t1, t2, t3, t4 = st.columns(4)
    if t1.button("AC", key="od_ac"):
        clear_all()
    if t2.button("DEL", key="od_del"):
        backspace()
    if t3.button("Ans", key="od_ans"):
        insert_text("Ans")
    if t4.button("ENG", key="od_eng"):
        st.info(f"ENG(Ans) = {eng_format(float(st.session_state.Ans), '')}")

    st.caption("Units: 230V, 1kOhm, 100uF, 50Hz | Use Ans | Use x for SOLVE/TABLE")

    r1 = st.columns(6)
    if r1[0].button("sin", key="od_sin"): insert_text("sin(")
    if r1[1].button("cos", key="od_cos"): insert_text("cos(")
    if r1[2].button("tan", key="od_tan"): insert_text("tan(")
    if r1[3].button("asin", key="od_asin"): insert_text("asin(")
    if r1[4].button("acos", key="od_acos"): insert_text("acos(")
    if r1[5].button("atan", key="od_atan"): insert_text("atan(")

    r2 = st.columns(6)
    if r2[0].button("sinh", key="od_sinh"): insert_text("sinh(")
    if r2[1].button("cosh", key="od_cosh"): insert_text("cosh(")
    if r2[2].button("tanh", key="od_tanh"): insert_text("tanh(")
    if r2[3].button("√", key="od_sqrt"): insert_text("sqrt(")
    if r2[4].button("x²", key="od_x2"): insert_text("**2")
    if r2[5].button("x^y", key="od_pow"): insert_text("**")

    r3 = st.columns(6)
    if r3[0].button("log", key="od_log"): insert_text("log(")
    if r3[1].button("ln", key="od_ln"): insert_text("ln(")
    if r3[2].button("10^x", key="od_10x"): insert_text("10**")
    if r3[3].button("e^x", key="od_ex"): insert_text("exp(")
    if r3[4].button("!", key="od_fact"): insert_text("fact(")
    if r3[5].button("π", key="od_pi"): insert_text("pi")

    r4 = st.columns(6)
    if r4[0].button("nCr", key="od_ncr"): insert_text("nCr(")
    if r4[1].button("nPr", key="od_npr"): insert_text("nPr(")
    if r4[2].button("×10^", key="od_exp10"): insert_text("*10**")
    if r4[3].button("(", key="od_lpar"): insert_text("(")
    if r4[4].button(")", key="od_rpar"): insert_text(")")
    if r4[5].button("mod", key="od_mod"): insert_text("%")

    d1 = st.columns(4)
    if d1[0].button("7", key="od_7"): insert_text("7")
    if d1[1].button("8", key="od_8"): insert_text("8")
    if d1[2].button("9", key="od_9"): insert_text("9")
    if d1[3].button("÷", key="od_div"): insert_text("/")

    d2 = st.columns(4)
    if d2[0].button("4", key="od_4"): insert_text("4")
    if d2[1].button("5", key="od_5"): insert_text("5")
    if d2[2].button("6", key="od_6"): insert_text("6")
    if d2[3].button("×", key="od_mul"): insert_text("*")

    d3 = st.columns(4)
    if d3[0].button("1", key="od_1"): insert_text("1")
    if d3[1].button("2", key="od_2"): insert_text("2")
    if d3[2].button("3", key="od_3"): insert_text("3")
    if d3[3].button("-", key="od_sub"): insert_text("-")

    d4 = st.columns(4)
    if d4[0].button("0", key="od_0"): insert_text("0")
    if d4[1].button(".", key="od_dot"): insert_text(".")
    if d4[2].button("+", key="od_add"): insert_text("+")
    if d4[3].button("=", key="od_eq"):
        try:
            expr_raw = st.session_state.calc_expr
            expr_parsed = replace_unit_literals(expr_raw)

            vars_ = {"Ans": st.session_state.Ans}

            if st.session_state.angle_mode == "Degrees":
                def _snap0(v):
                    return 0.0 if abs(v) < 1e-12 else v

                vars_["sin"] = lambda x: _snap0(float(np.sin(np.deg2rad(x))))
                vars_["cos"] = lambda x: _snap0(float(np.cos(np.deg2rad(x))))
                vars_["tan"] = lambda x: _snap0(float(np.tan(np.deg2rad(x))))
                vars_["asin"] = lambda x: float(np.rad2deg(np.arcsin(x)))
                vars_["acos"] = lambda x: float(np.rad2deg(np.arccos(x)))
                vars_["atan"] = lambda x: float(np.rad2deg(np.arctan(x)))

            result = safe_eval(expr_parsed, vars_)
            st.session_state.Ans = float(result)

            result_display = casio_format(result, digits=10, zero_eps=1e-12)

            if st.session_state.display_mode == "Fraction":
                fr = to_fraction_string(result)
                if fr is not None:
                    st.success(f"Result: {fr}  (≈ {result_display})")
                else:
                    st.success(f"Result: {result_display}")
            else:
                st.success(f"Result: {result_display}")

            st.caption(f"Parsed: {expr_parsed}")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()
    st.subheader("SOLVE (find x such that f(x)=0)")
    solve_expr = st.text_input("f(x) = 0", value="x**2 - 2", key="od_solve_expr")
    x0 = st.number_input("Start value x0", value=1.0, key="od_solve_x0")

    if st.button("Solve", key="od_solve_btn"):
        try:
            expr_parsed = replace_unit_literals(solve_expr)
            vars_ = {"Ans": st.session_state.Ans}
            if st.session_state.angle_mode == "Degrees":
                def _snap0(v):
                    return 0.0 if abs(v) < 1e-12 else v
                vars_["sin"] = lambda x: _snap0(float(np.sin(np.deg2rad(x))))
                vars_["cos"] = lambda x: _snap0(float(np.cos(np.deg2rad(x))))
                vars_["tan"] = lambda x: _snap0(float(np.tan(np.deg2rad(x))))
                vars_["asin"] = lambda x: float(np.rad2deg(np.arcsin(x)))
                vars_["acos"] = lambda x: float(np.rad2deg(np.arccos(x)))
                vars_["atan"] = lambda x: float(np.rad2deg(np.arctan(x)))

            root = solve_root(expr_parsed, x0, vars_)
            st.success(f"x ≈ {casio_format(root)}")
        except Exception as e:
            st.error(f"Error: {e}")

    st.subheader("TABLE")
    table_expr = st.text_input("f(x)", value="sin(x)", key="od_table_expr")
    cA, cB, cC = st.columns(3)
    with cA:
        x_start = st.number_input("x_start", value=0.0, key="od_txs")
    with cB:
        x_end = st.number_input("x_end", value=6.283185, key="od_txe")
    with cC:
        step = st.number_input("step", value=0.523598, min_value=1e-12, key="od_tstep")

    if st.button("Generate table", key="od_table_btn"):
        try:
            expr_parsed = replace_unit_literals(table_expr)
            vars_ = {"Ans": st.session_state.Ans}
            if st.session_state.angle_mode == "Degrees":
                def _snap0(v):
                    return 0.0 if abs(v) < 1e-12 else v
                vars_["sin"] = lambda x: _snap0(float(np.sin(np.deg2rad(x))))
                vars_["cos"] = lambda x: _snap0(float(np.cos(np.deg2rad(x))))
                vars_["tan"] = lambda x: _snap0(float(np.tan(np.deg2rad(x))))
                vars_["asin"] = lambda x: float(np.rad2deg(np.arcsin(x)))
                vars_["acos"] = lambda x: float(np.rad2deg(np.arccos(x)))
                vars_["atan"] = lambda x: float(np.rad2deg(np.arctan(x)))

            xs = np.arange(x_start, x_end + step / 2, step)
            ys = []
            for xx in xs:
                vv = dict(vars_)
                vv["x"] = float(xx)
                val = float(safe_eval(expr_parsed, vv))
                ys.append(val)

            df = pd.DataFrame({"x": xs, "f(x)": ys})
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")


# ==============================
# Matrix
# ==============================
with tab_matrix:
    st.subheader("Matrix Toolbox (up to 6×6)")
    n = st.slider("Matrix size n", 1, 6, 3, key="mat_n")
    op = st.selectbox(
        "Operation",
        ["A + B", "A - B", "A × B", "A × v (vector)", "det(A)", "inv(A)", "Solve A x = b"],
        key="mat_op"
    )
    st.caption("A×v is matrix-vector multiplication (e.g., 3×3 times 3×1).")

    A0 = pd.DataFrame(np.zeros((n, n), dtype=float), columns=[str(i + 1) for i in range(n)])
    A_df = st.data_editor(A0, key="A_mat", use_container_width=True, num_rows="fixed")
    A = A_df.to_numpy(dtype=float)

    B = None
    if op in ["A + B", "A - B", "A × B"]:
        B0 = pd.DataFrame(np.zeros((n, n), dtype=float), columns=[str(i + 1) for i in range(n)])
        B_df = st.data_editor(B0, key="B_mat", use_container_width=True, num_rows="fixed")
        B = B_df.to_numpy(dtype=float)

    bvec = None
    if op in ["Solve A x = b", "A × v (vector)"]:
        b0 = pd.DataFrame({"b": np.zeros(n, dtype=float)})
        b_df = st.data_editor(b0, key="b_vec", use_container_width=True, num_rows="fixed")
        bvec = b_df["b"].to_numpy(dtype=float)

    if st.button("Run operation", key="mat_run"):
        try:
            if op == "A + B":
                st.dataframe(A + B, use_container_width=True)
            elif op == "A - B":
                st.dataframe(A - B, use_container_width=True)
            elif op == "A × B":
                st.dataframe(A @ B, use_container_width=True)
            elif op == "A × v (vector)":
                y = A @ bvec
                st.success("Result y = A · v")
                st.dataframe(pd.DataFrame({"y": y}), use_container_width=True)
            elif op == "det(A)":
                st.success(f"det(A) = {float(np.linalg.det(A))}")
            elif op == "inv(A)":
                st.dataframe(np.linalg.inv(A), use_container_width=True)
            elif op == "Solve A x = b":
                x = np.linalg.solve(A, bvec)
                st.success("Solution x")
                st.dataframe(pd.DataFrame({"x": x}), use_container_width=True)

                Ax = A @ x
                residual = Ax - bvec
                st.write("Check (Ax vs b):")
                st.dataframe(pd.DataFrame({"Ax": Ax, "b": bvec, "Ax-b": residual}), use_container_width=True)
                st.info(f"Max |Ax-b| = {np.max(np.abs(residual)):.6g}")

        except np.linalg.LinAlgError as e:
            st.error(f"Linear algebra error: {e}")
        except Exception as e:
            st.error(f"Error: {e}")


# ==============================
# Save/Load + PDF
# ==============================
with tab_tools:
    st.subheader("📦 Save / Load Scenarios (JSON)")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Create snapshot", key="snap_btn"):
            snap = snapshot_state()
            st.session_state["_snapshot_json"] = json.dumps(snap, indent=2)
            st.success("Snapshot created.")

        if "_snapshot_json" in st.session_state:
            st.download_button(
                "Download scenario (.json)",
                data=st.session_state["_snapshot_json"],
                file_name="ee_toolkit_scenario.json",
                mime="application/json",
                key="snap_download"
            )

    with c2:
        up = st.file_uploader("Upload scenario (.json)", type=["json"], key="snap_upload")
        if up is not None:
            try:
                payload = json.loads(up.read().decode("utf-8"))
                load_state(payload)
                st.success("Scenario loaded.")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

    st.divider()
    st.subheader("📄 PDF Report")

    report_title = st.text_input("Report title", value="EE + Automation Toolkit Report", key="pdf_title")
    notes = st.text_area("Notes (optional)", value="", key="pdf_notes")

    if st.button("Generate PDF", key="pdf_btn"):
        lines = ["Generated from EE + Automation Toolkit (Streamlit)."]
        if notes.strip():
            lines.append("Notes: " + notes.strip())

        keys = ["Ans", "calc_expr", "display_mode", "angle_mode"]
        for k in keys:
            if k in st.session_state:
                lines.append(f"{k}: {st.session_state[k]}")

        pdf_bytes = make_pdf_report(report_title, lines)
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name="ee_toolkit_report.pdf",
            mime="application/pdf",
            key="pdf_download"
        )

st.divider()
st.caption('To publish updates later: git add .  |  git commit -m "update"  |  git push')
