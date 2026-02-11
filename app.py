import math
import streamlit as st

st.set_page_config(page_title="EE Toolkit", page_icon="⚡", layout="centered")

st.title("⚡ Electrical Engineering Toolkit")
st.caption("Single-phase / Three-phase + reverse calculations + checks (student/automation oriented).")

# ---------- helpers ----------
def to_base(value: float, unit: str) -> float:
    """Convert value to base unit."""
    factors = {
        "V": 1.0, "kV": 1e3,
        "A": 1.0, "mA": 1e-3,
        "Ω": 1.0, "kΩ": 1e3,
        "W": 1.0, "kW": 1e3,
        "VA": 1.0, "kVA": 1e3,
        "var": 1.0, "kvar": 1e3,
        "Hz": 1.0, "kHz": 1e3,
        "s": 1.0, "ms": 1e-3,
        "F": 1.0, "µF": 1e-6,
        "H": 1.0, "mH": 1e-3,
        "%": 0.01,
    }
    return float(value) * factors[unit]

def fmt_si(x: float, base_unit: str) -> str:
    """Pretty format with engineering prefixes."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    x = float(x)
    prefixes = [
        (1e9, "G"), (1e6, "M"), (1e3, "k"),
        (1.0, ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n")
    ]
    ax = abs(x)
    for f, p in prefixes:
        if ax >= f:
            return f"{x / f:.3f} {p}{base_unit}"
    return f"{x:.3e} {base_unit}"

def warn_if(condition: bool, msg: str):
    if condition:
        st.warning(msg)

# ---------- UI ----------
tab_calc, tab_rc, tab_pwm = st.tabs(["⚡ Power & Ohm", "⏱ RC/RL (Automation)", "〰 PWM"])

with tab_calc:
    colA, colB = st.columns([1, 1])
    with colA:
        system = st.selectbox("System", ["Single-phase", "Three-phase (balanced)"])
    with colB:
        mode = st.selectbox("Mode", ["Forward (compute)", "Goal-seeking (solve missing)"])

    st.divider()

    if system == "Single-phase":
        if mode == "Forward (compute)":
            st.subheader("Single-phase (forward)")
            c1, c2 = st.columns(2)
            with c1:
                U_val = st.number_input("Voltage", value=230.0, min_value=0.0)
                U_unit = st.selectbox("Unit", ["V", "kV"], index=0, key="u1")
            with c2:
                R_val = st.number_input("Resistance", value=10.0, min_value=1e-9)
                R_unit = st.selectbox("Unit", ["Ω", "kΩ"], index=0, key="r1")

            U = to_base(U_val, U_unit)
            R = to_base(R_val, R_unit)

            I = U / R
            P = U * I

            st.success(f"Current: {fmt_si(I, 'A')}")
            st.info(f"Power: {fmt_si(P, 'W')}")

            warn_if(I > 32, "Current > 32 A: check if this is realistic for your circuit / protection.")
            warn_if(P > 5000, "Power > 5 kW: check component ratings and safety assumptions.")
            warn_if(R < 1, "Very low resistance (<1 Ω): high currents likely (heating/short-circuit risk).")

        else:
            st.subheader("Single-phase (goal-seeking)")
            target = st.selectbox("Solve for", ["Resistance (R)", "Voltage (U)", "Current (I)", "Power (P)"])

            # Choose which formula set: Ohm + P=UI
            c1, c2 = st.columns(2)
            with c1:
                U_val = st.number_input("Voltage (known)", value=230.0, min_value=0.0)
                U_unit = st.selectbox("Unit", ["V", "kV"], index=0, key="u2")
                I_val = st.number_input("Current (known)", value=5.0, min_value=0.0)
                I_unit = st.selectbox("Unit", ["A", "mA"], index=0, key="i2")
            with c2:
                R_val = st.number_input("Resistance (known)", value=10.0, min_value=0.0)
                R_unit = st.selectbox("Unit", ["Ω", "kΩ"], index=0, key="r2")
                P_val = st.number_input("Power (known)", value=1000.0, min_value=0.0)
                P_unit = st.selectbox("Unit", ["W", "kW"], index=0, key="p2")

            U = to_base(U_val, U_unit)
            I = to_base(I_val, I_unit)
            R = to_base(R_val, R_unit)
            P = to_base(P_val, P_unit)

            # We will use whichever inputs are relevant; user may leave some at 0.
            result = None
            if target == "Resistance (R)":
                # Prefer U and I if I>0 else use U and P if P>0
                if I > 0:
                    result = U / I
                    st.success(f"R = {fmt_si(result, 'Ω')}")
                elif P > 0 and U > 0:
                    result = (U * U) / P
                    st.success(f"R = {fmt_si(result, 'Ω')}")
                else:
                    st.error("To solve R, provide (U and I) OR (U and P).")

            elif target == "Voltage (U)":
                if R > 0 and I > 0:
                    result = I * R
                    st.success(f"U = {fmt_si(result, 'V')}")
                elif P > 0 and I > 0:
                    result = P / I
                    st.success(f"U = {fmt_si(result, 'V')}")
                else:
                    st.error("To solve U, provide (R and I) OR (P and I).")

            elif target == "Current (I)":
                if R > 0 and U > 0:
                    result = U / R
                    st.success(f"I = {fmt_si(result, 'A')}")
                elif P > 0 and U > 0:
                    result = P / U
                    st.success(f"I = {fmt_si(result, 'A')}")
                else:
                    st.error("To solve I, provide (U and R) OR (P and U).")

            elif target == "Power (P)":
                if U > 0 and I > 0:
                    result = U * I
                    st.success(f"P = {fmt_si(result, 'W')}")
                elif U > 0 and R > 0:
                    result = (U * U) / R
                    st.success(f"P = {fmt_si(result, 'W')}")
                else:
                    st.error("To solve P, provide (U and I) OR (U and R).")

            if result is not None:
                warn_if(result < 0, "Negative result: check sign conventions / inputs.")
                warn_if(target == "Current (I)" and result > 32, "Current > 32 A: check protection/cable sizing assumptions.")

    else:
        if mode == "Forward (compute)":
            st.subheader("Three-phase (forward)")
            c1, c2 = st.columns(2)
            with c1:
                U_val = st.number_input("Line Voltage U_LL", value=400.0, min_value=0.0)
                U_unit = st.selectbox("Unit", ["V", "kV"], index=0, key="u3")
                I_val = st.number_input("Line Current I", value=5.0, min_value=0.0)
                I_unit = st.selectbox("Unit", ["A", "mA"], index=0, key="i3")
            with c2:
                pf = st.number_input("Power Factor cos(φ)", value=0.85, min_value=0.0, max_value=1.0)
                show_SQ = st.checkbox("Show S & Q", value=True)

            U = to_base(U_val, U_unit)
            I = to_base(I_val, I_unit)

            S = math.sqrt(3) * U * I
            P = S * pf
            Q = math.sqrt(max(S * S - P * P, 0.0))

            st.success(f"Active Power P: {fmt_si(P, 'W')}")
            if show_SQ:
                st.info(f"Apparent Power S: {fmt_si(S, 'VA')}")
                st.warning(f"Reactive Power Q: {fmt_si(Q, 'var')}")

            warn_if(pf < 0.7, "Low power factor (<0.7): consider compensation (capacitor bank) in real systems.")
            warn_if(P > 15000, "Power > 15 kW: check if this matches your installation/lab constraints.")

        else:
            st.subheader("Three-phase (goal-seeking)")
            target = st.selectbox("Solve for", ["Line Current (I)", "Line Voltage (U_LL)", "Power Factor (cosφ)", "Active Power (P)"])

            c1, c2 = st.columns(2)
            with c1:
                U_val = st.number_input("Line Voltage U_LL (known)", value=400.0, min_value=0.0)
                U_unit = st.selectbox("Unit", ["V", "kV"], index=0, key="u4")
                I_val = st.number_input("Line Current I (known)", value=5.0, min_value=0.0)
                I_unit = st.selectbox("Unit", ["A", "mA"], index=0, key="i4")
            with c2:
                pf = st.number_input("Power Factor cos(φ) (known)", value=0.85, min_value=0.0, max_value=1.0)
                P_val = st.number_input("Active Power P (known)", value=5000.0, min_value=0.0)
                P_unit = st.selectbox("Unit", ["W", "kW"], index=0, key="p4")

            U = to_base(U_val, U_unit)
            I = to_base(I_val, I_unit)
            P = to_base(P_val, P_unit)

            # P = sqrt(3) U I pf
            if target == "Line Current (I)":
                if U > 0 and pf > 0:
                    result = P / (math.sqrt(3) * U * pf) if P > 0 else 0.0
                    st.success(f"I = {fmt_si(result, 'A')}")
                else:
                    st.error("To solve I, provide U_LL and cosφ, and a non-zero P.")
            elif target == "Line Voltage (U_LL)":
                if I > 0 and pf > 0:
                    result = P / (math.sqrt(3) * I * pf) if P > 0 else 0.0
                    st.success(f"U_LL = {fmt_si(result, 'V')}")
                else:
                    st.error("To solve U_LL, provide I and cosφ, and a non-zero P.")
            elif target == "Power Factor (cosφ)":
                denom = math.sqrt(3) * U * I
                if denom > 0 and P > 0:
                    result = min(max(P / denom, 0.0), 1.0)
                    st.success(f"cos(φ) = {result:.3f}")
                else:
                    st.error("To solve cosφ, provide U_LL, I, and a non-zero P.")
            elif target == "Active Power (P)":
                if U > 0 and I > 0:
                    result = math.sqrt(3) * U * I * pf
                    st.success(f"P = {fmt_si(result, 'W')}")
                else:
                    st.error("To solve P, provide U_LL and I (and cosφ).")

with tab_rc:
    st.subheader("RC / RL quick tools (automation)")
    kind = st.radio("Choose circuit", ["RC (low-pass)", "RL"], horizontal=True)
    st.caption("Useful for step response / time constant τ and 2% settling time approximation (~4τ).")

    if kind.startswith("RC"):
        c1, c2 = st.columns(2)
        with c1:
            Rv = st.number_input("R", value=1000.0, min_value=0.0)
            Ru = st.selectbox("Unit", ["Ω", "kΩ"], index=0, key="rc_r")
        with c2:
            Cv = st.number_input("C", value=10.0, min_value=0.0)
            Cu = st.selectbox("Unit", ["µF", "F"], index=0, key="rc_c")

        R = to_base(Rv, Ru)
        C = to_base(Cv, Cu)

        tau = R * C
        t_settle = 4 * tau

        st.success(f"Time constant τ = {fmt_si(tau, 's')}")
        st.info(f"~2% settling time ≈ 4τ = {fmt_si(t_settle, 's')}")

        warn_if(tau > 10, "Large τ (>10 s): response will be slow.")

    else:
        c1, c2 = st.columns(2)
        with c1:
            Lv = st.number_input("L", value=50.0, min_value=0.0)
            Lu = st.selectbox("Unit", ["mH", "H"], index=0, key="rl_l")
        with c2:
            Rv = st.number_input("R", value=10.0, min_value=1e-9)
            Ru = st.selectbox("Unit", ["Ω", "kΩ"], index=0, key="rl_r")

        L = to_base(Lv, Lu)
        R = to_base(Rv, Ru)

        tau = L / R
        t_settle = 4 * tau

        st.success(f"Time constant τ = {fmt_si(tau, 's')}")
        st.info(f"~2% settling time ≈ 4τ = {fmt_si(t_settle, 's')}")

with tab_pwm:
    st.subheader("PWM helper (automation / drives)")
    st.caption("Duty cycle, frequency, period, on/off times. Great for PLC/MCU/drive basics.")

    c1, c2 = st.columns(2)
    with c1:
        f_val = st.number_input("Frequency", value=1000.0, min_value=0.0)
        f_unit = st.selectbox("Unit", ["Hz", "kHz"], index=0, key="pwm_f")
    with c2:
        duty_percent = st.number_input("Duty (%)", value=50.0, min_value=0.0, max_value=100.0)

    f = to_base(f_val, f_unit)
    duty = duty_percent / 100.0

    if f > 0:
        T = 1.0 / f
        Ton = duty * T
        Toff = (1 - duty) * T

        st.success(f"Period T = {fmt_si(T, 's')}")
        st.info(f"On-time Ton = {fmt_si(Ton, 's')}")
        st.warning(f"Off-time Toff = {fmt_si(Toff, 's')}")
    else:
        st.error("Frequency must be > 0.")

st.divider()
st.caption("Next: add motor current estimation, cable sizing hints, or PID step response plot.")
