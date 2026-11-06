"""
BARA-IFAM: Bara Integrated Flow Assurance Model
Author: Osimi Godsgift Gbubemi
Description: Coupled 1D Steady-State Thermal, Parametric Wax Deposition,
             Hydraulic (Case A & Case B), Sensitivity, and LCC Screening Engine.
Timeframe: 365 Operating Days
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set global publication plotting standards
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 300
})


class BaraIFAMEngine:
    def __init__(self):
        # 1. Central Parameter Register (Strict Classification: L, M, E, A)
        self.register = [
            ("Liquid Flow Rate", "Q_L", 465.0, "STB/day", "Hadi et al. (2018)", "L", "±20%", "Historical reference liquid throughput"),
            ("Historical Oil Rate", "Q_oil", 367.0, "STB/day", "Hadi et al. (2018)", "L", "±20%", "Historical oil production rate (31 Oct 2016)"),
            ("Water Cut", "WC", 21.0, "%", "Hadi et al. (2018)", "L", "Fixed", "Historical produced water fraction"),
            ("API Gravity (2018 Study)", "API", 35.8, "deg API", "Hadi et al. (2018)", "L", "32.4-35.8", "Verified 2018 thesis value (vs 32.4 in draft)"),
            ("Fluid Density", "rho", 845.0, "kg/m3", "Derived from 35.8 API", "A", "835-862", "Operating bulk liquid density"),
            ("Specific Heat Capacity", "Cp", 2200.0, "J/kg/K", "Mehrotra et al. (2020)", "E", "2000-2400", "Typical paraffinic crude heat capacity"),
            ("Pipeline Length", "L", 10000.0, "m", "Assumed Flowline", "A", "5000-15000", "Representative gathering flowline length"),
            ("Internal Pipe Diameter", "D", 0.1524, "m", "Assumed 6-inch ID", "A", "0.10-0.20", "Nominal 6-inch production flowline"),
            ("Pipe Wall Roughness", "epsilon", 4.5e-5, "m", "Commercial Steel", "E", "3e-5 - 9e-5", "Standard carbon steel internal roughness"),
            ("Inlet Temperature", "T_in", 60.0, "deg C", "Parametric Baseline", "A", "50.0-75.0", "Wellhead/manifold entry temperature"),
            ("Ambient Temperature", "T_amb", 35.0, "deg C", "Muglad Basin Surface", "L/A", "30.0-40.0", "Regional ambient ground temperature"),
            ("Overall Heat Transfer Coeff", "U", 5.0, "W/m2/K", "Uninsulated Steel", "A", "3.5-6.5", "Baseline overall heat loss coefficient"),
            ("Assumed WAT", "T_wat", 36.5, "deg C", "Parametric / Draft M", "M/A", "33.5-39.5", "Unverified in 2018 lab; treated parametrically"),
            ("Effective Deposition Coeff", "k_m", 2.5e-6, "kg/m2/s", "Screening Parameter", "A", "±30%", "Lumped diffusive mass transfer parameter"),
            ("Critical Shear Stress", "tau_c", 0.15, "Pa", "Literature Estimate", "E", "0.10-0.25", "Shear threshold for mechanical sloughing"),
            ("Shear Removal Coeff", "k_r", 1.0e-7, "kg/m2/s/Pa", "Screening Parameter", "A", "0 - 5e-7", "Sloughing kinetic rate constant"),
            ("Wax Deposit Density", "rho_dep", 850.0, "kg/m3", "Azevedo & Teixeira (2003)", "E", "820-880", "Solid-gel wax deposit bulk density"),
            ("Effective Dynamic Viscosity", "mu", 0.015, "Pa.s", "Parametric / 15 cP", "A", "±20%", "Effective single-phase operating viscosity"),
            ("Simulation Operating Time", "t_days", 365.0, "days", "Project Standard", "A", "180-365", "Full 1-year uninhibited accumulation period")
        ]

        # Base numerical attributes
        self.Q_STB = 465.0
        self.rho = 845.0
        self.Cp = 2200.0
        self.L = 10000.0
        self.D = 0.1524
        self.epsilon = 4.5e-5
        self.T_in = 60.0
        self.T_amb = 35.0
        self.U = 5.0
        self.WAT = 36.5
        self.k_m = 2.5e-6
        self.tau_c = 0.15
        self.k_r = 1.0e-7
        self.rho_dep = 850.0
        self.mu = 0.015
        self.t_days = 365.0

        self.N = 100
        self.x = np.linspace(0, self.L, self.N + 1)
        self.dx = self.L / self.N

    def solve_system(self, Q_mult=1.0, T_in_delta=0.0, T_amb_delta=0.0,
                     U_mult=1.0, WAT_delta=0.0, km_mult=1.0, mu_mult=1.0, days=None):
        """Executes coupled thermal, deposition, and hydraulic modules."""
        if days is None:
            days = self.t_days

        Q_stb = self.Q_STB * Q_mult
        Q_m3s = Q_stb * 0.15898729 / 86400.0
        m_dot = Q_m3s * self.rho

        T_i = self.T_in + T_in_delta
        T_a = self.T_amb + T_amb_delta
        U_val = self.U * U_mult
        WAT_val = self.WAT + WAT_delta
        km_val = self.k_m * km_mult
        mu_val = self.mu * mu_mult

        # 1. Thermal Profile
        P_perim = np.pi * self.D
        lam = (U_val * P_perim) / (m_dot * self.Cp)
        T_prof = T_a + (T_i - T_a) * np.exp(-lam * self.x)

        if T_i > WAT_val and WAT_val > T_a:
            x_wat = min(self.L, (1.0 / lam) * np.log((T_i - T_a) / (WAT_val - T_a)))
        elif T_i <= WAT_val:
            x_wat = 0.0
        else:
            x_wat = np.nan

        # Clean-pipe reference hydraulics
        A_clean = np.pi * (self.D ** 2) / 4.0
        v_clean = Q_m3s / A_clean
        Re_clean = (self.rho * v_clean * self.D) / mu_val
        if Re_clean < 2100.0:
            f_clean = 64.0 / Re_clean
        else:
            f_clean = 0.25 / (np.log10(self.epsilon / (3.7 * self.D) + 5.74 / (Re_clean ** 0.9))) ** 2
        tau_w_clean = f_clean * self.rho * (v_clean ** 2) / 8.0
        dP_clean = f_clean * (self.L / self.D) * (self.rho * (v_clean ** 2) / 2.0)

        # 2. Wax Deposition Profile over time t
        t_sec = days * 86400.0
        driving_force = np.maximum(0.0, (WAT_val - T_prof) / WAT_val)
        J_d = km_val * driving_force
        J_r = self.k_r * np.maximum(0.0, tau_w_clean - self.tau_c)
        J_net = np.maximum(0.0, J_d - J_r)

        delta_prof = (J_net * t_sec) / self.rho_dep
        delta_prof = np.minimum(delta_prof, 0.475 * self.D)  # Cap at 95% bore

        # 3. Hydraulic Performance (Discretized Segment Integration)
        D_eff_nodes = self.D - 2.0 * delta_prof
        D_eff_seg = 0.5 * (D_eff_nodes[:-1] + D_eff_nodes[1:])
        A_eff_seg = np.pi * (D_eff_seg ** 2) / 4.0
        v_seg = Q_m3s / A_eff_seg
        Re_seg = (self.rho * v_seg * D_eff_seg) / mu_val

        f_seg = np.where(
            Re_seg < 2100.0,
            64.0 / Re_seg,
            0.25 / (np.log10(self.epsilon / (3.7 * D_eff_seg) + 5.74 / (Re_seg ** 0.9))) ** 2
        )
        dP_seg = f_seg * (self.dx / D_eff_seg) * (self.rho * (v_seg ** 2) / 2.0)
        dP_total = np.sum(dP_seg)
        dP_mult = dP_total / dP_clean

        # Case B: Achievable Throughput under Fixed Pressure Differential
        Q_case_B = Q_stb / dP_mult

        return {
            "x": self.x,
            "T_prof": T_prof,
            "delta_mm": delta_prof * 1000.0,
            "D_eff_mm": D_eff_nodes * 1000.0,
            "T_in": T_i,
            "T_out": T_prof[-1],
            "T_avg": np.mean(T_prof),
            "WAT_margin_out": T_prof[-1] - WAT_val,
            "x_WAT_m": x_wat,
            "max_delta_mm": np.max(delta_prof) * 1000.0,
            "min_D_eff_mm": np.min(D_eff_nodes) * 1000.0,
            "diam_red_pct": (1.0 - np.min(D_eff_nodes) / self.D) * 100.0,
            "dP_clean_kPa": dP_clean / 1000.0,
            "dP_dep_kPa": dP_total / 1000.0,
            "dP_mult": dP_mult,
            "dP_inc_pct": (dP_mult - 1.0) * 100.0,
            "Q_caseB_STB": Q_case_B
        }

    def run_all_scenarios(self):
        configs = {
            "S0_Baseline": {},
            "S1_Reduced_Flow": {"Q_mult": 0.80},
            "S2_Lower_Inlet_T": {"T_in_delta": -10.0},
            "S3_Higher_WAT": {"WAT_delta": 3.0},
            "S4_Increased_Heat_Loss": {"U_mult": 1.30},
            "S5_Improved_Insulation": {"U_mult": 0.70},
            "S6_Chemical_Inhibition": {"km_mult": 0.65},
            "S7_Integrated_Mitigation": {"T_in_delta": 15.0, "U_mult": 0.70, "km_mult": 0.65}
        }
        rows = []
        profiles = {}
        for name, kw in configs.items():
            res = self.solve_system(**kw)
            profiles[name] = res
            rows.append({
                "Scenario": name,
                "Outlet_T_C": round(res["T_out"], 2),
                "Avg_T_C": round(res["T_avg"], 2),
                "x_WAT_m": round(res["x_WAT_m"], 1) if not np.isnan(res["x_WAT_m"]) else np.nan,
                "Max_Deposit_mm": round(res["max_delta_mm"], 2),
                "Min_Deff_mm": round(res["min_D_eff_mm"], 2),
                "Diam_Reduction_%": round(res["diam_red_pct"], 2),
                "Clean_dP_kPa": round(res["dP_clean_kPa"], 2),
                "Deposited_dP_kPa": round(res["dP_dep_kPa"], 2),
                "dP_Multiplier": round(res["dP_mult"], 3),
                "CaseB_Flow_STB_d": round(res["Q_caseB_STB"], 1)
            })
        return pd.DataFrame(rows), profiles

    def run_sensitivity_analysis(self):
        base = self.solve_system()
        params = [
            ("Flow Rate (Q)", "Q_mult", 0.80, 1.0, 1.20, 0.40),
            ("Inlet Temp (T_in)", "T_in_delta", -10.0, 0.0, 10.0, 20.0 / self.T_in),
            ("Ambient Temp (T_amb)", "T_amb_delta", -5.0, 0.0, 5.0, 10.0 / self.T_amb),
            ("Heat Transfer (U)", "U_mult", 0.80, 1.0, 1.20, 0.40),
            ("Assumed WAT", "WAT_delta", -3.0, 0.0, 3.0, 6.0 / self.WAT),
            ("Deposition Coeff (k_m)", "km_mult", 0.70, 1.0, 1.30, 0.60),
            ("Viscosity (mu)", "mu_mult", 0.80, 1.0, 1.20, 0.40)
        ]
        sens_rows = []
        for label, key, low_v, _, high_v, dX_rel in params:
            res_low = self.solve_system(**{key: low_v})
            res_high = self.solve_system(**{key: high_v})

            S_thick = ((res_high["max_delta_mm"] - res_low["max_delta_mm"]) / base["max_delta_mm"]) / dX_rel
            S_mult = ((res_high["dP_mult"] - res_low["dP_mult"]) / base["dP_mult"]) / dX_rel

            sens_rows.append({
                "Parameter": label,
                "Low_Thickness_mm": round(res_low["max_delta_mm"], 2),
                "Base_Thickness_mm": round(base["max_delta_mm"], 2),
                "High_Thickness_mm": round(res_high["max_delta_mm"], 2),
                "S_i_Thickness": round(S_thick, 3),
                "Low_dP_Mult": round(res_low["dP_mult"], 3),
                "Base_dP_Mult": round(base["dP_mult"], 3),
                "High_dP_Mult": round(res_high["dP_mult"], 3),
                "S_i_dP_Mult": round(S_mult, 3)
            })
        df_sens = pd.DataFrame(sens_rows)
        df_sens["Rank_dP_Mult"] = df_sens["S_i_dP_Mult"].abs().rank(ascending=False).astype(int)
        return df_sens.sort_values("Rank_dP_Mult")

    def export_deliverables(self, df_scen, profiles, df_sens):
        # Export multi-sheet Excel Workbook
        reg_df = pd.DataFrame(self.register, columns=[
            "Parameter", "Symbol", "Value", "Unit", "Source", "Class", "Range", "Justification"
        ])
        with pd.ExcelWriter("BARA_IFAM_Deliverables.xlsx", engine="openpyxl") as writer:
            reg_df.to_excel(writer, sheet_name="Parameter_Register", index=False)
            df_scen.to_excel(writer, sheet_name="Scenario_Matrix_365d", index=False)
            df_sens.to_excel(writer, sheet_name="Sensitivity_Analysis", index=False)

            # Spatial profiles sheet
            prof_df = pd.DataFrame({"Distance_m": self.x})
            for k, v in profiles.items():
                prof_df[f"{k}_T_C"] = v["T_prof"]
                prof_df[f"{k}_Wax_mm"] = v["delta_mm"]
            prof_df.to_excel(writer, sheet_name="Axial_Profiles", index=False)

        # Figure 1: Axial Temperature Profile
        plt.figure(figsize=(8, 4.5))
        for k in ["S0_Baseline", "S2_Lower_Inlet_T", "S5_Improved_Insulation", "S7_Integrated_Mitigation"]:
            plt.plot(self.x / 1000.0, profiles[k]["T_prof"], label=k, linewidth=2)
        plt.axhline(self.WAT, color='r', linestyle='--', label='Assumed WAT (36.5°C [M/A])')
        plt.xlabel("Axial Distance Along Pipeline (km)")
        plt.ylabel("Bulk Fluid Temperature (°C)")
        plt.title("BARA-IFAM: Axial Temperature Distribution (Calculated from Assumed Geometry)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("Fig1_Axial_Temperature_Profile.png")
        plt.close()

        # Figure 2: Deposit Thickness vs Distance (365 Days)
        plt.figure(figsize=(8, 4.5))
        for k in ["S0_Baseline", "S3_Higher_WAT", "S5_Improved_Insulation", "S6_Chemical_Inhibition", "S7_Integrated_Mitigation"]:
            plt.plot(self.x / 1000.0, profiles[k]["delta_mm"], label=k, linewidth=2)
        plt.xlabel("Axial Distance Along Pipeline (km)")
        plt.ylabel("Wax Deposit Thickness (mm)")
        plt.title("BARA-IFAM: Spatial Wax Deposit Profile After 365 Days (Parametric Simulation)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("Fig3_Spatial_Wax_Thickness.png")
        plt.close()


if __name__ == "__main__":
    engine = BaraIFAMEngine()
    df_scenarios, spatial_profiles = engine.run_all_scenarios()
    df_sensitivity = engine.run_sensitivity_analysis()
    engine.export_deliverables(df_scenarios, spatial_profiles, df_sensitivity)
    print(df_scenarios.to_string(index=False))
    print("\n", df_sensitivity.to_string(index=False))
