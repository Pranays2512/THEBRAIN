#pragma once

#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <iomanip>

namespace brain2 {
namespace discovery {

struct ObservationPoint {
    std::map<std::string, double> inputs;
    double output;
};

struct DiscoveredLaw {
    bool verified = false;
    std::string law_name;
    std::string target_var;
    std::vector<std::string> input_vars;
    std::string equation;
    double r2_score = 0.0;
    double mse = 0.0;
    std::vector<std::string> discovery_steps;
    std::string explanation;

    std::string to_json() const {
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"verified\": " << (verified ? "true" : "false") << ",\n";
        oss << "  \"law_name\": \"" << law_name << "\",\n";
        oss << "  \"target_var\": \"" << target_var << "\",\n";
        oss << "  \"equation\": \"" << equation << "\",\n";
        oss << "  \"r2_score\": " << r2_score << ",\n";
        oss << "  \"mse\": " << mse << ",\n";
        oss << "  \"discovery_steps\": [\n";
        for (size_t i = 0; i < discovery_steps.size(); ++i) {
            oss << "    \"" << discovery_steps[i] << "\"";
            if (i + 1 < discovery_steps.size()) oss << ",";
            oss << "\n";
        }
        oss << "  ],\n";
        oss << "  \"explanation\": \"" << explanation << "\"\n";
        oss << "}";
        return oss.str();
    }
};

class DiscoveryEngine {
private:
    std::map<std::string, std::vector<ObservationPoint>> preloaded_datasets;

public:
    DiscoveryEngine() {
        preload_canonical_scientific_datasets();
    }

    void preload_canonical_scientific_datasets() {
        // 1. Kepler's 3rd Law Dataset (Distance R in AU -> Orbital Period T in Years)
        // Planets: Mercury(0.387, 0.241), Venus(0.723, 0.615), Earth(1.0, 1.0), Mars(1.524, 1.881), Jupiter(5.204, 11.862)
        std::vector<ObservationPoint> kepler = {
            {{{"R", 0.387}}, 0.241},
            {{{"R", 0.723}}, 0.615},
            {{{"R", 1.000}}, 1.000},
            {{{"R", 1.524}}, 1.881},
            {{{"R", 5.204}}, 11.862}
        };
        preloaded_datasets["kepler"] = kepler;
        preloaded_datasets["kepler_planetary"] = kepler;

        // 2. Newton's 2nd Law Dataset (mass m, accel a -> Force F)
        std::vector<ObservationPoint> newton = {
            {{{"m", 2.0}, {"a", 3.0}}, 6.0},
            {{{"m", 5.0}, {"a", 4.0}}, 20.0},
            {{{"m", 10.0}, {"a", 2.5}}, 25.0},
            {{{"m", 3.5}, {"a", 8.0}}, 28.0},
            {{{"m", 12.0}, {"a", 1.5}}, 18.0}
        };
        preloaded_datasets["newton"] = newton;
        preloaded_datasets["newton_force"] = newton;

        // 3. Ohm's Law Dataset (Current I, Resistance R -> Voltage V)
        std::vector<ObservationPoint> ohm = {
            {{{"I", 1.5}, {"R", 10.0}}, 15.0},
            {{{"I", 2.0}, {"R", 25.0}}, 50.0},
            {{{"I", 0.5}, {"R", 100.0}}, 50.0},
            {{{"I", 3.0}, {"R", 4.0}}, 12.0},
            {{{"I", 4.0}, {"R", 5.5}}, 22.0}
        };
        preloaded_datasets["ohm"] = ohm;
        preloaded_datasets["ohm_circuit"] = ohm;

        // 4. Boyle's Ideal Gas Law (Volume V -> Pressure P) (P * V = 100)
        std::vector<ObservationPoint> boyle = {
            {{{"V", 2.0}}, 50.0},
            {{{"V", 4.0}}, 25.0},
            {{{"V", 5.0}}, 20.0},
            {{{"V", 10.0}}, 10.0},
            {{{"V", 20.0}}, 5.0}
        };
        preloaded_datasets["boyle"] = boyle;
        preloaded_datasets["boyle_gas"] = boyle;

        // 5. Kinetic Energy Dataset (mass m, velocity v -> Energy E = 0.5 * m * v^2)
        std::vector<ObservationPoint> kinetic = {
            {{{"m", 2.0}, {"v", 3.0}}, 9.0},
            {{{"m", 4.0}, {"v", 2.0}}, 8.0},
            {{{"m", 10.0}, {"v", 4.0}}, 80.0},
            {{{"m", 6.0}, {"v", 5.0}}, 75.0},
            {{{"m", 1.0}, {"v", 10.0}}, 50.0}
        };
        preloaded_datasets["kinetic_energy"] = kinetic;
        preloaded_datasets["energy"] = kinetic;

        // 6. Coulomb's Law Dataset (q1, q2, r -> Force F = 9.0 * q1 * q2 / r^2)
        std::vector<ObservationPoint> coulomb = {
            {{{"q1", 2.0}, {"q2", 3.0}, {"r", 1.0}}, 54.0},
            {{{"q1", 4.0}, {"q2", 2.0}, {"r", 2.0}}, 18.0},
            {{{"q1", 1.0}, {"q2", 1.0}, {"r", 3.0}}, 1.0},
            {{{"q1", 3.0}, {"q2", 3.0}, {"r", 3.0}}, 9.0}
        };
        preloaded_datasets["coulomb"] = coulomb;
        preloaded_datasets["coulomb_electrostatic"] = coulomb;

        // 7. Ideal Gas Law (n, T, V -> Pressure P = 8.314 * n * T / V)
        std::vector<ObservationPoint> ideal_gas = {
            {{{"n", 1.0}, {"T", 300.0}, {"V", 10.0}}, 249.42},
            {{{"n", 2.0}, {"T", 300.0}, {"V", 10.0}}, 498.84},
            {{{"n", 1.0}, {"T", 600.0}, {"V", 10.0}}, 498.84},
            {{{"n", 2.0}, {"T", 400.0}, {"V", 20.0}}, 332.56}
        };
        preloaded_datasets["ideal_gas"] = ideal_gas;
        preloaded_datasets["ideal_gas_pv"] = ideal_gas;

        // 8. Carnot Heat Engine Efficiency (Tc, Th -> eta = 1.0 - Tc / Th)
        std::vector<ObservationPoint> carnot = {
            {{{"Tc", 300.0}, {"Th", 600.0}}, 0.50},
            {{{"Tc", 200.0}, {"Th", 800.0}}, 0.75},
            {{{"Tc", 400.0}, {"Th", 500.0}}, 0.20},
            {{{"Tc", 300.0}, {"Th", 1200.0}}, 0.75}
        };
        preloaded_datasets["carnot"] = carnot;
        preloaded_datasets["carnot_efficiency"] = carnot;

        // 9. Poiseuille Fluid Resistance & Flow (r, dP -> Flow Q = 2.0 * r^4 * dP)
        std::vector<ObservationPoint> poiseuille = {
            {{{"r", 1.0}, {"dP", 10.0}}, 20.0},
            {{{"r", 2.0}, {"dP", 5.0}}, 160.0},
            {{{"r", 1.0}, {"dP", 25.0}}, 50.0},
            {{{"r", 2.0}, {"dP", 10.0}}, 320.0}
        };
        preloaded_datasets["poiseuille"] = poiseuille;
        preloaded_datasets["poiseuille_fluid"] = poiseuille;
    }

    bool has_dataset(const std::string& name) const {
        return preloaded_datasets.find(name) != preloaded_datasets.end();
    }

    const std::vector<ObservationPoint>* get_dataset(const std::string& name) const {
        auto it = preloaded_datasets.find(name);
        if (it != preloaded_datasets.end()) return &it->second;
        return nullptr;
    }

    // ── Goodness-of-fit, actually computed ───────────────────────────────────
    // Every branch that claims `verified` must pass its candidate through
    // this first. r2_score/mse used to be assigned as literal constants, so a
    // structureless 5-point dataset was reported as a verified scientific law
    // with r2 = 0.999. Measured false-discovery rate on pure noise was 100%
    // (200/200), see eval/heldout_probe.cpp section L.
    struct FitStats { double r2 = 0.0, mse = 0.0, max_rel_resid = 0.0; };

    template <typename Pred>
    static FitStats fit_stats(const std::vector<ObservationPoint>& data, Pred predict) {
        FitStats fs;
        if (data.empty()) return fs;
        double mean = 0.0;
        for (const auto& pt : data) mean += pt.output;
        mean /= (double)data.size();

        double ss_res = 0.0, ss_tot = 0.0;
        for (const auto& pt : data) {
            const double pred = predict(pt);
            if (!std::isfinite(pred)) {
                fs.r2 = -1.0; fs.mse = INFINITY; fs.max_rel_resid = INFINITY;
                return fs;
            }
            const double r = pt.output - pred;
            ss_res += r * r;
            ss_tot += (pt.output - mean) * (pt.output - mean);
            fs.max_rel_resid = std::max(fs.max_rel_resid,
                std::abs(r) / std::max(1e-12, std::abs(pt.output)));
        }
        fs.mse = ss_res / (double)data.size();
        fs.r2  = (ss_tot > 1e-12) ? (1.0 - ss_res / ss_tot)
                                  : (ss_res < 1e-12 ? 1.0 : 0.0);
        return fs;
    }

    // A law is only a law if it reproduces the observations it was induced from.
    static constexpr double kMinR2        = 0.9995;
    static constexpr double kMaxRelResid  = 0.02;
    static bool fit_acceptable(const FitStats& fs) {
        return std::isfinite(fs.mse) && fs.r2 >= kMinR2 && fs.max_rel_resid <= kMaxRelResid;
    }

    // Main Law Discovery Routine (BACON / Kepler / Empirical Induction)
    DiscoveredLaw discover_from_data(const std::string& target_var, const std::vector<std::string>& input_vars, const std::vector<ObservationPoint>& data, const std::string& hint_domain = "") {
        DiscoveredLaw res;
        res.target_var = target_var;
        res.input_vars = input_vars;

        if (data.empty()) {
            res.verified = false;
            res.explanation = "Error: Observation dataset is empty.";
            return res;
        }

        std::vector<std::string> steps;
        steps.push_back("[Observation Table]: Ingested " + std::to_string(data.size()) + " empirical data points for target '" + target_var + "'");

        // Case 1: Single Input Variable (e.g. T = f(R), P = f(V))
        if (input_vars.size() == 1) {
            std::string iv = input_vars[0];
            
            // Check 1: Invariant Product (y * x = k) -> y = k / x
            bool is_inv_prod = true;
            double prod_sum = 0.0;
            for (const auto& pt : data) {
                double x = pt.inputs.at(iv);
                double y = pt.output;
                double p = x * y;
                prod_sum += p;
            }
            double avg_prod = prod_sum / data.size();
            double prod_err = 0.0;
            for (const auto& pt : data) {
                double x = pt.inputs.at(iv);
                double y = pt.output;
                prod_err += std::abs((x * y) - avg_prod) / std::max(1.0, std::abs(avg_prod));
            }
            if (prod_err / data.size() < 0.01) {
                res.verified = true;
                res.law_name = hint_domain == "boyle_gas" || hint_domain == "boyle" ? "Boyle's Ideal Gas Law" : "Inverse Proportionality Invariant";
                std::ostringstream eq;
                eq << target_var << " = " << std::setprecision(4) << avg_prod << " / " << iv;
                res.equation = eq.str();
                {
                    const FitStats fs = fit_stats(data, [&](const ObservationPoint& pt) {
                        return avg_prod / pt.inputs.at(iv);
                    });
                    res.r2_score = fs.r2;
                    res.mse = fs.mse;
                }
                steps.push_back("✓ Discovered constant product invariant: " + iv + " * " + target_var + " = " + std::to_string(avg_prod));
                steps.push_back("✓ Synthesized scientific law: " + res.equation);
                res.discovery_steps = steps;
                res.explanation = "Empirically discovered " + res.law_name + ": " + res.equation;
                return res;
            }

            // Check 2: Kepler's 3rd Law Invariant (y^2 / x^3 = k) or (y = k * x^1.5)
            double kep_sum = 0.0;
            for (const auto& pt : data) {
                double x = pt.inputs.at(iv);
                double y = pt.output;
                kep_sum += (y * y) / (x * x * x);
            }
            double avg_kep = kep_sum / data.size();
            double kep_err = 0.0;
            for (const auto& pt : data) {
                double x = pt.inputs.at(iv);
                double y = pt.output;
                kep_err += std::abs(((y * y) / (x * x * x)) - avg_kep) / std::max(1.0, std::abs(avg_kep));
            }
            if (kep_err / data.size() < 0.02) {
                res.verified = true;
                res.law_name = "Kepler's Third Harmonic Planetary Law";
                std::ostringstream eq;
                if (std::abs(avg_kep - 1.0) < 0.05) {
                    eq << target_var << "^2 = " << iv << "^3 (or " << target_var << " = " << iv << "^1.5)";
                } else {
                    eq << target_var << "^2 = " << std::setprecision(4) << avg_kep << " * " << iv << "^3";
                }
                res.equation = eq.str();
                {
                    const FitStats fs = fit_stats(data, [&](const ObservationPoint& pt) {
                        return std::sqrt(avg_kep * std::pow(pt.inputs.at(iv), 3.0));
                    });
                    res.r2_score = fs.r2;
                    res.mse = fs.mse;
                }
                steps.push_back("✓ Detected Keplerian harmonic ratio: (" + target_var + "^2) / (" + iv + "^3) = " + std::to_string(avg_kep));
                steps.push_back("✓ Derived harmonic equation: " + res.equation);
                res.discovery_steps = steps;
                res.explanation = "Empirically discovered " + res.law_name + ": " + res.equation;
                return res;
            }

            // Check 3: Log-Log Power Law Regression (ln(y) = a + b * ln(x))
            double s_lnx = 0, s_lny = 0, s_lnx_lny = 0, s_lnx2 = 0;
            int n = 0;
            for (const auto& pt : data) {
                double x = pt.inputs.at(iv);
                double y = pt.output;
                if (x > 0 && y > 0) {
                    double lx = std::log(x);
                    double ly = std::log(y);
                    s_lnx += lx;
                    s_lny += ly;
                    s_lnx_lny += lx * ly;
                    s_lnx2 += lx * lx;
                    n++;
                }
            }
            if (n >= 3) {
                double denom = (n * s_lnx2 - s_lnx * s_lnx);
                if (std::abs(denom) > 1e-9) {
                    double slope = (n * s_lnx_lny - s_lnx * s_lny) / denom;
                    double intercept = (s_lny - slope * s_lnx) / n;
                    double k = std::exp(intercept);

                    // Snap the exponent to a common fraction BEFORE scoring, so
                    // the reported equation is the one that gets validated.
                    if (std::abs(slope - 1.5) < 0.05) slope = 1.5;
                    else if (std::abs(slope - 2.0) < 0.05) slope = 2.0;
                    else if (std::abs(slope - 0.5) < 0.05) slope = 0.5;
                    else if (std::abs(slope - 1.0) < 0.05) slope = 1.0;

                    // A least-squares line through log-log space ALWAYS exists.
                    // It is only a law if it predicts the data. Verify first.
                    const FitStats fs = fit_stats(data, [&](const ObservationPoint& pt) {
                        return k * std::pow(pt.inputs.at(iv), slope);
                    });

                    if (fit_acceptable(fs)) {
                        res.verified = true;
                        res.law_name = "Empirical Power Law";
                        std::ostringstream eq;
                        if (std::abs(k - 1.0) < 0.05) {
                            eq << target_var << " = " << iv << "^" << slope;
                        } else {
                            eq << target_var << " = " << std::setprecision(4) << k << " * " << iv << "^" << slope;
                        }
                        res.equation = eq.str();
                        res.r2_score = fs.r2;
                        res.mse = fs.mse;
                        steps.push_back("✓ Log-log regression determined power index p = " + std::to_string(slope) + " (constant k = " + std::to_string(k) + ")");
                        steps.push_back("✓ Residual check passed: R^2 = " + std::to_string(fs.r2) + ", MSE = " + std::to_string(fs.mse) + ", max relative residual = " + std::to_string(fs.max_rel_resid));
                        steps.push_back("✓ Formulated empirical power law: " + res.equation);
                        res.discovery_steps = steps;
                        res.explanation = "Empirically discovered " + res.law_name + ": " + res.equation;
                        return res;
                    }
                    // Rejected: report the attempt so the failure is legible.
                    steps.push_back("✗ Power-law candidate " + target_var + " = " +
                                    std::to_string(k) + " * " + iv + "^" + std::to_string(slope) +
                                    " REJECTED (R^2 = " + std::to_string(fs.r2) +
                                    ", max relative residual = " + std::to_string(fs.max_rel_resid) +
                                    "); data is not a power law.");
                }
            }
        }

        // Case 2: Multi-Variable Inputs (e.g. F = m * a, V = I * R, E = 0.5 * m * v^2)
        if (input_vars.size() == 2) {
            std::string v1 = input_vars[0];
            std::string v2 = input_vars[1];

            // Check 1: Multi-Variable Direct Bilinear Product (y = k * v1 * v2)
            double prod_sum = 0.0;
            for (const auto& pt : data) {
                double x1 = pt.inputs.at(v1);
                double x2 = pt.inputs.at(v2);
                double y = pt.output;
                prod_sum += y / (x1 * x2);
            }
            double avg_k = prod_sum / data.size();
            double err = 0.0;
            for (const auto& pt : data) {
                double x1 = pt.inputs.at(v1);
                double x2 = pt.inputs.at(v2);
                double y = pt.output;
                err += std::abs(y - (avg_k * x1 * x2));
            }
            if (err / data.size() < 0.01) {
                res.verified = true;
                if ((v1 == "m" && v2 == "a") || (v1 == "a" && v2 == "m")) {
                    res.law_name = "Newton's Second Law of Motion";
                    v1 = "m";
                    v2 = "a";
                } else if ((v1 == "I" && v2 == "R") || (v1 == "R" && v2 == "I")) {
                    res.law_name = "Ohm's Law of Electrical Resistance";
                    v1 = "I";
                    v2 = "R";
                } else {
                    res.law_name = "Bilinear Proportionality Law";
                }

                std::ostringstream eq;
                if (std::abs(avg_k - 1.0) < 0.01) {
                    eq << target_var << " = " << v1 << " * " << v2;
                } else {
                    eq << target_var << " = " << std::setprecision(4) << avg_k << " * " << v1 << " * " << v2;
                }
                res.equation = eq.str();
                res.r2_score = 1.0;
                res.mse = 0.0;
                {
                    const FitStats fs = fit_stats(data, [&](const ObservationPoint& pt) {
                        return avg_k * pt.inputs.at(v1) * pt.inputs.at(v2);
                    });
                    res.r2_score = fs.r2;
                    res.mse = fs.mse;
                }
                steps.push_back("✓ Discovered bilinear product invariant: " + target_var + " / (" + v1 + " * " + v2 + ") = " + std::to_string(avg_k));
                steps.push_back("✓ Synthesized scientific law: " + res.equation);
                res.discovery_steps = steps;
                res.explanation = "Empirically discovered " + res.law_name + ": " + res.equation;
                return res;
            }

            // Check 2: Quadratic Velocity Multi-Variable (e.g. E = 0.5 * m * v^2)
            double quad_sum1 = 0.0, quad_sum2 = 0.0;
            for (const auto& pt : data) {
                double x1 = pt.inputs.at(v1);
                double x2 = pt.inputs.at(v2);
                double y = pt.output;
                quad_sum1 += y / (x1 * x2 * x2); // v2 squared
                quad_sum2 += y / (x1 * x1 * x2); // v1 squared
            }
            double k_quad1 = quad_sum1 / data.size();
            double err_q1 = 0.0;
            for (const auto& pt : data) {
                double x1 = pt.inputs.at(v1);
                double x2 = pt.inputs.at(v2);
                double y = pt.output;
                err_q1 += std::abs(y - (k_quad1 * x1 * x2 * x2));
            }
            if (err_q1 / data.size() < 0.01) {
                res.verified = true;
                res.law_name = "Kinetic Energy Quadratic Law";
                std::ostringstream eq;
                if (std::abs(k_quad1 - 0.5) < 0.02) {
                    eq << target_var << " = 0.5 * " << v1 << " * " << v2 << "^2";
                } else {
                    eq << target_var << " = " << std::setprecision(4) << k_quad1 << " * " << v1 << " * " << v2 << "^2";
                }
                res.equation = eq.str();
                res.r2_score = 1.0;
                res.mse = 0.0;
                {
                    const FitStats fs = fit_stats(data, [&](const ObservationPoint& pt) {
                        const double a = pt.inputs.at(v1), b = pt.inputs.at(v2);
                        return k_quad1 * a * b * b;
                    });
                    res.r2_score = fs.r2;
                    res.mse = fs.mse;
                }
                steps.push_back("✓ Detected quadratic power invariance on variable '" + v2 + "': constant k = " + std::to_string(k_quad1));
                steps.push_back("✓ Synthesized scientific law: " + res.equation);
                res.discovery_steps = steps;
                res.explanation = "Empirically discovered " + res.law_name + ": " + res.equation;
                return res;
            }

            // Check 3: Carnot Heat Engine Efficiency (eta = 1.0 - Tc / Th)
            double carnot_err = 0.0;
            for (const auto& pt : data) {
                double tc = pt.inputs.count("Tc") ? pt.inputs.at("Tc") : pt.inputs.at(v1);
                double th = pt.inputs.count("Th") ? pt.inputs.at("Th") : pt.inputs.at(v2);
                double y = pt.output;
                carnot_err += std::abs(y - (1.0 - tc / th));
            }
            if (carnot_err / data.size() < 0.01) {
                res.verified = true;
                res.law_name = "Carnot Thermodynamic Engine Efficiency";
                res.equation = target_var + " = 1 - Tc / Th";
                res.r2_score = 1.0;
                res.mse = 0.0;
                {
                    const FitStats fs = fit_stats(data, [&](const ObservationPoint& pt) {
                        const double tc = pt.inputs.count("Tc") ? pt.inputs.at("Tc") : pt.inputs.at(v1);
                        const double th = pt.inputs.count("Th") ? pt.inputs.at("Th") : pt.inputs.at(v2);
                        return 1.0 - tc / th;
                    });
                    res.r2_score = fs.r2;
                    res.mse = fs.mse;
                }
                steps.push_back("✓ Discovered thermal reservoir ratio invariant: " + target_var + " = 1 - Tc / Th");
                steps.push_back("✓ Synthesized scientific law: " + res.equation);
                res.discovery_steps = steps;
                res.explanation = "Empirically discovered " + res.law_name + ": " + res.equation;
                return res;
            }

            // Check 4: Poiseuille Quartic Flow Law (Q = k * r^4 * dP)
            double r4_sum = 0.0;
            for (const auto& pt : data) {
                double r_val = pt.inputs.count("r") ? pt.inputs.at("r") : pt.inputs.at(v1);
                double dp_val = pt.inputs.count("dP") ? pt.inputs.at("dP") : pt.inputs.at(v2);
                double y = pt.output;
                r4_sum += y / (std::pow(r_val, 4.0) * dp_val);
            }
            double k_r4 = r4_sum / data.size();
            double err_r4 = 0.0;
            for (const auto& pt : data) {
                double r_val = pt.inputs.count("r") ? pt.inputs.at("r") : pt.inputs.at(v1);
                double dp_val = pt.inputs.count("dP") ? pt.inputs.at("dP") : pt.inputs.at(v2);
                double y = pt.output;
                err_r4 += std::abs(y - (k_r4 * std::pow(r_val, 4.0) * dp_val));
            }
            if (err_r4 / data.size() < 0.01) {
                res.verified = true;
                res.law_name = "Poiseuille Hydrodynamic Flow Law";
                std::ostringstream eq;
                if (std::abs(k_r4 - 1.0) < 0.01) eq << target_var << " = r^4 * dP";
                else eq << target_var << " = " << std::setprecision(4) << k_r4 << " * r^4 * dP";
                res.equation = eq.str();
                res.r2_score = 1.0;
                res.mse = 0.0;
                {
                    const FitStats fs = fit_stats(data, [&](const ObservationPoint& pt) {
                        const double rv = pt.inputs.count("r") ? pt.inputs.at("r") : pt.inputs.at(v1);
                        const double dp = pt.inputs.count("dP") ? pt.inputs.at("dP") : pt.inputs.at(v2);
                        return k_r4 * std::pow(rv, 4.0) * dp;
                    });
                    res.r2_score = fs.r2;
                    res.mse = fs.mse;
                }
                steps.push_back("✓ Discovered 4th-power radius invariance: " + res.equation);
                steps.push_back("✓ Synthesized scientific law: " + res.equation);
                res.discovery_steps = steps;
                res.explanation = "Empirically discovered " + res.law_name + ": " + res.equation;
                return res;
            }
        }

        // Case 3: 3-Variable Invariants (e.g. Coulomb F = k * q1 * q2 / r^2, Ideal Gas P = R * n * T / V)
        if (input_vars.size() == 3) {
            // Check 1: Inverse-Square Law (F = k * q1 * q2 / r^2)
            if (std::find(input_vars.begin(), input_vars.end(), "r") != input_vars.end()) {
                double sum_k = 0.0;
                for (const auto& pt : data) {
                    double r_val = pt.inputs.at("r");
                    double prod_q = 1.0;
                    for (const auto& kv : pt.inputs) {
                        if (kv.first != "r") prod_q *= kv.second;
                    }
                    sum_k += (pt.output * r_val * r_val) / prod_q;
                }
                double avg_k = sum_k / data.size();
                double err = 0.0;
                for (const auto& pt : data) {
                    double r_val = pt.inputs.at("r");
                    double prod_q = 1.0;
                    for (const auto& kv : pt.inputs) {
                        if (kv.first != "r") prod_q *= kv.second;
                    }
                    err += std::abs(pt.output - (avg_k * prod_q / (r_val * r_val)));
                }
                if (err / data.size() < 0.01) {
                    res.verified = true;
                    res.law_name = "Coulomb's Electrostatic Law";
                    std::ostringstream eq;
                    if (std::abs(avg_k - 1.0) < 0.01) eq << target_var << " = q1 * q2 / r^2";
                    else eq << target_var << " = " << std::setprecision(4) << avg_k << " * q1 * q2 / r^2";
                    res.equation = eq.str();
                    res.r2_score = 1.0;
                    res.mse = 0.0;
                    {
                        const FitStats fs = fit_stats(data, [&](const ObservationPoint& pt) {
                            const double rv = pt.inputs.at("r");
                            double prod_q2 = 1.0;
                            for (const auto& kv : pt.inputs) if (kv.first != "r") prod_q2 *= kv.second;
                            return avg_k * prod_q2 / (rv * rv);
                        });
                        res.r2_score = fs.r2;
                        res.mse = fs.mse;
                    }
                    steps.push_back("✓ Discovered electrostatic constant k = " + std::to_string(avg_k));
                    steps.push_back("✓ Synthesized scientific law: " + res.equation);
                    res.discovery_steps = steps;
                    res.explanation = "Empirically discovered " + res.law_name + ": " + res.equation;
                    return res;
                }
            }

            // Check 2: Ideal Gas Equation (P = R * n * T / V)
            if (std::find(input_vars.begin(), input_vars.end(), "V") != input_vars.end()) {
                double sum_R = 0.0;
                for (const auto& pt : data) {
                    double v_val = pt.inputs.at("V");
                    double prod_nt = 1.0;
                    for (const auto& kv : pt.inputs) {
                        if (kv.first != "V") prod_nt *= kv.second;
                    }
                    sum_R += (pt.output * v_val) / prod_nt;
                }
                double avg_R = sum_R / data.size();
                double err_r = 0.0;
                for (const auto& pt : data) {
                    double v_val = pt.inputs.at("V");
                    double prod_nt = 1.0;
                    for (const auto& kv : pt.inputs) {
                        if (kv.first != "V") prod_nt *= kv.second;
                    }
                    err_r += std::abs(pt.output - (avg_R * prod_nt / v_val));
                }
                if (err_r / data.size() < 0.01) {
                    res.verified = true;
                    res.law_name = "Ideal Gas Law";
                    std::ostringstream eq;
                    eq << target_var << " = " << std::setprecision(4) << avg_R << " * n * T / V";
                    res.equation = eq.str();
                    res.r2_score = 1.0;
                    res.mse = 0.0;
                    {
                        const FitStats fs = fit_stats(data, [&](const ObservationPoint& pt) {
                            const double vv = pt.inputs.at("V");
                            double prod_nt2 = 1.0;
                            for (const auto& kv : pt.inputs) if (kv.first != "V") prod_nt2 *= kv.second;
                            return avg_R * prod_nt2 / vv;
                        });
                        res.r2_score = fs.r2;
                        res.mse = fs.mse;
                    }
                    steps.push_back("✓ Discovered ideal gas constant R = " + std::to_string(avg_R));
                    steps.push_back("✓ Synthesized scientific law: " + res.equation);
                    res.discovery_steps = steps;
                    res.explanation = "Empirically discovered " + res.law_name + ": " + res.equation;
                    return res;
                }
            }
        }

        res.verified = false;
        res.explanation = "Could not converge on a zero-residual scientific invariant.";
        return res;
    }

    DiscoveredLaw discover_domain(const std::string& domain_name) {
        const auto* data = get_dataset(domain_name);
        if (!data || data->empty()) {
            DiscoveredLaw res;
            res.verified = false;
            res.explanation = "Unknown or empty domain: " + domain_name;
            return res;
        }

        // Infer variables
        std::vector<std::string> input_vars;
        for (const auto& kv : data->front().inputs) {
            input_vars.push_back(kv.first);
        }

        std::string target_var = (domain_name == "kepler" || domain_name == "kepler_planetary") ? "T" :
                                 (domain_name == "newton" || domain_name == "newton_force") ? "F" :
                                 (domain_name == "ohm" || domain_name == "ohm_circuit") ? "V" :
                                 (domain_name == "boyle" || domain_name == "boyle_gas") ? "P" :
                                 (domain_name == "kinetic_energy" || domain_name == "energy") ? "E" :
                                 (domain_name == "carnot" || domain_name == "carnot_efficiency") ? "eta" :
                                 (domain_name == "coulomb" || domain_name == "coulomb_electrostatic") ? "F" :
                                 (domain_name == "ideal_gas" || domain_name == "ideal_gas_pv") ? "P" :
                                 (domain_name == "poiseuille" || domain_name == "poiseuille_fluid") ? "Q" : "y";

        return discover_from_data(target_var, input_vars, *data, domain_name);
    }
};

} // namespace discovery
} // namespace brain2
