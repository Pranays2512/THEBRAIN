#pragma once
#include <string>
#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <memory>
#include <iostream>
#include <sstream>
#include <regex>
#include <algorithm>

#include "crisp/engines/math/math_parser.hpp"
#include "crisp/engines/math/calculus_engine.hpp"
#include "crisp/engines/math/algebra_engine.hpp"
#include "crisp/engines/math/physics_engine.hpp"
#include "crisp/engines/math/integral_engine.hpp"
#include "crisp/engines/knowledge/nl_query.hpp"
#include "crisp/engines/knowledge/policy_pack.hpp"

namespace brain2 {
namespace math {

// Helper to evaluate an AST numerically
inline double eval_expr(std::shared_ptr<ExprNode> e, double x_val = 0.0) {
    if (!e) return 0.0;
    if (e->op == "val") return e->val;
    if (e->op == "var") return x_val;
    
    if (e->op == "neg") return -eval_expr(e->children[0], x_val);
    if (e->op == "sin") return std::sin(eval_expr(e->children[0], x_val));
    if (e->op == "cos") return std::cos(eval_expr(e->children[0], x_val));
    if (e->op == "exp") return std::exp(eval_expr(e->children[0], x_val));
    if (e->op == "ln")  return std::log(eval_expr(e->children[0], x_val));
    
    double a = e->children.size() > 0 ? eval_expr(e->children[0], x_val) : 0.0;
    double b = e->children.size() > 1 ? eval_expr(e->children[1], x_val) : 0.0;
    
    if (e->op == "+") return a + b;
    if (e->op == "-") return a - b;
    if (e->op == "*") return a * b;
    if (e->op == "/") return b != 0.0 ? a / b : std::numeric_limits<double>::infinity();
    if (e->op == "^") return std::pow(a, b);
    
    return 0.0;
}

// Numerical derivative for ground truth
inline double numerical_diff(std::shared_ptr<ExprNode> e, double x, double h = 1e-5) {
    return (eval_expr(e, x + h) - eval_expr(e, x - h)) / (2 * h);
}

// Symbolic differentiation
inline std::shared_ptr<ExprNode> diff(const std::shared_ptr<ExprNode>& e, const std::string& var = "x") {
    if (!e) return ExprNode::make_num(0);
    if (is_num(e)) return ExprNode::make_num(0);
    if (is_var(e)) return (e->var == var) ? ExprNode::make_num(1) : ExprNode::make_num(0);

    const auto& op = e->op;
    auto& ch = e->children;

    if (op == "neg") return ExprNode::make_op("neg", {diff(ch[0], var)});
    if (op == "+") return ExprNode::make_op("+", {diff(ch[0], var), diff(ch[1], var)});
    if (op == "-") return ExprNode::make_op("-", {diff(ch[0], var), diff(ch[1], var)});
    if (op == "*")
        return ExprNode::make_op("+",
            {ExprNode::make_op("*", {diff(ch[0], var), ch[1]}),
             ExprNode::make_op("*", {ch[0], diff(ch[1], var)})});
    if (op == "/")
        return ExprNode::make_op("/",
            {ExprNode::make_op("-",
                {ExprNode::make_op("*", {diff(ch[0], var), ch[1]}),
                 ExprNode::make_op("*", {ch[0], diff(ch[1], var)})}),
             ExprNode::make_op("*", {ch[1], ch[1]})});
    if (op == "^" && is_num(ch[1])) {
        double n = ch[1]->val;
        return ExprNode::make_op("*",
            {ExprNode::make_num(n),
             ExprNode::make_op("^", {ch[0], ExprNode::make_num(n - 1)})});
    }
    if (op == "sin") return ExprNode::make_op("*", {ExprNode::make_op("cos", {ch[0]}), diff(ch[0], var)});
    if (op == "cos") return ExprNode::make_op("*", {ExprNode::make_op("neg", {ExprNode::make_op("sin", {ch[0]})}), diff(ch[0], var)});
    if (op == "exp") return ExprNode::make_op("*", {ExprNode::make_op("exp", {ch[0]}), diff(ch[0], var)});
    if (op == "ln")  return ExprNode::make_op("/", {diff(ch[0], var), ch[0]});
    return ExprNode::make_num(0);
}

// Unified Math & Physics Problem Solving Result
struct MathSolveResult {
    bool success = false;
    std::string op;             // "algebra", "physics", "integral", "diff", "eval", "word"
    std::string target;         // target variable or expression
    double numeric_val = 0.0;
    std::string symbolic_val;
    std::vector<std::string> steps;
    std::string explanation;
};

class MathEngine {
public:
    AlgebraEngine algebra;
    PhysicsEngine physics;
    IntegralEngine integral;
    knowledge::NLQueryParser nl_parser;
    std::vector<std::string> discovered_rules;

    MathEngine() {
        init_default_physics_laws();
    }

    void init_default_physics_laws() {
        // F = m*a
        physics.add_law("newton2", "F", ExprNode::make_op("*", {ExprNode::make_var("m"), ExprNode::make_var("a")}));
        physics.add_law("force", "F", ExprNode::make_op("*", {ExprNode::make_var("m"), ExprNode::make_var("a")}));
        
        // v = d/t
        physics.add_law("speed", "v", ExprNode::make_op("/", {ExprNode::make_var("d"), ExprNode::make_var("t")}));
        physics.add_law("velocity", "v", ExprNode::make_op("/", {ExprNode::make_var("d"), ExprNode::make_var("t")}));
        
        // p = m*v
        physics.add_law("momentum", "p", ExprNode::make_op("*", {ExprNode::make_var("m"), ExprNode::make_var("v")}));
        
        // ke = 0.5 * m * v^2
        physics.add_law("ke", "ke", ExprNode::make_op("*", {ExprNode::make_num(0.5), ExprNode::make_op("*", {ExprNode::make_var("m"), ExprNode::make_op("^", {ExprNode::make_var("v"), ExprNode::make_num(2.0)})})}));
        
        // W = F * d
        physics.add_law("work", "W", ExprNode::make_op("*", {ExprNode::make_var("F"), ExprNode::make_var("d")}));
        
        // P = W / t
        physics.add_law("power", "P", ExprNode::make_op("/", {ExprNode::make_var("W"), ExprNode::make_var("t")}));
        
        // V = I * R (Ohm's law)
        physics.add_law("ohms_law", "V", ExprNode::make_op("*", {ExprNode::make_var("I"), ExprNode::make_var("R")}));
        
        // density = m / V
        physics.add_law("density", "rho", ExprNode::make_op("/", {ExprNode::make_var("m"), ExprNode::make_var("V")}));

        // Carnot Engine Efficiency: eta = 1 - Tc / Th
        physics.add_law("carnot", "eta", ExprNode::make_op("-", {ExprNode::make_num(1.0), ExprNode::make_op("/", {ExprNode::make_var("Tc"), ExprNode::make_var("Th")})}));

        // Ideal Gas Law: P = n * R * T / V
        physics.add_law("ideal_gas", "P", ExprNode::make_op("/", {ExprNode::make_op("*", {ExprNode::make_var("n"), ExprNode::make_op("*", {ExprNode::make_var("R"), ExprNode::make_var("T")})}), ExprNode::make_var("V")}));

        // Universal Gravitation: F = G * m1 * m2 / r^2, G = 6.674e-11 m^3/(kg*s^2)
        // (constant folded in as a literal multiplier node so eval/isolate stay exact)
        physics.add_law("gravitation", "F", ExprNode::make_op("*", {
            ExprNode::make_num(6.674e-11),
            ExprNode::make_op("/", {ExprNode::make_op("*", {ExprNode::make_var("m1"), ExprNode::make_var("m2")}), ExprNode::make_op("^", {ExprNode::make_var("r"), ExprNode::make_num(2.0)})})}));

        // Coulomb's Law: F = k_e * q1 * q2 / r^2, k_e = 8.9875517923e9 N*m^2/C^2
        physics.add_law("coulomb", "F", ExprNode::make_op("*", {
            ExprNode::make_num(8.9875517923e9),
            ExprNode::make_op("/", {ExprNode::make_op("*", {ExprNode::make_var("q1"), ExprNode::make_var("q2")}), ExprNode::make_op("^", {ExprNode::make_var("r"), ExprNode::make_num(2.0)})})}));

        // De Broglie Wavelength: lambda = h / p
        physics.add_law("de_broglie", "lambda", ExprNode::make_op("/", {ExprNode::make_var("h"), ExprNode::make_var("p")}));

        // Henderson-Hasselbalch Buffer pH: pH = pKa + log10(base / acid)
        // Stored as log10(x) = ln(x) / ln(10) since the evaluator only has ln.
        physics.add_law("buffer_ph", "pH", ExprNode::make_op("+", {ExprNode::make_var("pKa"), ExprNode::make_op("/", {ExprNode::make_op("ln", {ExprNode::make_op("/", {ExprNode::make_var("base"), ExprNode::make_var("acid")})}), ExprNode::make_num(2.302585092994046)})}));

        // Poiseuille Fluid Resistance: R = 8 * eta * L / r^4
        physics.add_law("poiseuille", "R", ExprNode::make_op("/", {ExprNode::make_op("*", {ExprNode::make_num(8.0), ExprNode::make_op("*", {ExprNode::make_var("eta"), ExprNode::make_var("L")})}), ExprNode::make_op("^", {ExprNode::make_var("r"), ExprNode::make_num(4.0)})}));
    }

    // Attempt to discover the power rule: d/dx(x^n) = n*x^(n-1)
    bool discover_power_rule() {
        std::vector<double> test_points = {1.0, 2.0, 3.0};
        auto expr = ExprNode::make_op("^", {ExprNode::make_var("x"), ExprNode::make_num(3.0)});
        auto cand = ExprNode::make_op("*", {ExprNode::make_num(3.0), ExprNode::make_op("^", {ExprNode::make_var("x"), ExprNode::make_num(2.0)})});
        
        for (double p : test_points) {
            double truth = numerical_diff(expr, p);
            double cand_val = eval_expr(cand, p);
            if (std::abs(truth - cand_val) > 1e-3) return false;
        }
        
        discovered_rules.push_back("power rule: d/dx(x^n) = n*x^(n-1)");
        return true;
    }

    // ── 1. Equation Solver (Algebra) ──────────────────────────────────────────
    MathSolveResult solve_equation(const std::string& eq_str, const std::string& var = "x") {
        MathSolveResult res;
        res.op = "algebra";
        res.target = var;
        try {
            auto expr = parse(eq_str);
            auto [val, steps] = algebra.solve(expr, var);
            res.success = true;
            res.numeric_val = val;
            res.symbolic_val = std::to_string(val);
            // remove trailing zeros
            if (res.symbolic_val.find('.') != std::string::npos) {
                res.symbolic_val.erase(res.symbolic_val.find_last_not_of('0') + 1, std::string::npos);
                if (res.symbolic_val.back() == '.') res.symbolic_val.pop_back();
            }
            res.steps = steps;
            res.explanation = var + " = " + res.symbolic_val;
        } catch (const std::exception& e) {
            res.success = false;
            res.explanation = e.what();
        }
        return res;
    }

    // ── 2. Physics Law Solver ─────────────────────────────────────────────────
    MathSolveResult solve_physics(const std::string& law, const std::string& target, const std::map<std::string, double>& knowns) {
        MathSolveResult res;
        res.op = "physics";
        res.target = target;
        try {
            auto [val, steps] = physics.solve(law, target, knowns);
            res.success = true;
            res.numeric_val = val;
            res.symbolic_val = std::to_string(val);
            if (res.symbolic_val.find('.') != std::string::npos) {
                res.symbolic_val.erase(res.symbolic_val.find_last_not_of('0') + 1, std::string::npos);
                if (res.symbolic_val.back() == '.') res.symbolic_val.pop_back();
            }
            res.steps = steps;
            res.explanation = target + " = " + res.symbolic_val + " [law: " + law + "]";
        } catch (const std::exception& e) {
            res.success = false;
            res.explanation = e.what();
        }
        return res;
    }

    // ── 3. Symbolic Calculus (Integrals & Derivatives) ────────────────────────
    MathSolveResult solve_integral(const std::string& expr_str) {
        MathSolveResult res;
        res.op = "integral";
        try {
            auto expr = parse(expr_str);
            auto int_expr = integral.integrate(expr);
            if (int_expr) {
                res.success = true;
                res.symbolic_val = render(int_expr) + " + C";
                res.explanation = "integral(" + expr_str + ") = " + res.symbolic_val;
            } else {
                res.success = false;
                res.explanation = "Could not find closed-form integral";
            }
        } catch (const std::exception& e) {
            res.success = false;
            res.explanation = e.what();
        }
        return res;
    }

    MathSolveResult solve_derivative(const std::string& expr_str, const std::string& var = "x") {
        MathSolveResult res;
        res.op = "diff";
        try {
            auto expr = parse(expr_str);
            auto diff_expr = diff(expr, var);
            if (diff_expr) {
                res.success = true;
                res.symbolic_val = render(diff_expr);
                res.explanation = "d/d" + var + "(" + expr_str + ") = " + res.symbolic_val;
            } else {
                res.success = false;
                res.explanation = "Could not differentiate expression";
            }
        } catch (const std::exception& e) {
            res.success = false;
            res.explanation = e.what();
        }
        return res;
    }

    // ── 4. Natural Language Word Math & Physics Problem Solver ────────────────
    MathSolveResult solve_word_problem(const std::string& text) {
        MathSolveResult res;
        res.op = "word";
        
        std::string lower = text;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        // Canonical variable alias table
        std::map<std::string, std::string> var_aliases = {
            {"mass", "m"}, {"weight", "m"}, {"m", "m"},
            {"accel", "a"}, {"acceleration", "a"}, {"a", "a"},
            {"force", "F"}, {"f", "F"},
            {"speed", "v"}, {"velocity", "v"}, {"v", "v"},
            {"dist", "d"}, {"distance", "d"}, {"d", "d"},
            {"time", "t"}, {"t", "t"},
            {"work", "W"}, {"energy", "W"}, {"w", "W"},
            {"power", "P"}, {"p", "P"},
            {"momentum", "p"},
            {"ke", "ke"}, {"kinetic_energy", "ke"},
            {"current", "I"}, {"i", "I"},
            {"resistance", "R"}, {"r", "R"},
            {"voltage", "V"},
            {"density", "rho"}, {"rho", "rho"},
            {"volume", "V"},
            {"efficiency", "eta"}, {"eta", "eta"},
            {"tc", "Tc"}, {"cold_temp", "Tc"},
            {"th", "Th"}, {"hot_temp", "Th"},
            {"pressure", "P"}, {"p", "P"},
            {"moles", "n"}, {"n", "n"},
            {"gas_const", "R"},
            {"temp", "T"}, {"temperature", "T"},
            {"m1", "m1"}, {"mass1", "m1"},
            {"m2", "m2"}, {"mass2", "m2"},
            {"q1", "q1"}, {"charge1", "q1"},
            {"q2", "q2"}, {"charge2", "q2"},
            {"radius", "r"},
            {"planck", "h"}, {"h", "h"},
            {"wavelength", "lambda"}, {"lambda", "lambda"},
            {"pka", "pKa"}, {"ph", "pH"},
            {"base", "base"}, {"acid", "acid"},
            {"viscosity", "eta"},
            {"length", "L"}, {"l", "L"}
        };

        // 1. Extract known variables and numbers
        std::map<std::string, double> knowns;
        
        // Pattern A: "var = number" or "var: number"
        std::regex eq_regex(R"(([a-zA-Z_0-9]+)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?))");
        auto words_begin = std::sregex_iterator(lower.begin(), lower.end(), eq_regex);
        auto words_end = std::sregex_iterator();
        for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
            std::smatch match = *i;
            std::string k = match[1].str();
            double v = std::stod(match[2].str());
            if (var_aliases.count(k)) {
                knowns[var_aliases[k]] = v;
            }
        }

        // Pattern B: "number units" or "word is number"
        std::regex phrase_regex(R"(([a-zA-Z_0-9]+)\s+(?:is|of|=|measures|equals)?\s*([0-9]+(?:\.[0-9]+)?))");
        auto phrase_begin = std::sregex_iterator(lower.begin(), lower.end(), phrase_regex);
        auto phrase_end = std::sregex_iterator();
        for (std::sregex_iterator i = phrase_begin; i != phrase_end; ++i) {
            std::smatch match = *i;
            std::string k = match[1].str();
            double v = std::stod(match[2].str());
            if (var_aliases.count(k)) {
                knowns[var_aliases[k]] = v;
            }
        }

        // Add cross-symbol aliases if present
        if (knowns.count("R") && !knowns.count("r")) knowns["r"] = knowns["R"];
        if (knowns.count("r") && !knowns.count("R")) knowns["R"] = knowns["r"];
        if (knowns.count("v") && !knowns.count("V")) knowns["V"] = knowns["v"];
        if (knowns.count("V") && !knowns.count("v")) knowns["v"] = knowns["V"];
        if (knowns.count("t") && !knowns.count("T")) knowns["T"] = knowns["t"];
        if (knowns.count("T") && !knowns.count("t")) knowns["t"] = knowns["T"];
        if (knowns.count("P") && !knowns.count("p")) knowns["p"] = knowns["P"];
        if (knowns.count("p") && !knowns.count("P")) knowns["P"] = knowns["p"];

        // 2. Identify target variable
        std::string target = "";
        std::vector<std::string> query_phrases = {"what is the ", "find the ", "calculate ", "determine ", "find ", "what is "};
        for (const auto& qp : query_phrases) {
            size_t pos = lower.find(qp);
            if (pos != std::string::npos) {
                std::string rest = lower.substr(pos + qp.length());
                std::stringstream ss(rest);
                std::string cand;
                ss >> cand;
                cand.erase(std::remove_if(cand.begin(), cand.end(), [](char c){ return std::ispunct(c); }), cand.end());
                if (var_aliases.count(cand)) {
                    target = var_aliases[cand];
                    break;
                }
            }
        }

        // If target still not found, check for unassigned variable names mentioned in text
        if (target.empty()) {
            std::stringstream ss(lower);
            std::string tok;
            while (ss >> tok) {
                tok.erase(std::remove_if(tok.begin(), tok.end(), [](char c){ return std::ispunct(c); }), tok.end());
                if (var_aliases.count(tok)) {
                    std::string var = var_aliases[tok];
                    if (knowns.find(var) == knowns.end() && var != "V" && var != "r") {
                        target = var;
                    }
                }
            }
        }

        // Check if query specifically mentions target tokens at end or explicitly
        if (target.empty() || knowns.count(target)) {
            std::stringstream ss(lower);
            std::string tok, last_tok;
            while (ss >> tok) {
                tok.erase(std::remove_if(tok.begin(), tok.end(), [](char c){ return std::ispunct(c); }), tok.end());
                if (!tok.empty()) last_tok = tok;
            }
            if (var_aliases.count(last_tok)) {
                target = var_aliases[last_tok];
            }
        }

        if (target.empty()) {
            // fallback: find unassigned variable from matched laws
            if (knowns.count("m") && knowns.count("a")) target = "F";
            else if (knowns.count("F") && knowns.count("m")) target = "a";
            else if (knowns.count("v") && knowns.count("t")) target = "d";
            else if (knowns.count("d") && knowns.count("t")) target = "v";
            else if (knowns.count("m") && knowns.count("v")) target = "p";
            else if (knowns.count("Tc") && knowns.count("Th")) target = "eta";
            else if (knowns.count("n") && knowns.count("R") && knowns.count("T") && knowns.count("V")) target = "P";
            else if (knowns.count("m1") && knowns.count("m2") && knowns.count("r")) target = "F";
            else if (knowns.count("q1") && knowns.count("q2") && knowns.count("r")) target = "F";
            else if (knowns.count("h") && knowns.count("p")) target = "lambda";
            else if (knowns.count("pKa") && knowns.count("base") && knowns.count("acid")) target = "pH";
            else if (knowns.count("eta") && knowns.count("L") && knowns.count("r")) target = "R";
            else if (knowns.count("I") && knowns.count("R")) target = "V";
            else if (knowns.count("m") && knowns.count("V")) target = "rho";
        }

        // 3. Match against physics laws
        std::vector<std::string> candidate_laws = {
            "newton2", "speed", "momentum", "ke", "work", "power", "ohms_law", "density",
            "carnot", "ideal_gas", "gravitation", "coulomb", "de_broglie", "buffer_ph", "poiseuille"
        };
        
        std::vector<std::string> target_candidates = {target};
        if (target == "v") target_candidates.push_back("V");
        else if (target == "V") target_candidates.push_back("v");
        if (target == "r") target_candidates.push_back("R");
        else if (target == "R") target_candidates.push_back("r");
        if (target == "p") target_candidates.push_back("P");
        else if (target == "P") target_candidates.push_back("p");
        if (target == "t") target_candidates.push_back("T");
        else if (target == "T") target_candidates.push_back("t");

        for (const auto& tgt : target_candidates) {
            for (const auto& law : candidate_laws) {
                try {
                    auto pres = solve_physics(law, tgt, knowns);
                    if (pres.success) {
                        pres.op = "word";
                        return pres;
                    }
                } catch (...) {
                    continue;
                }
            }
        }

        res.success = false;
        res.explanation = "Could not resolve word problem: insufficient knowns or unmapped law.";
        return res;
    }

    // ── 5. General Dispatcher ─────────────────────────────────────────────────
    MathSolveResult solve(const std::string& query) {
        std::string q = query;
        // trim
        q.erase(q.begin(), std::find_if(q.begin(), q.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        q.erase(std::find_if(q.rbegin(), q.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), q.end());

        if (q.find("=") != std::string::npos) {
            // Equation solver
            return solve_equation(q);
        } else if (q.find("int ") == 0 || q.find("integrate ") == 0 || q.find("integral ") == 0) {
            size_t space_pos = q.find(' ');
            return solve_integral(q.substr(space_pos + 1));
        } else if (q.find("diff ") == 0 || q.find("derivative ") == 0 || q.find("d/dx ") == 0) {
            size_t space_pos = q.find(' ');
            return solve_derivative(q.substr(space_pos + 1));
        } else {
            // Natural language word problem
            return solve_word_problem(q);
        }
    }
};

} // namespace math
} // namespace brain2
