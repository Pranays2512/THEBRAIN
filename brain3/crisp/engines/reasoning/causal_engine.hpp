#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <sstream>
#include <iostream>
#include <cmath>
#include <memory>
#include <algorithm>
#include <stdexcept>
#include <cctype>

namespace brain2 {
namespace reasoning {

// ── Recursive descent parser for arithmetic expressions with variables ────────
class CausalExpr {
public:
    virtual ~CausalExpr() = default;
    virtual double eval(const std::map<std::string, double>& env) const = 0;
    virtual std::set<std::string> get_variables() const = 0;
    virtual std::string to_string() const = 0;
};

using CausalExprPtr = std::shared_ptr<CausalExpr>;

class CausalLiteral : public CausalExpr {
    double val_;
public:
    CausalLiteral(double val) : val_(val) {}
    double eval(const std::map<std::string, double>&) const override { return val_; }
    std::set<std::string> get_variables() const override { return {}; }
    std::string to_string() const override {
        std::string s = std::to_string(val_);
        if (s.find('.') != std::string::npos) {
            s.erase(s.find_last_not_of('0') + 1, std::string::npos);
            if (s.back() == '.') s.pop_back();
        }
        return s;
    }
};

class CausalVar : public CausalExpr {
    std::string name_;
public:
    CausalVar(const std::string& name) : name_(name) {}
    double eval(const std::map<std::string, double>& env) const override {
        auto it = env.find(name_);
        if (it == env.end()) throw std::runtime_error("Unknown variable: " + name_);
        return it->second;
    }
    std::set<std::string> get_variables() const override { return {name_}; }
    std::string to_string() const override { return name_; }
};

class CausalBinaryOp : public CausalExpr {
    char op_;
    CausalExprPtr left_;
    CausalExprPtr right_;
public:
    CausalBinaryOp(char op, CausalExprPtr left, CausalExprPtr right)
        : op_(op), left_(left), right_(right) {}

    double eval(const std::map<std::string, double>& env) const override {
        double l = left_->eval(env);
        double r = right_->eval(env);
        switch (op_) {
            case '+': return l + r;
            case '-': return l - r;
            case '*': return l * r;
            case '/': 
                if (std::abs(r) < 1e-9) throw std::runtime_error("Division by zero");
                return l / r;
            case '^': return std::pow(l, r);
            default: throw std::runtime_error(std::string("Unknown op: ") + op_);
        }
    }

    std::set<std::string> get_variables() const override {
        auto vars = left_->get_variables();
        auto rvars = right_->get_variables();
        vars.insert(rvars.begin(), rvars.end());
        return vars;
    }

    std::string to_string() const override {
        return "(" + left_->to_string() + " " + op_ + " " + right_->to_string() + ")";
    }
};

class CausalParser {
    std::string text_;
    size_t pos_ = 0;

    char peek() {
        while (pos_ < text_.size() && std::isspace(text_[pos_])) pos_++;
        return pos_ < text_.size() ? text_[pos_] : '\0';
    }

    char get() {
        char c = peek();
        if (pos_ < text_.size()) pos_++;
        return c;
    }

public:
    CausalParser(const std::string& text) : text_(text) {}

    CausalExprPtr parse() {
        auto e = parse_expr();
        return e;
    }

private:
    CausalExprPtr parse_expr() {
        auto left = parse_term();
        while (peek() == '+' || peek() == '-') {
            char op = get();
            auto right = parse_term();
            left = std::make_shared<CausalBinaryOp>(op, left, right);
        }
        return left;
    }

    CausalExprPtr parse_term() {
        auto left = parse_factor();
        while (peek() == '*' || peek() == '/') {
            char op = get();
            auto right = parse_factor();
            left = std::make_shared<CausalBinaryOp>(op, left, right);
        }
        return left;
    }

    CausalExprPtr parse_factor() {
        auto left = parse_primary();
        while (peek() == '^') {
            char op = get();
            auto right = parse_primary();
            left = std::make_shared<CausalBinaryOp>(op, left, right);
        }
        return left;
    }

    CausalExprPtr parse_primary() {
        char c = peek();
        if (c == '(') {
            get(); // eat '('
            auto e = parse_expr();
            if (peek() == ')') get(); // eat ')'
            return e;
        }
        if (c == '-' || std::isdigit(c) || c == '.') {
            std::string num;
            if (c == '-') { num += get(); }
            while (std::isdigit(peek()) || peek() == '.') {
                num += get();
            }
            return std::make_shared<CausalLiteral>(std::stod(num));
        }
        if (std::isalpha(c) || c == '_') {
            std::string id;
            while (std::isalnum(peek()) || peek() == '_') {
                id += get();
            }
            return std::make_shared<CausalVar>(id);
        }
        throw std::runtime_error(std::string("Unexpected token in causal expr: ") + c);
    }
};

// ── Structural Causal Model (SCM) Engine ─────────────────────────────────────
struct CausalResult {
    bool success = false;
    std::string target;
    double value = 0.0;
    std::string value_str;
    std::vector<std::string> steps;
    std::string explanation;
};

class CausalEngine {
public:
    std::map<std::string, CausalExprPtr> equations;
    std::map<std::string, double> factual_evidence;

    CausalEngine() {
        init_default_causal_models();
    }

    void init_default_causal_models() {
        // Physics standard structural equations:
        // accel = force / mass
        define_equation("accel", "force / mass");
        // velocity = accel * time
        define_equation("velocity", "accel * time");
        // distance = 0.5 * accel * (time ^ 2)
        define_equation("distance", "0.5 * accel * (time ^ 2)");
        // power = force * velocity
        define_equation("power", "force * velocity");
        // work = force * distance
        define_equation("work", "force * distance");
        // momentum = mass * velocity
        define_equation("momentum", "mass * velocity");
        // voltage = current * resistance
        define_equation("voltage", "current * resistance");
        // profit = (price - cost) * quantity
        define_equation("profit", "(price - cost) * quantity");

        // ── Legal Tort & Liability SCM ──
        // accident_risk = baseline_hazard + 10 * defect_present + 5 * breach_of_duty
        define_equation("accident_risk", "baseline_hazard + (10 * defect_present) + (5 * breach_of_duty)");
        // damages = accident_occurred * (medical_cost + lost_wages + 2000)
        define_equation("damages", "accident_occurred * (medical_cost + lost_wages + 2000)");
        // liability = duty_of_care * defect_present * accident_occurred
        define_equation("liability", "duty_of_care * defect_present * accident_occurred");

        // ── Biomedical Clinical Trial & Drug Efficacy SCM ──
        // biomarker = baseline_biomarker - (25 * drug_treatment) + (10 * disease_severity)
        define_equation("biomarker", "baseline_biomarker - (25 * drug_treatment) + (10 * disease_severity)");
        // recovery_rate = 80 + (30 * drug_treatment) - (20 * disease_severity) - (0.5 * patient_age)
        define_equation("recovery_rate", "80 + (30 * drug_treatment) - (20 * disease_severity) - (0.5 * patient_age)");
        // adverse_events = 5 + (0.2 * patient_age) + (2 * drug_treatment)
        define_equation("adverse_events", "5 + (0.2 * patient_age) + (2 * drug_treatment)");

        // ── Macro-Economic Policy & Central Bank SCM ──
        // inflation = 10 - (1.5 * interest_rate) + (0.8 * money_growth)
        define_equation("inflation", "10 - (1.5 * interest_rate) + (0.8 * money_growth)");
        // unemployment = 3 + (0.6 * interest_rate) - (0.2 * gdp_stimulus)
        define_equation("unemployment", "3 + (0.6 * interest_rate) - (0.2 * gdp_stimulus)");
        // gdp_growth = 5 - (0.7 * interest_rate) + (0.5 * gdp_stimulus)
        define_equation("gdp_growth", "5 - (0.7 * interest_rate) + (0.5 * gdp_stimulus)");
    }

    void define_equation(const std::string& target, const std::string& expr_str) {
        CausalParser parser(expr_str);
        equations[target] = parser.parse();
    }

    void observe(const std::string& var, double val) {
        factual_evidence[var] = val;
    }

    // Level 1: Associational / Factual Prediction
    CausalResult predict(const std::string& target) {
        std::map<std::string, double> env = factual_evidence;
        std::vector<std::string> steps;

        auto eval_res = evaluate_dag(env, steps);
        if (eval_res.count(target)) {
            double val = eval_res[target];
            std::string val_str = format_val(val);
            return {true, target, val, val_str, steps, "Factual: " + target + " = " + val_str};
        }
        return {false, target, 0.0, "", steps, "Could not resolve variable: " + target};
    }

    // Level 2: Interventional Reasoning (do-calculus / graph surgery)
    CausalResult intervene(const std::string& do_var, double do_val, const std::string& target) {
        std::map<std::string, double> env = factual_evidence;
        env[do_var] = do_val; // Graph surgery: sever incoming equation, set constant

        std::vector<std::string> steps;
        steps.push_back("1. Graph Surgery: Sever all incoming causal links to [" + do_var + "] and set do(" + do_var + " = " + format_val(do_val) + ")");

        // Mutilated equations (remove do_var from equations)
        auto mutilated_eqs = equations;
        mutilated_eqs.erase(do_var);

        auto eval_res = evaluate_dag_custom(env, mutilated_eqs, steps);
        if (eval_res.count(target)) {
            double val = eval_res[target];
            std::string val_str = format_val(val);
            steps.push_back("2. Downstream Propagation: " + target + " = " + val_str);
            return {true, target, val, val_str, steps, "Intervention: do(" + do_var + " = " + format_val(do_val) + ") -> " + target + " = " + val_str};
        }
        return {false, target, 0.0, "", steps, "Could not resolve intervention target: " + target};
    }

    // Level 3: Counterfactual Inference (Pearl's 3-Step: Abduction -> Action -> Prediction)
    CausalResult counterfactual(const std::string& hyp_var, double hyp_val, const std::string& target) {
        std::vector<std::string> steps;

        // Step 1: Abduction (Infer background exogenous state U from factual evidence)
        std::map<std::string, double> U = factual_evidence;
        auto factual_state = evaluate_dag(U, steps);
        steps.push_back("Step 1 (Abduction): Inferred factual background state: " + describe_env(factual_state));

        // Step 2: Action / Graph Mutilation (Apply counterfactual hypothesis X <- x*)
        std::map<std::string, double> counterfactual_env = factual_evidence;
        counterfactual_env[hyp_var] = hyp_val;
        steps.push_back("Step 2 (Action): Mutilated model with counterfactual hypothesis: " + hyp_var + " <- " + format_val(hyp_val));

        // Step 3: Prediction (Compute counterfactual outcome under mutilated model)
        auto mutilated_eqs = equations;
        mutilated_eqs.erase(hyp_var);

        auto counterfactual_state = evaluate_dag_custom(counterfactual_env, mutilated_eqs, steps);
        if (counterfactual_state.count(target)) {
            double val = counterfactual_state[target];
            std::string val_str = format_val(val);
            steps.push_back("Step 3 (Prediction): Counterfactual outcome " + target + "* = " + val_str);
            return {true, target, val, val_str, steps, "Counterfactual: IF " + hyp_var + " = " + format_val(hyp_val) + " THEN " + target + " = " + val_str};
        }
        return {false, target, 0.0, "", steps, "Could not resolve counterfactual target: " + target};
    }

private:
    std::string format_val(double val) const {
        std::string s = std::to_string(val);
        if (s.find('.') != std::string::npos) {
            s.erase(s.find_last_not_of('0') + 1, std::string::npos);
            if (s.back() == '.') s.pop_back();
        }
        return s;
    }

    std::string describe_env(const std::map<std::string, double>& env) const {
        std::string res;
        for (const auto& kv : env) {
            if (!res.empty()) res += ", ";
            res += kv.first + "=" + format_val(kv.second);
        }
        return res;
    }

    std::map<std::string, double> evaluate_dag(std::map<std::string, double> env, std::vector<std::string>& steps) {
        return evaluate_dag_custom(env, equations, steps);
    }

    std::map<std::string, double> evaluate_dag_custom(std::map<std::string, double> env,
                                                      const std::map<std::string, CausalExprPtr>& eqs,
                                                      std::vector<std::string>& steps) {
        bool progress = true;
        int passes = 0;
        while (progress && passes < 20) {
            progress = false;
            passes++;
            for (const auto& kv : eqs) {
                if (env.count(kv.first)) continue;
                try {
                    double v = kv.second->eval(env);
                    env[kv.first] = v;
                    steps.push_back("Derived " + kv.first + " = " + format_val(v) + " via " + kv.second->to_string());
                    progress = true;
                } catch (...) {
                    // dependencies not yet ready
                }
            }
        }
        return env;
    }
};

}} // namespace brain2::reasoning
