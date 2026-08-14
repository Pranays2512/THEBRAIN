#pragma once
#include <string>
#include <vector>
#include <map>
#include <set>
#include <tuple>
#include <variant>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <sstream>

#include "crisp/engines/reasoning/reasoning_engine.hpp"

namespace brain2 {
namespace reasoning {

// ── Expressions ─────────────────────────────────────────────────────────────
struct Expr;
using ExprPtr = std::shared_ptr<Expr>;

struct LiteralExpr { double val; };
struct VarExpr { std::string name; };
struct OpExpr { std::string op; std::vector<ExprPtr> args; };

struct Expr : std::variant<LiteralExpr, VarExpr, OpExpr> {
    using variant::variant;
};

inline ExprPtr lit(double v) { return std::make_shared<Expr>(LiteralExpr{v}); }
inline ExprPtr var(const std::string& n) { return std::make_shared<Expr>(VarExpr{n}); }
inline ExprPtr op(const std::string& o, const std::vector<ExprPtr>& args) { return std::make_shared<Expr>(OpExpr{o, args}); }

inline double ev(const ExprPtr& e, const std::map<std::string, double>& env) {
    return std::visit([&](auto&& v) -> double {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, LiteralExpr>) return v.val;
        if constexpr (std::is_same_v<T, VarExpr>) {
            auto it = env.find(v.name);
            if (it == env.end()) throw std::runtime_error("unknown value for " + v.name);
            return it->second;
        }
        if constexpr (std::is_same_v<T, OpExpr>) {
            if (v.op == "neg") return -ev(v.args[0], env);
            double a = ev(v.args[0], env);
            double b = ev(v.args[1], env);
            if (v.op == "+") return a + b;
            if (v.op == "-") return a - b;
            if (v.op == "*") return a * b;
            if (v.op == "/") return a / b;
            if (v.op == "^") return std::pow(a, b);
            throw std::runtime_error("unknown op " + v.op);
        }
        return 0;
    }, *e);
}

// ── Need (Goal) ─────────────────────────────────────────────────────────────
struct Need {
    std::string subject;
    std::string rel;
    bool operator<(const Need& o) const {
        if (subject != o.subject) return subject < o.subject;
        return rel < o.rel;
    }
    std::string to_string() const { return subject + "." + rel; }
};

// ── Policy Memory ───────────────────────────────────────────────────────────
struct Policy {
    std::string target;
    std::vector<std::string> inputs;
    ExprPtr expr;
};

class PolicyMemory {
public:
    std::map<std::string, std::shared_ptr<Policy>> by_target;

    PolicyMemory() {
        init_default_policies();
    }

    void init_default_policies() {
        // force = mass * accel
        add({"force", {"mass", "accel"}, op("*", {var("mass"), var("accel")})});
        
        // power = force * speed
        add({"power", {"force", "speed"}, op("*", {var("force"), var("speed")})});
        
        // momentum = mass * velocity
        add({"momentum", {"mass", "velocity"}, op("*", {var("mass"), var("velocity")})});
        
        // work = force * distance
        add({"work", {"force", "distance"}, op("*", {var("force"), var("distance")})});
        
        // voltage = current * resistance
        add({"voltage", {"current", "resistance"}, op("*", {var("current"), var("resistance")})});
        
        // density = mass / volume
        add({"density", {"mass", "volume"}, op("/", {var("mass"), var("volume")})});
        
        // speed = distance / time
        add({"speed", {"distance", "time"}, op("/", {var("distance"), var("time")})});
    }

    void add(const Policy& p) {
        by_target[p.target] = std::make_shared<Policy>(p);
    }
    std::shared_ptr<Policy> get(const std::string& target) const {
        auto it = by_target.find(target);
        return it != by_target.end() ? it->second : nullptr;
    }
    bool contains(const std::string& target) const {
        return by_target.count(target) > 0;
    }
};

// ── Blackboard ──────────────────────────────────────────────────────────────
class Blackboard {
public:
    std::map<Need, double> solved;
    std::vector<std::string> trace;

    void log(const std::string& msg, int depth = 0) {
        std::string pad(depth * 2, ' ');
        trace.push_back(pad + msg);
    }
    void show() const {
        for (const auto& t : trace) std::cout << t << "\n";
    }
};

class MeansEndsSolver;

// ── Knowledge Sources ───────────────────────────────────────────────────────
class KnowledgeSource {
public:
    virtual ~KnowledgeSource() = default;
    virtual bool can_handle(const Need& need) = 0;
    virtual std::optional<double> contribute(const Need& need, MeansEndsSolver* solver, int depth) = 0;
};

class MeansEndsSolver {
public:
    std::vector<KnowledgeSource*> sources;
    Blackboard bb;

    MeansEndsSolver(const std::vector<KnowledgeSource*>& srcs) : sources(srcs) {}

    std::optional<double> solve(const Need& need, int depth = 0) {
        if (bb.solved.count(need)) {
            bb.log("· " + need.to_string() + " = " + std::to_string(bb.solved[need]) + "   (memo)", depth);
            return bb.solved[need];
        }
        for (auto* src : sources) {
            if (src->can_handle(need)) {
                auto val = src->contribute(need, this, depth);
                if (val) {
                    bb.solved[need] = *val;
                    return val;
                }
            }
        }
        return std::nullopt;
    }
};

class FactSource : public KnowledgeSource {
private:
    ReasoningEngine* kb;
public:
    FactSource(ReasoningEngine* kb) : kb(kb) {}

    bool can_handle(const Need& need) override {
        auto res = kb->ask(need.subject, need.rel);
        return !res.first.empty();
    }

    std::optional<double> contribute(const Need& need, MeansEndsSolver* solver, int depth) override {
        auto res = kb->ask(need.subject, need.rel);
        if (res.first.empty()) return std::nullopt;
        solver->bb.log("✓ " + need.to_string() + " = " + res.first + "   (fact: " + res.second + ")", depth);
        try {
            return std::stod(res.first);
        } catch (...) {
            return std::nullopt;
        }
    }
};

class PolicySource : public KnowledgeSource {
private:
    PolicyMemory* mem;
public:
    PolicySource(PolicyMemory* mem) : mem(mem) {}

    bool can_handle(const Need& need) override {
        return mem->contains(need.rel);
    }

    std::optional<double> contribute(const Need& need, MeansEndsSolver* solver, int depth) override {
        auto p = mem->get(need.rel);
        if (!p) return std::nullopt;

        std::string inputs_str;
        for (size_t i = 0; i < p->inputs.size(); i++) {
            if (i > 0) inputs_str += ", ";
            inputs_str += p->inputs[i];
        }
        solver->bb.log("? " + need.to_string() + " not a fact - policy needs " + inputs_str, depth);

        std::map<std::string, double> env;
        for (const auto& r : p->inputs) {
            auto v = solver->solve(Need{need.subject, r}, depth + 1);
            if (!v) {
                solver->bb.log("x could not get " + need.subject + "." + r, depth + 1);
                return std::nullopt;
            }
            env[r] = *v;
        }

        try {
            double result = ev(p->expr, env);
            solver->bb.log("= " + need.to_string() + " = " + std::to_string(result) + "   (policy " + need.rel + ")", depth);
            return result;
        } catch (...) {
            return std::nullopt;
        }
    }
};

}}
