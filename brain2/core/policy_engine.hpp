// policy_engine.hpp — the CRISP symbolic reasoner, in C++.
//
// The C++ port of the Python neurosymbolic executive (means_ends.py +
// policy_proposer.py). It lives BESIDE the vector-symbolic BindingMemory as the
// crisp half of the brain: discrete string symbols, exact tuple-formula
// evaluation, verified policy learning — the membrane preserved in C++.
//
//   * PolicyMemory  — stored composition rules (target = f(inputs)), separate
//                     from the fuzzy binding memory.
//   * PolicyEngine  — means-ends solver with memoization (tabling) + cycle guard
//                     + a groundability PROPOSER that orders candidate policies,
//                     and a conjecture -> verify -> admit loop for learning new
//                     policies by composition (numeric gate; no fuzzy leakage).
//
// Facts come through a callback (std::function), so the engine can be fed by the
// Python KB or, later, a crispified view of the C++ BindingMemory. Header-only,
// no external dependencies.

#pragma once
#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace brain2 {

// ── tuple-formula AST: ("*", mass, accel) etc. ──────────────────────────────
struct Expr {
  enum Kind { NUM, VAR, OP } kind;
  double num = 0.0;
  std::string var;            // VAR
  char op = 0;                // OP: + - * / ^
  std::shared_ptr<Expr> a, b;
};
using ExprPtr = std::shared_ptr<Expr>;

inline ExprPtr num(double v) { auto e = std::make_shared<Expr>(); e->kind = Expr::NUM; e->num = v; return e; }
inline ExprPtr var(const std::string& s) { auto e = std::make_shared<Expr>(); e->kind = Expr::VAR; e->var = s; return e; }
inline ExprPtr opx(char o, ExprPtr a, ExprPtr b) { auto e = std::make_shared<Expr>(); e->kind = Expr::OP; e->op = o; e->a = a; e->b = b; return e; }

// Evaluate lazily (one branch) — never computes a**b for a '+' op, so no overflow.
inline double eval(const ExprPtr& e, const std::map<std::string, double>& env) {
  switch (e->kind) {
    case Expr::NUM: return e->num;
    case Expr::VAR: {
      auto it = env.find(e->var);
      if (it == env.end()) throw std::runtime_error("unknown value for '" + e->var + "'");
      return it->second;
    }
    default: break;
  }
  double a = eval(e->a, env), b = eval(e->b, env);
  switch (e->op) {
    case '+': return a + b;
    case '-': return a - b;
    case '*': return a * b;
    case '/': return a / b;
    case '^': return std::pow(a, b);
  }
  throw std::runtime_error("unknown op");
}

// Replace every VAR==name with sub (for conjecture composition).
inline ExprPtr substitute(const ExprPtr& e, const std::string& name, const ExprPtr& sub) {
  if (e->kind == Expr::VAR) return e->var == name ? sub : e;
  if (e->kind == Expr::OP) return opx(e->op, substitute(e->a, name, sub), substitute(e->b, name, sub));
  return e;
}

// ── serialize / parse an Expr (prefix s-expr: "(* mass #0.5)") ───────────────
inline std::string serialize_expr(const ExprPtr& e) {
  if (e->kind == Expr::NUM) { std::ostringstream o; o << "#" << e->num; return o.str(); }
  if (e->kind == Expr::VAR) return e->var;
  return std::string("(") + e->op + " " + serialize_expr(e->a) + " " + serialize_expr(e->b) + ")";
}

inline ExprPtr parse_tokens(const std::vector<std::string>& t, size_t& i) {
  const std::string& tok = t.at(i++);
  if (tok == "(") {
    char op = t.at(i++)[0];
    ExprPtr a = parse_tokens(t, i), b = parse_tokens(t, i);
    ++i;  // ')'
    return opx(op, a, b);
  }
  if (!tok.empty() && tok[0] == '#') return num(std::stod(tok.substr(1)));
  return var(tok);
}

inline ExprPtr parse_expr(const std::string& s) {
  std::vector<std::string> toks;
  std::string cur;
  auto flush = [&]() { if (!cur.empty()) { toks.push_back(cur); cur.clear(); } };
  for (char c : s) {
    if (c == '(' || c == ')') { flush(); toks.push_back(std::string(1, c)); }
    else if (c == ' ') flush();
    else cur += c;
  }
  flush();
  size_t i = 0;
  return parse_tokens(toks, i);
}

// ── policy + memory ─────────────────────────────────────────────────────────
struct Policy {
  std::string target;
  std::vector<std::string> inputs;
  ExprPtr expr;
};

class PolicyMemory {
 public:
  void add(const Policy& p) { by_target_[p.target].push_back(p); }
  const std::vector<Policy>& candidates(const std::string& t) const {
    static const std::vector<Policy> empty;
    auto it = by_target_.find(t);
    return it == by_target_.end() ? empty : it->second;
  }
  bool contains(const std::string& t) const { return by_target_.count(t) > 0; }
  std::vector<std::string> targets() const {
    std::vector<std::string> out;
    for (auto& kv : by_target_) out.push_back(kv.first);
    return out;
  }

  void save(const std::string& path) const {
    std::ofstream f(path);
    for (auto& kv : by_target_)
      for (auto& p : kv.second) {
        f << p.target << '\t';
        for (size_t i = 0; i < p.inputs.size(); ++i)
          f << (i ? "," : "") << p.inputs[i];
        f << '\t' << serialize_expr(p.expr) << '\n';
      }
  }
  void load(const std::string& path) {
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
      auto t1 = line.find('\t');
      if (t1 == std::string::npos) continue;
      auto t2 = line.find('\t', t1 + 1);
      Policy p;
      p.target = line.substr(0, t1);
      std::string ins = line.substr(t1 + 1, t2 - t1 - 1);
      std::stringstream ss(ins);
      std::string in;
      while (std::getline(ss, in, ',')) if (!in.empty()) p.inputs.push_back(in);
      p.expr = parse_expr(line.substr(t2 + 1));
      add(p);
    }
  }
 private:
  std::map<std::string, std::vector<Policy>> by_target_;
};

// fact callback: returns true and sets `out` if (entity, rel) is a known fact.
using FactFn = std::function<bool(const std::string&, const std::string&, double&)>;

// ── the executive ─────────────────────────────────────────────────────────--
class PolicyEngine {
 public:
  PolicyEngine(PolicyMemory* mem, FactFn facts, bool use_proposer = true)
      : mem_(mem), facts_(std::move(facts)), use_proposer_(use_proposer) {}

  // Means-ends solve for (entity, rel). Memoized + cycle-guarded. work() counts
  // sub-goal attempts (the search cost the proposer reduces).
  std::optional<double> solve(const std::string& entity, const std::string& rel) {
    std::map<std::string, double> memo;
    std::set<std::string> seen;
    work_ = 0;
    return solve_(entity, rel, memo, seen);
  }
  long work() const { return work_; }

  // conjecture -> verify -> admit: compose a sub-policy into the target's policy,
  // verify it numerically against the step-by-step derivation, admit only if equal.
  bool learn(const std::string& target, const std::string& entity, double tol = 1e-6) {
    Policy conj;
    if (!conjecture(target, conj)) return false;
    auto ref = solve(entity, target);                 // ground truth (current policies)
    if (!ref) return false;
    std::map<std::string, double> env;
    for (auto& inp : conj.inputs) {
      auto v = solve(entity, inp);
      if (!v) return false;
      env[inp] = *v;
    }
    double got;
    try { got = eval(conj.expr, env); } catch (...) { return false; }
    if (std::fabs(got - *ref) < tol) { mem_->add(conj); return true; }  // admit
    return false;                                                       // reject
  }

 private:
  std::optional<double> solve_(const std::string& e, const std::string& rel,
                               std::map<std::string, double>& memo,
                               std::set<std::string>& seen) {
    work_++;
    std::string key = e + "\x1f" + rel;
    if (seen.count(key)) return std::nullopt;          // cycle guard
    auto m = memo.find(key);
    if (m != memo.end()) return m->second;             // tabling
    double fv;
    if (facts_(e, rel, fv)) { memo[key] = fv; return fv; }

    std::vector<const Policy*> cands;
    for (auto& p : mem_->candidates(rel)) cands.push_back(&p);
    if (use_proposer_)                                 // premise selection: best first
      std::stable_sort(cands.begin(), cands.end(), [&](const Policy* x, const Policy* y) {
        return score(*x, e) > score(*y, e);
      });

    seen.insert(key);
    for (const Policy* p : cands) {
      std::map<std::string, double> env;
      bool ok = true;
      for (auto& inp : p->inputs) {
        auto v = solve_(e, inp, memo, seen);
        if (!v) { ok = false; break; }
        env[inp] = *v;
      }
      if (ok) {
        seen.erase(key);
        double val = eval(p->expr, env);
        memo[key] = val;
        return val;
      }
    }
    seen.erase(key);
    return std::nullopt;
  }

  // proposer: mean groundability of a policy's inputs (higher = likelier to solve).
  double score(const Policy& p, const std::string& entity) {
    double s = 0.0;
    std::set<std::string> seen;
    for (auto& inp : p.inputs) s += groundable(inp, entity, seen);
    return p.inputs.empty() ? 0.0 : s / p.inputs.size();
  }
  double groundable(const std::string& rel, const std::string& entity, std::set<std::string>& seen) {
    if (seen.count(rel)) return 0.0;
    double fv;
    if (facts_(entity, rel, fv)) return 1.0;
    auto& cands = mem_->candidates(rel);
    if (cands.empty()) return 0.0;
    seen.insert(rel);
    double best = 0.0;
    for (auto& p : cands) {
      double s = 0.0;
      for (auto& inp : p.inputs) s += groundable(inp, entity, seen);
      best = std::max(best, p.inputs.empty() ? 0.0 : s / p.inputs.size());
    }
    seen.erase(rel);
    return best;
  }

  bool conjecture(const std::string& target, Policy& out) {
    auto& cands = mem_->candidates(target);
    if (cands.empty()) return false;
    const Policy& p = cands.front();
    for (auto& inp : p.inputs) {
      auto& q = mem_->candidates(inp);
      if (q.empty()) continue;
      out.target = target;
      out.expr = substitute(p.expr, inp, q.front().expr);
      for (auto& i : p.inputs) if (i != inp) out.inputs.push_back(i);
      for (auto& j : q.front().inputs)
        if (std::find(p.inputs.begin(), p.inputs.end(), j) == p.inputs.end())
          out.inputs.push_back(j);
      return true;
    }
    return false;
  }

  PolicyMemory* mem_;
  FactFn facts_;
  bool use_proposer_;
  long work_ = 0;
};

}  // namespace brain2
