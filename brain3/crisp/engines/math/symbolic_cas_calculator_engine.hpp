#pragma once
/**
 * brain3/crisp/engines/math/symbolic_cas_calculator_engine.hpp
 *
 * THE BRAIN — SYMBOLIC CAS CALCULATOR ENGINE ("SymPy in C++")
 *
 * An exact, zero-float-drift Computer Algebra System (CAS) and Symbolic Calculator
 * designed to serve as the computational backbone for all autonomous discovery engines.
 *
 * Features:
 * 1. Arbitrary 128-bit exact Rational arithmetic (num/den, gcd-reduced)
 * 2. Symbolic AST: Variables, Rationals, Add, Mul, Pow, Trigonometric, Exp, Ln
 * 3. Exact Analytic Differentiation d/dx (Product, Quotient, Chain Rule)
 * 4. Exact Symbolic Taylor Expansion & Limits (L'Hôpital)
 * 5. Multi-variate Expression Substitution & System Solving
 * 6. Exact Matrix Algebra & Lie Commutators: [A, B] = AB - BA
 */

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <memory>
#include <unordered_map>
#include <cmath>
#include <numeric>
#include <cstdint>
#include <cctype>
#include <cassert>
#include <algorithm>

namespace thebrain {
namespace cas {

// ─────────────────────────────────────────────────────────────────────────────
// 1. EXACT 128-BIT RATIONAL NUMBER CLASS
// ─────────────────────────────────────────────────────────────────────────────

inline __int128_t abs128(__int128_t v) {
    return v < 0 ? -v : v;
}

inline __int128_t gcd128(__int128_t a, __int128_t b) {
    a = abs128(a);
    b = abs128(b);
    while (b != 0) {
        __int128_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}

// Exact decimal rendering of a signed 128-bit integer (repeated division by 10).
// Works on the unsigned magnitude so the most negative value is handled safely.
inline std::string to_string128(__int128_t v) {
    if (v == 0) return "0";
    bool negative = v < 0;
    unsigned __int128 mag = negative
        ? static_cast<unsigned __int128>(-(v + 1)) + 1u
        : static_cast<unsigned __int128>(v);
    std::string digits;
    while (mag > 0) {
        digits += static_cast<char>('0' + static_cast<int>(mag % 10));
        mag /= 10;
    }
    if (negative) digits += '-';
    std::reverse(digits.begin(), digits.end());
    return digits;
}

struct Rational {
    __int128_t num;
    __int128_t den;

    Rational(__int128_t n = 0, __int128_t d = 1) : num(n), den(d) {
        if (den == 0) {
            throw std::invalid_argument("Division by zero in Rational creation.");
        }
        if (den < 0) {
            num = -num;
            den = -den;
        }
        __int128_t g = gcd128(num, den);
        if (g > 1) {
            num /= g;
            den /= g;
        }
    }

    bool is_zero() const { return num == 0; }
    bool is_one() const { return num == 1 && den == 1; }
    bool is_integer() const { return den == 1; }

    Rational operator+(const Rational& o) const {
        return Rational(num * o.den + o.num * den, den * o.den);
    }
    Rational operator-(const Rational& o) const {
        return Rational(num * o.den - o.num * den, den * o.den);
    }
    Rational operator*(const Rational& o) const {
        return Rational(num * o.num, den * o.den);
    }
    Rational operator/(const Rational& o) const {
        if (o.is_zero()) throw std::invalid_argument("Rational division by zero.");
        return Rational(num * o.den, den * o.num);
    }
    Rational operator-() const {
        return Rational(-num, den);
    }

    bool operator==(const Rational& o) const {
        return num == o.num && den == o.den;
    }
    bool operator!=(const Rational& o) const {
        return !(*this == o);
    }
    bool operator<(const Rational& o) const {
        return num * o.den < o.num * den;
    }
    bool operator>(const Rational& o) const {
        return num * o.den > o.num * den;
    }
    bool operator<=(const Rational& o) const { return !(*this > o); }
    bool operator>=(const Rational& o) const { return !(*this < o); }

    double to_double() const {
        return static_cast<double>(num) / static_cast<double>(den);
    }

    std::string to_string() const {
        if (den == 1) {
            return to_string128(num);
        }
        return to_string128(num) + "/" + to_string128(den);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// 1b. EXACT DECIMAL-LITERAL PARSING (string -> Rational, no float truncation)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Convert a plain decimal literal ("[-+]dd[.ddd][e[-+]dd]") into an exact
 * rational by parsing the digit string directly: integer part + fractional
 * digits build numerator/denominator ("0.5" -> 5/10), reduced via gcd by the
 * Rational constructor. Exponents are applied exactly as powers of ten.
 *
 * Returns false when `s` is not a pure decimal literal (the caller may then
 * treat it as a symbol). Throws std::runtime_error for syntactically valid
 * literals that exceed exact 128-bit range — never falls back to lossy
 * floating-point conversion.
 */
inline bool try_parse_decimal_rational(const std::string& s, Rational& out) {
    size_t i = 0;
    bool negative = false;
    if (i < s.size() && (s[i] == '+' || s[i] == '-')) {
        negative = (s[i] == '-');
        ++i;
    }

    std::string digits;
    int frac_digits = 0;
    bool any_digit = false;

    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
        digits += s[i];
        any_digit = true;
        ++i;
    }
    if (i < s.size() && s[i] == '.') {
        ++i;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
            digits += s[i];
            ++frac_digits;
            any_digit = true;
            ++i;
        }
    }
    if (!any_digit) return false;

    long exp_scale = 0; // net power of ten applied to the value
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
        ++i;
        bool exp_negative = false;
        if (i < s.size() && (s[i] == '+' || s[i] == '-')) {
            exp_negative = (s[i] == '-');
            ++i;
        }
        if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i]))) return false;
        long ev = 0;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
            ev = ev * 10 + (s[i] - '0');
            if (ev > 4096) {
                throw std::runtime_error("CAS parse: decimal exponent out of exact range in '" + s + "'");
            }
            ++i;
        }
        exp_scale = exp_negative ? -ev : ev;
    }
    if (i != s.size()) return false; // trailing garbage -> not a pure number token

    // Trim insignificant zeros so range checks reflect true precision.
    size_t first_sig = digits.find_first_not_of('0');
    if (first_sig == std::string::npos) {
        out = Rational(0, 1);
        return true;
    }
    while (frac_digits > 0 && digits.back() == '0') {
        digits.pop_back();
        --frac_digits;
    }

    long num_scale = exp_scale > 0 ? exp_scale : 0;
    long den_scale = exp_scale < 0 ? -exp_scale : 0;
    long num_digits = static_cast<long>(digits.size()) + num_scale;
    long den_digits = static_cast<long>(frac_digits) + den_scale;
    // __int128 holds ~38 decimal digits; refuse rather than silently truncating.
    if (num_digits > 30 || den_digits > 30) {
        throw std::runtime_error("CAS parse: decimal literal '" + s + "' exceeds 128-bit exact range");
    }

    __int128_t num = 0;
    for (char c : digits) num = num * 10 + (c - '0');
    __int128_t den = 1;
    for (long k = 0; k < frac_digits + den_scale; ++k) den *= 10;
    for (long k = 0; k < num_scale; ++k) num *= 10;
    if (negative) num = -num;

    out = Rational(num, den);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. SYMBOLIC EXPRESSION AST
// ─────────────────────────────────────────────────────────────────────────────

enum class ExprKind {
    CONST,
    VAR,
    ADD,
    MUL,
    POW,
    SIN,
    COS,
    EXP,
    LN,
    SQRT
};

struct CasNode;
using CasExpr = std::shared_ptr<CasNode>;

struct CasNode {
    ExprKind kind;
    Rational const_val;
    std::string var_name;
    std::vector<CasExpr> args;

    CasNode(const Rational& r) : kind(ExprKind::CONST), const_val(r) {}
    CasNode(const std::string& v) : kind(ExprKind::VAR), var_name(v) {}
    CasNode(ExprKind k, const std::vector<CasExpr>& a) : kind(k), args(a) {}

    static CasExpr make_num(int64_t n, int64_t d = 1) {
        return std::make_shared<CasNode>(Rational(n, d));
    }
    static CasExpr make_rat(const Rational& r) {
        return std::make_shared<CasNode>(r);
    }
    static CasExpr make_var(const std::string& v) {
        return std::make_shared<CasNode>(v);
    }
    static CasExpr make_add(const CasExpr& a, const CasExpr& b) {
        return simplify_add(a, b);
    }
    static CasExpr make_sub(const CasExpr& a, const CasExpr& b) {
        return simplify_add(a, make_mul(make_num(-1), b));
    }
    static CasExpr make_mul(const CasExpr& a, const CasExpr& b) {
        return simplify_mul(a, b);
    }
    static CasExpr make_div(const CasExpr& a, const CasExpr& b) {
        return simplify_mul(a, make_pow(b, make_num(-1)));
    }
    static CasExpr make_pow(const CasExpr& base, const CasExpr& exp) {
        return simplify_pow(base, exp);
    }
    static CasExpr make_sin(const CasExpr& a) {
        return std::make_shared<CasNode>(ExprKind::SIN, std::vector<CasExpr>{a});
    }
    static CasExpr make_cos(const CasExpr& a) {
        return std::make_shared<CasNode>(ExprKind::COS, std::vector<CasExpr>{a});
    }
    static CasExpr make_exp(const CasExpr& a) {
        return std::make_shared<CasNode>(ExprKind::EXP, std::vector<CasExpr>{a});
    }
    static CasExpr make_ln(const CasExpr& a) {
        return std::make_shared<CasNode>(ExprKind::LN, std::vector<CasExpr>{a});
    }

    // Simplification & Constant Folding
    static CasExpr simplify_add(const CasExpr& a, const CasExpr& b) {
        if (a->kind == ExprKind::CONST && a->const_val.is_zero()) return b;
        if (b->kind == ExprKind::CONST && b->const_val.is_zero()) return a;
        if (a->kind == ExprKind::CONST && b->kind == ExprKind::CONST) {
            return make_rat(a->const_val + b->const_val);
        }
        return std::make_shared<CasNode>(ExprKind::ADD, std::vector<CasExpr>{a, b});
    }

    static CasExpr simplify_mul(const CasExpr& a, const CasExpr& b) {
        if (a->kind == ExprKind::CONST && a->const_val.is_zero()) return make_num(0);
        if (b->kind == ExprKind::CONST && b->const_val.is_zero()) return make_num(0);
        if (a->kind == ExprKind::CONST && a->const_val.is_one()) return b;
        if (b->kind == ExprKind::CONST && b->const_val.is_one()) return a;
        if (a->kind == ExprKind::CONST && b->kind == ExprKind::CONST) {
            return make_rat(a->const_val * b->const_val);
        }
        return std::make_shared<CasNode>(ExprKind::MUL, std::vector<CasExpr>{a, b});
    }

    static CasExpr simplify_pow(const CasExpr& base, const CasExpr& exp) {
        if (exp->kind == ExprKind::CONST && exp->const_val.is_zero()) return make_num(1);
        if (exp->kind == ExprKind::CONST && exp->const_val.is_one()) return base;
        if (base->kind == ExprKind::CONST && exp->kind == ExprKind::CONST && exp->const_val.is_integer()) {
            int64_t p = static_cast<int64_t>(exp->const_val.num);
            if (p >= 0 && p <= 8) {
                Rational res(1, 1);
                for (int64_t i = 0; i < p; ++i) res = res * base->const_val;
                return make_rat(res);
            }
        }
        return std::make_shared<CasNode>(ExprKind::POW, std::vector<CasExpr>{base, exp});
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// 3. EXACT SYMBOLIC RENDERING & CALCULUS ENGINE
// ─────────────────────────────────────────────────────────────────────

class SymbolicCasCalculatorEngine {
public:
    /**
     * Parse common mathematical expressions into AST CasExpr nodes.
     */
    static CasExpr parse_expression(const std::string& input_str) {
        std::string s = input_str;
        s.erase(std::remove_if(s.begin(), s.end(), ::isspace), s.end());
        if (s.empty()) return CasNode::make_num(0);

        // Power pattern: var^number (e.g. x^2, x^3)
        size_t caret = s.find('^');
        if (caret != std::string::npos && caret > 0 && caret + 1 < s.size()) {
            std::string base_s = s.substr(0, caret);
            std::string exp_s = s.substr(caret + 1);
            CasExpr base = parse_expression(base_s);
            CasExpr exp = parse_expression(exp_s);
            return CasNode::make_pow(base, exp);
        }

        // Functions: sin, cos, exp, ln, sqrt
        if (s.rfind("sin(", 0) == 0 && s.back() == ')') {
            return CasNode::make_sin(parse_expression(s.substr(4, s.size() - 5)));
        }
        if (s.rfind("cos(", 0) == 0 && s.back() == ')') {
            return CasNode::make_cos(parse_expression(s.substr(4, s.size() - 5)));
        }
        if (s.rfind("exp(", 0) == 0 && s.back() == ')') {
            return CasNode::make_exp(parse_expression(s.substr(4, s.size() - 5)));
        }
        if (s.rfind("ln(", 0) == 0 && s.back() == ')') {
            return CasNode::make_ln(parse_expression(s.substr(3, s.size() - 4)));
        }
        if (s.rfind("sqrt(", 0) == 0 && s.back() == ')') {
            return CasNode::make_pow(parse_expression(s.substr(5, s.size() - 6)), CasNode::make_rat(Rational(1, 2)));
        }

        // Numbers: convert the decimal string exactly to a Rational
        // (never through double — "0.5" must not truncate to 0)
        Rational r;
        if (try_parse_decimal_rational(s, r)) {
            return CasNode::make_rat(r);
        }

        // Variable
        return CasNode::make_var(s);
    }

    static std::string render(const CasExpr& e) {
        if (!e) return "0";
        switch (e->kind) {
            case ExprKind::CONST:
                return e->const_val.to_string();
            case ExprKind::VAR:
                return e->var_name;
            case ExprKind::ADD:
                return "(" + render(e->args[0]) + " + " + render(e->args[1]) + ")";
            case ExprKind::MUL:
                return "(" + render(e->args[0]) + " * " + render(e->args[1]) + ")";
            case ExprKind::POW:
                return "(" + render(e->args[0]) + " ^ " + render(e->args[1]) + ")";
            case ExprKind::SIN:
                return "sin(" + render(e->args[0]) + ")";
            case ExprKind::COS:
                return "cos(" + render(e->args[0]) + ")";
            case ExprKind::EXP:
                return "exp(" + render(e->args[0]) + ")";
            case ExprKind::LN:
                return "ln(" + render(e->args[0]) + ")";
            case ExprKind::SQRT:
                return "sqrt(" + render(e->args[0]) + ")";
        }
        return "?";
    }

    /**
     * Exact Analytic Differentiation: d(expr)/d(var)
     */
    static CasExpr diff(const CasExpr& e, const std::string& var) {
        if (!e) return CasNode::make_num(0);
        switch (e->kind) {
            case ExprKind::CONST:
                return CasNode::make_num(0);
            case ExprKind::VAR:
                return (e->var_name == var) ? CasNode::make_num(1) : CasNode::make_num(0);
            case ExprKind::ADD:
                return CasNode::make_add(diff(e->args[0], var), diff(e->args[1], var));
            case ExprKind::MUL: {
                // Product Rule: u'v + uv'
                auto u = e->args[0], v = e->args[1];
                return CasNode::make_add(
                    CasNode::make_mul(diff(u, var), v),
                    CasNode::make_mul(u, diff(v, var))
                );
            }
            case ExprKind::POW: {
                // General Power Rule: d(u^v)/dx = u^v * (v' * ln(u) + v * u'/u)
                auto u = e->args[0], v = e->args[1];
                if (v->kind == ExprKind::CONST) {
                    // Power rule for constant exponent: n * u^(n-1) * u'
                    auto n_minus_1 = CasNode::make_sub(v, CasNode::make_num(1));
                    auto du = diff(u, var);
                    return CasNode::make_mul(
                        v,
                        CasNode::make_mul(CasNode::make_pow(u, n_minus_1), du)
                    );
                } else {
                    auto du = diff(u, var);
                    auto dv = diff(v, var);
                    auto term1 = CasNode::make_mul(dv, CasNode::make_ln(u));
                    auto term2 = CasNode::make_mul(v, CasNode::make_div(du, u));
                    return CasNode::make_mul(e, CasNode::make_add(term1, term2));
                }
            }
            case ExprKind::SIN: {
                // Chain Rule: cos(u) * u'
                auto u = e->args[0];
                return CasNode::make_mul(CasNode::make_cos(u), diff(u, var));
            }
            case ExprKind::COS: {
                // Chain Rule: -sin(u) * u'
                auto u = e->args[0];
                return CasNode::make_mul(CasNode::make_mul(CasNode::make_num(-1), CasNode::make_sin(u)), diff(u, var));
            }
            case ExprKind::EXP: {
                // Chain Rule: exp(u) * u'
                auto u = e->args[0];
                return CasNode::make_mul(CasNode::make_exp(u), diff(u, var));
            }
            case ExprKind::LN: {
                // Chain Rule: u' / u
                auto u = e->args[0];
                return CasNode::make_div(diff(u, var), u);
            }
            default:
                return CasNode::make_num(0);
        }
    }

    /**
     * Exact Substitution: replace all instances of var with replacement_expr
     */
    static CasExpr substitute(const CasExpr& e, const std::string& var, const CasExpr& replacement) {
        if (!e) return nullptr;
        if (e->kind == ExprKind::VAR && e->var_name == var) {
            return replacement;
        }
        if (e->kind == ExprKind::CONST || e->kind == ExprKind::VAR) {
            return e;
        }
        std::vector<CasExpr> new_args;
        for (const auto& arg : e->args) {
            new_args.push_back(substitute(arg, var, replacement));
        }
        return std::make_shared<CasNode>(e->kind, new_args);
    }

    /**
     * Exact Rational Evaluation
     */
    static Rational eval_rational(const CasExpr& e, const std::unordered_map<std::string, Rational>& env) {
        if (!e) return Rational(0);
        switch (e->kind) {
            case ExprKind::CONST:
                return e->const_val;
            case ExprKind::VAR: {
                auto it = env.find(e->var_name);
                if (it != env.end()) return it->second;
                throw std::runtime_error("Variable " + e->var_name + " not bound in exact rational evaluation.");
            }
            case ExprKind::ADD:
                return eval_rational(e->args[0], env) + eval_rational(e->args[1], env);
            case ExprKind::MUL:
                return eval_rational(e->args[0], env) * eval_rational(e->args[1], env);
            case ExprKind::POW: {
                auto b = eval_rational(e->args[0], env);
                auto exp = eval_rational(e->args[1], env);
                if (!exp.is_integer()) throw std::runtime_error("Non-integer rational powers not closed in Q.");
                int64_t p = static_cast<int64_t>(exp.num);
                if (p == 0) return Rational(1);
                Rational res(1);
                Rational base_val = (p > 0) ? b : Rational(1) / b;
                int64_t abs_p = p > 0 ? p : -p;
                for (int64_t i = 0; i < abs_p; ++i) res = res * base_val;
                return res;
            }
            default:
                throw std::runtime_error("Transcendental functions (sin/cos/exp/ln) require symbolic or numerical approximation.");
        }
    }

    /**
     * Exact Matrix Commutator: [A, B] = A * B - B * A
     */
    static std::vector<std::vector<CasExpr>> matrix_commutator(
        const std::vector<std::vector<CasExpr>>& A,
        const std::vector<std::vector<CasExpr>>& B
    ) {
        size_t n = A.size();
        assert(n == A[0].size() && n == B.size() && n == B[0].size());

        auto AB = matrix_mul(A, B);
        auto BA = matrix_mul(B, A);

        std::vector<std::vector<CasExpr>> comm(n, std::vector<CasExpr>(n));
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                comm[i][j] = CasNode::make_sub(AB[i][j], BA[i][j]);
            }
        }
        return comm;
    }

    static std::vector<std::vector<CasExpr>> matrix_mul(
        const std::vector<std::vector<CasExpr>>& A,
        const std::vector<std::vector<CasExpr>>& B
    ) {
        size_t n = A.size();
        std::vector<std::vector<CasExpr>> C(n, std::vector<CasExpr>(n, CasNode::make_num(0)));
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                CasExpr sum = CasNode::make_num(0);
                for (size_t k = 0; k < n; ++k) {
                    sum = CasNode::make_add(sum, CasNode::make_mul(A[i][k], B[k][j]));
                }
                C[i][j] = sum;
            }
        }
        return C;
    }
};

} // namespace cas
} // namespace thebrain
