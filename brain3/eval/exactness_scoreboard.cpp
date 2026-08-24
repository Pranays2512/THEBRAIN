// eval/exactness_scoreboard.cpp — Sprint 3: THE EXACTNESS SCOREBOARD.
//
// Blind, seeded problem suite inside native-engine coverage. Contestants see
// identical natural-language questions; an independent typed grader (numeric
// tolerance / polynomial-canonical form) decides correctness without knowing
// who answered.
//
// Contestants:
//   brain    — native engines called directly (capability ceiling; NL routing
//              is graded separately in Suite 1)
//   ollama:* — local Ollama models (auto-detected; qwen3:1.7B default)
//   frontier — OpenAI-compatible endpoint, enabled only when OPENAI_API_KEY
//              and EXACTNESS_FRONTIER_MODEL are set (never silently off)
//
// Metrics per category: exactness %, wall-clock mean s, cost proxy.
// Exit 0 iff brain exactness >= GATE_BRAIN_EXACT overall.
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <random>
#include <map>
#include <vector>
#include <algorithm>
#include <functional>
#include <iomanip>
#include <cstring>
#include <set>

#include "eval/eval_suite.hpp"
#include "crisp/engines/math/math_engine.hpp"
#include "crisp/engines/math/math_engine.hpp"

using namespace brain2::math;

static constexpr double GATE_BRAIN_EXACT = 0.95;

// ── problems ─────────────────────────────────────────────────────────────────
enum class Cat { Arith, Deriv, Integral, Newton, Gravitation, Ph };
static const char* cat_name(Cat c) {
    switch (c) {
        case Cat::Arith: return "exact_arith";
        case Cat::Deriv: return "derivative";
        case Cat::Integral: return "integral";
        case Cat::Newton: return "newton_fma";
        case Cat::Gravitation: return "gravitation";
        case Cat::Ph: return "buffer_ph";
    }
    return "?";
}

struct Problem {
    int id;
    Cat cat;
    std::string question;         // identical text to every contestant
    // grading payload (blind: grader sees only these + contestant answer)
    double expect_num = 0;
    bool symbolic = false;
    std::string expect_sym;       // canonical polynomial, e.g. "6x^2+4x"
    // structured engine input for the brain contestant (never shown to LLMs)
    std::string law, target, expr;
    std::map<std::string, double> knowns;
};

static std::string fmt_g(double v) {
    std::ostringstream o; o << v; return o.str();
}
static std::string poly_to_string(const std::vector<double>& c) { // c[i]=coef of x^i, desc order build
    // produce descending canonical like "6x^2+4x+1" (no spaces)
    std::string out;
    for (int p = (int)c.size() - 1; p >= 0; --p) {
        double k = c[p];
        if (std::fabs(k) < 1e-12) continue;
        std::string term;
        if (!out.empty()) term += (k > 0 ? "+" : "-");
        else if (k < 0) term += "-";
        double a = std::fabs(k);
        if (p == 0 || std::fabs(a - 1.0) > 1e-12) {
            if (a == (long long)a) { std::ostringstream t; t << (long long)a; term += t.str(); }
            else { std::ostringstream t; t << std::setprecision(12) << a; term += t.str(); }
        }
        if (p > 0) { term += "x"; if (p > 1) { term += "^"; term += std::to_string(p); } }
        out += term;
    }
    if (out.empty()) out = "0";
    return out;
}

static std::vector<Problem> generate_problems(unsigned seed, int per_cat) {
    std::mt19937 rng(seed);
    std::vector<Problem> ps; int id = 0;
    auto ri = [&](int lo, int hi){ return (int)(lo + rng() % (hi - lo + 1)); };
    auto rd = [&](double lo, double hi){ return lo + (hi - lo) * (rng() % 1000) / 1000.0; };

    for (int i = 0; i < per_cat; ++i) {
        {   // exact arithmetic: a/b + c*d - e
            long long a = ri(2, 40), b = ri(2, 30), c = ri(2, 25), d = ri(2, 12), e = ri(2, 60);
            Problem p; p.id = id++; p.cat = Cat::Arith;
            p.expr = std::to_string(a) + "/" + std::to_string(b) +
                     " + " + std::to_string(c) + "*" + std::to_string(d) +
                     " - " + std::to_string(e);
            p.question = "Compute exactly: " + p.expr +
                         ". Reply with only the numeric value.";
            p.expect_num = (double)a / b + (double)(c * d) - e;
            ps.push_back(p);
        }
        {   // derivative of random polynomial
            int deg = ri(2, 4);
            std::vector<double> c(deg + 1);
            for (int k = 0; k <= deg; ++k) c[k] = ri(-9, 9);
            std::string px;
            for (int k = deg; k >= 0; --k) {
                if (c[k] == 0) continue;
                if (!px.empty()) px += (c[k] > 0 ? " + " : " - ");
                else if (c[k] < 0) px += "-";
                double a = std::fabs(c[k]);
                if (k == 0 || a != 1) px += fmt_g(a);
                if (k > 0) { px += "x"; if (k > 1) px += "^" + std::to_string(k); }
            }
            if (px.empty()) px = "0";
            Problem p; p.id = id++; p.cat = Cat::Deriv; p.symbolic = true;
            p.expr = px;
            p.question = "Differentiate with respect to x: " + px +
                         ". Reply with only the derivative polynomial in x.";
            std::vector<double> dc(deg);
            for (int k = 1; k <= deg; ++k) dc[k - 1] = c[k] * k;
            p.expect_sym = poly_to_string(dc);
            ps.push_back(p);
        }
        {   // integral of polynomial (no +C needed)
            int deg = ri(1, 3);
            std::vector<double> c(deg + 1);
            for (int k = 0; k <= deg; ++k) c[k] = ri(1, 8);
            std::string px;
            for (int k = deg; k >= 0; --k) {
                if (!px.empty()) px += " + ";
                px += fmt_g(c[k]);
                if (k > 0) { px += "x"; if (k > 1) px += "^" + std::to_string(k); }
            }
            Problem p; p.id = id++; p.cat = Cat::Integral; p.symbolic = true;
            p.expr = px;
            p.question = "Integrate with respect to x: " + px +
                         ". Reply with only the antiderivative polynomial in x "
                         "(omit the constant of integration).";
            std::vector<double> ic(deg + 2, 0.0);
            for (int k = 0; k <= deg; ++k) ic[k + 1] = c[k] / (k + 1);
            p.expect_sym = poly_to_string(ic);
            ps.push_back(p);
        }
        {   // Newton F = m a
            double m = rd(1, 100), a = rd(1, 20);
            Problem p; p.id = id++; p.cat = Cat::Newton;
            p.question = "A mass of " + fmt_g(m) + " kg accelerates at " +
                         fmt_g(a) + " m/s^2. Compute the force in newtons. " +
                         "Reply with only the numeric value.";
            p.law = "newton2"; p.target = "F";
            p.knowns = {{"m", m}, {"a", a}};
            p.expect_num = m * a;
            ps.push_back(p);
        }
        {   // gravitation with explicit constant (fair to all contestants)
            double m1 = rd(10, 100), m2 = rd(10, 100), r = rd(1, 50);
            Problem p; p.id = id++; p.cat = Cat::Gravitation;
            p.question = "Using G = 6.674e-11, two masses m1=" + fmt_g(m1) +
                         " kg and m2=" + fmt_g(m2) + " kg sit " + fmt_g(r) +
                         " m apart. Compute the gravitational force in newtons. " +
                         "Reply with only the numeric value.";
            p.law = "gravitation"; p.target = "F";
            p.knowns = {{"m1", m1}, {"m2", m2}, {"r", r}};
            p.expect_num = 6.674e-11 * m1 * m2 / (r * r);
            ps.push_back(p);
        }
        {   // buffer pH = pKa + log10(ratio)
            double pKa = ri(6, 18) / 2.0;
            double ratio = (i % 3 == 0) ? 0.1 : (i % 3 == 1 ? 1.0 : 10.0);
            Problem p; p.id = id++; p.cat = Cat::Ph;
            p.question = "A buffer has pKa " + fmt_g(pKa) +
                         " and base-to-acid ratio " + fmt_g(ratio) +
                         ". Using pH = pKa + log10(base/acid), compute pH. " +
                         "Reply with only the numeric value.";
            p.law = "buffer_ph"; p.target = "pH";
            p.knowns = {{"pKa", pKa}, {"base", ratio}, {"acid", 1.0}};
            p.expect_num = pKa + std::log10(ratio);
            ps.push_back(p);
        }
    }
    return ps;
}


// ── tiny expression evaluator ────────────────────────────────────────────────
// numbers, x, + - * / ^ (right-assoc, binds tighter than unary minus),
// parentheses, implicit multiplication. Native ^ means NO textual x
// substitution anywhere — immune to the digit-gluing bug class.
namespace miniexpr {
struct P {
    const std::string& s; size_t i = 0; double x;
    P(const std::string& s_, double xv) : s(s_), x(xv) {}
    void ws() { while (i < s.size() && std::isspace((unsigned char)s[i])) ++i; }
    double expr() { double v = term(); ws();
        while (i < s.size() && (s[i]=='+'||s[i]=='-')) {
            char o=s[i++]; double r=term(); v = o=='+' ? v+r : v-r; ws(); }
        return v; }
    double term() { double v = unary(); ws();
        while (i < s.size()) { char c = s[i];
            if (c=='*' || c=='/') { ++i; double r=unary(); v = c=='*'?v*r:v/r; }
            else if (std::isdigit((unsigned char)c) || c=='x' || c=='(') {
                double r = unary(); v = v*r; }          // implicit multiply
            else break;
            ws(); }
        return v; }
    double unary() { ws();
        if (i<s.size() && (s[i]=='-'||s[i]=='+')) {
            char c=s[i++]; double v=unary(); return c=='-'?-v:v; }
        return powlevel(); }
    double powlevel() { double b = prim(); ws();
        if (i<s.size() && s[i]=='^') { ++i; double e = powlevel(); return std::pow(b,e); }
        return b; }
    double prim() { ws();
        if (i<s.size() && s[i]=='(') { ++i; double v=expr(); ws();
            if (i<s.size()&&s[i]==')') ++i; return v; }
        if (i<s.size() && s[i]=='x') { ++i; return x; }
        size_t b=i;
        while (i<s.size() && (std::isdigit((unsigned char)s[i])||s[i]=='.')) ++i;
        if (i>b) return std::strtod(s.c_str()+b,nullptr);
        throw std::runtime_error("bad token"); }
};
inline double eval_at(const std::string& e, double x) {
    std::string t; for (char ch : e)
        if (!std::isspace((unsigned char)ch)) t += std::tolower(ch);
    // strip latex noise from chatty answers
    std::string clean;
    for (char ch : t)
        if (ch != '\\' && ch != '$' && ch != '{' && ch != '}')
            clean += ch;
    // antiderivatives carry an arbitrary constant: strip a trailing +c/-c
    for (const char* suf : {"+c", "-c"}) {
        size_t pos = clean.rfind(suf);
        if (pos != std::string::npos && pos + 2 == clean.size())
            clean = clean.substr(0, pos);
    }
    P p(clean, x); return p.expr();
}
} // namespace miniexpr

// ── blind graders ────────────────────────────────────────────────────────────
static bool grab_number(const std::string& s, double& out) {
    // exact-fraction form "a/b" (optionally signed) evaluates precisely
    {
        std::string t;
        for (char ch : s) if (!std::isspace((unsigned char)ch)) t += ch;
        size_t slash = t.find('/');
        if (slash != std::string::npos && slash > 0 && slash + 1 < t.size()) {
            bool clean = true;
            for (size_t i = 0; i < t.size(); ++i)
                if (i != slash && !(std::isdigit((unsigned char)t[i]) ||
                    (i == 0 && (t[i]=='-'||t[i]=='+')))) { clean = false; break; }
            if (clean) {
                double num = std::strtod(t.substr(0, slash).c_str(), nullptr);
                double den = std::strtod(t.substr(slash + 1).c_str(), nullptr);
                if (den != 0) { out = num / den; return true; }
            }
        }
    }
    // forward scan; keep the LAST complete numeric token
    bool have = false;
    for (size_t i = 0; i < s.size(); ) {
        if (std::isdigit((unsigned char)s[i]) ||
            ((s[i]=='-'||s[i]=='+') && i+1<s.size() &&
             (std::isdigit((unsigned char)s[i+1]) || s[i+1]=='.')) ||
            (s[i]=='.' && i+1<s.size() && std::isdigit((unsigned char)s[i+1]))) {
            char* end = nullptr;
            double v = std::strtod(s.c_str()+i, &end);
            if (end != s.c_str()+i) { out = v; have = true; i = end - s.c_str(); continue; }
        }
        ++i;
    }
    return have;
}
static bool parse_poly(const std::string& in, std::vector<double>& c) {
    std::string s;
    for (char ch : in) if (!std::isspace((unsigned char)ch)) s += std::tolower(ch);
    // drop constant-of-integration suffixes like "+c" / "-c"
    for (const char* suf : {"+c", "-c"}) {
        size_t pos = s.rfind(suf);
        if (pos != std::string::npos && pos + 2 == s.size()) s = s.substr(0, pos);
    }
    if (s.empty()) return false;
    // tokenize signed terms
    std::vector<std::pair<double, std::string>> terms; // sign folded into coef magnitude path
    size_t i = 0; bool neg = false;
    while (i < s.size()) {
        if (s[i] == '+') { ++i; continue; }
        if (s[i] == '-') { neg = !neg; ++i; continue; }
        std::string term;
        while (i < s.size() && s[i] != '+' && s[i] != '-') { term += s[i]; ++i; }
        if (term.empty()) continue;
        // strip '*' and leading digits/coefficient
        double coef = 1.0; int power = 0;
        size_t xpos = term.find('x');
        std::string coefpart = xpos == std::string::npos ? term : term.substr(0, xpos);
        if (!coefpart.empty()) {
            if (coefpart.back() == '*') coefpart.pop_back();
            if (!coefpart.empty()) {
                char* end = nullptr;
                double v = std::strtod(coefpart.c_str(), &end);
                if (end != coefpart.c_str()) coef = v;
            }
        }
        if (xpos != std::string::npos) {
            power = 1;
            if (xpos + 1 < term.size() && term[xpos + 1] == '^')
                power = std::atoi(term.c_str() + xpos + 2);
        }
        if (neg) coef = -coef;
        neg = false;
        if ((int)c.size() <= power) c.resize(power + 1, 0.0);
        c[power] += coef;
    }
    return !c.empty();
}

static bool grade(const Problem& p, const std::string& ans) {
    if (p.symbolic) {
        // format-free blind grading: evaluate BOTH expressions at probe
        // points. Polynomials agreeing at 5 non-symmetric points (deg<=5)
        // are identical — handles unsimplified traces and LLM prose alike.
        static const double xs[5] = {0.7, 1.3, 2.9, -1.7, 4.1};
        try {
            for (double x : xs) {
                double w = miniexpr::eval_at(p.expect_sym, x);
                double g = miniexpr::eval_at(ans, x);
                if (!(std::fabs(g - w) <= 1e-4 * std::max(1.0, std::fabs(w))))
                    return false;
            }
            return true;
        } catch (...) { return false; }   // unparseable answer == wrong answer
    }
    double v;
    if (!grab_number(ans, v)) return false;
    return std::fabs(v - p.expect_num) <=
           1e-4 * std::max(1.0, std::fabs(p.expect_num));
}

// ── contestants ──────────────────────────────────────────────────────────────
struct Answer { std::string raw; double seconds = 0; bool error = false; };

struct IContestant {
    virtual ~IContestant() = default;
    virtual std::string name() const = 0;
    virtual bool available() const { return true; }
    virtual Answer respond(const Problem& p) = 0;
};

static MathEngine& brain_math() { static MathEngine me; return me; }

// math_parser grammar needs explicit multiplication: "3x^2" -> "3*x^2",
// and NO spaces anywhere in the expression.
static std::string starize(const std::string& e) {
    std::string out;
    for (size_t i = 0; i < e.size(); ++i) {
        char ch = e[i];
        if (std::isspace((unsigned char)ch)) continue;
        out += ch;
        char nx = (i + 1 < e.size()) ? e[i + 1] : '\0';
        bool a_num = std::isdigit((unsigned char)ch);
        if ((a_num && (nx == 'x' || nx == '(')) ||
            (ch == 'x' && (nx == '(' || std::isdigit((unsigned char)nx))))
            out += '*';
    }
    return out;
}

class BrainContestant : public IContestant {
public:
    std::string name() const override { return "brain3-native"; }
    Answer respond(const Problem& p) override {
        Answer a;
        auto t0 = std::chrono::steady_clock::now();
        try {
            MathSolveResult r;
            switch (p.cat) {
                case Cat::Arith:
                    // route pure arithmetic through the equation solver
                    r = brain_math().solve_equation("x = " + p.expr, "x");
                    break;
                case Cat::Deriv:
                    r = brain_math().solve_derivative(starize(p.expr), "x");
                    break;
                case Cat::Integral:
                    r = brain_math().solve_integral(starize(p.expr));
                    break;
                case Cat::Newton:
                case Cat::Gravitation:
                case Cat::Ph:
                    r = brain_math().solve_physics(p.law, p.target, p.knowns);
                    break;
            }
            if (r.success) {
                if (!r.symbolic_val.empty() &&
                    r.symbolic_val.find_first_not_of(" \t") != std::string::npos)
                    a.raw = r.symbolic_val;
                else
                    a.raw = fmt_g(r.numeric_val);
            }
        } catch (...) { a.error = true; }
        auto t1 = std::chrono::steady_clock::now();
        a.seconds = std::chrono::duration<double>(t1 - t0).count();
        return a;
    }
};

// curl that captures output (write body to temp file to avoid stdin races)
static std::string http_post_capture(const std::string& url,
                                     const std::string& body,
                                     const std::vector<std::string>& headers,
                                     int timeout_s) {
    const char* tmpin = "/tmp/opencode/exact_req.json";
    const char* tmpout = "/tmp/opencode/exact_resp.json";
    { std::ofstream f(tmpin); f << body; }
    std::string cmd = "curl -s --max-time " + std::to_string(timeout_s) +
                      " -X POST '" + url + "'";
    for (auto& h : headers) cmd += " -H \"" + h + "\"";
    cmd += std::string(" --data @") + tmpin + " -o " + tmpout;
    int rc = std::system(cmd.c_str());
    (void)rc;
    std::ifstream f(tmpout);
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

static std::string strip_think(const std::string& s) {
    size_t b = s.find("<think>");
    if (b == std::string::npos) return s;
    size_t e = s.find("</think>");
    if (e == std::string::npos) return s;
    return s.substr(0, b) + s.substr(e + 8);
}
static std::string json_str_field(const std::string& js, const std::string& field) {
    std::string key = "\"" + field + "\":";
    size_t k = js.find(key);
    if (k == std::string::npos) { key = "\"" + field + "\": "; k = js.find(key); }
    if (k == std::string::npos) return "";
    size_t q1 = js.find('"', k + key.size());
    if (q1 == std::string::npos) return "";
    std::string out;
    for (size_t i = q1 + 1; i < js.size(); ++i) {
        if (js[i] == '\\' && i + 1 < js.size()) { out += js[i]; out += js[i + 1]; ++i; continue; }
        if (js[i] == '"') break;
        out += js[i];
    }
    return out;
}

class OllamaContestant : public IContestant {
public:
    explicit OllamaContestant(std::string model) : model_(std::move(model)) {}
    std::string name() const override { return "ollama:" + model_; }
    bool available() const override {
        std::string cmd = "curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1";
        return std::system(cmd.c_str()) == 0;
    }
    bool broken_ = false;
    Answer respond(const Problem& p) override {
        Answer a;
        if (broken_) { a.error = true; return a; }
        std::string body = "{\"model\":\"" + model_ +
            "\",\"prompt\":" + json_quote(p.question) +
            ",\"stream\":false,\"think\":false," +
            "\"options\":{\"temperature\":0,\"num_predict\":700}}";
        auto t0 = std::chrono::steady_clock::now();
        std::string resp = http_post_capture(
            "http://localhost:11434/api/generate", body,
            {"Content-Type: application/json"}, 120);
        auto t1 = std::chrono::steady_clock::now();
        a.seconds = std::chrono::duration<double>(t1 - t0).count();
        std::string errtext = json_str_field(resp, "error");
        if (!errtext.empty()) {
            if (!broken_) { std::cout << "[skip] " << model_ << ": "
                                      << errtext << "\n"; broken_ = true; }
            a.error = true;
            return a;
        }
        std::string text = json_str_field(resp, "response");
        if (text.empty()) text = json_str_field(resp, "thinking");
        a.raw = strip_think(text);
        return a;
    }
private:
    static std::string json_quote(const std::string& s) {
        std::string o = "\"";
        for (char c : s) {
            if (c == '"' || c == '\\') o += '\\';
            if (c == '\n') { o += "\\n"; continue; }
            o += c;
        }
        return o + "\"";
    }
    std::string model_;
};

class FrontierContestant : public IContestant {
public:
    std::string name() const override { return "frontier:" + model_; }
    bool available() const override {
        const char* k = std::getenv("OPENAI_API_KEY");
        const char* m = std::getenv("EXACTNESS_FRONTIER_MODEL");
        return k && *k && m && *m;
    }
    explicit FrontierContestant(std::string model) : model_(std::move(model)) {}
    Answer respond(const Problem& p) override {
        Answer a;
        const char* key = std::getenv("OPENAI_API_KEY");
        std::string body = "{\"model\":\"" + model_ +
            "\",\"messages\":[{\"role\":\"user\",\"content\":" +
            json_quote(p.question) + "}],\"temperature\":0}";
        auto t0 = std::chrono::steady_clock::now();
        std::string resp = http_post_capture(
            "https://api.openai.com/v1/chat/completions", body,
            {"Content-Type: application/json",
             std::string("Authorization: Bearer ") + (key ? key : "")}, 90);
        auto t1 = std::chrono::steady_clock::now();
        a.seconds = std::chrono::duration<double>(t1 - t0).count();
        a.raw = json_str_field(resp, "content");
        return a;
    }
private:
    static std::string json_quote(const std::string& s) {
        std::string o = "\"";
        for (char c : s) {
            if (c == '"' || c == '\\') o += '\\';
            if (c == '\n') { o += "\\n"; continue; }
            o += c;
        }
        return o + "\"";
    }
    std::string model_;
};

int main(int argc, char** argv) {
    unsigned seed = 20260824;
    int per_cat = 30;
    if (argc > 1) per_cat = std::atoi(argv[1]);

    auto problems = generate_problems(seed, per_cat);

    std::vector<std::unique_ptr<IContestant>> contestants;
    contestants.push_back(std::make_unique<BrainContestant>());
    // LLM opponents are COMPARISON BASELINES ONLY — never part of the brain.
    // Pure-native scoreboard by default; set EXACTNESS_ENABLE_LLM=1 to add
    // the baseline column.
    if (std::getenv("EXACTNESS_ENABLE_LLM")) {
        const char* list = std::getenv("EXACTNESS_OLLAMA_MODELS");
        const char* one = std::getenv("EXACTNESS_OLLAMA_MODEL");
        std::vector<std::string> models;
        if (list && *list) {
            std::string s = list;
            for (size_t p = 0; p < s.size(); ++p)
                if (s[p] == ',') s[p] = ' ';
            std::istringstream iss(s);
            std::string m; while (iss >> m) models.push_back(m);
        } else if (one && *one) models.push_back(one);
        else models.push_back("qwen3:1.7B");
        bool any_up = false;
        for (auto& m : models) {
            auto oc = std::make_unique<OllamaContestant>(m);
            if (oc->available()) { contestants.push_back(std::move(oc)); any_up = true; }
            else std::cout << "[skip] " << m << " not reachable\n";
        }
        if (!any_up) std::cout << "[skip] no ollama models reachable\n";
    } else {
        std::cout << "[pure] LLM baselines disabled (EXACTNESS_ENABLE_LLM=1 to compare)\n";
    }
    {
        const char* m = std::getenv("EXACTNESS_FRONTIER_MODEL");
        if (m && *m && std::getenv("OPENAI_API_KEY")) {
            contestants.push_back(std::make_unique<FrontierContestant>(m));
            std::cout << "[info] frontier contestant enabled: " << m << "\n";
        } else {
            std::cout << "[skip] frontier (set OPENAI_API_KEY + EXACTNESS_FRONTIER_MODEL)\n";
        }
    }

    struct Row { std::string contestant, cat; int ok = 0, n = 0; double secs = 0; };
    std::vector<Row> rows;

    std::cout << "=== EXACTNESS SCOREBOARD === problems=" << problems.size() << "\n";
    for (auto& c : contestants) {
        std::cout << "-- contestant: " << c->name() << "\n";
        const bool dbg = std::getenv("EXACTNESS_DEBUG") != nullptr;
        std::map<std::string,int> shown;
        for (auto& p : problems) {
            Answer a = c->respond(p);
            bool correct = !a.error && grade(p, a.raw);
            if ((dbg && shown[cat_name(p.cat)]++ < 1) || (dbg && !correct)) {
                std::cout << "   DBG[" << c->name() << "/" << cat_name(p.cat)
                          << "] raw='" << a.raw.substr(0, 160)
                          << "' err=" << a.error
                          << " want_num=" << p.expect_num
                          << " want_sym='" << p.expect_sym << "' -> "
                          << (correct ? "OK" : "WRONG")
                          << " q='" << p.question.substr(0, 80) << "'\n";
            }
            bool found = false;
            for (auto& r : rows)
                if (r.contestant == c->name() && r.cat == cat_name(p.cat)) {
                    r.ok += correct; r.n++; r.secs += a.seconds; found = true; break;
                }
            if (!found) rows.push_back({c->name(), cat_name(p.cat),
                                        correct ? 1 : 0, 1, a.seconds});
        }
    }

    // scorecard
    std::cout << "\n" << std::left;
    std::printf("%-18s %-14s %8s %10s %12s\n", "contestant", "category",
                "exact", "avg_s", "exact%");
    double brain_all_ok = 0, brain_all_n = 0;
    for (auto& r : rows) {
        std::printf("%-18s %-14s %8s %10.4f %11.1f%%\n",
                    r.contestant.c_str(), r.cat.c_str(),
                    (std::to_string(r.ok) + "/" + std::to_string(r.n)).c_str(),
                    r.n ? r.secs / r.n : 0,
                    r.n ? 100.0 * r.ok / r.n : 0);
        if (r.contestant == "brain3-native") { brain_all_ok += r.ok; brain_all_n += r.n; }
    }
    double brain_exact = brain_all_n ? brain_all_ok / brain_all_n : 0;
    std::printf("\nBRAIN OVERALL EXACTNESS: %.1f%% (%.0f/%.0f)  gate>=%.0f%%\n",
                100 * brain_exact, brain_all_ok, brain_all_n, 100 * GATE_BRAIN_EXACT);

    std::ofstream rf("eval_scoreboard.json");
    rf << "{\"gate_pass\":" << (brain_exact >= GATE_BRAIN_EXACT)
       << ", \"problems\":" << problems.size() << "}\n";

    return brain_exact >= GATE_BRAIN_EXACT ? 0 : 1;
}
