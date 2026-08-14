#pragma once
/**
 * math_parser.hpp — recursive-descent math expression parser.
 * Port of brain2/engines/math/math_parser.py to C++.
 *
 * Grammar (precedence low -> high):
 *   equation := expr ('=' expr)?
 *   expr     := term (('+' | '-') term)*
 *   term     := power (('*' | '/') power)*
 *   power    := unary ('^' power)?        [right-associative]
 *   unary    := '-' unary | atom
 *   atom     := number | func '(' expr ')' | ident | '(' expr ')'
 *
 * Produces ExprNode trees (from calculus_engine.hpp) consumed by
 * calculus_engine, algebra_engine, integral_engine, and physics_engine.
 */

#include <string>
#include <vector>
#include <stdexcept>
#include <cctype>
#include <cstdlib>
#include <set>
#include "crisp/engines/math/calculus_engine.hpp"

namespace brain2 {
namespace math {

class ParseError : public std::runtime_error {
public:
    explicit ParseError(const std::string& msg) : std::runtime_error(msg) {}
};

// ── Tokenizer ────────────────────────────────────────────────────────────────

inline std::vector<std::string> tokenize(const std::string& s) {
    std::vector<std::string> tokens;
    size_t i = 0;
    while (i < s.size()) {
        if (std::isspace((unsigned char)s[i])) { ++i; continue; }

        // number: integer or float
        if (std::isdigit((unsigned char)s[i]) || (s[i] == '.' && i+1 < s.size() && std::isdigit((unsigned char)s[i+1]))) {
            size_t j = i;
            while (j < s.size() && (std::isdigit((unsigned char)s[j]) || s[j] == '.')) ++j;
            tokens.push_back(s.substr(i, j - i));
            i = j;
            continue;
        }

        // identifier or keyword
        if (std::isalpha((unsigned char)s[i]) || s[i] == '_') {
            size_t j = i;
            while (j < s.size() && (std::isalnum((unsigned char)s[j]) || s[j] == '_')) ++j;
            tokens.push_back(s.substr(i, j - i));
            i = j;
            continue;
        }

        // ** => ^
        if (s[i] == '*' && i+1 < s.size() && s[i+1] == '*') {
            tokens.push_back("^");
            i += 2;
            continue;
        }

        // single-char operators and parentheses
        if (std::string("+-*/^()=").find(s[i]) != std::string::npos) {
            tokens.push_back(std::string(1, s[i]));
            ++i;
            continue;
        }

        throw ParseError(std::string("unexpected character: ") + s[i]);
    }
    return tokens;
}

// ── Parser ────────────────────────────────────────────────────────────────────

class MathParser {
    static const std::set<std::string> FUNCS;
    std::vector<std::string> toks;
    size_t pos = 0;

    const std::string* peek() const {
        return pos < toks.size() ? &toks[pos] : nullptr;
    }

    std::string consume() {
        if (pos >= toks.size()) throw ParseError("unexpected end of expression");
        return toks[pos++];
    }

    std::string expect(const std::string& what) {
        auto t = consume();
        if (t != what) throw ParseError("expected '" + what + "', got '" + t + "'");
        return t;
    }

    bool at(const std::string& t) const {
        auto p = peek();
        return p && *p == t;
    }

public:
    explicit MathParser(const std::vector<std::string>& tokens) : toks(tokens) {}

    ExprPtr parse() {
        auto node = expr();
        if (at("=")) {
            consume();
            node = ExprNode::make_op("=", {node, expr()});
        }
        if (peek()) throw ParseError("trailing tokens: " + *peek());
        return node;
    }

    ExprPtr expr() {
        auto node = term();
        while (peek() && (*peek() == "+" || *peek() == "-")) {
            auto op = consume();
            node = ExprNode::make_op(op, {node, term()});
        }
        return node;
    }

    ExprPtr term() {
        auto node = power();
        while (peek() && (*peek() == "*" || *peek() == "/")) {
            auto op = consume();
            node = ExprNode::make_op(op, {node, power()});
        }
        return node;
    }

    ExprPtr power() {
        auto node = unary();
        if (at("^")) {
            consume();
            node = ExprNode::make_op("^", {node, power()}); // right-assoc
        }
        return node;
    }

    ExprPtr unary() {
        if (at("-")) {
            consume();
            return ExprNode::make_op("neg", {unary()});
        }
        return atom();
    }

    ExprPtr atom() {
        if (!peek()) throw ParseError("unexpected end of expression");
        std::string t = consume();

        // Parenthesized sub-expression
        if (t == "(") {
            auto node = expr();
            expect(")");
            return node;
        }

        // Number
        if (std::isdigit((unsigned char)t[0]) || t[0] == '.') {
            double v = std::stod(t);
            return ExprNode::make_num(v);
        }

        // Identifier: function or variable
        if (std::isalpha((unsigned char)t[0]) || t[0] == '_') {
            if (FUNCS.count(t)) {
                expect("(");
                auto arg = expr();
                expect(")");
                return ExprNode::make_op(t, {arg});
            }
            return ExprNode::make_var(t);
        }

        throw ParseError("unexpected token: " + t);
    }
};

const std::set<std::string> MathParser::FUNCS = {"sin", "cos", "exp", "ln"};

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Parse a math expression string into an ExprNode tree.
 * Throws ParseError on syntax errors.
 */
inline ExprPtr parse(const std::string& s) {
    auto tokens = tokenize(s);
    if (tokens.empty()) throw ParseError("empty expression");
    return MathParser(tokens).parse();
}

} // namespace math
} // namespace brain2
