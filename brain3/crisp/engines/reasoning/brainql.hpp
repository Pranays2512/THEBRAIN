#pragma once
#include <string>
#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <sstream>
#include <optional>
#include <stdexcept>
#include <algorithm>

#include "crisp/engines/reasoning/reasoning_engine.hpp"
#include "crisp/engines/math/math_engine.hpp"
#include "crisp/engines/code/code_engine.hpp"
#include "crisp/engines/reasoning/means_ends.hpp"
#include "crisp/engines/vision/vision_engine.hpp"
#include "crisp/engines/reasoning/causal_engine.hpp"
#include "crisp/engines/reasoning/analogy_engine.hpp"
#include "crisp/engines/reasoning/metacognitive_engine.hpp"
#include "crisp/engines/discovery/discovery_engine.hpp"
#include "crisp/engines/reasoning/curiosity_engine.hpp"
#include "crisp/engines/reasoning/instinct_engine.hpp"
#include "crisp/core/refuter.hpp"

namespace brain2 {
namespace reasoning {

class BrainQLParseError : public std::runtime_error {
public:
    BrainQLParseError(const std::string& msg) : std::runtime_error(msg) {}
};

struct BrainQLQuery {
    std::string op;
    std::string subj;
    std::string rel;
    std::string obj;
    std::string prem1;
    std::string prem2;
    std::string concl;
    int hops = 8;
    std::string raw;
};

struct BrainQLResult {
    std::string op;
    std::string subj;
    std::string rel;
    std::string obj;
    std::string value;
    std::vector<std::string> chain;
    bool verified = false;
    bool known = false;
    std::string note;
};

inline BrainQLQuery parse_bql(std::string text) {
    size_t pos = text.find('#');
    if (pos != std::string::npos) text = text.substr(0, pos);
    
    // trim
    text.erase(text.begin(), std::find_if(text.begin(), text.end(), [](unsigned char ch) { return !std::isspace(ch); }));
    text.erase(std::find_if(text.rbegin(), text.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), text.end());
    if (text.empty()) throw BrainQLParseError("empty instruction");

    std::vector<std::string> parts;
    std::stringstream ss(text);
    std::string part;
    while (ss >> part) parts.push_back(part);

    std::string op = parts[0];
    for (auto& c : op) c = std::toupper(c);

    BrainQLQuery q;
    q.op = op;
    q.raw = text;

    if (op == "CAUSAL_DEFINE") {
        size_t eq_pos = text.find('=');
        if (eq_pos == std::string::npos) throw BrainQLParseError("CAUSAL_DEFINE requires '=' (e.g. CAUSAL_DEFINE accel = force / mass)");
        std::string left = text.substr(13, eq_pos - 13);
        std::string right = text.substr(eq_pos + 1);
        left.erase(left.begin(), std::find_if(left.begin(), left.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        left.erase(std::find_if(left.rbegin(), left.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), left.end());
        right.erase(right.begin(), std::find_if(right.begin(), right.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        right.erase(std::find_if(right.rbegin(), right.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), right.end());
        q.subj = left;
        q.obj = right;
        q.rel = "define";
        return q;
    }

    if (op == "CAUSAL_OBSERVE") {
        if (parts.size() < 2) throw BrainQLParseError("CAUSAL_OBSERVE requires variables and values");
        q.subj = "causal";
        q.rel = "multi";
        q.obj = text.substr(op.length());
        q.obj.erase(q.obj.begin(), std::find_if(q.obj.begin(), q.obj.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        q.obj.erase(std::find_if(q.obj.rbegin(), q.obj.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), q.obj.end());
        return q;
    }

    if (op == "INTERVENE") {
        size_t q_pos = text.find("QUERY");
        if (q_pos == std::string::npos) q_pos = text.find("query");
        
        std::string do_clause, target;
        if (q_pos != std::string::npos) {
            do_clause = text.substr(op.length(), q_pos - op.length());
            target = text.substr(q_pos + 5);
        } else if (parts.size() >= 3) {
            // Support: INTERVENE <var>=<val> <target>
            do_clause = parts[1];
            target = parts[2];
        } else {
            throw BrainQLParseError("INTERVENE requires target (e.g. INTERVENE var=val QUERY target)");
        }
        
        do_clause.erase(do_clause.begin(), std::find_if(do_clause.begin(), do_clause.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        do_clause.erase(std::find_if(do_clause.rbegin(), do_clause.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), do_clause.end());
        target.erase(target.begin(), std::find_if(target.begin(), target.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        target.erase(std::find_if(target.rbegin(), target.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), target.end());
        
        if (do_clause.find("do(") == 0 && do_clause.back() == ')') {
            do_clause = do_clause.substr(3, do_clause.length() - 4);
        }
        size_t eq = do_clause.find('=');
        if (eq == std::string::npos) throw BrainQLParseError("INTERVENE requires <var> = <val>");
        std::string var = do_clause.substr(0, eq);
        std::string val = do_clause.substr(eq + 1);
        var.erase(var.begin(), std::find_if(var.begin(), var.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        var.erase(std::find_if(var.rbegin(), var.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), var.end());
        val.erase(val.begin(), std::find_if(val.begin(), val.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        val.erase(std::find_if(val.rbegin(), val.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), val.end());
        
        q.subj = var;
        q.rel = val;
        q.obj = target;
        return q;
    }

    if (op == "COUNTERFACTUAL" || op == "WHAT_IF") {
        size_t then_pos = text.find("THEN");
        if (then_pos == std::string::npos) then_pos = text.find("then");
        
        std::string hyp_clause, target;
        if (then_pos != std::string::npos) {
            hyp_clause = text.substr(op.length(), then_pos - op.length());
            target = text.substr(then_pos + 4);
        } else if (parts.size() >= 3) {
            // Support: COUNTERFACTUAL <var>=<val> <target>
            hyp_clause = parts[1];
            target = parts[2];
        } else {
            throw BrainQLParseError("COUNTERFACTUAL requires target (e.g. COUNTERFACTUAL var=val THEN target)");
        }
        
        hyp_clause.erase(hyp_clause.begin(), std::find_if(hyp_clause.begin(), hyp_clause.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        hyp_clause.erase(std::find_if(hyp_clause.rbegin(), hyp_clause.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), hyp_clause.end());
        target.erase(target.begin(), std::find_if(target.begin(), target.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        target.erase(std::find_if(target.rbegin(), target.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), target.end());

        if (hyp_clause.find("IF ") == 0 || hyp_clause.find("if ") == 0) {
            hyp_clause = hyp_clause.substr(3);
            hyp_clause.erase(hyp_clause.begin(), std::find_if(hyp_clause.begin(), hyp_clause.end(), [](unsigned char ch) { return !std::isspace(ch); }));
            hyp_clause.erase(std::find_if(hyp_clause.rbegin(), hyp_clause.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), hyp_clause.end());
        }
        size_t eq = hyp_clause.find('=');
        if (eq == std::string::npos) throw BrainQLParseError("COUNTERFACTUAL requires <var> = <val>");
        std::string var = hyp_clause.substr(0, eq);
        std::string val = hyp_clause.substr(eq + 1);
        var.erase(var.begin(), std::find_if(var.begin(), var.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        var.erase(std::find_if(var.rbegin(), var.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), var.end());
        val.erase(val.begin(), std::find_if(val.begin(), val.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        val.erase(std::find_if(val.rbegin(), val.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), val.end());

        q.subj = var;
        q.rel = val;
        q.obj = target;
        return q;
    }

    if (op == "ANALOGY_DEFINE") {
        if (parts.size() < 5) throw BrainQLParseError("ANALOGY_DEFINE requires <domain> <subj> <rel> <obj>");
        q.subj = parts[1]; // domain
        q.prem1 = parts[2]; // subj
        q.rel = parts[3];   // rel
        q.obj = parts[4];   // obj
        return q;
    }

    if (op == "ANALOGY") {
        std::string mode = "map";
        std::string proj_entity = "";
        std::string src_domain = "";
        std::string tgt_domain = "";

        // Check for PROJECT keyword
        size_t proj_pos = text.find("PROJECT ");
        if (proj_pos == std::string::npos) proj_pos = text.find("project ");
        if (proj_pos != std::string::npos) {
            proj_entity = text.substr(proj_pos + 8);
            proj_entity.erase(proj_entity.begin(), std::find_if(proj_entity.begin(), proj_entity.end(), [](unsigned char ch) { return !std::isspace(ch); }));
            proj_entity.erase(std::find_if(proj_entity.rbegin(), proj_entity.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), proj_entity.end());
        }

        // Find source and target domains
        auto to_it = std::find(parts.begin() + 1, parts.end(), "TO");
        if (to_it == parts.end()) to_it = std::find(parts.begin() + 1, parts.end(), "to");

        if (to_it != parts.end() && to_it + 1 != parts.end()) {
            src_domain = *(to_it - 1);
            tgt_domain = *(to_it + 1);
        } else if (parts.size() >= 3) {
            if (parts[1] == "PROJECT" || parts[1] == "project") {
                src_domain = parts[2];
                tgt_domain = (parts.size() > 3) ? parts[3] : "";
                if (parts.size() > 4) proj_entity = parts[4];
            } else {
                src_domain = parts[1];
                tgt_domain = parts[2];
            }
        } else {
            throw BrainQLParseError("ANALOGY requires: ANALOGY <source_domain> [TO] <target_domain>");
        }

        q.subj = src_domain;
        q.obj = tgt_domain;
        q.rel = mode;
        q.prem1 = proj_entity;
        return q;
    }

    if (op == "PERCEIVE_IMAGE" || op == "VISION") {
        if (parts.size() < 2) throw BrainQLParseError(op + " requires image path (e.g. PERCEIVE_IMAGE test.ppm)");
        q.obj = parts[1];
        q.subj = "vision";
        q.rel = "perceive";
        return q;
    }

    if (op == "SYNTH") {
        if (parts.size() < 2) throw BrainQLParseError("SYNTH requires input-output spec (e.g. SYNTH [1, 2] -> [2, 4])");
        size_t synth_pos = text.find("SYNTH ");
        if (synth_pos == std::string::npos) synth_pos = text.find("synth ");
        if (synth_pos != std::string::npos) {
            q.obj = text.substr(synth_pos + 6);
        } else {
            q.obj = text.substr(parts[0].length());
        }
        // trim q.obj
        q.obj.erase(q.obj.begin(), std::find_if(q.obj.begin(), q.obj.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        q.obj.erase(std::find_if(q.obj.rbegin(), q.obj.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), q.obj.end());
        q.subj = "code";
        q.rel = "synth";
        return q;
    }

    if (op == "SOLVE") {
        if (parts.size() < 2) throw BrainQLParseError("SOLVE requires an expression or problem");
        size_t solve_pos = text.find("SOLVE ");
        if (solve_pos == std::string::npos) solve_pos = text.find("solve ");
        if (solve_pos != std::string::npos) {
            q.obj = text.substr(solve_pos + 6);
        } else {
            q.obj = text.substr(parts[0].length());
        }
        // trim q.obj
        q.obj.erase(q.obj.begin(), std::find_if(q.obj.begin(), q.obj.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        q.obj.erase(std::find_if(q.obj.rbegin(), q.obj.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), q.obj.end());
        q.subj = "math";
        q.rel = "solve";
        return q;
    }

    if (op == "TEACH_RULE") {
        auto it = std::find(parts.begin(), parts.end(), "->");
        if (it == parts.end()) throw BrainQLParseError("TEACH_RULE requires '->'");
        int arrow_idx = std::distance(parts.begin(), it);
        if (arrow_idx < 3) throw BrainQLParseError("TEACH_RULE requires two premises before '->'");
        q.prem1 = parts[1];
        q.prem2 = parts[2];
        if (arrow_idx + 1 < parts.size()) q.concl = parts[arrow_idx + 1];
        if (q.concl.empty()) throw BrainQLParseError("TEACH_RULE: missing conclusion after '->'");
        return q;
    }

    if (op == "TEACH") {
        if (parts.size() < 4) throw BrainQLParseError("TEACH requires: TEACH <subj> <rel> <obj>");
        q.subj = parts[1];
        q.rel = parts[2];
        q.obj = parts[3];
        return q;
    }

    if (op == "DISCOVER" || op == "INFER_EQUATION") {
        if (parts.size() < 2) throw BrainQLParseError(op + " requires domain name or specification (e.g. DISCOVER LAW kepler or DISCOVER kepler)");
        if (parts.size() >= 3 && (parts[1] == "LAW" || parts[1] == "law")) {
            q.subj = parts[2];
            q.rel = "law";
            q.obj = (parts.size() > 3) ? parts[3] : "";
            return q;
        }
        size_t data_pos = text.find("DATA ");
        if (data_pos == std::string::npos) data_pos = text.find("data ");
        if (data_pos != std::string::npos) {
            q.subj = parts[1];
            q.rel = "data";
            q.obj = text.substr(data_pos + 5);
            return q;
        }
        q.subj = parts[1];
        q.rel = "domain";
        q.obj = (parts.size() > 2) ? parts[2] : "";
        return q;
    }

    if (op == "CURIOSITY_GAPS" || op == "CURIOSITY_TICK" || op == "AUTONOMOUS_CYCLE" || op == "CURIOSITY_OBSERVE") {
        q.subj = parts.size() > 1 ? parts[1] : "";
        q.rel = parts.size() > 2 ? parts[2] : "";
        q.obj = text.length() > op.length() ? text.substr(op.length() + 1) : "";
        return q;
    }

    if (op == "INSTINCT" || op == "INSTINCT_FIRE") {
        q.subj = "instinct";
        q.obj = text.length() > op.length() ? text.substr(op.length() + 1) : (parts.size() > 1 ? parts[1] : "");
        // trim obj
        q.obj.erase(q.obj.begin(), std::find_if(q.obj.begin(), q.obj.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        q.obj.erase(std::find_if(q.obj.rbegin(), q.obj.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), q.obj.end());
        return q;
    }

    if (op == "INSTINCT_TRAIN") {
        size_t arrow = text.find("->");
        if (arrow != std::string::npos) {
            std::string lhs = text.substr(op.length(), arrow - op.length());
            std::string rhs = text.substr(arrow + 2);
            lhs.erase(lhs.begin(), std::find_if(lhs.begin(), lhs.end(), [](unsigned char ch) { return !std::isspace(ch); }));
            lhs.erase(std::find_if(lhs.rbegin(), lhs.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), lhs.end());
            rhs.erase(rhs.begin(), std::find_if(rhs.begin(), rhs.end(), [](unsigned char ch) { return !std::isspace(ch); }));
            rhs.erase(std::find_if(rhs.rbegin(), rhs.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), rhs.end());
            q.subj = lhs;
            q.obj = rhs;
            q.rel = "train";
            return q;
        }
        if (parts.size() >= 3) {
            q.subj = parts[1];
            q.obj = parts[2];
            q.rel = "train";
            return q;
        }
        throw BrainQLParseError("INSTINCT_TRAIN requires: INSTINCT_TRAIN <signature> -> <action>");
    }

    if (op == "INSTINCT_STATUS") {
        q.subj = "status";
        return q;
    }

    if (op == "INSTINCT_PENALIZE") {
        q.subj = parts.size() > 1 ? parts[1] : "";
        return q;
    }

    if (op == "REFUTE" || op == "META_VERIFY" || op == "CRITIQUE") {
        if (parts.size() < 3) throw BrainQLParseError(op + " requires at least: " + op + " <subj> <rel> [obj]");
        q.subj = parts[1];
        q.rel = parts[2];
        if (q.rel == "=" && parts.size() > 3) {
            q.rel = "val";
            q.obj = parts[3];
        } else if (parts.size() > 3) {
            q.obj = parts[3];
            for (size_t i = 4; i < parts.size(); ++i) q.obj += " " + parts[i];
        } else {
            q.obj = "";
        }
        return q;
    }

    if (parts.size() < 3) throw BrainQLParseError(op + " requires at least: " + op + " <subj> <rel>");
    q.subj = parts[1];
    q.rel = parts[2];
    q.obj = parts.size() > 3 ? parts[3] : "";

    for (size_t i = 3; i < parts.size(); ++i) {
        if (parts[i].find("hops=") == 0) {
            q.hops = std::stoi(parts[i].substr(5));
        }
    }
    return q;
}

class BrainQLExecutor {
private:
    ReasoningEngine* re;
    MeansEndsSolver* mes;
    math::MathEngine* me;
    CodeEngine* ce;
    PolicyMemory* pm;
    brain3::engines::vision::VisionEngine* ve;
    CausalEngine* ca;
    AnalogyEngine* ae;
    MetacognitiveEngine* mce;
    discovery::DiscoveryEngine* de;
    CuriosityEngine* cue;
    InstinctEngine* ie;

public:
    BrainQLExecutor(ReasoningEngine* re, MeansEndsSolver* mes = nullptr, math::MathEngine* me = nullptr, CodeEngine* ce = nullptr, PolicyMemory* pm = nullptr, brain3::engines::vision::VisionEngine* ve = nullptr, CausalEngine* ca = nullptr, AnalogyEngine* ae = nullptr, MetacognitiveEngine* mce = nullptr, discovery::DiscoveryEngine* de = nullptr, CuriosityEngine* cue = nullptr, InstinctEngine* ie = nullptr) 
        : re(re), mes(mes), me(me), ce(ce), pm(pm), ve(ve), ca(ca), ae(ae), mce(mce), de(de), cue(cue), ie(ie) {}

    BrainQLResult run(const BrainQLQuery& q) {
        if (q.op == "LOOKUP") return _lookup(q);
        if (q.op == "CHAIN") return _chain(q);
        if (q.op == "INHERIT") return _inherit(q);
        if (q.op == "DERIVE") return _derive(q);
        if (q.op == "TEACH") return _teach(q);
        if (q.op == "TEACH_RULE") return _teach_rule(q);
        if (q.op == "COMPUTE") return _compute(q);
        if (q.op == "EXPLAIN") return _explain(q);
        if (q.op == "SOLVE") return _solve(q);
        if (q.op == "SYNTH") return _synth(q);
        if (q.op == "PERCEIVE_IMAGE" || q.op == "VISION") return _perceive_image(q);
        if (q.op == "CAUSAL_DEFINE") return _causal_define(q);
        if (q.op == "CAUSAL_OBSERVE") return _causal_observe(q);
        if (q.op == "INTERVENE") return _intervene(q);
        if (q.op == "COUNTERFACTUAL" || q.op == "WHAT_IF") return _counterfactual(q);
        if (q.op == "ANALOGY") return _analogy(q);
        if (q.op == "ANALOGY_DEFINE") return _analogy_define(q);
        if (q.op == "REFUTE") return _refute(q);
        if (q.op == "META_VERIFY") return _meta_verify(q);
        if (q.op == "CRITIQUE") return _critique(q);
        if (q.op == "DISCOVER" || q.op == "INFER_EQUATION") return _discover(q);
        if (q.op == "CURIOSITY_GAPS") return _curiosity_gaps(q);
        if (q.op == "CURIOSITY_TICK") return _curiosity_tick(q);
        if (q.op == "AUTONOMOUS_CYCLE") return _autonomous_cycle(q);
        if (q.op == "CURIOSITY_OBSERVE") return _curiosity_observe(q);
        if (q.op == "INSTINCT" || q.op == "INSTINCT_FIRE") return _instinct(q);
        if (q.op == "INSTINCT_TRAIN") return _instinct_train(q);
        if (q.op == "INSTINCT_STATUS") return _instinct_status(q);
        if (q.op == "INSTINCT_PENALIZE") return _instinct_penalize(q);
        return {q.op, "", "", "", "", {}, false, false, "unknown op: " + q.op};
    }

private:
    BrainQLResult _instinct(const BrainQLQuery& q) {
        InstinctEngine default_ie;
        InstinctEngine* active_ie = ie ? ie : &default_ie;

        std::string query = q.obj.empty() ? q.subj : q.obj;
        auto res = active_ie->evaluate_instinct(query);

        return {"INSTINCT", query, res.domain, res.action, res.action, res.steps, res.has_reflex, res.has_reflex, res.explanation};
    }

    BrainQLResult _instinct_train(const BrainQLQuery& q) {
        InstinctEngine default_ie;
        InstinctEngine* active_ie = ie ? ie : &default_ie;

        active_ie->crystallize_reflex(q.subj, "learned_instinct", q.obj, 0.90);
        std::vector<std::string> steps = {
            "⚡ Crystallized System 2 reasoning trace into System 1 reflex arc",
            "Signature: '" + q.subj + "' -> Action: '" + q.obj + "'",
            "Initial Reflex Confidence: 0.90"
        };
        return {"INSTINCT_TRAIN", q.subj, "train", q.obj, q.obj, steps, true, true, "Reflex arc crystallized into instinct memory"};
    }

    BrainQLResult _instinct_status(const BrainQLQuery& q) {
        InstinctEngine default_ie;
        InstinctEngine* active_ie = ie ? ie : &default_ie;

        std::string json_status = active_ie->get_status_json();
        std::vector<std::string> steps = {
            "Total Reflex Arcs: " + std::to_string(active_ie->reflex_arcs.size()),
            "Total Fires: " + std::to_string(active_ie->total_reflex_fires),
            "Total Hits: " + std::to_string(active_ie->total_reflex_hits),
            "Reflex Threshold: " + std::to_string(active_ie->reflex_threshold)
        };
        return {"INSTINCT_STATUS", "status", "telemetry", json_status, json_status, steps, true, true, "Instinct engine status telemetry"};
    }

    BrainQLResult _instinct_penalize(const BrainQLQuery& q) {
        InstinctEngine default_ie;
        InstinctEngine* active_ie = ie ? ie : &default_ie;

        active_ie->penalize_reflex(q.subj, 0.35);
        std::vector<std::string> steps = {
            "⚡ Applied anti-Hebbian penalty to reflex signature: '" + q.subj + "'",
            "Suppression penalty: -0.35 confidence"
        };
        return {"INSTINCT_PENALIZE", q.subj, "penalize", "penalized", "penalized", steps, true, true, "Reflex arc penalized due to refutation"};
    }

    BrainQLResult _curiosity_gaps(const BrainQLQuery& q) {
        CuriosityEngine default_cue;
        CuriosityEngine* active_cue = cue ? cue : &default_cue;

        int limit = 5;
        if (!q.subj.empty()) {
            try { limit = std::stoi(q.subj); } catch (...) {}
        }

        auto gaps = active_cue->curiosity_gaps(limit);
        std::vector<std::string> steps;
        std::ostringstream val_ss;
        val_ss << "Curiosity Gaps: " << gaps.size() << " active (prediction error: " << std::fixed << std::setprecision(2) << active_cue->compute_prediction_error() << ")";

        for (const auto& g : gaps) {
            std::string item = "• [" + g.target_entity + "] uncertainty=" + std::to_string(g.uncertainty_score) + (g.is_unlearnable ? " [UNLEARNABLE/STOCHASTIC]" : " [LEARNABLE]");
            steps.push_back(item);
        }

        return {"CURIOSITY_GAPS", q.subj, q.rel, val_ss.str(), val_ss.str(), steps, true, true, "Scanned epistemic knowledge gaps based on prediction error"};
    }

    BrainQLResult _curiosity_tick(const BrainQLQuery& q) {
        CuriosityEngine default_cue;
        CuriosityEngine* active_cue = cue ? cue : &default_cue;

        auto res = active_cue->tick(re, mce, de);
        return {"CURIOSITY_TICK", q.subj, q.rel, res.conjecture_name, res.explanation, res.trace, res.verified, true, res.explanation};
    }

    BrainQLResult _autonomous_cycle(const BrainQLQuery& q) {
        CuriosityEngine default_cue;
        CuriosityEngine* active_cue = cue ? cue : &default_cue;

        int ticks = 3;
        if (!q.subj.empty()) {
            try { ticks = std::stoi(q.subj); } catch (...) {}
        }

        auto report = active_cue->run_autonomous_cycle(ticks, re, mce, de);
        std::ostringstream val_ss;
        val_ss << "Autonomous Cycle Complete: " << report.gaps_resolved << " gaps resolved across " << report.total_ticks << " ticks (error: " << std::fixed << std::setprecision(2) << report.initial_error << " -> " << report.final_error << ")";

        std::vector<std::string> steps = report.discovery_log;
        for (const auto& kv : report.banked_laws) {
            steps.push_back("Banked Law: " + kv.first + " = " + kv.second);
        }

        return {"AUTONOMOUS_CYCLE", q.subj, q.rel, val_ss.str(), val_ss.str(), steps, true, true, "Completed autonomous curiosity and self-directed discovery cycle"};
    }

    BrainQLResult _curiosity_observe(const BrainQLQuery& q) {
        CuriosityEngine default_cue;
        CuriosityEngine* active_cue = cue ? cue : &default_cue;

        // Parse sequences e.g. "rain,wet_ground,puddles ; study,pass ; dice,one"
        std::string raw = q.obj.empty() ? q.subj : q.obj;
        std::stringstream ss(raw);
        std::string ep_str;
        int count = 0;

        while (std::getline(ss, ep_str, ';')) {
            ep_str.erase(ep_str.begin(), std::find_if(ep_str.begin(), ep_str.end(), [](unsigned char ch) { return !std::isspace(ch); }));
            ep_str.erase(std::find_if(ep_str.rbegin(), ep_str.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), ep_str.end());
            if (ep_str.empty()) continue;

            std::stringstream token_ss(ep_str);
            std::string tok;
            std::vector<std::string> tokens;
            while (std::getline(token_ss, tok, ',')) {
                tok.erase(tok.begin(), std::find_if(tok.begin(), tok.end(), [](unsigned char ch) { return !std::isspace(ch); }));
                tok.erase(std::find_if(tok.rbegin(), tok.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), tok.end());
                if (!tok.empty()) tokens.push_back(tok);
            }
            if (!tokens.empty()) {
                active_cue->observe(tokens);
                count++;
            }
        }

        double err = active_cue->compute_prediction_error();
        std::string val = "Ingested " + std::to_string(count) + " transition episode(s); prediction error: " + std::to_string(err);
        return {"CURIOSITY_OBSERVE", q.subj, q.rel, val, val, {"[Curiosity Observer]: Stored " + std::to_string(count) + " transition sequences"}, true, true, val};
    }

    BrainQLResult _discover(const BrainQLQuery& q) {
        discovery::DiscoveryEngine default_de;
        discovery::DiscoveryEngine* active_de = de ? de : &default_de;

        discovery::DiscoveredLaw res;
        if (q.rel == "data") {
            std::vector<discovery::ObservationPoint> pts;
            std::stringstream ss(q.obj);
            std::string item;
            while (std::getline(ss, item, ';')) {
                item.erase(item.begin(), std::find_if(item.begin(), item.end(), [](unsigned char ch) { return !std::isspace(ch); }));
                item.erase(std::find_if(item.rbegin(), item.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), item.end());
                if (item.empty()) continue;
                size_t colon = item.find(':');
                if (colon != std::string::npos) {
                    std::string in_str = item.substr(0, colon);
                    std::string out_str = item.substr(colon + 1);
                    double y = std::stod(out_str);
                    std::map<std::string, double> in_map;
                    std::stringstream in_ss(in_str);
                    std::string in_val;
                    int v_idx = 1;
                    while (std::getline(in_ss, in_val, ',')) {
                        in_map["x" + std::to_string(v_idx++)] = std::stod(in_val);
                    }
                    pts.push_back({in_map, y});
                }
            }
            std::vector<std::string> in_vars;
            if (!pts.empty()) {
                for (const auto& kv : pts.front().inputs) in_vars.push_back(kv.first);
            }
            res = active_de->discover_from_data(q.subj.empty() ? "y" : q.subj, in_vars, pts);
        } else {
            res = active_de->discover_domain(q.subj);
        }

        if (!res.verified) {
            return {"DISCOVER", q.subj, q.rel, "", "", res.discovery_steps, false, false, res.explanation};
        }

        return {"DISCOVER", q.subj, q.rel, res.equation, res.equation, res.discovery_steps, true, true, res.explanation};
    }

    BrainQLResult _refute(const BrainQLQuery& q) {
        MetacognitiveEngine default_mce;
        MetacognitiveEngine* active_mce = mce ? mce : &default_mce;
        auto res = active_mce->refute(q.subj, q.rel, q.obj, re, ca);
        std::string val = res.verdict_str + ": " + (res.is_refuted ? res.falsification_reason : "sound");
        return {"REFUTE", q.subj, q.rel, q.obj, val, res.proof_trace, !res.is_refuted, true, res.falsification_reason};
    }

    BrainQLResult _meta_verify(const BrainQLQuery& q) {
        MetacognitiveEngine default_mce;
        MetacognitiveEngine* active_mce = mce ? mce : &default_mce;
        auto res = active_mce->refute(q.subj, q.rel, q.obj, re, ca);
        std::string val = res.verdict_str + ": " + (res.is_refuted ? res.corrected_truth : "sound");
        return {"META_VERIFY", q.subj, q.rel, q.obj, val, res.proof_trace, !res.is_refuted, true, res.falsification_reason};
    }

    BrainQLResult _critique(const BrainQLQuery& q) {
        MetacognitiveEngine default_mce;
        MetacognitiveEngine* active_mce = mce ? mce : &default_mce;
        auto res = active_mce->refute(q.subj, q.rel, q.obj, re, ca);
        std::string val = res.system1_intuition + " | " + res.verdict_str + ": " + (res.is_refuted ? res.falsification_reason : "validated");
        return {"CRITIQUE", q.subj, q.rel, q.obj, val, res.proof_trace, !res.is_refuted, true, res.falsification_reason};
    }

private:
    BrainQLResult _analogy_define(const BrainQLQuery& q) {
        AnalogyEngine default_ae;
        AnalogyEngine* active_ae = ae ? ae : &default_ae;
        active_ae->define_triple(q.subj, q.prem1, q.rel, q.obj);
        return {"ANALOGY_DEFINE", q.subj, q.rel, q.prem1 + " " + q.rel + " " + q.obj, "defined", {"defined domain triple: " + q.subj + " -> " + q.prem1 + " " + q.rel + " " + q.obj}, true, true, ""};
    }

    BrainQLResult _analogy(const BrainQLQuery& q) {
        AnalogyEngine default_ae;
        AnalogyEngine* active_ae = ae ? ae : &default_ae;
        auto res = active_ae->map_analogy(q.subj, q.obj);
        if (!res.success) {
            return {"ANALOGY", q.subj, q.obj, "", "", {}, false, false, res.explanation};
        }
        
        std::vector<std::string> steps;
        for (const auto& m : res.matched_triples) {
            steps.push_back("matched: (" + m.first + ") <-> (" + m.second + ")");
        }
        for (const auto& inf : res.candidate_inferences) {
            steps.push_back("inferred: " + inf.to_string() + " [from: " + inf.source_origin + "]");
        }

        // If specific entity projection was requested
        if (!q.prem1.empty()) {
            auto it = res.entity_map.find(q.prem1);
            if (it != res.entity_map.end()) {
                return {"ANALOGY", q.subj, q.obj, it->second, it->second, steps, true, true, "Projected: " + q.prem1 + " -> " + it->second};
            }
            for (const auto& kv : res.entity_map) {
                if (kv.second == q.prem1) {
                    return {"ANALOGY", q.subj, q.obj, kv.first, kv.first, steps, true, true, "Projected: " + q.prem1 + " -> " + kv.first};
                }
            }
            for (const auto& inf : res.candidate_inferences) {
                if (inf.target_subj == q.prem1 || inf.target_obj == q.prem1 || inf.source_origin == q.prem1) {
                    return {"ANALOGY", q.subj, q.obj, inf.to_string(), inf.to_string(), steps, true, true, "Projected: " + inf.to_string()};
                }
            }
        }

        std::string inf_summary = "";
        for (size_t i = 0; i < res.candidate_inferences.size(); ++i) {
            inf_summary += res.candidate_inferences[i].to_string();
            if (i + 1 < res.candidate_inferences.size()) inf_summary += "; ";
        }
        if (inf_summary.empty()) inf_summary = "score=" + std::to_string(res.score);

        return {"ANALOGY", q.subj, q.obj, inf_summary, inf_summary, steps, true, true, res.explanation};
    }

private:
    BrainQLResult _causal_define(const BrainQLQuery& q) {
        CausalEngine default_ca;
        CausalEngine* active_ca = ca ? ca : &default_ca;
        active_ca->define_equation(q.subj, q.obj);
        return {"CAUSAL_DEFINE", q.subj, "equation", q.obj, q.subj + " = " + q.obj, {"defined structural equation: " + q.subj + " = " + q.obj}, true, true, ""};
    }

    BrainQLResult _causal_observe(const BrainQLQuery& q) {
        CausalEngine default_ca;
        CausalEngine* active_ca = ca ? ca : &default_ca;
        
        std::string raw = q.obj.empty() ? q.subj : q.obj;
        std::vector<std::string> steps;
        std::stringstream ss(raw);
        std::string token;
        int count = 0;

        while (std::getline(ss, token, ',')) {
            token.erase(token.begin(), std::find_if(token.begin(), token.end(), [](unsigned char ch) { return !std::isspace(ch); }));
            token.erase(std::find_if(token.rbegin(), token.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), token.end());
            if (token.empty()) continue;

            size_t eq = token.find('=');
            if (eq != std::string::npos) {
                std::string k = token.substr(0, eq);
                std::string v_str = token.substr(eq + 1);
                k.erase(k.begin(), std::find_if(k.begin(), k.end(), [](unsigned char ch) { return !std::isspace(ch); }));
                k.erase(std::find_if(k.rbegin(), k.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), k.end());
                v_str.erase(v_str.begin(), std::find_if(v_str.begin(), v_str.end(), [](unsigned char ch) { return !std::isspace(ch); }));
                v_str.erase(std::find_if(v_str.rbegin(), v_str.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), v_str.end());
                if (!k.empty() && !v_str.empty()) {
                    try {
                        double v = std::stod(v_str);
                        active_ca->observe(k, v);
                        steps.push_back("Observed fact: " + k + " = " + v_str);
                        count++;
                    } catch (...) {}
                }
            } else {
                std::stringstream tok_ss(token);
                std::string k, v_str;
                if (tok_ss >> k >> v_str) {
                    try {
                        double v = std::stod(v_str);
                        active_ca->observe(k, v);
                        steps.push_back("Observed fact: " + k + " = " + v_str);
                        count++;
                    } catch (...) {}
                }
            }
        }

        std::string val = "Observed " + std::to_string(count) + " facts";
        return {"CAUSAL_OBSERVE", q.subj, "observed", val, val, steps, true, true, ""};
    }

    BrainQLResult _intervene(const BrainQLQuery& q) {
        CausalEngine default_ca;
        CausalEngine* active_ca = ca ? ca : &default_ca;
        double do_val = std::stod(q.rel);
        auto res = active_ca->intervene(q.subj, do_val, q.obj);
        if (!res.success) {
            return {"INTERVENE", q.subj, q.obj, "", "", res.steps, false, false, res.explanation};
        }
        return {"INTERVENE", q.subj, q.obj, res.value_str, res.value_str, res.steps, true, true, res.explanation};
    }

    BrainQLResult _counterfactual(const BrainQLQuery& q) {
        CausalEngine default_ca;
        CausalEngine* active_ca = ca ? ca : &default_ca;
        double hyp_val = std::stod(q.rel);
        auto res = active_ca->counterfactual(q.subj, hyp_val, q.obj);
        if (!res.success) {
            return {"COUNTERFACTUAL", q.subj, q.obj, "", "", res.steps, false, false, res.explanation};
        }
        return {"COUNTERFACTUAL", q.subj, q.obj, res.value_str, res.value_str, res.steps, true, true, res.explanation};
    }

private:
    BrainQLResult _perceive_image(const BrainQLQuery& q) {
        brain3::engines::vision::VisionEngine default_ve;
        brain3::engines::vision::VisionEngine* active_ve = ve ? ve : &default_ve;

        auto blobs = active_ve->parse_image(q.obj, *re);
        if (blobs.empty()) {
            return {"PERCEIVE_IMAGE", "vision", "blobs", "0", "0 blobs detected or failed to load image", {}, false, false, "empty image or load error"};
        }

        std::vector<std::string> steps;
        for (const auto& b : blobs) {
            steps.push_back("blob_" + std::to_string(b.id) + ": color=" + b.color_name + " pos=" + b.position_name + " area=" + std::to_string(b.area));
        }

        return {"PERCEIVE_IMAGE", "vision", "blobs", std::to_string(blobs.size()), "detected " + std::to_string(blobs.size()) + " physical objects", steps, true, true, "grounded in KB"};
    }

    BrainQLResult _synth(const BrainQLQuery& q) {
        CodeEngine local_ce;
        CodeEngine* active_ce = ce ? ce : &local_ce;
        
        auto res = active_ce->synthesize_spec(q.obj);
        if (!res.tree) {
            return {"SYNTH", "code", "synthesis", "", "failed to synthesize program", {}, false, false, "search space exhausted"};
        }
        std::vector<std::string> steps = {
            "searched " + std::to_string(res.candidates_searched) + " candidates",
            res.code
        };
        return {"SYNTH", "code", "solution", res.code, res.code, steps, true, true, res.code_cpp};
    }

    BrainQLResult _solve(const BrainQLQuery& q) {
        math::MathEngine local_me;
        math::MathEngine* active_me = me ? me : &local_me;
        
        auto res = active_me->solve(q.obj);
        if (!res.success) {
            return {"SOLVE", res.op.empty() ? "math" : res.op, res.target.empty() ? "solve" : res.target, "", "", {}, false, false, res.explanation};
        }
        return {"SOLVE", res.op, res.target.empty() ? "solution" : res.target, res.symbolic_val, res.explanation, res.steps, true, true, res.explanation};
    }

    BrainQLResult _lookup(const BrainQLQuery& q) {
        auto res = re->ask(q.subj, q.rel);
        if (res.first.empty()) return {"LOOKUP", q.subj, q.rel, "", "", {}, false, false, "no direct fact"};
        return {"LOOKUP", q.subj, q.rel, "", res.first, {res.second}, true, true, ""};
    }

    BrainQLResult _chain(const BrainQLQuery& q) {
        auto ancestors = re->closure(q.subj, q.rel);
        if (ancestors.empty()) return {"CHAIN", q.subj, q.rel, "", "", {}, false, false, "no ancestors"};
        std::string vals;
        for (const auto& kv : ancestors) {
            if (!vals.empty()) vals += ", ";
            vals += kv.first;
        }
        return {"CHAIN", q.subj, q.rel, "", vals, {}, true, true, ""};
    }

    BrainQLResult _inherit(const BrainQLQuery& q) {
        re->set_transitive("isa");
        auto direct = re->ask(q.subj, q.rel);
        if (!direct.first.empty()) return {"INHERIT", q.subj, q.rel, "", direct.first, {q.subj + " " + q.rel + " " + direct.first}, true, true, ""};

        std::set<std::string> visited;
        std::vector<std::vector<std::string>> frontier = {{q.subj}};
        while (!frontier.empty()) {
            auto path = frontier.front();
            frontier.erase(frontier.begin());
            std::string node = path.back();
            if (visited.count(node)) continue;
            visited.insert(node);

            auto obj = re->ask(node, q.rel);
            if (!obj.first.empty()) {
                std::vector<std::string> steps;
                for (size_t i = 0; i + 1 < path.size(); ++i) steps.push_back(path[i] + " isa " + path[i+1]);
                steps.push_back(node + " " + q.rel + " " + obj.first);
                return {"INHERIT", q.subj, q.rel, "", obj.first, steps, true, true, ""};
            }

            auto closure = re->closure(node, "isa");
            for (const auto& kv : closure) {
                if (!visited.count(kv.first)) {
                    auto npath = path;
                    npath.push_back(kv.first);
                    frontier.push_back(npath);
                }
            }
        }
        return {"INHERIT", q.subj, q.rel, "", "", {}, false, false, "not found"};
    }

    BrainQLResult _derive(const BrainQLQuery& q) {
        auto obj = re->ask(q.subj, q.rel, q.hops);
        if (obj.first.empty()) return {"DERIVE", q.subj, q.rel, "", "", {}, false, false, "could not derive"};
        return {"DERIVE", q.subj, q.rel, "", obj.first, {obj.second}, true, true, ""};
    }

    BrainQLResult _teach(const BrainQLQuery& q) {
        re->learn(q.subj, q.rel, q.obj);
        return {"TEACH", q.subj, q.rel, q.obj, q.obj, {}, true, true, "taught"};
    }

    BrainQLResult _teach_rule(const BrainQLQuery& q) {
        re->add_rule(q.prem1, q.prem2, q.concl);
        return {"TEACH_RULE", "", "", "", q.prem1 + " + " + q.prem2 + " -> " + q.concl, {}, true, true, ""};
    }

    BrainQLResult _compute(const BrainQLQuery& q) {
        PolicyMemory default_pm;
        PolicyMemory* active_pm = pm ? pm : &default_pm;

        FactSource fs(re);
        PolicySource ps(active_pm);
        std::vector<KnowledgeSource*> srcs = {&fs, &ps};
        MeansEndsSolver solver(srcs);

        Need need{q.subj, q.rel};
        auto val = solver.solve(need);
        if (!val) {
            return {"COMPUTE", q.subj, q.rel, "", "", solver.bb.trace, false, false, "could not derive goal " + need.to_string()};
        }

        std::string val_str = std::to_string(*val);
        if (val_str.find('.') != std::string::npos) {
            val_str.erase(val_str.find_last_not_of('0') + 1, std::string::npos);
            if (val_str.back() == '.') val_str.pop_back();
        }

        return {"COMPUTE", q.subj, q.rel, val_str, val_str, solver.bb.trace, true, true, ""};
    }

    BrainQLResult _explain(const BrainQLQuery& q) {
        return _derive(q); // Simplified for C++ port
    }
};

} // namespace reasoning
} // namespace brain2
