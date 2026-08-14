#include "brain3_BrainNative.h"
#include "../../fuzzy/core/brain.hpp"
#include <string>
#include <vector>
#include <stdexcept>
#include <sstream>
#include "crisp/engines/reasoning/brainql.hpp"

// Helper to get Brain instance
inline brain2::Brain* get_brain(jlong handle) {
    return reinterpret_cast<brain2::Brain*>(handle);
}

// Helper to convert jstring to std::string
std::string jstring2string(JNIEnv *env, jstring jstr) {
    if (!jstr) return "";
    const char *str = env->GetStringUTFChars(jstr, 0);
    std::string s(str);
    env->ReleaseStringUTFChars(jstr, str);
    return s;
}

// Helper to convert std::string to jstring
jstring string2jstring(JNIEnv *env, const std::string& str) {
    return env->NewStringUTF(str.c_str());
}

// Helper to convert jfloatArray to std::vector<float>
std::vector<float> jfloatArray2vector(JNIEnv *env, jfloatArray jarr) {
    if (!jarr) return {};
    jsize len = env->GetArrayLength(jarr);
    std::vector<float> vec(len);
    env->GetFloatArrayRegion(jarr, 0, len, vec.data());
    return vec;
}

// Helper to convert std::vector<float> to jfloatArray
jfloatArray vector2jfloatArray(JNIEnv *env, const std::vector<float>& vec) {
    jfloatArray jarr = env->NewFloatArray(vec.size());
    if (!vec.empty()) {
        env->SetFloatArrayRegion(jarr, 0, vec.size(), vec.data());
    }
    return jarr;
}

/*
 * Class:     brain3_BrainNative
 * Method:    init
 * Signature: (III)J
 */
JNIEXPORT jlong JNICALL Java_brain3_BrainNative_init
  (JNIEnv *env, jobject obj, jint somRows, jint somCols, jint nDims) {
    auto* b = new brain2::Brain(somRows, somCols, nDims);
    return reinterpret_cast<jlong>(b);
}

/*
 * Class:     brain3_BrainNative
 * Method:    destroy
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_brain3_BrainNative_destroy
  (JNIEnv *env, jobject obj, jlong handle) {
    auto* b = get_brain(handle);
    if (b) {
        delete b;
    }
}

/*
 * Class:     brain3_BrainNative
 * Method:    loadComponents
 */
JNIEXPORT void JNICALL Java_brain3_BrainNative_loadComponents
  (JNIEnv *env, jobject obj, jlong handle, jstring predictorPath, jstring languagePath, jstring somPath, jstring episodicPath, jstring emotionPath, jstring selfPath, jstring symbolicPath, jstring bindingPath, jstring bgPath, jstring proceduresPath, jstring hpredPath) {
    auto* b = get_brain(handle);
    if (!b) return;
    
    b->load_components(
        jstring2string(env, predictorPath),
        jstring2string(env, languagePath),
        jstring2string(env, somPath),
        jstring2string(env, episodicPath),
        jstring2string(env, emotionPath),
        jstring2string(env, selfPath),
        jstring2string(env, symbolicPath),
        jstring2string(env, bindingPath),
        jstring2string(env, bgPath),
        jstring2string(env, proceduresPath),
        jstring2string(env, hpredPath)
    );
    
    b->language.freeze_vocabulary(true);
    std::vector<int> active_indices;
    std::string dirPath = jstring2string(env, predictorPath);
    dirPath = dirPath.substr(0, dirPath.find_last_of('/'));
    std::ifstream av_file(dirPath + "/active_vocab.txt");
    if (av_file.is_open()) {
        int idx;
        while (av_file >> idx) active_indices.push_back(idx);
        av_file.close();
    } else {
        active_indices.resize(b->language.vocab_size());
        for (int i = 0; i < b->language.vocab_size(); i++) active_indices[i] = i;
    }
    b->set_active_vocab(active_indices);
}

/*
 * Class:     brain3_BrainNative
 * Method:    saveComponents
 */
JNIEXPORT void JNICALL Java_brain3_BrainNative_saveComponents
  (JNIEnv *env, jobject obj, jlong handle, jstring dirPath) {
    auto* b = get_brain(handle);
    if (b) b->save_components(jstring2string(env, dirPath));
}

/*
 * Class:     brain3_BrainNative
 * Method:    seedMathSymbols
 */
JNIEXPORT void JNICALL Java_brain3_BrainNative_seedMathSymbols
  (JNIEnv *env, jobject obj, jlong handle) {
    auto* b = get_brain(handle);
    if (b) b->symbolic.seed_math_symbols();
}

/*
 * Class:     brain3_BrainNative
 * Method:    bindSymbol
 */
JNIEXPORT void JNICALL Java_brain3_BrainNative_bindSymbol
  (JNIEnv *env, jobject obj, jlong handle, jstring word) {
    auto* b = get_brain(handle);
    if (b) b->symbolic.bind(jstring2string(env, word), std::vector<float>(b->n_dims, 0.0f));
}

/*
 * Class:     brain3_BrainNative
 * Method:    knowsSymbol
 */
JNIEXPORT jboolean JNICALL Java_brain3_BrainNative_knowsSymbol
  (JNIEnv *env, jobject obj, jlong handle, jstring word) {
    auto* b = get_brain(handle);
    if (b) return b->symbolic.knows(jstring2string(env, word)) ? JNI_TRUE : JNI_FALSE;
    return JNI_FALSE;
}

/*
 * Class:     brain3_BrainNative
 * Method:    registerWord
 */
JNIEXPORT void JNICALL Java_brain3_BrainNative_registerWord
  (JNIEnv *env, jobject obj, jlong handle, jstring word) {
    auto* b = get_brain(handle);
    if (b) b->language.register_word(jstring2string(env, word));
}

/*
 * Class:     brain3_BrainNative
 * Method:    knowsWord
 */
JNIEXPORT jboolean JNICALL Java_brain3_BrainNative_knowsWord
  (JNIEnv *env, jobject obj, jlong handle, jstring word) {
    auto* b = get_brain(handle);
    if (b) return b->language.knows(jstring2string(env, word)) ? JNI_TRUE : JNI_FALSE;
    return JNI_FALSE;
}

/*
 * Class:     brain3_BrainNative
 * Method:    encodeWord
 */
JNIEXPORT jfloatArray JNICALL Java_brain3_BrainNative_encodeWord
  (JNIEnv *env, jobject obj, jlong handle, jstring word) {
    auto* b = get_brain(handle);
    if (!b) return env->NewFloatArray(0);
    return vector2jfloatArray(env, b->language.encode(jstring2string(env, word)));
}

/*
 * Class:     brain3_BrainNative
 * Method:    bestWord
 */
JNIEXPORT jstring JNICALL Java_brain3_BrainNative_bestWord
  (JNIEnv *env, jobject obj, jlong handle, jfloatArray vec) {
    auto* b = get_brain(handle);
    if (!b) return env->NewStringUTF("");
    auto v = jfloatArray2vector(env, vec);
    return string2jstring(env, b->language.best_word(v));
}

/*
 * Class:     brain3_BrainNative
 * Method:    getEmotionArousal
 */
JNIEXPORT jfloat JNICALL Java_brain3_BrainNative_getEmotionArousal
  (JNIEnv *env, jobject obj, jlong handle) {
    auto* b = get_brain(handle);
    return b ? b->emotion.arousal : 0.0f;
}

/*
 * Class:     brain3_BrainNative
 * Method:    getEmotionValence
 */
JNIEXPORT jfloat JNICALL Java_brain3_BrainNative_getEmotionValence
  (JNIEnv *env, jobject obj, jlong handle) {
    auto* b = get_brain(handle);
    return b ? b->emotion.valence : 0.0f;
}

/*
 * Class:     brain3_BrainNative
 * Method:    getSelfMeanRecentError
 */
JNIEXPORT jfloat JNICALL Java_brain3_BrainNative_getSelfMeanRecentError
  (JNIEnv *env, jobject obj, jlong handle) {
    auto* b = get_brain(handle);
    return b ? b->self_model.mean_recent_error() : 0.0f;
}

/*
 * Class:     brain3_BrainNative
 * Method:    getLastEpisode
 */
JNIEXPORT jfloatArray JNICALL Java_brain3_BrainNative_getLastEpisode
  (JNIEnv *env, jobject obj, jlong handle) {
    auto* b = get_brain(handle);
    return vector2jfloatArray(env, b->episodic.get_last_episode());
}

/*
 * Class:     brain3_BrainNative
 * Method:    commitEpisode
 */
JNIEXPORT void JNICALL Java_brain3_BrainNative_commitEpisode
  (JNIEnv *env, jobject obj, jlong handle, jfloat salience, jfloatArray vec) {
    auto* b = get_brain(handle);
    if (b) b->commit_episode(salience, jfloatArray2vector(env, vec));
}

/*
 * Class:     brain3_BrainNative
 * Method:    bindingQueryAll
 */
JNIEXPORT jobjectArray JNICALL Java_brain3_BrainNative_bindingQueryAll
  (JNIEnv *env, jobject obj, jlong handle, jfloatArray vec, jfloat threshold) {
    auto* b = get_brain(handle);
    if (!b) return nullptr;
    auto res = b->binding.query_all(jfloatArray2vector(env, vec), threshold);
    jclass floatArrayClass = env->FindClass("[F");
    jobjectArray result = env->NewObjectArray(res.size(), floatArrayClass, nullptr);
    for (size_t i = 0; i < res.size(); i++) {
        env->SetObjectArrayElement(result, i, vector2jfloatArray(env, res[i]));
    }
    return result;
}

/*
 * Class:     brain3_BrainNative
 * Method:    bindingBind
 */
JNIEXPORT void JNICALL Java_brain3_BrainNative_bindingBind
  (JNIEnv *env, jobject obj, jlong handle, jfloatArray subj, jfloatArray rel, jfloatArray obj_arr) {
    auto* b = get_brain(handle);
    if (b) b->binding.bind(jfloatArray2vector(env, subj), jfloatArray2vector(env, rel), jfloatArray2vector(env, obj_arr));
}

/*
 * Class:     brain3_BrainNative
 * Method:    scratchpadClear
 */
JNIEXPORT void JNICALL Java_brain3_BrainNative_scratchpadClear
  (JNIEnv *env, jobject obj, jlong handle) {
    auto* b = get_brain(handle);
    if (b) b->scratchpad.clear();
}

/*
 * Class:     brain3_BrainNative
 * Method:    scratchpadWrite
 */
JNIEXPORT void JNICALL Java_brain3_BrainNative_scratchpadWrite
  (JNIEnv *env, jobject obj, jlong handle, jstring slotName, jfloatArray vec, jstring historyCtx) {
    auto* b = get_brain(handle);
    if (b) b->scratchpad.write(jstring2string(env, slotName), jfloatArray2vector(env, vec), jstring2string(env, historyCtx));
}

/*
 * Class:     brain3_BrainNative
 * Method:    scratchpadRead
 */
JNIEXPORT jfloatArray JNICALL Java_brain3_BrainNative_scratchpadRead
  (JNIEnv *env, jobject obj, jlong handle, jstring slotName) {
    auto* b = get_brain(handle);
    if (!b) return env->NewFloatArray(0);
    return vector2jfloatArray(env, b->scratchpad.read(jstring2string(env, slotName)));
}

/*
 * Class:     brain3_BrainNative
 * Method:    startReasoning
 */
JNIEXPORT void JNICALL Java_brain3_BrainNative_startReasoning
  (JNIEnv *env, jobject obj, jlong handle) {
    auto* b = get_brain(handle);
    if (b) b->start_reasoning();
}

/*
 * Class:     brain3_BrainNative
 * Method:    forceReasonStep
 */
JNIEXPORT void JNICALL Java_brain3_BrainNative_forceReasonStep
  (JNIEnv *env, jobject obj, jlong handle, jint op, jstring context) {
    auto* b = get_brain(handle);
    if (b) b->force_reason_step(op, jstring2string(env, context));
}

/*
 * Class:     brain3_BrainNative
 * Method:    getSpokenWords
 */
JNIEXPORT jobjectArray JNICALL Java_brain3_BrainNative_getSpokenWords
  (JNIEnv *env, jobject obj, jlong handle) {
    auto* b = get_brain(handle);
    if (!b) return nullptr;
    auto words = b->get_spoken_words();
    jclass stringClass = env->FindClass("java/lang/String");
    jobjectArray result = env->NewObjectArray(words.size(), stringClass, nullptr);
    for (size_t i = 0; i < words.size(); i++) {
        env->SetObjectArrayElement(result, i, string2jstring(env, words[i]));
    }
    return result;
}

/*
 * Class:     brain3_BrainNative
 * Method:    clearSpokenWords
 */
JNIEXPORT void JNICALL Java_brain3_BrainNative_clearSpokenWords
  (JNIEnv *env, jobject obj, jlong handle) {
    auto* b = get_brain(handle);
    if (b) b->clear_spoken_words();
}

/*
 * Class:     brain3_BrainNative
 * Method:    proceduresRetrieve
 */
JNIEXPORT jintArray JNICALL Java_brain3_BrainNative_proceduresRetrieve
  (JNIEnv *env, jobject obj, jlong handle, jfloatArray vec) {
    auto* b = get_brain(handle);
    if (!b) return env->NewIntArray(0);
    auto* proc = b->procedures.retrieve(jfloatArray2vector(env, vec));
    if (!proc) return env->NewIntArray(0);
    jintArray result = env->NewIntArray(proc->steps.size());
    std::vector<int> steps_int(proc->steps.size());
    for(size_t i=0; i<proc->steps.size(); i++) steps_int[i] = (int)proc->steps[i];
    env->SetIntArrayRegion(result, 0, steps_int.size(), steps_int.data());
    return result;
}

/*
 * Class:     brain3_BrainNative
 * Method:    workingMemGate
 */
JNIEXPORT void JNICALL Java_brain3_BrainNative_workingMemGate
  (JNIEnv *env, jobject obj, jlong handle, jfloatArray vec, jfloat salience) {
    auto* b = get_brain(handle);
    if (b) b->working_mem.gate(jfloatArray2vector(env, vec), salience);
}

/*
 * Class:     brain3_BrainNative
 * Method:    workingMemTick
 */
JNIEXPORT void JNICALL Java_brain3_BrainNative_workingMemTick
  (JNIEnv *env, jobject obj, jlong handle) {
    auto* b = get_brain(handle);
    if (b) b->working_mem.tick();
}

/*
 * Class:     brain3_BrainNative
 * Method:    workingMemContext
 */
JNIEXPORT jfloatArray JNICALL Java_brain3_BrainNative_workingMemContext
  (JNIEnv *env, jobject obj, jlong handle) {
    auto* b = get_brain(handle);
    if (!b) return env->NewFloatArray(0);
    return vector2jfloatArray(env, b->working_mem.context());
}

/*
 * Class:     brain3_BrainNative
 * Method:    somActivationMap
 */
JNIEXPORT jfloatArray JNICALL Java_brain3_BrainNative_somActivationMap
  (JNIEnv *env, jobject obj, jlong handle, jfloatArray vec) {
    auto* b = get_brain(handle);
    if (!b) return env->NewFloatArray(0);
    return vector2jfloatArray(env, b->som.activation_map(jfloatArray2vector(env, vec)));
}

/*
 * Class:     brain3_BrainNative
 * Method:    daydream
 */
JNIEXPORT void JNICALL Java_brain3_BrainNative_daydream
  (JNIEnv *env, jobject obj, jlong handle) {
    auto* b = get_brain(handle);
    if (b) b->daydream();
}

/*
 * Class:     brain3_BrainNative
 * Method:    sleep
 */
JNIEXPORT jstring JNICALL Java_brain3_BrainNative_sleep
  (JNIEnv *env, jobject obj, jlong handle, jstring gateLogPath, jstring checkpointDir) {
    auto* b = get_brain(handle);
    if (!b) return env->NewStringUTF("{\"error\": \"Brain instance is null\"}");

    const char* glp = env->GetStringUTFChars(gateLogPath, 0);
    const char* cpd = env->GetStringUTFChars(checkpointDir, 0);

    std::string gl_str(glp ? glp : "associative_gate.jsonl");
    std::string cp_str(cpd ? cpd : "./out/brain_fluent");

    if (glp) env->ReleaseStringUTFChars(gateLogPath, glp);
    if (cpd) env->ReleaseStringUTFChars(checkpointDir, cpd);

    auto report = b->sleep(gl_str, cp_str);

    std::string json = "{";
    json += "\"phase1_rules\": " + std::to_string(report.phase1_rules_created) + ", ";
    json += "\"phase1_facts_pruned\": " + std::to_string(report.phase1_facts_pruned) + ", ";
    json += "\"phase1_exceptions\": " + std::to_string(report.phase1_exceptions_added) + ", ";
    json += "\"phase2_records\": " + std::to_string(report.phase2_telemetry_records) + ", ";
    json += "\"phase2_triples_trained\": " + std::to_string(report.phase2_triples_trained) + ", ";
    json += "\"phase2_loss_before\": " + std::to_string(report.phase2_avg_loss_before) + ", ";
    json += "\"phase2_loss_after\": " + std::to_string(report.phase2_avg_loss_after) + ", ";
    json += "\"phase3_som_decayed\": " + std::to_string(report.phase3_som_nodes_decayed) + ", ";
    json += "\"phase4_checkpoint\": " + std::string(report.phase4_checkpoint_success ? "true" : "false");
    json += "}";

    return env->NewStringUTF(json.c_str());
}

/*
 * Class:     brain3_BrainNative
 * Method:    perceive
 */
JNIEXPORT jfloatArray JNICALL Java_brain3_BrainNative_perceive
  (JNIEnv *env, jobject obj, jlong handle, jfloatArray vec) {
    auto* b = get_brain(handle);
    if (!b) return env->NewFloatArray(0);
    auto res = b->perceive(jfloatArray2vector(env, vec));
    std::vector<float> ret = { (float)res.bmu, res.prediction_error, res.valence, res.arousal, res.salience };
    return vector2jfloatArray(env, ret);
}

/*
 * Class:     brain3_BrainNative
 * Method:    getLastConfidence
 */
JNIEXPORT jfloat JNICALL Java_brain3_BrainNative_getLastConfidence
  (JNIEnv *env, jobject obj, jlong handle) {
    auto* b = get_brain(handle);
    return 0.0f;
}

/*
 * Class:     brain3_BrainNative
 * Method:    resetSequence
 */
JNIEXPORT void JNICALL Java_brain3_BrainNative_resetSequence
  (JNIEnv *env, jobject obj, jlong handle) {
    auto* b = get_brain(handle);
    if (b) b->reset_sequence();
}

/*
 * Class:     brain3_BrainNative
 * Method:    executeBrainQL
 */
static std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\b') out += "\\b";
        else if (c == '\f') out += "\\f";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out;
}

static std::string to_lower(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c){ return std::tolower(c); });
    return out;
}

static void log_gate_decision(const std::string& subj, const std::string& rel, const std::string& guess, const std::string& store_truth, const std::string& verdict) {
    std::ofstream out("associative_gate.jsonl", std::ios_base::app);
    if (out.is_open()) {
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
        out << "{\"timestamp\": " << timestamp 
            << ", \"inputs\": [\"" << json_escape(subj) << "\", \"" << json_escape(rel) << "\"]"
            << ", \"guess\": \"" << json_escape(guess) << "\""
            << ", \"store_truth\": \"" << json_escape(store_truth) << "\""
            << ", \"verdict\": \"" << verdict << "\"}\n";
    }
}

JNIEXPORT jstring JNICALL Java_brain3_BrainNative_executeBrainQL(
    JNIEnv* env, jobject obj, jlong handle, jstring query) {
    auto* b = get_brain(handle);
    if (!b) return env->NewStringUTF("{\"error\": \"Brain instance is null\"}");
    
    std::string qstr = jstring2string(env, query);
    
    if (qstr.find("CHAT ") == 0) {
        std::string chat_text = qstr.substr(5);
        std::string lower_chat = chat_text;
        std::transform(lower_chat.begin(), lower_chat.end(), lower_chat.begin(), ::tolower);
        
        std::string subj = "unknown";
        std::string rel = "unknown";
        std::string obj = "unknown";
        std::string objType = "atomic";
        std::string source = "unknown";
        bool verified = false;
        bool known = false;
        
        if (lower_chat.find("hello") != std::string::npos || 
            lower_chat.find("hi") != std::string::npos ||
            lower_chat.find("greeting") != std::string::npos) {
            subj = "user";
            rel = "intent";
            obj = "greeting";
            objType = "atomic";
            source = "greeting";
            verified = true;
            known = true;
        } else if (lower_chat.find("how are you") != std::string::npos ||
                   lower_chat.find("status") != std::string::npos) {
            subj = "self";
            rel = "emotion";
            float val = b->emotion.valence;
            if (val > 0.6f) obj = "happy";
            else if (val < 0.4f) obj = "sad";
            else obj = "neutral";
            objType = "atomic";
            source = "emotion";
            verified = true;
            known = true;
        } else {
            // Open chit-chat: use neural predictor for associative musing
            b->reset_sequence();
            std::stringstream ss(chat_text);
            std::string word;
            std::vector<float> last_vec;
            while (ss >> word) {
                std::string clean_word;
                for (char c : word) {
                    if (std::isalpha(c)) clean_word += std::tolower(c);
                }
                if (!clean_word.empty()) {
                    auto vec = b->language.encode(clean_word);
                    b->perceive(vec);
                    last_vec = vec;
                }
            }
            
            auto tr = b->think(4);
            
            if (tr.words.empty() && !last_vec.empty()) {
                auto nearest = b->language.decode(last_vec, 5);
                for (const auto& res : nearest) {
                    if (chat_text.find(res.first) == std::string::npos) {
                        tr.words.push_back(res.first);
                    }
                }
            }
            
            // TEST BACKDOORS for deterministic gate testing
            if (chat_text == "TEST_GATE_1") tr.words = {"Paris", "capital", "France"};
            else if (chat_text == "TEST_GATE_2") tr.words = {"paris", "capital", "france"};
            else if (chat_text == "TEST_GATE_3") tr.words = {"paris", "capital", "london"};
            else if (chat_text == "TEST_GATE_4") tr.words = {"dog", "drives", "car"};

            std::string raw_thoughts = "";
            for (const auto& w : tr.words) {
                if (!raw_thoughts.empty()) raw_thoughts += " ";
                raw_thoughts += w;
            }
            if (raw_thoughts.empty()) raw_thoughts = "silence";
            
            // Associative Triple Pre-Verification Gate
            bool gate_passed = false;
            if (tr.words.size() == 3) {
                std::string p_subj = tr.words[0];
                std::string p_rel = tr.words[1];
                std::string p_obj = tr.words[2];
                
                std::string p_subj_lower = to_lower(p_subj);
                std::string p_rel_lower = to_lower(p_rel);
                std::string p_obj_lower = to_lower(p_obj);
                
                // Case-insensitive lookup against the crisp store facts
                std::string store_truth = "";
                for (const auto& f : b->brainql_engine.facts) {
                    if (to_lower(f.subj) == p_subj_lower && to_lower(f.rel) == p_rel_lower) {
                        store_truth = to_lower(f.obj);
                        break;
                    }
                }
                
                if (store_truth.empty()) {
                    log_gate_decision(p_subj, p_rel, p_obj, "", "not_found");
                } else if (store_truth == p_obj_lower) {
                    log_gate_decision(p_subj, p_rel, p_obj, store_truth, "verified_atomic");
                    subj = p_subj;
                    rel = p_rel;
                    obj = p_obj;
                    objType = "atomic";
                    source = "gate_verified_associative";
                    verified = true;
                    known = true;
                    gate_passed = true;
                } else {
                    log_gate_decision(p_subj, p_rel, p_obj, store_truth, "rejected_mismatch");
                }
            }
            
            if (!gate_passed) {
                subj = "internal";
                rel = "association";
                obj = raw_thoughts;
                objType = "freetext";
                source = "associative_musing";
                verified = false;
                known = false;
            }
        }
        
        std::string escaped_obj = json_escape(obj);
        std::string escaped_res = json_escape(subj + " " + rel + " " + obj);

        std::string json = "{";
        json += "\"op\": \"CHAT\",";
        json += "\"subj\": \"" + subj + "\",";
        json += "\"rel\": \"" + rel + "\",";
        json += "\"obj\": \"" + escaped_obj + "\",";
        json += "\"objType\": \"" + objType + "\",";
        json += "\"result\": \"" + escaped_res + "\",";
        json += "\"verified\": " + std::string(verified ? "true" : "false") + ",";
        json += "\"known\": " + std::string(known ? "true" : "false") + ",";
        json += "\"source\": \"" + source + "\"";
        json += "}";
        return env->NewStringUTF(json.c_str());
    }

    try {
        brain2::reasoning::BrainQLQuery bq = brain2::reasoning::parse_bql(qstr);
        brain2::reasoning::BrainQLExecutor exec(&b->brainql_engine, nullptr, &b->math_engine, &b->code_engine, &b->policy_memory, &b->vision_engine, &b->causal_engine, &b->analogy_engine, &b->metacognitive_engine, &b->discovery_engine, &b->curiosity_engine, &b->instinct_engine);
        brain2::reasoning::BrainQLResult res = exec.run(bq);
        
        std::string src = (res.op == "SOLVE") ? "math_engine" : 
                          (res.op == "SYNTH") ? "code_engine" : 
                          (res.op == "COMPUTE") ? "means_ends" : 
                          (res.op == "PERCEIVE_IMAGE" || res.op == "VISION") ? "vision_engine" : 
                          (res.op == "INTERVENE" || res.op == "COUNTERFACTUAL" || res.op == "WHAT_IF" || res.op == "CAUSAL_DEFINE" || res.op == "CAUSAL_OBSERVE") ? "causal_engine" : 
                          (res.op == "ANALOGY" || res.op == "ANALOGY_DEFINE") ? "analogy_engine" : 
                          (res.op == "REFUTE" || res.op == "META_VERIFY" || res.op == "CRITIQUE") ? "metacognitive_engine" : 
                          (res.op == "DISCOVER" || res.op == "INFER_EQUATION") ? "discovery_engine" : 
                          (res.op == "CURIOSITY_GAPS" || res.op == "CURIOSITY_TICK" || res.op == "AUTONOMOUS_CYCLE" || res.op == "CURIOSITY_OBSERVE") ? "curiosity_engine" : 
                          (res.op == "INSTINCT" || res.op == "INSTINCT_FIRE" || res.op == "INSTINCT_TRAIN" || res.op == "INSTINCT_STATUS" || res.op == "INSTINCT_PENALIZE") ? "instinct_engine" : "lookup";
        std::string escaped_obj = json_escape(res.obj);
        std::string escaped_val = json_escape(res.value);

        // Simple JSON serialization
        std::string json = "{";
        json += "\"op\": \"" + res.op + "\",";
        json += "\"subj\": \"" + res.subj + "\",";
        json += "\"rel\": \"" + res.rel + "\",";
        json += "\"obj\": \"" + escaped_obj + "\",";
        json += "\"objType\": \"atomic\",";
        json += "\"result\": \"" + escaped_val + "\",";
        json += "\"verified\": " + std::string(res.verified ? "true" : "false") + ",";
        json += "\"known\": " + std::string(res.known ? "true" : "false") + ",";
        json += "\"source\": \"" + src + "\",";
        json += "\"explanation\": \"" + json_escape(res.note) + "\",";
        json += "\"steps\": [";
        for (size_t si = 0; si < res.chain.size(); ++si) {
            json += "\"" + json_escape(res.chain[si]) + "\"";
            if (si + 1 < res.chain.size()) json += ",";
        }
        json += "]";
        json += "}";
        return env->NewStringUTF(json.c_str());
    } catch (const std::exception& e) {
        std::string err = "{\"error\": \"" + json_escape(std::string(e.what())) + "\"}";
        return env->NewStringUTF(err.c_str());
    }
}
