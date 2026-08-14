#pragma once
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <stdexcept>

namespace brain3 {
namespace engines {
namespace synthesis {

struct ClassField {
    std::string name;
    std::string type;
};

struct ClassMethod {
    std::string name;
    std::vector<ClassField> params;
    std::string ret = "void";
};

struct ClassSpec {
    std::string name;
    std::vector<ClassField> fields;
    std::vector<ClassMethod> methods;
};

class CodeGen {
private:
    std::string map_type(const std::string& lang, const std::string& generic_type) const {
        static std::map<std::string, std::map<std::string, std::string>> TYPES = {
            {"python", {{"int", "int"}, {"float", "float"}, {"string", "str"}, {"bool", "bool"}, {"void", "None"}}},
            {"cpp",    {{"int", "int"}, {"float", "double"}, {"string", "std::string"}, {"bool", "bool"}, {"void", "void"}}},
            {"java",   {{"int", "int"}, {"float", "double"}, {"string", "String"}, {"bool", "boolean"}, {"void", "void"}}}
        };
        auto it = TYPES.find(lang);
        if (it != TYPES.end()) {
            auto jt = it->second.find(generic_type);
            if (jt != it->second.end()) return jt->second;
        }
        return generic_type;
    }

    std::string java_default(const std::string& mapped_type) const {
        if (mapped_type == "int") return "0";
        if (mapped_type == "double") return "0.0";
        if (mapped_type == "boolean") return "false";
        if (mapped_type == "String") return "null";
        return "null";
    }

    std::string gen_python(const ClassSpec& s) const {
        std::ostringstream L;
        L << "class " << s.name << ":\n";
        if (!s.fields.empty()) {
            L << "    def __init__(self";
            for (const auto& f : s.fields)
                L << ", " << f.name << ": " << map_type("python", f.type);
            L << "):\n";
            for (const auto& f : s.fields)
                L << "        self." << f.name << " = " << f.name << "\n";
        } else {
            L << "    def __init__(self):\n";
            L << "        pass\n";
        }
        for (const auto& m : s.methods) {
            L << "\n    def " << m.name << "(self";
            for (const auto& p : m.params)
                L << ", " << p.name << ": " << map_type("python", p.type);
            L << ") -> " << map_type("python", m.ret) << ":\n";
            L << "        pass\n";
        }
        return L.str();
    }

    std::string gen_cpp(const ClassSpec& s) const {
        std::ostringstream L;
        L << "class " << s.name << " {\npublic:\n";
        for (const auto& f : s.fields)
            L << "    " << map_type("cpp", f.type) << " " << f.name << ";\n";
        
        if (!s.fields.empty()) {
            L << "    " << s.name << "(";
            bool first = true;
            for (const auto& f : s.fields) {
                if (!first) L << ", ";
                L << map_type("cpp", f.type) << " " << f.name;
                first = false;
            }
            L << ") : ";
            first = true;
            for (const auto& f : s.fields) {
                if (!first) L << ", ";
                L << f.name << "(" << f.name << ")";
                first = false;
            }
            L << " {}\n";
        }

        for (const auto& m : s.methods) {
            std::string ret = map_type("cpp", m.ret);
            L << "    " << ret << " " << m.name << "(";
            bool first = true;
            for (const auto& p : m.params) {
                if (!first) L << ", ";
                L << map_type("cpp", p.type) << " " << p.name;
                first = false;
            }
            L << ") {\n        // TODO\n";
            if (ret != "void")
                L << "        return {};\n";
            L << "    }\n";
        }
        L << "};\n";
        return L.str();
    }

    std::string gen_java(const ClassSpec& s) const {
        std::ostringstream L;
        L << "public class " << s.name << " {\n";
        for (const auto& f : s.fields)
            L << "    private " << map_type("java", f.type) << " " << f.name << ";\n";
        
        if (!s.fields.empty()) {
            L << "    public " << s.name << "(";
            bool first = true;
            for (const auto& f : s.fields) {
                if (!first) L << ", ";
                L << map_type("java", f.type) << " " << f.name;
                first = false;
            }
            L << ") {\n";
            for (const auto& f : s.fields)
                L << "        this." << f.name << " = " << f.name << ";\n";
            L << "    }\n";
        }

        for (const auto& m : s.methods) {
            std::string ret = map_type("java", m.ret);
            L << "    public " << ret << " " << m.name << "(";
            bool first = true;
            for (const auto& p : m.params) {
                if (!first) L << ", ";
                L << map_type("java", p.type) << " " << p.name;
                first = false;
            }
            L << ") {\n        // TODO\n";
            if (ret != "void")
                L << "        return " << java_default(ret) << ";\n";
            L << "    }\n";
        }
        L << "}\n";
        return L.str();
    }

public:
    std::string generate(const ClassSpec& spec, const std::string& lang) const {
        if (lang == "python") return gen_python(spec);
        if (lang == "cpp") return gen_cpp(spec);
        if (lang == "java") return gen_java(spec);
        throw std::invalid_argument("unsupported language: " + lang);
    }
};

}}}
