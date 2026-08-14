package brain3;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class NLQuery {

    public static final Set<String> STOP = new HashSet<>(Arrays.asList(
        "what", "whats", "is", "are", "the", "of", "a", "an", "how", "much",
        "many", "does", "do", "did", "have", "has", "for", "compute", "find",
        "calculate", "give", "me", "tell", "s", "to", "in", "on", "value"
    ));

    private final Set<String> entities;
    private final List<String> relations;
    private final Map<String, float[]> glove;
    private final Map<String, float[]> relVec;
    private final Map<String, String> lex;

    public NLQuery(Set<String> entities, List<String> relations, Map<String, float[]> glove) {
        this.entities = entities;
        this.relations = relations;
        this.glove = glove;
        this.relVec = new HashMap<>();
        this.lex = new HashMap<>();

        for (String r : relations) {
            List<float[]> vs = new ArrayList<>();
            for (String w : getRelWords(r)) {
                if (glove.containsKey(w)) {
                    vs.add(glove.get(w));
                }
                lex.put(w, r);
            }
            lex.put(r, r);
            if (!vs.isEmpty()) {
                relVec.put(r, mean(vs));
            }
        }
    }

    private List<String> getRelWords(String rel) {
        // Simplified mapping for port
        return Arrays.asList(rel.split("_"));
    }

    private float[] mean(List<float[]> vecs) {
        if (vecs.isEmpty()) return new float[0];
        int dim = vecs.get(0).length;
        float[] res = new float[dim];
        for (float[] v : vecs) {
            for (int i = 0; i < dim; i++) {
                res[i] += v[i];
            }
        }
        for (int i = 0; i < dim; i++) {
            res[i] /= vecs.size();
        }
        return res;
    }

    private double cosine(float[] a, float[] b) {
        double dot = 0.0, normA = 0.0, normB = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return (normA == 0 || normB == 0) ? 0.0 : dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    public MatchResult matchRelation(List<String> tokens) {
        // 1. exact lexical hit
        for (String t : tokens) {
            if (lex.containsKey(t)) {
                return new MatchResult(lex.get(t), 1.0);
            }
        }
        // 2. morphological hit
        for (String t : tokens) {
            if (t.length() >= 4) {
                String sub = t.substring(0, 4);
                for (String r : relations) {
                    if (r.startsWith(sub) || t.startsWith(r.length() >= 4 ? r.substring(0, 4) : r)) {
                        return new MatchResult(r, 0.9);
                    }
                }
            }
        }
        // 3. embedding nearest
        double bestScore = 0.35;
        String bestRel = null;
        for (String t : tokens) {
            if (!glove.containsKey(t)) continue;
            float[] tVec = glove.get(t);
            for (Map.Entry<String, float[]> entry : relVec.entrySet()) {
                double c = cosine(tVec, entry.getValue());
                if (c > bestScore) {
                    bestScore = c;
                    bestRel = entry.getKey();
                }
            }
        }
        return new MatchResult(bestRel, bestScore);
    }

    public ParseResult parse(String sentence) {
        String[] words = sentence.toLowerCase().replaceAll("[^a-z_\\s]", "").split("\\s+");
        List<String> toks = new ArrayList<>();
        for (String w : words) {
            if (!w.isEmpty() && !STOP.contains(w)) toks.add(w);
        }

        String entity = null;
        for (String t : toks) {
            if (entities.contains(t)) {
                entity = t;
                break;
            }
        }

        List<String> content = new ArrayList<>();
        for (String t : toks) {
            if (!t.equals(entity)) content.add(t);
        }

        MatchResult mr = matchRelation(content);
        return new ParseResult(entity, mr.relation, mr.score);
    }

    public static class MatchResult {
        public final String relation;
        public final double score;
        public MatchResult(String r, double s) { relation = r; score = s; }
    }

    public static class ParseResult {
        public final String entity;
        public final String relation;
        public final double score;
        public ParseResult(String e, String r, double s) { entity = e; relation = r; score = s; }
    }
}
