package brain3;

import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class NLFront {

    private static final double STRONG = 0.9;
    private static final double CONF_FLOOR = 0.85;

    private final NLQuery lexical;
    // Student, BrainNative, etc would be integrated here

    public NLFront(NLQuery lexical) {
        this.lexical = lexical;
    }

    public ResolveResult resolve(String q) {
        NLQuery.ParseResult parsed = lexical.parse(q);
        
        // 1. strong lexical 
        if (parsed.relation != null && parsed.score >= STRONG) {
            return new ResolveResult(parsed.entity, parsed.relation, "lexical");
        }

        // Student and LLM fallbacks would be implemented here in a full port.
        // For now, if lexical is weak, we abstain.
        return new ResolveResult(parsed.entity, null, "none");
    }

    public AnswerResult answer(String q) {
        ResolveResult res = resolve(q);
        if (res.relation == null) {
            return new AnswerResult("I don't know.", "none");
        }
        
        // Value resolution would call BrainNative policy engine
        // Mocked response for port structure
        return new AnswerResult(res.entity + "." + res.relation + " = [computed]", res.source);
    }

    public static class ResolveResult {
        public final String entity;
        public final String relation;
        public final String source;
        public ResolveResult(String e, String r, String s) { entity = e; relation = r; source = s; }
    }

    public static class AnswerResult {
        public final String answer;
        public final String source;
        public AnswerResult(String a, String s) { answer = a; source = s; }
    }
}
