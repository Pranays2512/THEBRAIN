/*
brain_gru.cpp — FastGRU: C++ GRU language model for THEBRAIN
=============================================================
Exposed via brain_gru Python module (built alongside brain_core).
Replaces Python GRUWordTP — same algorithm, full C++ training loop.
Speed: ~100x faster fit() than numpy Python equivalent.
*/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstring>

namespace py = pybind11;
using namespace std;

class FastGRU {
public:
    static const int SEP_IDX = 0;
    static const int END_IDX = 1;

    int H, V;
    bool trained;
    int V_at_train;

    vector<float> Wz, Wr, Wh, Wo;
    vector<float> bz, br, bh, bo;

    vector<string>           vocab;
    unordered_map<string,int> w2i;
    unordered_map<int, unordered_map<int,float>> counts;
    vector<vector<int>>      sequences;

    int          prev_word;
    vector<int>  cur_seq;
    mt19937      rng;

    FastGRU() : FastGRU(64) {}
    explicit FastGRU(int hidden_dim)
        : H(hidden_dim), V(2), trained(false), V_at_train(0),
          prev_word(-1), rng(42)
    {
        vocab = {"<SEP>","<END>"};
        w2i   = {{"<SEP>",0},{"<END>",1}};
        cur_seq = {SEP_IDX};
    }

    int add_word(const string& w) {
        auto it = w2i.find(w);
        if (it != w2i.end()) return it->second;
        int idx = (int)vocab.size();
        vocab.push_back(w);
        w2i[w] = idx;
        V = (int)vocab.size();
        return idx;
    }

    void observe(const string& word, float weight = 1.0f) {
        int idx = add_word(word);
        if (prev_word >= 0) counts[prev_word][idx] += weight;
        prev_word = idx;
        cur_seq.push_back(idx);
    }

    void separator() {
        cur_seq.push_back(SEP_IDX);
        prev_word = -1;
    }

    void end_seq() {
        prev_word = -1;
        if ((int)cur_seq.size() >= 4) sequences.push_back(cur_seq);
        cur_seq = {SEP_IDX};
    }

    int n_words()       const { return max(0, V-2); }
    int n_transitions() const {
        int t=0; for(auto&[k,m]:counts) t+=(int)m.size(); return t;
    }

    // ── Weight init ────────────────────────────────────────────────────

    void init_weights() {
        int HV = H+V;
        float s = sqrtf(2.0f/HV);
        normal_distribution<float> d1(0.0f,s);
        auto fill=[&](int n){ vector<float> v(n); for(auto&x:v)x=d1(rng); return v; };
        Wz=fill(H*HV); Wr=fill(H*HV); Wh=fill(H*HV);
        bz.assign(H,0); br.assign(H,0); bh.assign(H,0);
        float s2=sqrtf(2.0f/(V+H));
        normal_distribution<float> d2(0.0f,s2);
        Wo.resize(V*H); for(auto&x:Wo)x=d2(rng);
        bo.assign(V,0);
        V_at_train=V;
    }

    void resize_weights() {
        int HV_old=H+V_at_train, HV_new=H+V;
        auto expand=[&](vector<float>& W){
            vector<float> W2(H*HV_new,0.0f);
            for(int i=0;i<H;i++)
                memcpy(W2.data()+i*HV_new, W.data()+i*HV_old, HV_old*sizeof(float));
            W=move(W2);
        };
        expand(Wz); expand(Wr); expand(Wh);
        float s=sqrtf(2.0f/(V+H));
        normal_distribution<float> d(0.0f,s);
        Wo.resize(V*H); for(int i=V_at_train*H;i<V*H;i++) Wo[i]=d(rng);
        bo.resize(V,0.0f);
        V_at_train=V;
    }

    // ── Forward (one-hot trick) ────────────────────────────────────────

    struct Cache {
        vector<float> z,r,ht,hp,hn,rh;
        int wi;
    };

    pair<vector<float>,vector<float>> fwd(int wi, const vector<float>& h, Cache& c) const {
        int HV=H+V;
        auto gp=[&](const vector<float>& W, const vector<float>& b,
                    const vector<float>& hin){
            vector<float> o(H);
            for(int i=0;i<H;i++){
                const float* row=W.data()+i*HV;
                float s=b[i]+row[H+wi];
                for(int j=0;j<H;j++) s+=row[j]*hin[j];
                o[i]=s;
            }
            return o;
        };
        auto zp=gp(Wz,bz,h), rp=gp(Wr,br,h);
        vector<float> z(H),r(H),rh(H);
        for(int i=0;i<H;i++){
            z[i]=1.0f/(1.0f+expf(-zp[i]));
            r[i]=1.0f/(1.0f+expf(-rp[i]));
            rh[i]=r[i]*h[i];
        }
        auto htp=gp(Wh,bh,rh);
        vector<float> ht(H),hn(H);
        for(int i=0;i<H;i++){
            ht[i]=tanhf(htp[i]);
            hn[i]=(1.0f-z[i])*h[i]+z[i]*ht[i];
        }
        vector<float> logits(V);
        for(int i=0;i<V;i++){
            const float* row=Wo.data()+i*H;
            float s=bo[i];
            for(int j=0;j<H;j++) s+=row[j]*hn[j];
            logits[i]=s;
        }
        c={z,r,ht,h,hn,rh,wi};
        return {hn,logits};
    }

    // ── Backward ──────────────────────────────────────────────────────

    void bwd(const Cache& c, int target, const vector<float>& logits, float lr) {
        float mx=*max_element(logits.begin(),logits.end());
        vector<float> dl(V); float se=0;
        for(int i=0;i<V;i++){dl[i]=expf(logits[i]-mx);se+=dl[i];}
        for(int i=0;i<V;i++) dl[i]/=se;
        dl[target]-=1.0f;

        vector<float> dhn(H,0.0f);
        for(int i=0;i<V;i++){
            float d=dl[i]*lr;
            float* row=Wo.data()+i*H;
            for(int j=0;j<H;j++){dhn[j]+=row[j]*dl[i]; row[j]-=d*c.hn[j];}
            bo[i]-=dl[i]*lr;
        }

        int HV=H+V;
        const auto &z=c.z,&r=c.r,&ht=c.ht,&hp=c.hp;

        vector<float> dht(H),dz(H);
        for(int i=0;i<H;i++){
            dht[i]=z[i]*dhn[i]*(1.0f-ht[i]*ht[i]);
            dz[i]=(ht[i]-hp[i])*dhn[i]*z[i]*(1.0f-z[i]);
        }

        vector<float> WhdhtH(H,0.0f);
        for(int j=0;j<H;j++){
            float s=0;
            for(int i=0;i<H;i++) s+=Wh[i*HV+j]*dht[i];
            WhdhtH[j]=s;
        }
        vector<float> dr(H);
        for(int i=0;i<H;i++) dr[i]=WhdhtH[i]*hp[i]*r[i]*(1.0f-r[i]);

        auto upd=[&](vector<float>& W, vector<float>& b,
                     const vector<float>& dg, const vector<float>& hin){
            for(int i=0;i<H;i++){
                float d=dg[i]*lr;
                float* row=W.data()+i*HV;
                b[i]-=d; row[H+c.wi]-=d;
                for(int j=0;j<H;j++) row[j]-=d*hin[j];
            }
        };
        upd(Wz,bz,dz,hp);
        upd(Wr,br,dr,hp);
        upd(Wh,bh,dht,c.rh);
    }

    // ── Training loop ─────────────────────────────────────────────────

    void fit(int epochs=15, float lr=0.02f, int max_sequences=800) {
        if((int)sequences.size()<3) return;
        if(Wz.empty()) init_weights();
        else if(V>V_at_train) resize_weights();

        int start=max(0,(int)sequences.size()-max_sequences);
        int n=(int)sequences.size()-start;
        vector<int> perm(n); iota(perm.begin(),perm.end(),0);

        for(int ep=0;ep<epochs;ep++){
            float ep_lr=lr*powf(0.95f,(float)ep);
            shuffle(perm.begin(),perm.end(),rng);
            for(int si:perm){
                const auto& seq=sequences[start+si];
                vector<float> h(H,0.0f);
                for(int t=0;t+1<(int)seq.size();t++){
                    Cache c;
                    auto [hn,logits]=fwd(seq[t],h,c);
                    bwd(c,seq[t+1],logits,ep_lr);
                    h=move(hn);
                }
            }
        }
        trained=true;
        if((int)sequences.size()>max_sequences)
            sequences.erase(sequences.begin(),sequences.end()-max_sequences);
    }

    // ── Generation ────────────────────────────────────────────────────

    vector<string> generate(const vector<string>& ctx, int max_len=8,
                             float temperature=1.0f,
                             const vector<string>& forbidden_words={}) {
        unordered_set<int> fb;
        for(auto& w:forbidden_words){ auto it=w2i.find(w); if(it!=w2i.end()) fb.insert(it->second); }

        if(!trained) return bigram_gen(ctx,max_len,fb);
        if(V>V_at_train) resize_weights();

        vector<float> h(H,0.0f);
        for(auto& w:ctx){
            auto it=w2i.find(w); if(it==w2i.end()) continue;
            Cache c; auto [hn,_]=fwd(it->second,h,c); h=move(hn);
        }

        vector<string> result;
        int cur=SEP_IDX;
        uniform_real_distribution<float> ud(0.0f,1.0f);
        for(int i=0;i<max_len;i++){
            Cache c; auto [hn,logits]=fwd(cur,h,c); h=move(hn);
            float mx=*max_element(logits.begin(),logits.end());
            vector<float> probs(V); float se=0;
            for(int j=0;j<V;j++){
                if(fb.count(j)){probs[j]=0;continue;}
                probs[j]=expf((logits[j]-mx)/temperature); se+=probs[j];
            }
            if(se==0) break;
            float rv=ud(rng)*se; int next=END_IDX;
            for(int j=0;j<V;j++){rv-=probs[j]; if(rv<=0){next=j;break;}}
            if(next==END_IDX||next==SEP_IDX) break;
            result.push_back(vocab[next]); cur=next;
        }
        return result;
    }

    vector<string> bigram_gen(const vector<string>& ctx, int max_len,
                               const unordered_set<int>& fb) {
        int cur=SEP_IDX;
        for(auto& w:ctx){ auto it=w2i.find(w); if(it!=w2i.end()) cur=it->second; }
        vector<string> result;
        uniform_real_distribution<float> ud(0.0f,1.0f);
        for(int i=0;i<max_len;i++){
            auto it=counts.find(cur); if(it==counts.end()) break;
            float tot=0; for(auto&[j,c]:it->second) if(!fb.count(j)) tot+=c;
            if(tot==0) break;
            float rv=ud(rng)*tot; int next=END_IDX;
            for(auto&[j,c]:it->second){if(fb.count(j)) continue; rv-=c; if(rv<=0){next=j;break;}}
            if(next==END_IDX||next==SEP_IDX) break;
            result.push_back(vocab[next]); cur=next;
        }
        return result;
    }

    // ── State serialisation ───────────────────────────────────────────

    py::dict get_state() const {
        py::dict d;
        d["H"]=H; d["V"]=V; d["trained"]=trained; d["V_at_train"]=V_at_train;
        d["vocab"]=vocab;
        auto ta=[](const vector<float>&v,vector<ssize_t> sh){
            return py::array_t<float>(sh,v.data());
        };
        if(!Wz.empty()){
            d["Wz"]=ta(Wz,{H,H+V}); d["Wr"]=ta(Wr,{H,H+V}); d["Wh"]=ta(Wh,{H,H+V});
            d["Wo"]=ta(Wo,{V,H});
            d["bz"]=ta(bz,{H}); d["br"]=ta(br,{H}); d["bh"]=ta(bh,{H}); d["bo"]=ta(bo,{V});
        }
        py::dict cd;
        for(auto&[k,m]:counts){
            py::dict inner; for(auto&[j,v]:m) inner[py::int_(j)]=v;
            cd[py::int_(k)]=inner;
        }
        d["counts"]=cd;
        int start=max(0,(int)sequences.size()-800);
        py::list sl;
        for(int i=start;i<(int)sequences.size();i++) sl.append(py::cast(sequences[i]));
        d["sequences"]=sl;
        d["gru"]=true;
        return d;
    }

    void set_state(const py::dict& d) {
        H=d["H"].cast<int>(); trained=d["trained"].cast<bool>();
        V_at_train=d.contains("V_at_train")?d["V_at_train"].cast<int>():0;
        vocab=d["vocab"].cast<vector<string>>();
        V=(int)vocab.size();
        w2i.clear(); for(int i=0;i<V;i++) w2i[vocab[i]]=i;
        auto lv=[&](const string& k)->vector<float>{
            if(!d.contains(k.c_str())) return {};
            auto a=d[k.c_str()].cast<py::array_t<float>>(); auto b=a.request();
            return vector<float>((float*)b.ptr,(float*)b.ptr+b.size);
        };
        Wz=lv("Wz");Wr=lv("Wr");Wh=lv("Wh");Wo=lv("Wo");
        bz=lv("bz");br=lv("br");bh=lv("bh");bo=lv("bo");
        counts.clear();
        if(d.contains("counts"))
            for(auto&[k,m]:d["counts"].cast<py::dict>())
                for(auto&[j,v]:m.cast<py::dict>())
                    counts[k.cast<int>()][j.cast<int>()]=v.cast<float>();
        sequences.clear();
        if(d.contains("sequences"))
            for(auto& s:d["sequences"]) sequences.push_back(s.cast<vector<int>>());
        prev_word=-1; cur_seq={SEP_IDX};
    }
};

PYBIND11_MODULE(brain_gru, m) {
    m.doc() = "FastGRU — C++ GRU word-level language model for THEBRAIN";

    py::class_<FastGRU>(m, "FastGRU")
        .def(py::init<int>(), py::arg("hidden_dim")=64)
        .def("observe",       &FastGRU::observe,
             py::arg("word"), py::arg("weight")=1.0f)
        .def("separator",     &FastGRU::separator)
        .def("end",           &FastGRU::end_seq)
        .def("fit",           &FastGRU::fit,
             py::arg("epochs")=15, py::arg("lr")=0.02f,
             py::arg("max_sequences")=800)
        .def("generate",      &FastGRU::generate,
             py::arg("context_words"), py::arg("max_len")=8,
             py::arg("temperature")=1.0f,
             py::arg("forbidden_words")=vector<string>{})
        .def("n_words",       &FastGRU::n_words)
        .def("n_transitions", &FastGRU::n_transitions)
        .def("get_state",     &FastGRU::get_state)
        .def("set_state",     &FastGRU::set_state);
}
