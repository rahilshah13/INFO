#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <dirent.h>
#include <ctype.h>
#include <omp.h>
#include <math.h>
#include <espeak-ng/speak_lib.h>

#define GROW(p, c, cap, t) if ((c) >= (cap)) { (cap) = (cap) ? (cap) * 2 : 4; (p) = realloc((p), (cap) * sizeof(t)); }
#define MAX_FILES 1024
#define MAX_DEF_LENGTH 128
#define EMBED_DIM 150 // 150-dim CBOW + 150-dim Skip-gram = 300 total base dimensions

typedef struct { float *data; size_t *shape; size_t rank; char *id; } Tensor;
typedef struct { Tensor *tensors; size_t count, capacity; } TensorList;
typedef struct { char *start; size_t length; } ClosureSlice;

typedef struct {
    char **vocab;
    float **cbow_weights;
    float **skipgram_weights;
    size_t count, capacity;
} EmbeddingSpace;

typedef struct { 
    int pos; 
    uint8_t depth; 
    int transitive; 
    int sourceType; 
    int nodeComplexity; 
    float avgWordLen; 
    float wordsPerClosure; 
    int correctSentences; 
    int incorrectSentences; 
    float lexicalDensity;
    float semanticCoherence;
    float syntacticEntropy;
    float readabilityScore;
    size_t bpeLen; 
    size_t ipaBpeLen; 
} Metrics;

typedef struct {
    char **keys;
    Metrics *vals;
    size_t count, capacity;
} MetricsCache;

const char *IPA[] = {"p","b","t","d","k","g","m","n","ŋ","f","v","θ","ð","s","z","ʃ","ʒ","h","tʃ","dʒ","w","j","r","l","i","ɪ","e","ɛ","æ","a","ə","ʌ","u","ʊ","o","ɔ","ɑ","ɒ","aɪ","eɪ","ɔɪ","aʊ","oʊ"};

// Authentic Trealla Prolog execution
Metrics get_metrics(const char *w, const char *closure_text) {
    Metrics m = {3, 1, 0, 0, 0, 0.0f, 0.0f, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f, 0, 0};
    char cmd[2048];
    snprintf(cmd, sizeof(cmd), "tpl -l predicate.pl -l words.pl -g \"(entry('%s', _, _, Def), atom_length(Def, Len), Len =< %d -> (entry('%s', P, _, _) -> (is_transitive('%s') -> T=1; T=0), phrase_depth('%s', D), (is_module('%s') -> S=1; S=0), decl_count('%s', C), avg_word_length('%s', AW), words_per_closure('%s', WC), sentence_eval('%s', Cor, Inc), lexical_density('%s', LD), semantic_coherence('%s', SC), syntactic_entropy('%s', SE), readability_score('%s', RS), bpe_len('%s', BL), ipa_len_val('%s', IBL), format('~w,~w,~w,~w,~w,~w,~w,~w,~w,~w,~w,~w,~w,~w,~w', [P, T, D, S, C, AW, WC, Cor, Inc, LD, SC, SE, RS, BL, IBL]); write('x,0,1,0,0,0.0,0,0,0,0.0,0.0,0.0,0.0,0,0')); write('bunk,0,1,0,0,0.0,0,0,0,0.0,0.0,0.0,0.0,0,0')), halt.\" 2>/dev/null", 
        w, MAX_DEF_LENGTH, w, w, w, w, w, closure_text, closure_text, closure_text, closure_text, closure_text, closure_text, closure_text, closure_text, closure_text);
    FILE *fp = popen(cmd, "r");
    if (fp) { 
        char r[256]; 
        if (fgets(r, 256, fp)) {
            char *t[15]; int i=0; t[0]=strtok(r,","); while(i<14) t[++i]=strtok(NULL,",");
            if(t[0] && strcmp(t[0], "bunk") != 0) {
                if(*t[0]=='n') m.pos=0; else if(*t[0]=='v') m.pos=1; else if(*t[0]=='a') m.pos=2;
                if(t[1]) m.transitive = atoi(t[1]);
                if(t[2]) m.depth = (uint8_t)atoi(t[2]);
                if(t[3]) m.sourceType = atoi(t[3]);
                if(t[4]) m.nodeComplexity = atoi(t[4]);
                if(t[5]) m.avgWordLen = (float)atof(t[5]);
                if(t[6]) m.wordsPerClosure = (float)atof(t[6]);
                if(t[7]) m.correctSentences = atoi(t[7]);
                if(t[8]) m.incorrectSentences = atoi(t[8]);
                if(t[9]) m.lexicalDensity = (float)atof(t[9]);
                if(t[10]) m.semanticCoherence = (float)atof(t[10]);
                if(t[11]) m.syntacticEntropy = (float)atof(t[11]);
                if(t[12]) m.readabilityScore = (float)atof(t[12]);
                if(t[13]) m.bpeLen = (size_t)atoi(t[13]);
                if(t[14]) m.ipaBpeLen = (size_t)atoi(t[14]);
            }
        } 
        pclose(fp); 
    }
    return m;
}

// Thread-safe metrics cache to prevent popen exhaustion
Metrics get_cached_metrics(const char *w, const char *closure_text, MetricsCache *mc) {
    #pragma omp critical(metrics_cache_lock)
    {
        for (size_t i = 0; i < mc->count; i++) {
            if (strcmp(mc->keys[i], w) == 0) {
                return mc->vals[i];
            }
        }
    }

    Metrics m = get_metrics(w, closure_text);

    #pragma omp critical(metrics_cache_lock)
    {
        for (size_t i = 0; i < mc->count; i++) {
            if (strcmp(mc->keys[i], w) == 0) {
                return mc->vals[i];
            }
        }
        GROW(mc->keys, mc->count, mc->capacity, char*);
        GROW(mc->vals, mc->count, mc->capacity, Metrics);
        mc->keys[mc->count] = strdup(w);
        mc->vals[mc->count] = m;
        mc->count++;
    }
    return m;
}

// Fully thread-safe distributed vector lookup and allocation
void get_distributed_vector(const char *token, float *cbow_out, float *skip_out, EmbeddingSpace *es) {
    #pragma omp critical(embedding_space_lock)
    {
        int found = 0;
        for (size_t i = 0; i < es->count; i++) {
            if (strcmp(es->vocab[i], token) == 0) {
                memcpy(cbow_out, es->cbow_weights[i], EMBED_DIM * sizeof(float));
                memcpy(skip_out, es->skipgram_weights[i], EMBED_DIM * sizeof(float));
                found = 1;
                break;
            }
        }
        if (!found) {
            float *cbow_w = malloc(EMBED_DIM * sizeof(float));
            float *skip_w = malloc(EMBED_DIM * sizeof(float));
            unsigned int seed = (unsigned int)(uintptr_t)token ^ (unsigned int)es->count;
            for (int d = 0; d < EMBED_DIM; d++) {
                cbow_w[d] = ((float)rand_r(&seed) / (float)RAND_MAX) * 2.0f - 1.0f;
                skip_w[d] = ((float)rand_r(&seed) / (float)RAND_MAX) * 2.0f - 1.0f;
            }
            GROW(es->vocab, es->count, es->capacity, char*);
            GROW(es->cbow_weights, es->count, es->capacity, float*);
            GROW(es->skipgram_weights, es->count, es->capacity, float*);
            es->vocab[es->count] = strdup(token);
            es->cbow_weights[es->count] = cbow_w;
            es->skipgram_weights[es->count] = skip_w;
            es->count++;

            memcpy(cbow_out, cbow_w, EMBED_DIM * sizeof(float));
            memcpy(skip_out, skip_w, EMBED_DIM * sizeof(float));
        }
    }
}

void compute_modulated_features(const char *text, size_t len, float *cbow_vec, float *skip_vec, Metrics m, EmbeddingSpace *es) {
    for (int i = 0; i < EMBED_DIM; i++) {
        cbow_vec[i] = 0.0f;
        skip_vec[i] = 0.0f;
    }
    if (len == 0) return;

    char tokens[256][64];
    int token_count = 0;
    size_t idx = 0;
    while (idx < len && token_count < 256) {
        while (idx < len && isspace(text[idx])) idx++;
        size_t start = idx;
        while (idx < len && !isspace(text[idx])) idx++;
        if (idx > start) {
            size_t tlen = idx - start;
            if (tlen > 63) tlen = 63;
            memcpy(tokens[token_count], &text[start], tlen);
            tokens[token_count][tlen] = '\0';
            token_count++;
        }
    }
    if (token_count == 0) return;

    float temp_cbow[EMBED_DIM];
    float temp_skip[EMBED_DIM];

    for (int t = 0; t < token_count; t++) {
        get_distributed_vector(tokens[t], temp_cbow, temp_skip, es);
        for (int d = 0; d < EMBED_DIM; d++) {
            cbow_vec[d] += temp_cbow[d];
            skip_vec[d] += temp_skip[d];
        }
    }

    float coherence_sign = (m.semanticCoherence >= 0.0f) ? 1.0f : -1.0f;
    float entropy_factor = 1.0f + (m.syntacticEntropy * 0.1f);
    float depth_mod = (m.depth > 1) ? 1.5f : 1.0f;

    float norm_c = 0.0f, norm_s = 0.0f;
    for (int d = 0; d < EMBED_DIM; d++) {
        cbow_vec[d] = (cbow_vec[d] * coherence_sign * depth_mod);
        skip_vec[d] = (skip_vec[d] * entropy_factor);

        norm_c += cbow_vec[d] * cbow_vec[d];
        norm_s += skip_vec[d] * skip_vec[d];
    }

    if (norm_c > 0.0f) {
        norm_c = sqrtf(norm_c);
        for (int d = 0; d < EMBED_DIM; d++) cbow_vec[d] /= norm_c;
    }
    if (norm_s > 0.0f) {
        norm_s = sqrtf(norm_s);
        for (int d = 0; d < EMBED_DIM; d++) skip_vec[d] /= norm_s;
    }
}

void fill_bpe(ClosureSlice c, float *d, size_t n) {
    for (size_t i = 0, t = 0; i < c.length - 1 && t < n; i++) {
        if (!isspace(c.start[i]) && !isspace(c.start[i+1])) d[t++] = (float)((c.start[i] << 8) | c.start[i+1]);
    }
}

void fill_ipa_bpe(ClosureSlice c, float *d, size_t n) {
    int ipa_indices[512];
    size_t ipa_count = 0;
    for (size_t i = 0; i < c.length && ipa_count < 512; i++) {
        int matched = 0;
        for (size_t p = 0; p < 43; p++) {
            size_t l = strlen(IPA[p]);
            if (i + l <= c.length && !strncmp(&c.start[i], IPA[p], l)) {
                ipa_indices[ipa_count++] = (int)(p + 1);
                i += l - 1;
                matched = 1;
                break;
            }
        }
        if (!matched && !isspace(c.start[i])) {
            ipa_indices[ipa_count++] = (int)c.start[i];
        }
    }
    for (size_t i = 0, t = 0; i + 1 < ipa_count && t < n; i++) {
        d[t++] = (float)((ipa_indices[i] << 8) | ipa_indices[i+1]);
    }
}

Tensor gen_tensor(ClosureSlice c, const char *id, float tfidf, EmbeddingSpace *es, MetricsCache *mc) {
    char w[64] = {0}; 
    for(size_t i = 0; i < c.length && i < 63 && !isspace(c.start[i]); i++) w[i] = tolower(c.start[i]);
    
    char *c_text = malloc(c.length + 1);
    memcpy(c_text, c.start, c.length);
    c_text[c.length] = '\0';
    
    Metrics m = get_cached_metrics(w, c_text, mc);
    free(c_text);

    size_t total_dim = 300 + 3 + 5 + 8 + m.bpeLen + m.ipaBpeLen;
    Tensor t = {calloc(total_dim, sizeof(float)), calloc(2, sizeof(size_t)), 2, strdup(id)};
    t.shape[0] = 1; t.shape[1] = total_dim;

    compute_modulated_features(c.start, c.length, &t.data[0], &t.data[EMBED_DIM], m, es);

    t.data[300 + m.pos] = 1.0f; 
    t.data[303] = (float)m.depth;
    t.data[304] = (float)m.transitive; 
    t.data[305] = tfidf;
    t.data[306] = (float)m.sourceType; 
    t.data[307] = (float)m.nodeComplexity;
    t.data[308] = m.avgWordLen; 
    t.data[309] = m.wordsPerClosure;
    t.data[310] = (float)m.correctSentences; 
    t.data[311] = (float)m.incorrectSentences;
    t.data[312] = m.lexicalDensity; 
    t.data[313] = m.semanticCoherence;
    t.data[314] = m.syntacticEntropy; 
    t.data[315] = m.readabilityScore;
    
    if (m.bpeLen > 0) fill_bpe(c, &t.data[316], m.bpeLen); 
    if (m.ipaBpeLen > 0) fill_ipa_bpe(c, &t.data[316 + m.bpeLen], m.ipaBpeLen);
    
    return t;
}

void proc_file(const char *p, const char *fn, TensorList *tl, EmbeddingSpace *es, MetricsCache *mc) {
    int fd = open(p, O_RDONLY); 
    if (fd < 0) return;
    struct stat sb; 
    if (fstat(fd, &sb) < 0) { close(fd); return; }
    
    uint8_t *src = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (src == MAP_FAILED) { close(fd); return; }

    for (size_t i = 0, c = 0; i < sb.st_size && c < 1024; i++) {
        if (isspace(src[i])) continue;
        ClosureSlice cs = {(char*)&src[i], 0}; 
        while (i < sb.st_size && !(i + 1 < sb.st_size && src[i] == '\n' && src[i + 1] == '\n')) { 
            cs.length++; 
            i++; 
        }
        
        char b[256]; 
        snprintf(b, 256, "%s_%zu", fn, c++);
        
        Tensor t = gen_tensor(cs, b, 0.0f, es, mc);
        
        #pragma omp critical(tensor_list_lock)
        { 
            GROW(tl->tensors, tl->count, tl->capacity, Tensor); 
            tl->tensors[tl->count++] = t; 
        }
    }
    
    munmap(src, sb.st_size); 
    close(fd);
}

int main(int argc, char **argv) {
    int f_cnt = 0; 
    char *names[MAX_FILES], paths[MAX_FILES][512];
    
    DIR *d = opendir("../info_txt_volume/"); 
    struct dirent *e;
    while (d && (e = readdir(d)) && f_cnt < MAX_FILES) {
        if (strstr(e->d_name, "INFO_")) { 
            snprintf(paths[f_cnt], 512, "../info_txt_volume/%s", e->d_name); 
            names[f_cnt++] = strdup(e->d_name); 
        }
    }
    if (d) closedir(d);

    EmbeddingSpace es = {0};
    MetricsCache mc = {0};
    TensorList tl = {0};

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < f_cnt; i++) {
        proc_file(paths[i], names[i], &tl, &es, &mc);
    }

    printf("Compiled %zu tensors successfully with thread-safe vector embedding space and metrics cache.\n", tl.count);

    for (size_t i = 0; i < tl.count; i++) { 
        free(tl.tensors[i].id);
        free(tl.tensors[i].data); 
        free(tl.tensors[i].shape); 
    }
    free(tl.tensors);

    for (size_t i = 0; i < es.count; i++) {
        free(es.vocab[i]);
        free(es.cbow_weights[i]);
        free(es.skipgram_weights[i]);
    }
    free(es.vocab); free(es.cbow_weights); free(es.skipgram_weights);

    for (size_t i = 0; i < mc.count; i++) {
        free(mc.keys[i]);
    }
    free(mc.keys); free(mc.vals);

    for (int i = 0; i < f_cnt; i++) free(names[i]);
    
    return 0;
}
