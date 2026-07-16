// clang -Xpreprocessor -fopenmp -lomp -I$(brew --prefix libomp)/include -L$(brew --prefix libomp)/lib info_txt_compiler.c -o info_txt_compiler
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

#define GROW(p, c, cap, t) if ((c) >= (cap)) { (cap) = (cap) ? (cap) * 2 : 4; (p) = realloc((p), (cap) * sizeof(t)); }
#define MAX_FILES 1024

typedef struct { float *data; size_t *shape; size_t rank; const char *id; } Tensor;
typedef struct { Tensor *tensors; size_t count, capacity; } TensorList;
typedef struct { const uint8_t *start; size_t length; } ClosureSlice;
typedef struct { const char *id; ClosureSlice slice; } IndexEntry;
typedef struct { IndexEntry *entries; size_t count, capacity; } InfoIndex;

const char *IPA[] = {"p","b","t","d","k","g","m","n","ŋ","f","v","θ","ð","s","z","ʃ","ʒ","h","tʃ","dʒ","w","j","r","l","i","ɪ","e","ɛ","æ","a","ə","ʌ","u","ʊ","o","ɔ","ɑ","ɒ","aɪ","eɪ","ɔɪ","aʊ","oʊ"};

typedef struct { int pos; uint8_t depth; int transitive; } Metrics;

Metrics get_metrics(const char *w) {
    Metrics m = {3, 1, 0};
    char cmd[512];
    snprintf(cmd, 512, "tpl -l predicate.pl -l words.pl -g \"(entry('%s', P, _, _) -> (is_transitive('%s') -> T=1; T=0), phrase_depth('%s', D), format('~w,~w,~w', [P, T, D]); write('x,0,1')), halt.\" 2>/dev/null", w, w, w);
    FILE *fp = popen(cmd, "r");
    if (fp) { char r[32]; if (fgets(r, 32, fp)) {
        char *p_str = strtok(r, ","), *t_str = strtok(NULL, ","), *d_str = strtok(NULL, ",");
        if(p_str) { if(*p_str=='n') m.pos=0; else if(*p_str=='v') m.pos=1; else if(*p_str=='a') m.pos=2; }
        if(t_str) m.transitive = atoi(t_str);
        if(d_str) m.depth = (uint8_t)atoi(d_str);
    } pclose(fp); }
    return m;
}

void fill_bpe(ClosureSlice c, float *d, size_t n) {
    for (size_t i = 0, t = 0; i < c.length - 1 && t < n; i++) {
        if (!isspace(c.start[i]) && !isspace(c.start[i+1])) d[t++] = (float)((c.start[i] << 8) | c.start[i+1]);
    }
}

void fill_ipa(ClosureSlice c, float *d, size_t n) {
    for (size_t i = 0, t = 0; i < c.length && t < n; i++) {
        for (size_t p = 0; p < 43; p++) {
            size_t l = strlen(IPA[p]);
            if (i + l <= c.length && !strncmp((char*)&c.start[i], IPA[p], l)) { d[t++] = (float)(p + 1); i += l - 1; break; }
        }
    }
}

Tensor gen_tensor(ClosureSlice c, const char *id, float tfidf) {
    Tensor t = {calloc(434, sizeof(float)), calloc(2, sizeof(size_t)), 2, id};
    t.shape[0] = 1; t.shape[1] = 434;
    char w[64] = {0}; for(size_t i=0; i<c.length && i<63 && !isspace(c.start[i]); i++) w[i] = tolower(c.start[i]);
    Metrics m = get_metrics(w);
    for (int i = 0; i < 300; i++) t.data[i] = (0.01f * i) * (m.depth > 1 ? 1.5f : 1.0f);
    t.data[300 + m.pos] = 1.0f; t.data[303] = (float)m.depth;
    t.data[304] = (float)m.transitive; t.data[305] = tfidf;
    fill_bpe(c, &t.data[306], 64); fill_ipa(c, &t.data[370], 64);
    return t;
}

void proc_file(const char *p, const char *fn, InfoIndex *idx, TensorList *tl) {
    int fd = open(p, O_RDONLY); struct stat sb; fstat(fd, &sb);
    uint8_t *src = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    for (size_t i = 0, c = 0; i < sb.st_size && c < 1024; i++) {
        if (isspace(src[i])) continue;
        ClosureSlice cs = {&src[i], 0}; while (i < sb.st_size && !(i+1 < sb.st_size && src[i]=='\n' && src[i+1]=='\n')) { cs.length++; i++; }
        char b[256]; snprintf(b, 256, "%s_%zu", fn, c++);
        #pragma omp critical
        { GROW(idx->entries, idx->count, idx->capacity, IndexEntry); idx->entries[idx->count++] = (IndexEntry){strdup(b), cs}; }
        Tensor t = gen_tensor(cs, idx->entries[idx->count-1].id, 0.0f);
        #pragma omp critical
        { GROW(tl->tensors, tl->count, tl->capacity, Tensor); tl->tensors[tl->count++] = t; }
    }
    munmap(src, sb.st_size); close(fd);
}

int main(int argc, char **argv) {
    int f_cnt = 0; char *names[MAX_FILES], paths[MAX_FILES][512];
    DIR *d = opendir("../info_txt_volume/"); struct dirent *e;
    while (d && (e = readdir(d)) && f_cnt < MAX_FILES) if (strstr(e->d_name, "INFO_")) { snprintf(paths[f_cnt], 512, "../info_txt_volume/%s", e->d_name); names[f_cnt++] = strdup(e->d_name); }
    if(d) closedir(d);
    InfoIndex idx = {0}; TensorList tl = {0};
    #pragma omp parallel for
    for (int i = 0; i < f_cnt; i++) proc_file(paths[i], names[i], &idx, &tl);
    printf("Compiled %zu tensors.\n", tl.count);
    for (size_t i = 0; i < tl.count; i++) { free(tl.tensors[i].data); free(tl.tensors[i].shape); }
    for (size_t i = 0; i < idx.count; i++) free((void*)idx.entries[i].id);
    free(idx.entries); free(tl.tensors); for(int i=0; i<f_cnt; i++) free(names[i]);
    return 0;
}
