// brew install libomp
// clang -Xpreprocessor -fopenmp -lomp -I$(brew --prefix libomp)/include -L$(brew --prefix libomp)/lib info_txt_compiler.c -o info_txt_compiler

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <dirent.h>
#include <ctype.h>

#define IR(u, s, m) if(strstr(u, s)) printf("%s", m)
#define FLAG(n) (strcmp(argv[i], n) == 0)
#define ARG(n, l) (strncmp(argv[i], n, l) == 0)
#define MAX_FILES 1024

int main(int argc, char** argv) {
    int safe = 0, f_cnt = 0;
    char *l_mem = "inf", *l_time = "inf", *f_cont[MAX_FILES], *f_names[MAX_FILES];

    for (int i = 1; i < argc; i++) {
        if (FLAG("--safe")) safe = 1;
        else if (ARG("--linker-memory=", 16)) l_mem = argv[i] + 16;
        else if (ARG("--linker-allotted-time=", 23)) l_time = argv[i] + 23;
    }

    DIR *d = opendir("../info_txt_volume/");
    struct dirent *e;
    while (d && (e = readdir(d)) && f_cnt < MAX_FILES) {
        if (strstr(e->d_name, "INFO_") && strstr(e->d_name, ".txt")) {
            char p[512];
            snprintf(p, 512, "../info_txt_volume/%s", e->d_name);
            FILE *f = fopen(p, "rb");
            if (f) {
                fseek(f, 0, SEEK_END);
                long s = ftell(f);
                rewind(f);
                f_cont[f_cnt] = malloc(s + 1);
                fread(f_cont[f_cnt], 1, s, f);
                f_cont[f_cnt][s] = 0;
                f_names[f_cnt++] = strdup(e->d_name);
                fclose(f);
            }
        }
    }
    if(d) closedir(d);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < f_cnt; i++) {
        if (safe) {
            int l = 1, c = 1;
            for (char *p = f_cont[i]; *p; p++) {
                if (!isalnum(*p) && !strchr(" ->\n\t-", *p))
                    printf("[L-ERR] %s | %d:%d | '%c'\n", f_names[i], l, c, *p);
                (*p == '\n') ? (l++, c = 1) : c++;
            }
        }

        char *s1, *s2, *cl = strtok_r(f_cont[i], "\n\n", &s1);
        while (cl) {
            char *ln = strtok_r(cl, "\n", &s2);
            while (ln) {
                printf("[%s] %s: ", f_names[i], (ln[0]==' '||ln[0]=='\t') ? "CLOS" : "UNIT");
                IR(ln, "INCREMENT", "INC "); 
                IR(ln, "DECREMENT", "DEC "); 
                IR(ln, "->", "PTR ");
                puts("| [0,1]");
                ln = strtok_r(NULL, "\n", &s2);
            }
            cl = strtok_r(NULL, "\n\n", &s1);
        }
    }

    printf("Linker: MEM=%s TIME=%s. Compiled.\n", l_mem, l_time);
    for(int i = 0; i < f_cnt; i++) { free(f_cont[i]); free(f_names[i]); }
    return 0;
}
