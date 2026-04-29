#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define IR(u, s, m) if(strstr(u, s)) printf(m)
#define FLAG(n) (strcmp(argv[i], n) == 0)

int main(int argc, char** argv) {
    int unsafe = 0, time = 0;
    char *mem = "inf";
    
    for (int i = 1; i < argc; i++) {
        if (FLAG("--unsafe")) unsafe = 1;
        else if (FLAG("--time")) time = 1;
        else if (strncmp(argv[i], "--max-memory=", 13) == 0) mem = argv[i] + 13;
    }

    const char* cls[] = {"INCREMENT unit-alpha -> result", "DECREMENT unit-beta", "INCREMENT unit-gamma -> effect"};
    double start = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < 3; i++) {
        int tid = omp_get_thread_num();
        if (!unsafe) {
            for (const char *p = cls[i]; *p; p++) {
                if (!((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') || strchr(" ->", *p))) {
                    printf("[T%d] Invalid token: L1, P%ld\n", tid, p - cls[i]);
                    goto next;
                }
            }
        }
        printf("[T%d] IR %d: ", tid, i);
        IR(cls[i], "INCREMENT", "++ ");
        IR(cls[i], "DECREMENT", "-- ");
        IR(cls[i], "->", "-> ");
        printf("[Eff: 1]\n");
        next:;
    }

    if (time) printf("Time: %fs\n", omp_get_wtime() - start);
    printf("Linker: %s mem limit, .exe generated.\n", mem);
    return 0;
}
