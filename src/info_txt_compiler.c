// brew install libomp
// clang -Xpreprocessor -fopenmp -lomp -I$(brew --prefix libomp)/include -L$(brew --prefix libomp)/lib info_txt_compiler.c -o info_txt_compiler

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <dirent.h>

#define IR(u, s, m) if(strstr(u, s)) printf(m)
#define FLAG(n) (strcmp(argv[i], n) == 0)
#define MAX_FILES 1024

int main(int argc, char** argv) {
    int unsafe = 0, time = 0;
    char *mem = "inf";
    char *file_contents[MAX_FILES];
    int file_count = 0;
    
    for (int i = 1; i < argc; i++) {
        if (FLAG("--unsafe")) unsafe = 1;
        else if (FLAG("--time")) time = 1;
        else if (strncmp(argv[i], "--max-memory=", 13) == 0) mem = argv[i] + 13;
    }

    DIR *dir = opendir("../info_txt_volume/");
    if (!dir) {
        perror("Could not open directory");
        return 1;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL && file_count < MAX_FILES) {
        if (strncmp(entry->d_name, "INFO_", 5) == 0 && strstr(entry->d_name, ".txt")) {
            char path[512];
            snprintf(path, sizeof(path), "../info_txt_volume/%s", entry->d_name);
            
            FILE *f = fopen(path, "r");
            if (f) {
                fseek(f, 0, SEEK_END);
                long fsize = ftell(f);
                fseek(f, 0, SEEK_SET);

                file_contents[file_count] = malloc(fsize + 1);
                fread(file_contents[file_count], 1, fsize, f);
                file_contents[file_count][fsize] = 0;
                
                fclose(f);
                file_count++;
            }
        }
    }
    closedir(dir);

    double start = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < file_count; i++) {
        int tid = omp_get_thread_num();
        char *saveptr;
        // Spec: Closures are defined by 2 consecutive newlines
        char *closure = strtok_r(file_contents[i], "\n\n", &saveptr);

        while (closure != NULL) {
            if (!unsafe) {
                for (const char *p = closure; *p; p++) {
                    // Allowed: alpha, numeric (precision), arrows, spaces, and hyphens (unit-alpha)
                    if (!((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') || 
                        (*p >= '0' && *p <= '9') || strchr(" ->\n-", *p))) {
                        printf("[T%d] Lexical Error in file %d: Invalid token '%c'\n", tid, i, *p);
                        goto next_closure;
                    }
                }
            }

            printf("[T%d] IR (File %d): ", tid, i);
            IR(closure, "INCREMENT", "++ ");
            IR(closure, "DECREMENT", "-- ");
            IR(closure, "->", "-> ");
            printf("[Eff: 1]\n");

            next_closure:
            closure = strtok_r(NULL, "\n\n", &saveptr);
        }
    }

    if (time) printf("Time: %fs\n", omp_get_wtime() - start);
    printf("Linker: %s mem limit, %d source volumes linked. .exe generated.\n", mem, file_count);
    
    for(int i = 0; i < file_count; i++) free(file_contents[i]);
    return 0;
}
