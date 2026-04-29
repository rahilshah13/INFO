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
#define MAX_CONTENT 4096

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
                file_contents[file_count] = malloc(MAX_CONTENT);
                if (fgets(file_contents[file_count], MAX_CONTENT, f)) {
                    // Remove newline if exists
                    file_contents[file_count][strcspn(file_contents[file_count], "\r\n")] = 0;
                    file_count++;
                }
                fclose(f);
            }
        }
    }
    
    closedir(dir);
    double start = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < file_count; i++) {
        int tid = omp_get_thread_num();
        const char *current_line = file_contents[i];

        if (!unsafe) {
            for (const char *p = current_line; *p; p++) {
                if (!((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') || strchr(" ->", *p))) {
                    printf("[T%d] Invalid token in file %d: P%ld\n", tid, i, p - current_line);
                    goto next;
                }
            }
        }

        printf("[T%d] IR %d: ", tid, i);
        IR(current_line, "INCREMENT", "++ ");
        IR(current_line, "DECREMENT", "-- ");
        IR(current_line, "->", "-> ");
        printf("[Eff: 1]\n");
        next:;
    }

    if (time) printf("Time: %fs\n", omp_get_wtime() - start);
    printf("Linker: %s mem limit, %d files processed.\n", mem, file_count);
    for(int i = 0; i < file_count; i++) free(file_contents[i]);
    return 0;
}
