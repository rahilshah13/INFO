#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

typedef enum { TOKEN_NEWLINE, TOKEN_ARROW, TOKEN_HYPHEN, TOKEN_WORD } TokenType;

const char* SYMBOLS[] = {"INCREMENT", "DECREMENT", "ARROW"};
const char* MINIFIED[] = {"++", "--", "->"};

void process_unit(const char* unit, int unit_id) {
    int tid = omp_get_thread_num();
    
    if (strstr(unit, "++")) printf("[Thread %d] Unit %d: Macro Stacked Inc\n", tid, unit_id);
    
    printf("[Thread %d] Minified IR %d: ", tid, unit_id);
    for (int i = 0; i < 3; i++) {
        if (strstr(unit, SYMBOLS[i])) printf("%s ", MINIFIED[i]);
    }
    printf("\n");
}

int main() {
    const char* closures[] = {
        "INCREMENT unit-alpha -> result",
        "DECREMENT unit-beta",
        "INCREMENT unit-gamma -> effect"
    };
    int total_units = 3;

    printf("--- info.txt Parallel Compiler (OpenMP) ---\n");
    printf("Processing %d units on %d cores...\n\n", total_units, omp_get_num_procs());

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < total_units; i++) {
        process_unit(closures[i], i);
    }

    return 0;
}
