#include <stdio.h>
#include <time.h>

void store_timestamped_hash(const char* hash_val) {
    time_t now = time(NULL);
    char *timestamp = ctime(&now);
    
    FILE *f = fopen("artifacts_volume/build_index.log", "a");
    if (f) {
        fprintf(f, "Build Hash: %s | Timestamp: %s", hash_val, timestamp);
        fclose(f);
    }
}