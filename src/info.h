#ifndef INFO_H
#define INFO_H

void process_token(int type, const char* value);
void link_linear_sources();
void load_info_modules();
void store_timestamped_hash(const char* hash_val);
void expand_stacked_macros(const char* op);

#endif