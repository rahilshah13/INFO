#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum { TOKEN_NEWLINE, TOKEN_INDENT, TOKEN_SPACE, TOKEN_COMMA, TOKEN_HYPHEN, TOKEN_ARROW, TOKEN_WORD } TokenType;

void expand_stacked_macros(const char* op) {
    if (strcmp(op, "++") == 0) {
        printf("Macro Expansion: Stacked Increment\n");
    } else if (strcmp(op, "--") == 0) {
        printf("Macro Expansion: Stacked Decrement\n");
    }
}

void process_token(TokenType type, const char* value) {
    switch(type) {
        case TOKEN_NEWLINE: printf("[Token: Newline]\n"); break;
        case TOKEN_INDENT:  printf("[Token: Indent]\n"); break;
        case TOKEN_ARROW:   printf("[Token: Arrow ->]\n"); break;
        case TOKEN_HYPHEN:  printf("[Token: Hyphen -]\n"); break;
        default: printf("Value: %s\n", value);
    }
}

int main(int argc, char *argv[]) {
    printf("--- info.txt Compiler Starting ---\n");
    FILE *file = fopen("info.txt", "r");
    if (!file) {
        perror("Error opening info.txt");
        return 1;
    }

    // Basic logic to demonstrate "Newline is a token"
    char c;
    while ((c = fgetc(file)) != EOF) {
        if (c == '\n') process_token(TOKEN_NEWLINE, NULL);
        else if (c == ' ') process_token(TOKEN_SPACE, " ");
        else if (c == '-') {
            char next = fgetc(file);
            if (next == '>') process_token(TOKEN_ARROW, "->");
            else { ungetc(next, file); process_token(TOKEN_HYPHEN, "-"); }
        }
    }

    fclose(file);
    return 0;
}