# Build: docker build -t info-compiler .
# Run: docker run --rm -v $(pwd)/artifacts:/INFO_ARTIFACTS -v $(pwd)/volume:/info_txt_volume info-compiler
FROM alpine:latest
RUN apk add --no-cache clang lld musl-dev libomp-dev python3 py3-requests py3-mwparserfromhell py3-lxml git
COPY . ./INFO_SRC
WORKDIR ./INFO_SRC

# Clone Prolog files and move to root for compiler access
RUN mkdir -p temp_facts && \
    git clone --depth 1 --filter=blob:none --sparse https://github.com/rahilshah13/FACTS.git temp_facts && \
    cd temp_facts && \
    git sparse-checkout set DICTIONARY/LANGUAGES/ENGLISH && \
    mv DICTIONARY/LANGUAGES/ENGLISH/predicates.pl ../ && \
    mv DICTIONARY/LANGUAGES/ENGLISH/words.pl ../ && \
    cd .. && rm -rf temp_facts

# Compile compiler
RUN clang -Xpreprocessor -fopenmp -lomp -I$(echo /usr/lib/llvm*/include) -L$(echo /usr/lib/llvm*/lib) info_txt_compiler.c -o INFO

VOLUME [ "/INFO_ARTIFACTS", "/info_txt_volume" ]

# Run download script then compiler
CMD ["sh", "-c", "python3 _download_INFO.py && ./INFO"]
