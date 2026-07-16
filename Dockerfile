# Build: docker build -t info-compiler .
# Run: docker run --rm -v $(pwd)/artifacts:/INFO_ARTIFACTS info-compiler
FROM alpine:latest
RUN apk add --no-cache clang lld musl-dev libomp-dev
COPY . ./INFO_SRC
WORKDIR ./INFO_SRC
RUN clang -Xpreprocessor -fopenmp -lomp -I$(echo /usr/lib/llvm*/include) -L$(echo /usr/lib/llvm*/lib) info_txt_compiler.c -o INFO
VOLUME [ "/INFO_ARTIFACTS" ]
CMD ["./INFO"]
