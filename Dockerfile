FROM alpine:latest
RUN apk add --no-cache clang lld musl-dev libomp-dev
COPY . ./INFO_SRC
WORKDIR ./INFO_SRC
RUN clang -fopenmp info_txt_compiler.c -o INFO
VOLUME [ "/INFO_ARTIFACTS" ]
CMD ["./INFO"]
