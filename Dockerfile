FROM alpine:latest
COPY . ./INFO_SRC
WORKDIR ./INFO_SRC
RUN MAKE
RUN apk add gcc buildtools
VOLUME [ "/INFO_ARTIFACTS" ]
CMD ./INFO