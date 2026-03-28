CC = gcc
CFLAGS = -Wall
TARGET = INFO

all: $(TARGET)

$(TARGET): info.txt_compiler.c info.txt_linker.c info.txt_index.c info.txt_modules.c info.txt_assembler.c
	$(CC) $(CFLAGS) -o $(TARGET) $^
	@echo "Build complete. Running verification..."
	./verify_hashes.sh

clean:
	rm -f $(TARGET)
	rm -rf artifacts_volume/*