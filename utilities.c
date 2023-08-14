#include "utilities.h"

#include <stdio.h>
#include <stdlib.h>

char* read_file(const char* filename, size_t* size) {
  FILE* source_file = fopen(filename, "rb");
  if(source_file == NULL){
    fprintf(stderr, "Could not open file:%s\n", filename);
    exit(EXIT_FAILURE);
  }
  fseek(source_file, 0L, SEEK_END);
  size_t file_size = (size_t)ftell(source_file);
  fseek(source_file, 0L, SEEK_SET);
  char* source_string = malloc(sizeof(*source_string)*file_size+1);
  source_string[file_size] = '\0';
  if(size != NULL)
    *size = file_size+1;

  size_t nb_bytes_read = fread((void*)source_string, 1, file_size, source_file);
  if(nb_bytes_read != file_size){
    fprintf(stderr, "Could not read file:%s properly\n", filename);
    exit(EXIT_FAILURE);
  }
  fclose(source_file);

  return source_string;
}
