#ifndef UTILITIES_H
#define UTILITIES_H

#include <stddef.h>

// opens file `filename', allocates enough bytes to store the content
// of the file and return a pointer to the file contents. The return
// value must be freed. Exits with EXIT_FAILURE and outputs to stderr
// if it was unable to open the file. This function adds a null
// terminating character at the end of the binary file. if size is a
// non null pointer, it is set to the size in bytes of the file
// including the null terminator
char* read_file(const char* filename, size_t* size);

#endif // UTILITIES_H
