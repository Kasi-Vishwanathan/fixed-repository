#include "zfstream.h"

int main() {
    // Construct a stream object with this filebuffer.
    gzofstream os(1, std::ios::out);

    // Compressed output to stdout (test with 'test | zcat')
    os << "Hello, Mommy" << std::endl;

    // Set compression level using function call
    setcompressionlevel(os, Z_NO_COMPRESSION);
    os << "hello, hello, hi, ho!" << std::endl;

    // Set compression level and chain output
    setcompressionlevel(os, Z_DEFAULT_COMPRESSION)
        << "I'm compressing again" << std::endl;

    os.close();
    return 0;
}