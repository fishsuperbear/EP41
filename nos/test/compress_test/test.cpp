#include <iostream>
#include "dc_compress.h"

using namespace hozon::netaos::dc;

void test_compress_file_zip() {
    Compress compress;
    compress.compress_file("README.md", ZIP);
}

void test_compress_files_zip() {
    Compress compress;
    std::vector<std::string> vec;
    vec.push_back("cmake");
    vec.push_back("tools");
    vec.push_back("README.md");
    compress.compress_file("compress_zip.zip", vec, ZIP);
}

void test_decompress_unzip() {
    Compress compress;
    compress.decompress_file("README.md.zip", "result1");
    compress.decompress_file("compress_zip.zip", "result2");
}

void test_compress_gz() {
    Compress compress;
    compress.compress_file("README.md", GZ);
}

void test_decompress_gz() {
    Compress compress;
    compress.decompress_file("README.md.gz", "result3");
}

void test_compress_lz4() {
    Compress compress;
    compress.compress_file("README.md", LZ4);
}

void test_decompress_lz4() {
    Compress compress;
    compress.decompress_file("README.md.lz4", "result4");

}

void test_compress_tar_gz() {
    Compress compress;
    std::vector<std::string> vec;
    vec.push_back("cmake");
    vec.push_back("tools");
    vec.push_back("README.md");
    compress.compress_file("compress_tar.tar.gz", vec, TAR_GZ);
}

void test_decompress_tar_gz() {
    Compress compress;
    compress.decompress_file("compress_tar.tar.gz", "result5");
}

void test_compress_tar_lz4() {
    Compress compress;
    std::vector<std::string> vec;
    vec.push_back("cmake");
    vec.push_back("tools");
    vec.push_back("README.md");
    compress.compress_file("compress_tar.tar.lz4", vec, TAR_LZ4);
}

void test_decompress_tar_lz4() {
    Compress compress;
    compress.decompress_file("compress_tar.tar.lz4", "result6");
}

int main() {
    test_compress_file_zip();
    test_compress_files_zip();
    test_compress_gz();
    test_compress_lz4();
    test_compress_tar_gz();
    test_compress_tar_lz4();

    test_decompress_unzip();
    test_decompress_gz();
    test_decompress_lz4();
    test_decompress_tar_gz();
    test_decompress_tar_lz4();
    return 0;
}