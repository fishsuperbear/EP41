#pragma once
#include <iostream>
#include <vector>

namespace hozon {
namespace netaos {
namespace dc {

#define BUF_SIZE (8192)

enum FileType : int32_t {
    ZIP = 0,
    TAR_GZ = 1,
    TAR_LZ4 = 2,
    GZ = 3,
    LZ4 = 4
};

class Compress {
public:
    Compress();
    ~Compress();
    int compress_file(std::string output_folder_path, std::string output_file_name, std::vector<std::string> input_file_vec, int output_file_type, std::string is_delete_path);
    int compress_file(std::string output_folder_path, std::string output_file_name, std::string input_file_path, int output_file_type);
    int decompress_file(std::string input_file_path, std::string output_dir_path = "./");
    std::string get_result_path();

private:
    int compress_zip(std::string output_folder_path, std::string output_file_name, std::vector<std::string> input_file_vec);
    int compress_tar(std::string output_folder_path, std::string output_file_name, std::vector<std::string> input_file_vec, FileType output_file_type, std::string is_delete_path);
    int compress_zip(std::string output_folder_path, std::string output_file_name, std::string input_file_path);
    int compress_raw(std::string output_folder_path, std::string output_file_name, std::string input_file_path, FileType output_file_type);
    int decompress_unzip(std::string input_file_path, std::string output_dir_path);
    int decompress_tar(std::string input_file_path, std::string output_dir_path);
    int decompress_raw(std::string input_file_path, std::string output_dir_path);
    void get_all_file(std::string input_file_path);
    std::string get_file_name(std::string path);
    std::string get_dir_name(std::string path);
    
    std::vector<std::string> m_input_file_path_vec;
    std::string m_result_path;
    static int m_occupies_memory_size;
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon