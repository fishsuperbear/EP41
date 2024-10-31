#include "dc_compress.h"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <dirent.h>
#include <string.h>
#include <chrono>
#include <thread>
#include "archive/archive.h"
#include "archive/archive_entry.h"
#include "zipper/zipper.h"
#include "zipper/unzipper.h"
#include "zipper/tools.h"
#include "rcpputils/filesystem_helper.hpp"
#include "utils/include/dc_logger.hpp"
#include "utils/include/path_utils.h"

namespace hozon {
namespace netaos {
namespace dc {

using namespace zipper;
#define MAX_OCCUPIES_MEMORY_SIZE (100 * 1024 * 1024)
#define SLEEP_TIME (1)
#define PERSISTENT_DIRECTORY ("/opt/usr/col/runinfo/compress/")
int Compress::m_occupies_memory_size = 0;

Compress::Compress() {}

Compress::~Compress() {}

// 多个文件打包+压缩函数
// output_file_path为压缩输出文件路径，注意output_path_path的后缀应与打包及压缩后文件格式一致
// input_file_vec为待处理的文件、文件夹列表
// output_file_type代表打包及压缩格式
int Compress::compress_file(std::string output_folder_path, std::string output_file_name, std::vector<std::string> input_file_vec, int output_file_type, std::string is_delete_path) {
    int compress_result = 0;
    rcpputils::fs::path db_path(PERSISTENT_DIRECTORY);
    if (!db_path.is_directory()) {
        bool dir_created = rcpputils::fs::create_directories(db_path);
        if (!dir_created) {
            DC_SERVER_LOG_ERROR << "failed to create output directory: " << db_path.string();
            return -1;
        }
    }
    switch (output_file_type)
    {
    case ZIP:
        compress_result = compress_zip(output_folder_path, output_file_name, input_file_vec);
        break;
    case TAR_GZ:
    case TAR_LZ4:
        compress_result = compress_tar(output_folder_path, output_file_name, input_file_vec, (FileType)output_file_type, is_delete_path);
        break;
    default:
        DC_SERVER_LOG_ERROR << "please check file type";
        return -1;
    }
    return compress_result;
}

// 单个文件压缩函数
// input_file_path为待压缩的文件路径
// output_file_type代表压缩格式
// 压缩好的文件路径为 input_file_path + "." + output_file_type
int Compress::compress_file(std::string output_folder_path, std::string output_file_name, std::string input_file_path, int output_file_type) {
    int compress_result = 0;
    rcpputils::fs::path db_path(PERSISTENT_DIRECTORY);
    if (!db_path.is_directory()) {
        bool dir_created = rcpputils::fs::create_directories(db_path);
        if (!dir_created) {
            DC_SERVER_LOG_ERROR << "failed to create output directory: " << db_path.string();
            return -1;
        }
    }
    switch (output_file_type)
    {
    case ZIP:
        compress_result = compress_zip(output_folder_path, output_file_name, input_file_path);
        break;
    case GZ: 
    case LZ4: 
        compress_result = compress_raw(output_folder_path, output_file_name, input_file_path, (FileType)output_file_type);
        break;
    default:
        DC_SERVER_LOG_ERROR << "please check file type";
        return -1;
    }
    return compress_result;
}

// 解压缩函数
// input_file_path为待解压的文件路径
// output_dir_path为解压路径
int Compress::decompress_file(std::string input_file_path, std::string output_dir_path) {
    int decompress_result = 0;
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm* now_tm = std::gmtime(&now_time);
    std::ostringstream oss;
    oss << std::put_time(now_tm, "%Y%m%d_%H%M%S");
    std::string timestamp = oss.str();
    if (input_file_path.find(".zip") == (input_file_path.size() - 4)) {
        if (output_dir_path == "") {
            output_dir_path = input_file_path.substr(0, input_file_path.size() - 4) + "_" + timestamp;
        }
        decompress_result = decompress_unzip(input_file_path, output_dir_path);
    } else if (input_file_path.find(".tar.gz") == (input_file_path.size() - 7)) {
        if (output_dir_path == "") {
            output_dir_path = input_file_path.substr(0, input_file_path.size() - 7) + "_" + timestamp;
        }
        decompress_result = decompress_tar(input_file_path, output_dir_path);
    } else if (input_file_path.find(".tar.lz4") == (input_file_path.size() - 8)) {
        if (output_dir_path == "") {
            output_dir_path = input_file_path.substr(0, input_file_path.size() - 8) + "_" + timestamp;
        }
        decompress_result = decompress_tar(input_file_path, output_dir_path);
    } else if (input_file_path.find(".gz") == (input_file_path.size() - 3)) {
        if (output_dir_path == "") {
            output_dir_path = input_file_path.substr(0, input_file_path.size() - 3) + "_" + timestamp;
        }
        decompress_result = decompress_raw(input_file_path, output_dir_path);
    } else if (input_file_path.find(".lz4") == (input_file_path.size() - 4)) {
        if (output_dir_path == "") {
            output_dir_path = input_file_path.substr(0, input_file_path.size() - 4) + "_" + timestamp;
        }
        decompress_result = decompress_raw(input_file_path, output_dir_path);
    } else {
        DC_SERVER_LOG_ERROR << "please check compress file type";
        return -1;
    }
    return decompress_result;
}

int Compress::compress_zip(std::string output_folder_path, std::string output_file_name, std::vector<std::string> input_file_vec) {
    if (output_file_name.find(".zip") != (output_file_name.size() - 4)) {
        DC_SERVER_LOG_ERROR << "please check output file type";
        return -1;
    }
    rcpputils::fs::path db_path(output_folder_path);
    if (!db_path.is_directory()) {
        bool dir_created = rcpputils::fs::create_directories(db_path);
        if (!dir_created) {
            DC_SERVER_LOG_ERROR << "failed to create output directory: " << db_path.string();
            return -1;
        }
    }
    std::string temp_output_file_path = PathUtils::getFilePath(PERSISTENT_DIRECTORY, output_file_name + ".temp");
    std::string output_file_path = PathUtils::getFilePath(output_folder_path, output_file_name);
    Zipper zipper(temp_output_file_path);
    for (auto input_file_path : input_file_vec) {
        bool add_zip_result = true;
        rcpputils::fs::path db_path(input_file_path);
        if (db_path.is_directory()) {
            add_zip_result = zipper.add(input_file_path);
        } else {
            std::ifstream input_file_fs(input_file_path);
            add_zip_result = zipper.add(input_file_fs, input_file_path);
            input_file_fs.close();
        }
        if (add_zip_result == false) {
            DC_SERVER_LOG_ERROR << "file: " << input_file_path << "compress failed";
            return -1;
        }
    }
    zipper.close();
    PathUtils::renameFile(temp_output_file_path, output_file_path);
    m_result_path = output_file_path;
    return 0;
}

int Compress::compress_zip(std::string output_folder_path, std::string output_file_name, std::string input_file_path) {
    if (output_file_name.find(".zip") != (output_file_name.size() - 4)) {
        DC_SERVER_LOG_ERROR << "please check output file type";
        return -1;
    }
    rcpputils::fs::path db_path(output_folder_path);
    if (!db_path.is_directory()) {
        bool dir_created = rcpputils::fs::create_directories(db_path);
        if (!dir_created) {
            DC_SERVER_LOG_ERROR << "failed to create output directory: " << db_path.string();
            return -1;
        }
    }
    std::string temp_output_file_path = PathUtils::getFilePath(PERSISTENT_DIRECTORY, output_file_name + ".temp");
    std::string output_file_path = PathUtils::getFilePath(output_folder_path, output_file_name);
    Zipper zipper(temp_output_file_path);
    std::ifstream input_file_fs(input_file_path);
    bool add_zip_result = zipper.add(input_file_fs, input_file_path);
    input_file_fs.close();
    if (add_zip_result == false) {
        DC_SERVER_LOG_ERROR << "file: " << input_file_path << "compress failed";
        zipper.close();
        return -1;
    }
    zipper.close();
    PathUtils::renameFile(temp_output_file_path, output_file_path);
    m_result_path = output_file_path;
    return 0;
}

int Compress::compress_tar(std::string output_folder_path, std::string output_file_name, std::vector<std::string> input_file_vec, FileType output_file_type, std::string is_delete_path) {
    rcpputils::fs::path db_path(output_folder_path);
    if (!db_path.is_directory()) {
        bool dir_created = rcpputils::fs::create_directories(db_path);
        if (!dir_created) {
            DC_SERVER_LOG_ERROR << "failed to create output directory: " << db_path.string();
            return -1;
        }
    }
    struct archive *arch = archive_write_new();
    if (output_file_type == TAR_GZ) {
        if (output_file_name.find(".tar.gz") != (output_file_name.size() - 7)) {
            DC_SERVER_LOG_ERROR << "please check output file type";
            archive_write_free(arch);
            return -1;
        }
        archive_write_add_filter_gzip(arch);
    } else {
        if (output_file_name.find(".tar.lz4") != (output_file_name.size() - 8)) {
            DC_SERVER_LOG_ERROR << "please check output file type";
            archive_write_free(arch);
            return -1;
        }
        archive_write_add_filter_lz4(arch);
    }
    archive_write_set_format_ustar(arch);
    int archive_result = 0;
    std::string temp_output_file_path = PathUtils::getFilePath(PERSISTENT_DIRECTORY, output_file_name + ".temp");
    std::string output_file_path = PathUtils::getFilePath(output_folder_path, output_file_name);
    archive_result = archive_write_open_filename(arch, temp_output_file_path.data());
    if (archive_result != ARCHIVE_OK) {
        DC_SERVER_LOG_ERROR << "error opening archive: " << archive_error_string(arch);
        archive_write_free(arch);
        return archive_result;
    }
    m_input_file_path_vec.clear();
    for (auto input_file : input_file_vec) {
        get_all_file(input_file);
    }
    for (auto input_file_path : m_input_file_path_vec) {
        std::ifstream input_file_fs(input_file_path, std::ios::in | std::ios::binary | std::ios::ate);
        input_file_fs.seekg(0, std::ios::end);
        int64_t input_file_size = input_file_fs.tellg();
        input_file_fs.seekg(0, std::ios::beg);
        struct archive_entry *entry = archive_entry_new();
        if (is_delete_path == "1") {
            archive_entry_set_pathname(entry, PathUtils::getFileName(input_file_path).data());
        } else {
            archive_entry_set_pathname(entry, input_file_path.data());
        }
        archive_entry_set_size(entry, input_file_size); // 文件大小，字节为单位
        archive_entry_set_filetype(entry, AE_IFREG); // 文件类型，这里是普通文件
        archive_entry_set_perm(entry, 0664); // 文件权限，这里是读写
        archive_result = archive_write_header(arch, entry);
        if (archive_result != ARCHIVE_OK) {
            DC_SERVER_LOG_ERROR << "error write header: " << archive_error_string(arch);
            input_file_fs.close();
            archive_entry_free(entry);
            archive_write_free(arch);
            return archive_result;
        }
        archive_entry_free(entry);
        char buf[BUF_SIZE];
        while (1) {
            input_file_fs.read(buf, BUF_SIZE);
            int read_size = input_file_fs.gcount();
            if (read_size == 0) {
                break;
            }
            m_occupies_memory_size += read_size;
            if (m_occupies_memory_size >= MAX_OCCUPIES_MEMORY_SIZE) {
                std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_TIME));
                m_occupies_memory_size = 0;
            }
            archive_result = archive_write_data(arch, buf, read_size);
            if (archive_result != read_size) { // 写入数据块，包括数据和大小
                if (archive_result < ARCHIVE_OK) {
                    DC_SERVER_LOG_ERROR << input_file_path << " error write data: " << archive_error_string(arch);
                    input_file_fs.close();
                    archive_write_free(arch);
                    return archive_result;
                } else {
                    DC_SERVER_LOG_WARN << input_file_path << " file changed as we read it";
                    break;
                }
            }
            if (input_file_fs.eof()) {
                break;
            }
        }
        input_file_fs.close();
    }
    archive_result = archive_write_finish_entry(arch);
    if (archive_result != ARCHIVE_OK) { // 结束写入条目
        DC_SERVER_LOG_ERROR << "error finishing entry: " << archive_error_string(arch);
        archive_write_free(arch);
        return archive_result;
    }
    archive_write_close(arch);
    archive_write_free(arch); // 释放资源
    PathUtils::renameFile(temp_output_file_path, output_file_path);
    m_result_path = output_file_path;
    return 0;
}

int Compress::compress_raw(std::string output_folder_path, std::string output_file_name, std::string input_file_path, FileType output_file_type) {
    rcpputils::fs::path db_path(output_folder_path);
    if (!db_path.is_directory()) {
        bool dir_created = rcpputils::fs::create_directories(db_path);
        if (!dir_created) {
            DC_SERVER_LOG_ERROR << "failed to create output directory: " << db_path.string();
            return -1;
        }
    }
    struct archive *arch = archive_write_new();
    if (output_file_type == GZ) {
        if (output_file_name.find(".gz") != (output_file_name.size() - 3)) {
            DC_SERVER_LOG_ERROR << "please check output file type";
            archive_write_free(arch);
            return -1;
        }
        archive_write_add_filter_gzip(arch);
    } else {
        if (output_file_name.find(".lz4") != (output_file_name.size() - 4)) {
            DC_SERVER_LOG_ERROR << "please check output file type";
            archive_write_free(arch);
            return -1;
        }
        archive_write_add_filter_lz4(arch);
    }
    archive_write_set_format_raw(arch);
    int archive_result = 0;
    std::string temp_output_file_path = PathUtils::getFilePath(PERSISTENT_DIRECTORY, output_file_name + ".temp");
    std::string output_file_path = PathUtils::getFilePath(output_folder_path, output_file_name);
    archive_result = archive_write_open_filename(arch, temp_output_file_path.data());
    if (archive_result != ARCHIVE_OK) {
        DC_SERVER_LOG_ERROR << "error opening archive: " << archive_error_string(arch);
        archive_write_free(arch);
        return archive_result;
    }
    std::ifstream input_file_fs(input_file_path, std::ios::in | std::ios::binary | std::ios::ate);
    input_file_fs.seekg(0, std::ios::end);
    int64_t input_file_size = input_file_fs.tellg();
    input_file_fs.seekg(0, std::ios::beg);
    struct archive_entry *entry = archive_entry_new();
    archive_entry_set_pathname(entry, input_file_path.data());
    archive_entry_set_size(entry, input_file_size); // 文件大小，字节为单位
    archive_entry_set_filetype(entry, AE_IFREG); // 文件类型，这里是普通文件
    archive_entry_set_perm(entry, 0664); // 文件权限，这里是读写
    archive_result = archive_write_header(arch, entry);
    if (archive_result != ARCHIVE_OK) {
        DC_SERVER_LOG_ERROR << "error write header: " << archive_error_string(arch);
        input_file_fs.close();
        archive_entry_free(entry);
        archive_write_free(arch);
        return archive_result;
    }
    archive_entry_free(entry);
    char buf[BUF_SIZE];
    while (1) {
        input_file_fs.read(buf, BUF_SIZE);
        int read_size = input_file_fs.gcount();
        if (read_size == 0) {
            break;
        }
        m_occupies_memory_size += read_size;
        if (m_occupies_memory_size >= MAX_OCCUPIES_MEMORY_SIZE) {
            std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_TIME));
            m_occupies_memory_size = 0;
        }
        archive_result = archive_write_data(arch, buf, read_size);
        if (archive_result != read_size) { // 写入数据块，包括数据和大小
            if (archive_result < ARCHIVE_OK) {
                DC_SERVER_LOG_ERROR << input_file_path << " error write data: " << archive_error_string(arch);
                input_file_fs.close();
                archive_write_free(arch);
                return archive_result;
            } else {
                DC_SERVER_LOG_WARN << input_file_path << " file changed as we read it";
                break;
            }
        }
        if (input_file_fs.eof()) {
            break;
        }
    }
    input_file_fs.close();
    archive_result = archive_write_finish_entry(arch);
    if (archive_result != ARCHIVE_OK) { // 结束写入条目
        DC_SERVER_LOG_ERROR << "error finishing entry: " << archive_error_string(arch);
        archive_write_free(arch);
        return archive_result;
    }
    archive_write_close(arch);
    archive_write_free(arch); // 释放资源
    PathUtils::renameFile(temp_output_file_path, output_file_path);
    m_result_path = output_file_path;
    return archive_result;
}

int Compress::decompress_unzip(std::string input_file_path, std::string output_dir_path) {
    rcpputils::fs::path db_path(output_dir_path);
    if (!db_path.is_directory()) {
        bool dir_created = rcpputils::fs::create_directories(db_path);
        if (!dir_created) {
            DC_SERVER_LOG_ERROR << "failed to create output directory: " << db_path.string();
            return -1;
        }
    }
    Unzipper zipper(input_file_path);
    bool extract_zip_result = zipper.extract(output_dir_path);
    if (extract_zip_result == false) {
        DC_SERVER_LOG_ERROR << "file: " << input_file_path << "decompress failed";
        zipper.close();
        return -1;
    }
    zipper.close();
    m_result_path = output_dir_path;
    return 0;
}

int Compress::decompress_tar(std::string input_file_path, std::string output_dir_path) {
    struct archive *arch = archive_read_new();
    archive_read_support_format_tar(arch);
    archive_read_support_filter_all(arch);
    int archive_result = 0;
    archive_result = archive_read_open_filename(arch, input_file_path.data(), BUF_SIZE);
    if (archive_result != ARCHIVE_OK) {
        DC_SERVER_LOG_ERROR << "error open archive: " << archive_error_string(arch);
        archive_read_free(arch);
        return archive_result;
    }
    struct archive_entry *entry;
    while (1) {
        archive_result = archive_read_next_header(arch, &entry);
        if (archive_result == ARCHIVE_EOF) {
            break;
        } else if (archive_result < ARCHIVE_OK) {
            DC_SERVER_LOG_ERROR << "error read head: " << archive_error_string(arch);
            archive_read_free(arch);
            return archive_result;
        } else {
            std::string output_file_path = output_dir_path + '/' + std::string(archive_entry_pathname(entry));
            std::string dir_path = get_dir_name(output_file_path);
            rcpputils::fs::path db_path(dir_path);
            if (!db_path.is_directory()) {
                bool dir_created = rcpputils::fs::create_directories(db_path);
                if (!dir_created) {
                    DC_SERVER_LOG_ERROR << "failed to create output directory: " << db_path.string();
                    archive_read_free(arch);
                    return -1;
                }
            }
            std::ofstream output_file_fs(output_file_path, std::ios::out | std::ios::binary);
            char buf[BUF_SIZE];
            while (1) {
                archive_result = archive_read_data(arch, buf, BUF_SIZE);
                if (archive_result == 0) {
                    break;
                } else if (archive_result < ARCHIVE_OK) {
                    DC_SERVER_LOG_ERROR << "error read data: " << archive_error_string(arch);
                    output_file_fs.close();
                    archive_read_free(arch);
                    return archive_result;
                } else {
                    output_file_fs.write(buf, archive_result);
                }
                if (archive_result < BUF_SIZE) {
                    break;
                }
                m_occupies_memory_size += archive_result;
                if (m_occupies_memory_size >= MAX_OCCUPIES_MEMORY_SIZE) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_TIME));
                    m_occupies_memory_size = 0;
                }
            }
            output_file_fs.close();
        }
    }
    archive_read_close(arch);
    archive_read_free(arch);
    m_result_path = output_dir_path;
    return 0;
}

int Compress::decompress_raw(std::string input_file_path, std::string output_dir_path) {
    rcpputils::fs::path db_path(output_dir_path);
    if (!db_path.is_directory()) {
        bool dir_created = rcpputils::fs::create_directories(db_path);
        if (!dir_created) {
            DC_SERVER_LOG_ERROR << "failed to create output directory: " << db_path.string();
            return -1;
        }
    }
    struct archive *arch = archive_read_new();
    archive_read_support_format_raw(arch);
    archive_read_support_filter_all(arch);
    int archive_result = 0;
    archive_result = archive_read_open_filename(arch, input_file_path.data(), BUF_SIZE);
    if (archive_result != ARCHIVE_OK) {
        DC_SERVER_LOG_ERROR << "error open archive: " << archive_error_string(arch);
        archive_read_free(arch);
        return archive_result;
    }
    struct archive_entry *entry;
    archive_result = archive_read_next_header(arch, &entry);
    if (archive_result < ARCHIVE_OK) {
        DC_SERVER_LOG_ERROR << "error read head: " << archive_error_string(arch);
        archive_read_free(arch);
        return archive_result;
    }
    std::string input_file_name = get_file_name(input_file_path);
    std::string output_file_name;
    if (input_file_path.find(".gz") == (input_file_path.size() - 3)) {
        output_file_name = input_file_name.substr(0, (input_file_name.size() - 3));
    } else {
        output_file_name = input_file_name.substr(0, (input_file_name.size() - 4));
    }
    std::string output_file_path = PathUtils::getFilePath(output_dir_path, output_file_name);
    std::ofstream output_file_fs(output_file_path, std::ios::out | std::ios::binary);
    char buf[BUF_SIZE];
    while (1) {
        archive_result = archive_read_data(arch, buf, BUF_SIZE);
        if (archive_result == 0) {
            break;
        } else if (archive_result < ARCHIVE_OK) {
            DC_SERVER_LOG_ERROR << "error read data: " << archive_error_string(arch);
            output_file_fs.close();
            archive_read_free(arch);
            return archive_result;
        } else {
            output_file_fs.write(buf, archive_result);
        }
        if (archive_result < BUF_SIZE) {
            break;
        }
        m_occupies_memory_size += archive_result;
        if (m_occupies_memory_size >= MAX_OCCUPIES_MEMORY_SIZE) {
            std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_TIME));
            m_occupies_memory_size = 0;
        }
    }
    output_file_fs.close();
    archive_read_close(arch);
    archive_read_free(arch);
    m_result_path = output_dir_path;
    return 0;
}

void Compress::get_all_file(std::string input_file_path) {
    rcpputils::fs::path db_path(input_file_path);
    if (db_path.exists()) {
        if (db_path.is_directory()) {
            DIR *dir = opendir(input_file_path.data());
            struct dirent *entry;
            while ((entry = readdir(dir)) != nullptr) {
                if (entry->d_type == DT_DIR) {
                    if ((strcmp(entry->d_name, ".") == 0) || (strcmp(entry->d_name, "..") == 0)) {
                        continue;
                    }
                    get_all_file(PathUtils::getFilePath(input_file_path, std::string(entry->d_name)));
                } else {
                    m_input_file_path_vec.push_back(PathUtils::getFilePath(input_file_path, std::string(entry->d_name)));
                }
            }
            // 关闭根目录
            closedir(dir);
        } else {
            m_input_file_path_vec.push_back(input_file_path);
        }
    } else {
        DC_SERVER_LOG_ERROR << input_file_path << " not exist";
        return;
    }
}

std::string Compress::get_file_name(std::string path) {
	size_t length = path.length();
	size_t pos = path.rfind('/');
	if (pos != std::string::npos) {
		return path.substr(pos + 1, length - pos -1);
	} else {
		return path;
	}
}

std::string Compress::get_dir_name(std::string path) {
    size_t pos = path.rfind('/');
	if (pos != std::string::npos) {
		return path.substr(0, pos);
	} else {
		return "";
	}
}

std::string Compress::get_result_path() {
    return m_result_path;
}

}  // namespace dc
}  // namespace netaos
}  // namespace hozon