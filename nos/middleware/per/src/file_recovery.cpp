/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: 文件恢复
 * Created on: Feb 7, 2023
 *
 */
#include "src/file_recovery.h"

#include <filesystem>
#include <iostream>
namespace hozon {
namespace netaos {
namespace per {
template <class T>
T FileRecovery::stringToNum(const std::string& str) {
    std::istringstream iss(str);
    T num;
    // num = iss.get();
    iss >> std::noskipws >> num;
    iss.clear();
    return num;
}
template <class T>
std::string FileRecovery::NumToString(const T t) {
    std::ostringstream os;
    os << t;
    return os.str();
}
int FileRecovery::DeleteFile(const std::string& filepath, const StorageConfig config) {
    std::string filecrcpath = filepath + "_crc";
    remove(filepath.c_str());
    remove(filecrcpath.c_str());
    PER_LOG_WARN << "remove filepath " << filepath << " filecrcpath " << filecrcpath;
    int32_t bakcount = config.redundancy_config.redundant_count;
    std::string filename;
    std::string::size_type pos = filepath.find_last_of("/");
    if (pos == filepath.npos) {
        filename = filepath;
    } else {
        filename = filepath.substr(pos + 1, filepath.size());
    }
    PER_LOG_INFO << "filename  " << filename << " bakcount  " << bakcount;
    for (int index = 1; index <= bakcount; index++) {
        std::string bakfilepath = config.redundancy_config.redundant_dirpath + "/" + filename + "_bak_" + std::to_string(index);
        std::string bakfilecrcpath = bakfilepath + "_crc";
        PER_LOG_WARN << "remove bakfilepath " << bakfilepath << " bakfilecrcpath " << bakfilecrcpath;
        remove(bakfilepath.c_str());
        remove(bakfilecrcpath.c_str());
    }
    return 0;
}
bool FileRecovery::FileExist(const std::string& filepath) {
    bool res = true;
    if (0 != access(filepath.c_str(), R_OK)) {
        res = false;
    }
    PER_LOG_INFO << "filepath " << filepath << " res: " << res;
    return res;
}

void FileRecovery::ClearFiles(const std::string& filepath, const StorageConfig config) {
    struct stat buffer;
    std::ofstream outfile(filepath.c_str());
    if (stat(filepath.c_str(), &buffer) == 0) {
        outfile.clear();
        PER_LOG_WARN << "stat clear " << filepath;
    }
    outfile.close();
    std::string filepathcrc = filepath + "_crc";
    std::ofstream outfilecrc(filepathcrc.c_str());
    if (stat(filepathcrc.c_str(), &buffer) == 0) {
        outfilecrc.clear();
        PER_LOG_WARN << "stat clear " << filepathcrc;
    }
    outfilecrc.close();
    int32_t bakcount = config.redundancy_config.redundant_count;
    std::string filename;
    std::string::size_type pos = filepath.find_last_of("/");
    if (pos == filepath.npos) {
        filename = filepath;
    } else {
        filename = filepath.substr(pos + 1, filepath.size());
    }
    PER_LOG_INFO << "filename  " << filename << " bakcount  " << bakcount;
    for (int index = 1; index <= bakcount; index++) {
        std::string bakfilepath = config.redundancy_config.redundant_dirpath + "/" + filename + "_bak_" + std::to_string(index);
        struct stat bufferbak;
        std::ofstream outfilebak(bakfilepath.c_str());
        if (stat(bakfilepath.c_str(), &bufferbak) == 0) {
            outfilebak.clear();
            PER_LOG_WARN << "stat clear " << bakfilepath;
        }
        outfilebak.close();
        std::string bakfilecrcpath = bakfilepath + "_crc";
        std::ofstream outfilecrcbak(bakfilecrcpath.c_str());
        if (stat(bakfilecrcpath.c_str(), &bufferbak) == 0) {
            outfilecrcbak.clear();
            PER_LOG_WARN << "stat clear " << bakfilecrcpath;
        }
        outfilecrcbak.close();
    }
}
bool FileRecovery::Mkdir(std::string filepath, bool isfile) {
    if (filepath.size() > 255) {
        PER_LOG_ERROR << "path is lager than 250 ", filepath.size();
        return false;
    }
    if (isfile) {
        std::string::size_type pos = filepath.find_last_of("/");
        if (pos != filepath.npos) {
            filepath = filepath.substr(0, pos);
        }
    }

    std::string::size_type tmp_pos_begin = 0;
    std::string::size_type tmp_pos = 0;
    tmp_pos = filepath.find('/', tmp_pos_begin);
    while (tmp_pos != filepath.npos) {
        std::string tmpdir = filepath.substr(0, tmp_pos);
        if (tmpdir.empty()) {
            tmp_pos_begin = tmp_pos + 1;
            tmp_pos = filepath.find('/', tmp_pos_begin);
            continue;
        }
        if (access(tmpdir.c_str(), 0) == -1) {
            int ret = mkdir(tmpdir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            if (!ret) {
                PER_LOG_INFO << "mkdir create directory success " << tmpdir;
            } else {
                PER_LOG_ERROR << "mkdir create directory failed" << tmpdir;
                return false;
            }
        } else {
            // PER_LOG_ERROR << "access(tmpdir.c_str(), 0) !=-1";
        }
        tmp_pos_begin = tmp_pos + 1;
        tmp_pos = filepath.find('/', tmp_pos_begin);
        if (tmp_pos == filepath.npos) {
            if (tmp_pos_begin < filepath.size()) {
                tmp_pos = filepath.size();
            }
        }
    }
    return true;
}
bool FileRecovery::RecoverHandle(const std::string& filepath, const StorageConfig config) {
    bool res = false;
    if (Mkdir(filepath, true)) {
        PER_LOG_INFO << "Mkdir  filepath is successful " << filepath;
    }
    if (Mkdir(config.redundancy_config.redundant_dirpath, false)) {
        PER_LOG_INFO << "Mkdir  redundant_dirpath is successful " << config.redundancy_config.redundant_dirpath;
    }
    if (Mkdir(config.original_file_path, false)) {
        PER_LOG_INFO << "Mkdir  original_file_path is successful " << config.original_file_path;
    }

    if (!PerUtils::CheckFreeSize(filepath)) {
        return res;
    }

    int32_t bakcount = config.redundancy_config.redundant_count;
    if (bakcount == 0) {
        res = true;
        PER_LOG_INFO << "bakcount  is 0";
        return res;
    }
    if (CheckCrc32(filepath)) {
        res = true;
        PER_LOG_INFO << "CheckCrc32 true  " << filepath;
    } else {
        PER_LOG_INFO << "CheckCrc32 false  " << filepath;
        if (config.redundancy_config.auto_recover) {
            std::string filename;
            std::string::size_type pos = filepath.find_last_of("/");
            if (pos == filepath.npos) {
                filename = filepath;
            } else {
                filename = filepath.substr(pos + 1, filepath.size());
            }
            PER_LOG_INFO << "filename  " << filename << " bakcount  " << bakcount;
            for (int index = 1; index <= bakcount; index++) {
                std::string bakfilepath = config.redundancy_config.redundant_dirpath + "/" + filename + "_bak_" + std::to_string(index);
                if (CheckCrc32(bakfilepath)) {
                    PER_LOG_INFO << "CheckCrc32 true  " << bakfilepath;
                    bool tmpres = CopybakToOriginFile(bakfilepath, filepath) && CopybakToOriginFile(bakfilepath + "_crc", filepath + "_crc");
                    if (tmpres) {
                        PER_LOG_INFO << "CopybakToOriginFile successful  " << bakfilepath << "->" << filepath;
                        PER_LOG_INFO << "CopybakToOriginFile successful  " << bakfilepath + "_crc"
                                     << "->" << filepath + "_crc";
                        res = true;
                        break;
                    } else {
                        PER_LOG_WARN << "CopybakToOriginFile failed  ";
                    }
                } else {
                    PER_LOG_WARN << "CheckCrc32 false  " << bakfilepath;
                }
            }
            if (!res) {
                ClearFiles(filepath, config);
                PER_LOG_INFO << "ClearFiles " << filepath;
            }
        } else {
            PER_LOG_INFO << "auto_recover is false   ";
        }
    }
    PerUtils::CheckFreeSize(filepath);
    PER_LOG_INFO << "res " << res;
    return res;
}

void FileRecovery::GetFiles(std::string dirPath, std::string nstr, std::deque<std::string>& files) {
    if (dirPath.empty()) {
        return;
    }
    struct dirent* filename = nullptr;
    DIR* dir = nullptr;
    dir = opendir(dirPath.c_str());
    if (dir == nullptr) {
        return;
    }
    while ((filename = readdir(dir)) != nullptr) {
        if ((0 == strcmp(filename->d_name, ".")) || (0 == strcmp(filename->d_name, "..")) || (0 == strstr(filename->d_name, nstr.c_str()))) continue;
        std::string path = dirPath + "/" + filename->d_name;
        struct stat s;
        lstat(path.c_str(), &s);
        const int32_t ret = lstat(path.c_str(), &s);
        if (ret == -1) {
            continue;
        }
        if (S_ISREG(s.st_mode)) {
            files.push_back(path);
        } else if (S_ISDIR(s.st_mode)) {
            GetFiles(path, nstr, files);
        } else {
        }
    }
    closedir(dir);
}

bool FileRecovery::BackUpHandle(const std::string& filepath, const StorageConfig config) {
    if (!PerUtils::CheckFreeSize(filepath)) {
        return false;
    }
    int32_t bakcount = config.redundancy_config.redundant_count;
    std::string filename;
    std::string::size_type pos = filepath.find_last_of("/");
    if (pos == filepath.npos) {
        filename = filepath;
    } else {
        filename = filepath.substr(pos + 1, filepath.size());
    }
    std::deque<std::string> file_list;
    GetFiles(config.redundancy_config.redundant_dirpath, filename + "_bak_", file_list);
    while (!file_list.empty()) {
        std::string bak_name = file_list.front();
        file_list.pop_front();
        bool exist = false;
        for (int index = 1; index <= bakcount; index++) {
            std::string bakfilepath = config.redundancy_config.redundant_dirpath + "/" + filename + "_bak_" + std::to_string(index);
            if (bak_name.find(bakfilepath) != std::string::npos) {
                exist = true;
            }
        }
        if (!exist) {
            remove(bak_name.c_str());
            PER_LOG_INFO << "remove bak_name  " << bak_name;
        }
    }
    if (bakcount == 0) {
        std::string filecrcpath = filepath + "_crc";
        PER_LOG_WARN << "remove crc_name  " << filecrcpath;
        remove(filecrcpath.c_str());
        return true;
    }
    std::string filestr;
    if (!readsp(filepath, filestr)) {
        PER_LOG_WARN << "file not find  " << filepath;
        return false;
    }
    uint32_t calcrc = crc32((uint8_t*)filestr.c_str(), filestr.size());
    // init_CRC32_table();
    // uint32_t calcrc = GetCRC32((unsigned char*)filestr.c_str(), filestr.size());
    PER_LOG_INFO << "calcrc crc:  " << calcrc;
    std::string fs_crc = filepath + "_crc";
    if (!writesp(fs_crc, calcrc)) {
        PER_LOG_WARN << "writesp failed  " << calcrc;
        return false;
    }

    PER_LOG_INFO << "filename  " << filename << " bakcount  " << bakcount;
    for (int index = 1; index <= bakcount; index++) {
        std::string bakfilepath = config.redundancy_config.redundant_dirpath + "/" + filename + "_bak_" + std::to_string(index);
        std::string bakfilecrcpath = bakfilepath + "_crc";
        bool res = CopybakToOriginFile(filepath, bakfilepath) && CopybakToOriginFile(fs_crc, bakfilecrcpath);
        if (res) {
            PER_LOG_INFO << "CopybakToOriginFile successful  " << filepath << "->" << bakfilepath;
        } else {
            PER_LOG_INFO << "CopybakToOriginFile failed  ";
            continue;
        }
    }
    PerUtils::CheckFreeSize(filepath);
    return true;
}

bool FileRecovery::ResetHandle(const std::string& filepath, const StorageConfig config) {
    int32_t bakcount = config.redundancy_config.redundant_count;
    DeleteFile(filepath, config);
    if (!PerUtils::CheckFreeSize(filepath)) {
        return false;
    }
    std::string filename;
    std::string::size_type pos = filepath.find_last_of("/");
    if (pos == filepath.npos) {
        filename = filepath;
    } else {
        filename = filepath.substr(pos + 1, filepath.size());
    }

    std::deque<std::string> file_list;
    GetFiles(config.redundancy_config.redundant_dirpath, filename + "_bak_", file_list);
    while (!file_list.empty()) {
        std::string bak_name = file_list.front();
        file_list.pop_front();
        bool exist = false;
        for (int index = 1; index <= bakcount; index++) {
            std::string bakfilepath = config.redundancy_config.redundant_dirpath + "/" + filename + "_bak_" + std::to_string(index);
            if (bak_name.find(bakfilepath) != std::string::npos) {
                exist = true;
            } else {
            }
        }
        if (!exist) {
            remove(bak_name.c_str());
            PER_LOG_INFO << "remove bak_name  " << bak_name;
        }
    }

    std::string originfilepath = config.original_file_path + "/" + filename;
    std::string filestr;
    if (!readsp(originfilepath, filestr)) {
        PER_LOG_WARN << "file not find  " << originfilepath;
        return false;
    }
    if (!CopybakToOriginFile(originfilepath, filepath)) {
        PER_LOG_WARN << "CopybakToOriginFile failed  " << originfilepath << "->" << filepath;
        return false;
    }
    uint32_t calcrc = crc32((uint8_t*)filestr.c_str(), filestr.size());
    // init_CRC32_table();
    // uint32_t calcrc = GetCRC32((unsigned char*)filestr.c_str(), filestr.size());
    PER_LOG_INFO << "calcrc crc:  " << calcrc;
    std::string fs_crc = filepath + "_crc";
    if (!writesp(fs_crc, calcrc)) {
        PER_LOG_WARN << "writesp failed  " << calcrc;
        return false;
    }
    PER_LOG_INFO << "filename  " << filename << " bakcount  " << bakcount;
    for (int index = 1; index <= bakcount; index++) {
        std::string bakfilepath = config.redundancy_config.redundant_dirpath + "/" + filename + "_bak_" + std::to_string(index);
        std::string bakfilecrcpath = bakfilepath + "_crc";
        bool res = CopybakToOriginFile(filepath, bakfilepath) && CopybakToOriginFile(fs_crc, bakfilecrcpath);
        if (res) {
            PER_LOG_INFO << "CopybakToOriginFile successful  " << filepath << "->" << bakfilepath;
        } else {
            PER_LOG_WARN << "CopybakToOriginFile failed  " << filepath << "->" << bakfilepath;
            continue;
        }
    }
    return true;
}

bool FileRecovery::writesp(const std::string& filepath, uint32_t sp) {
    std::fstream fs_;
    fs_.open(filepath, std::ios::out | std::ios::binary);
    if (!fs_.is_open()) {
        PER_LOG_WARN << "writesp is_open false  ";
        return false;
    }
    fs_.write(reinterpret_cast<char*>(&sp), sizeof(sp));
    fs_.close();
    PER_LOG_INFO << "writesp  " << filepath << " size " << sp;
    return true;
}

bool FileRecovery::readsp(const std::string& filepath, uint32_t& sp) {
    std::fstream fs_;
    fs_.open(filepath, std::ios::in | std::ios::binary);
    if (!fs_.is_open()) {
        PER_LOG_WARN << "readsp is_open false  ";
        return false;
    }
    fs_.read(reinterpret_cast<char*>(&sp), sizeof(sp));
    fs_.close();
    PER_LOG_INFO << "readsp  " << filepath << " size " << sp;
    return true;
}

bool FileRecovery::readsp(const std::string& filepath, std::string& sp) {
    std::fstream fs_;
    fs_.open(filepath);
    if (!fs_.is_open()) {
        PER_LOG_WARN << "readsp is_open false  ";
        return false;
    }
    std::ostringstream tmp;
    tmp << fs_.rdbuf();
    sp = tmp.str();
    fs_.close();
    PER_LOG_INFO << "readsp  " << filepath << " size " << sp.size();
    return true;
}

bool FileRecovery::CheckCrc32(const std::string& filepath) {
    std::string filestr;
    if (!readsp(filepath, filestr)) {
        PER_LOG_WARN << "file not find  " << filepath;
        return false;
    }
    uint32_t calcrc = crc32((uint8_t*)filestr.c_str(), filestr.size());
    // init_CRC32_table();
    // uint32_t calcrc = GetCRC32((unsigned char*)filestr.c_str(), filestr.size());
    PER_LOG_INFO << "calcrc crc:  " << calcrc;
    std::string fs_crc = filepath + "_crc";
    uint32_t rcrc;
    if (!readsp(fs_crc, rcrc)) {
        PER_LOG_WARN << "file not find  " << fs_crc;
        return false;
    }
    PER_LOG_INFO << "readcrc crc:  " << rcrc;
    if (calcrc == rcrc) {
        PER_LOG_INFO << "crc is same  " << filepath;
        return true;
    } else {
        PER_LOG_INFO << "crc is different  " << filepath << "   " << rcrc;
        if (rcrc == 0) {
            rcrc = calcrc;
            writesp(fs_crc, rcrc);
            PER_LOG_INFO << "crc is null ";
            return true;
        } else {
            std::string  rcrcstr;
            readsp(fs_crc, rcrcstr);
            rcrc = stringToNum<uint32_t>(rcrcstr);
            if (rcrc == calcrc) {
                PER_LOG_INFO << "crc is str " << rcrc;
                rcrc = calcrc;
                writesp(fs_crc, rcrc);
                return true;
            }
        }
        return false;
    }
}

uint32_t FileRecovery::crc32(uint8_t* buf, int len) {
    int i, j;
    uint32_t crc, mask;
    crc = 0xFFFFFFFF;
    for (i = 0; i < len; i++) {
        crc = crc ^ (uint32_t)buf[i];
        for (j = 7; j >= 0; j--) {  // Do eight times.
            mask = -(crc & 1);
            crc = (crc >> 1) ^ (0xEDB88320 & mask);
        }
    }
    return ~crc;
}

// unsigned int CRC;                // int的大小是32位，作32位CRC寄存器
// unsigned int CRC_32_Table[256];  //用来保存CRC码表
// void FileRecovery::GenerateCRC32_Table() {
//     for (int i = 0; i < 256; ++i)  //用++i以提高效率
//     {
//         CRC = i;
//         for (int j = 0; j < 8; ++j) {
//             if (CRC & 1)                        // LSM为1
//                 CRC = (CRC >> 1) ^ 0xEDB88320;  //采取反向校验
//             else                                // 0xEDB88320就是CRC-32多项表达式的reversed值
//                 CRC >>= 1;
//         }
//         CRC_32_Table[i] = CRC;
//     }
// }

// unsigned int CRC32_table[256] = {0};
// void FileRecovery::init_CRC32_table() {
//     for (int i = 0; i != 256; i++) {
//         unsigned int CRC1 = i;
//         for (int j = 0; j != 8; j++) {
//             if (CRC1 & 1)
//                 CRC1 = (CRC1 >> 1) ^ 0xEDB88320;
//             else
//                 CRC1 >>= 1;
//         }
//         CRC32_table[i] = CRC1;
//     }
// }
// unsigned int FileRecovery::GetCRC32(unsigned char* buf, unsigned int len) {
//     unsigned int CRC32_data = 0xFFFFFFFF;
//     for (unsigned int i = 0; i != len; ++i) {
//         unsigned int t = (CRC32_data ^ buf[i]) & 0xFF;
//         CRC32_data = ((CRC32_data >> 8) & 0xFFFFFF) ^ CRC32_table[t];
//     }
//     return ~CRC32_data;
// }

bool FileRecovery::CopybakToOriginFile(std::string src, std::string dest) {
    bool res = false;
    do {
        // Check    file exists.
        if (0 != access(src.c_str(), R_OK)) {
            PER_LOG_WARN << "file dose not exist or have no read permission. path " << src;
            break;
        }
        std::ofstream os(dest, std::ios::binary);
        if (os.fail()) {
            PER_LOG_WARN << "Cannot open file " << dest;
            break;
        }
        // Copy  .
        std::ifstream is(src, std::ios::binary);
        if (is.fail()) {
            os.close();
            PER_LOG_WARN << "Cannot open file " << src;
            break;
        }
        is.seekg(0, std::ios::end);
        uint64_t length = is.tellg();
        is.seekg(0);
        char buf[1024] = {0};
        while (length > 0) {
            int buf_size = (length >= 1024) ? 1024 : length;
            is.read(buf, buf_size);
            os.write(buf, buf_size);
            length -= buf_size;
        }
        PER_LOG_WARN << "size" << length;
        is.close();
        os.close();
        res = true;
    } while (0);
    return res;
}

}  // namespace per
}  // namespace netaos
}  // namespace hozon
