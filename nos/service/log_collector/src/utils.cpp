// 
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
// 
/// @file utils.cpp
/// @brief 
/// @author xumengjun@hozonauto.com
/// @version 1.0.0
/// @date 2023-11-15

#include "log_collector/include/utils.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <string.h>

#include <cstring>
#include <sstream>

namespace hozon {
namespace netaos {
namespace logcollector {

int CommonTool::FileType(const std::string &filename) {
    struct stat infos;
    stat(filename.c_str(), &infos);
    if (infos.st_mode & S_IFDIR) {
        return 0;
    } else if (infos.st_mode & S_IFREG) {
        return 1;
    } else if (infos.st_mode & S_IFLNK) {
        return 2;
    }

    return -1;
}

bool CommonTool::ListSubPaths(const std::string &directory_path,
            const unsigned char d_type, const std::string filter, std::vector<std::string> &result) {
    DIR *directory = opendir(directory_path.c_str());
    if (directory == nullptr) {
        return false; 
    }

    struct dirent *entry;
    while ((entry = readdir(directory)) != nullptr) {
        // Skip "." and "..".
        if (entry->d_type == d_type && strcmp(entry->d_name, ".") != 0 &&
                    strcmp(entry->d_name, "..") != 0) {
            if (!filter.empty() && strstr(entry->d_name, filter.c_str()) == nullptr) {
                std::string name(entry->d_name);
                if (!filter.empty()) {
                    if (name.substr(name.size() - filter.size(), filter.size()) == filter) {
                        result.emplace_back(directory_path + "/" + std::string(entry->d_name));
                    }
                } else {
                    result.emplace_back(directory_path + "/" + std::string(entry->d_name));
                }
            } else {
                result.emplace_back(directory_path + "/" + std::string(entry->d_name));
            }
        }
    }
    closedir(directory);
    return true;
}

void CommonTool::SplitStr(const std::string &str, char delim,
            std::vector<std::string> &elems) {
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, delim)) {
        if (!item.empty()) {
            elems.push_back(item);
        }
    }
}

time_t CommonTool::TransStrToTime(const std::string &str) {
	struct tm time_fields = {};
	time_t seconds;
/*	
    std::string s1 = str.substr(0,4);
    std::string s2 = str.substr(5,2);
    std::string s3 = str.substr(8,2);
	time_fields.tm_year = atoi(str.substr(0, 4).c_str()) - 1900;
	time_fields.tm_mon = atoi(str.substr(5, 2).c_str());
	time_fields.tm_mday = atoi(str.substr(8, 2).c_str());
	
	time_fields.tm_hour = atoi(str.substr(11, 2).c_str());
	time_fields.tm_min = atoi(str.substr(14, 2).c_str());
	time_fields.tm_sec = atoi(str.substr(17, 2).c_str());
	
	seconds = mktime(&time_fields);
*/
    strptime(str.c_str(), "%Y-%m-%d %H-%M-%S", &time_fields);
    seconds = mktime(&time_fields);

	return seconds;
}

unsigned long long CommonTool::GetTscNS() {
    unsigned long long io_tsc_ns;
    uint64_t tsc;
    __asm__ __volatile__ ("mrs %[tsc], cntvct_el0" : [tsc] "=r" (tsc));
    io_tsc_ns = tsc * 32;
    return io_tsc_ns;
}

size_t CommonTool::GetFileSize(const char* filename) {
    struct stat st;
    if (stat(filename, &st) != 0) {
        return -1;
    }
    return st.st_size;
}

std::tuple<std::string, std::string> CommonTool::SplitByExtension(const std::string &fname) {
     auto ext_index = fname.rfind('.');

     // no valid extension found - return whole path and empty string as
     // extension
     if (ext_index == std::string::npos || ext_index == 0 || ext_index == fname.size() - 1)
     {
         return std::make_tuple(fname, std::string());
     }

     // treat cases like "/etc/rc.d/somelogfile or "/abc/.hiddenfile"
     auto folder_index = fname.find_last_of("/");
     if (folder_index != std::string::npos && folder_index >= ext_index - 1)
     {
         return std::make_tuple(fname, std::string());
     }

     // finally - return a valid base and extension tuple
     return std::make_tuple(fname.substr(0, ext_index), fname.substr(ext_index));
}

bool CommonTool::PathExists(const std::string &path) {
    struct stat buffer;
    if (::stat(path.c_str(), &buffer) != 0) {
        return false;
    }

    return true;
}

bool CommonTool::CreateDir(const std::string &fname) {
    auto pos = fname.find_last_of("/");
    const std::string &path = pos != std::string::npos ? fname.substr(0, pos) : std::string();
    if (path.empty()) {
        return false;
    }

    auto mkdir_ = [=](const std::string &subdir) {
        if (::mkdir(path.c_str(), mode_t(0755)) != 0) {
            return false;
        }
        
        return true;
    };

    if (PathExists(path)) {
        return true;
    }

    size_t search_offset = 0;
    do
    {
        auto token_pos = path.find_first_of("/", search_offset);
        // treat the entire path as a folder if no folder separator not found
        if (token_pos == std::string::npos)
        {
            token_pos = path.size();
        }

        auto subdir = path.substr(0, token_pos);
        if (!subdir.empty() && !PathExists(subdir) && !mkdir_(subdir))
        {
            return false; // return error if failed creating dir
        }
        search_offset = token_pos + 1;
    } while (search_offset < path.size());

    return true;
}

void CommonTool::CurrTimeStr(std::string &str_time) {
    time_t t = time(nullptr);
    tm buf;
    char mbstr[64];

    if (localtime_r(&t, &buf) != nullptr) {
        size_t n = strftime(mbstr, sizeof(mbstr), "%Y%m%d%H%M%S", &buf);
        if (n > 0) {
            str_time.assign(mbstr, n);
        }
    }
    str_time = "19000101000000";
}

std::pair<long long, long long> CommonTool::GetFileModifyTime(const char *filename) {
    std::pair<long long, long long> secnsec;
    struct stat st;
    if (stat(filename, &st) == 0) {
        secnsec.first = (long long)st.st_mtim.tv_sec;
        secnsec.second = st.st_mtim.tv_nsec;
    }
    return secnsec; 
}

std::pair<uint64_t, uint32_t> CommonTool::GetFileCreateTime(const char *filename) {
    int dirfd = AT_FDCWD;
    int flags = AT_SYMLINK_NOFOLLOW;
    unsigned int mask = STATX_ALL;
    struct statx stxbuf;
    auto ret = statx(dirfd, filename, flags, mask, &stxbuf);
    if (ret == 0) {
        return {*&stxbuf.stx_btime.tv_sec, *&stxbuf.stx_btime.tv_nsec};
    }
    return {0, 0};
}

} // namespace logcollector 
} // namespace netaos
} // namespace hozon
