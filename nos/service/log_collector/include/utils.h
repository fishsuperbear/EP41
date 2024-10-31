// 
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
// 
/// @fil__LOG_COLLECTOR_INCLUDE_UTILS_H__e utils.h
/// @brief 
/// @author xumengjun@hozonauto.com
/// @version 1.0.0
/// @date 2023-11-15

#ifndef __LOG_COLLECTOR_INCLUDE_UTILS_H__
#define __LOG_COLLECTOR_INCLUDE_UTILS_H__ 

#include <dirent.h>
#include <vector>
#include <string>
#include <tuple>

namespace hozon {
namespace netaos {
namespace logcollector {

class CommonTool {
public:
    static bool ListSubPaths(const std::string &directory_path,
                const unsigned char d_type,
                const std::string filter,
                std::vector<std::string> &result);
    static int FileType(const std::string &filename);
    static size_t GetFileSize(const char* filename);
    static bool CreateDir(const std::string &fname);
    static bool PathExists(const std::string &path);


    static void SplitStr(const std::string &str, char delim,
                std::vector<std::string> &elems);
    static unsigned long long GetTscNS();
    static std::tuple<std::string, std::string> SplitByExtension(const std::string &fname);

    static void CurrTimeStr(std::string &str_time); 
    static time_t TransStrToTime(const std::string &str);
    static std::pair<long long, long long> GetFileModifyTime(const char *filename);
    static std::pair<uint64_t, uint32_t> GetFileCreateTime(const char *filename);
};

} // namespace logcollector
} // namespace netaos
} // namespace hozon

#endif // __LOG_COLLECTOR_INCLUDE_UTILS_H__
