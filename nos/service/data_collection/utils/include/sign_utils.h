/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: sign_utils.h
 * @Date: 2023/12/15
 * @Author: kun
 * @Desc: --
 */

#ifndef MIDDLEWARE_TOOLS_COMMON_UTILS_SIGN_UTILS_H
#define MIDDLEWARE_TOOLS_COMMON_UTILS_SIGN_UTILS_H

#include <iostream>
#include <sstream>
#include <iomanip>
#include <random>
#include <regex>
#include <chrono>
#include <unordered_map>
#include <openssl/md5.h>

namespace hozon {
namespace netaos {
namespace dc {

class SignUtils {
public:
    static std::string getMd5(const std::string& text);
    static std::string genSignUrl(const std::string& url);
    static bool WriteFileWithLock(std::string file_path, std::string& str);
    static bool ReadFileWithLock(std::string file_path, std::string& str);
    static uint32_t getTimeSpecificSec();

private:
    static std::string getRandomString(int length);
    static std::unordered_map<std::string, std::string> splitUrl(const std::string& url);
    static std::string genTypeAUrl(const std::string& url, const std::string& key, const std::string& signName,
                                   const std::string& uid, long ts);
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // MIDDLEWARE_TOOLS_COMMON_UTILS_SIGN_UTILS_H
