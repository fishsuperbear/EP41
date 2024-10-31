/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: sign_utils.cpp
 * @Date: 2023/12/15
 * @Author: kun
 * @Desc: --
 */

#include "utils/include/sign_utils.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <filesystem>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <iostream>
#include <map>
#include <filesystem>
#include <regex>
#include <fcntl.h>
#include <thread>
#include <fstream>
#include <sstream>

#include "config_param.h"
#include "utils/include/dc_logger.hpp"
#include "utils/include/time_utils.h"

namespace hozon {
namespace netaos {
namespace dc {

uint32_t SignUtils::getTimeSpecificSec() {
    struct timespec time = {0};
    auto cfgMgr = cfg::ConfigParam::Instance();
    int64_t value;
    auto res = cfgMgr->GetParam<int64_t>("time/mp_offset", value);
    if (cfg::CONFIG_OK != res || value == 0) {
        clock_gettime(CLOCK_VIRTUAL, &time);
        return static_cast<uint32_t>(time.tv_sec);
    } else {
        clock_gettime(CLOCK_REALTIME, &time);
        return static_cast<uint32_t>(time.tv_sec) + value;
    }
}

std::string SignUtils::getRandomString(int length) {
    static const char alphanum[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";

    std::string str;
    for (int i = 0; i < length; ++i) {
        str += alphanum[rand() % (sizeof(alphanum) - 1)];
    }
    return str;
}

std::unordered_map<std::string, std::string> SignUtils::splitUrl(const std::string& url) {
    std::regex pattern("^(http://|https://)?([^/?]+)(/[^?]*)?(\\?.*)?$");
    std::smatch match;
    std::unordered_map<std::string, std::string> result;

    if (std::regex_match(url, match, pattern)) {
        result["scheme"] = match[1].str().empty() ? "http://" : match[1].str();
        result["domain"] = match[2].str().empty() ? "" : match[2].str();
        result["uri"] = match[3].str().empty() ? "/" : match[3].str();
        result["args"] = match[4].str().empty() ? "" : match[4].str();
    }

    return result;
}

std::string SignUtils::genTypeAUrl(const std::string& url, const std::string& key, const std::string& signName,
                                   const std::string& uid, long ts) {
    std::unordered_map<std::string, std::string> uriInfo = splitUrl(url);
    std::string scheme = uriInfo["scheme"], domain = uriInfo["domain"], uri = uriInfo["uri"], args = uriInfo["args"];
    std::string rand = getRandomString(10);
    std::string text = uri + "-" + std::to_string(ts) + "-" + rand + "-" + uid + "-" + key;
    std::string hash = getMd5(text);
    std::cout << "hash: " << hash;
    std::string authArg = signName + "=" + std::to_string(ts) + "-" + rand + "-" + uid + "-" + hash;

    if (args.empty()) {
        return scheme + domain + uri + "?" + authArg;
    } else {
        return scheme + domain + uri + args + "&" + authArg;
    }
}

std::string SignUtils::getMd5(const std::string& text) {
    MD5_CTX md5Context;
    MD5_Init(&md5Context);
    MD5_Update(&md5Context, text.c_str(), text.length());
    unsigned char md5Digest[MD5_DIGEST_LENGTH];
    MD5_Final(md5Digest, &md5Context);
    std::ostringstream md5StringStream;
    for (int i = 0; i < MD5_DIGEST_LENGTH; ++i) {
        md5StringStream << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(md5Digest[i]);
    }
    return md5StringStream.str();
}

std::string SignUtils::genSignUrl(const std::string& url) {
    std::string primaryKey = "yExMWFDujA(QQDNZJ";
    std::string signName = "sign";
    std::string uid = "0";
    long ts = getTimeSpecificSec();
    // long ts = std::chrono::duration_cast<std::chrono::seconds>(
    //                 std::chrono::system_clock::now().time_since_epoch())
    //                 .count();
    return genTypeAUrl(url, primaryKey, signName, uid, ts);
}

bool SignUtils::WriteFileWithLock(std::string file_path, std::string& str) {
    bool ret = false;
    int retval = -1;
    std::unique_ptr<FILE, void (*)(FILE*)> file(::fopen(file_path.c_str(), "wb+"), [](FILE* f) { ::fclose(f); });
    struct flock lock {0};
    lock.l_type = F_WRLCK;
    lock.l_whence = SEEK_SET;
    lock.l_start = 0;
    lock.l_len = 0;
    int count = 0;
    while((retval = ::fcntl(fileno(file.get()), F_SETLK, &lock)) != 0) {
        ++count;
        if (count > 100) {
            DC_LOG_ERROR << "Write file failed, can't get write lock. file path: " << file_path;
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    std::ofstream output_file(file_path, std::ios::out | std::ios::binary);
    output_file << str;
    output_file.close();
    // if (file && (1 == ::fwrite(str.data(), str.size(), 1, file.get()))) {
    //     ret = true;
    // }

    lock.l_type = F_UNLCK;
    ::fcntl(fileno(file.get()), F_SETLK, &lock);
    return ret;
}

bool SignUtils::ReadFileWithLock(std::string file_path, std::string& str) {
    bool ret = false;
    int retval = -1;
    std::unique_ptr<FILE, void (*)(FILE*)> file(::fopen(file_path.c_str(), "rb"), [](FILE* f) { ::fclose(f); });
    struct flock lock {0};
    lock.l_type = F_RDLCK;
    lock.l_whence = SEEK_SET;
    lock.l_start = 0;
    lock.l_len = 0;
    int count = 0;
    while((retval = ::fcntl(fileno(file.get()), F_SETLK, &lock)) != 0) {
        ++count;
        if (count > 100) {
            DC_LOG_ERROR << "read file failed, can't get read lock. file path: " << file_path;
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    std::ifstream input_file(file_path, std::ios::in | std::ios::binary);
    std::stringstream ss;
    ss << input_file.rdbuf(); //读取整个文件
    str = ss.str();

    lock.l_type = F_UNLCK;
    ::fcntl(fileno(file.get()), F_SETLK, &lock);
    return ret;
}

}  // namespace dc
}  // namespace netaos
}  // namespace hozon
