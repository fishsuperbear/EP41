/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: 通用函数定义
 * Created on: Feb 7, 2023
 *
 */

#ifndef MIDDLEWARE_PER_INCLUDE_PER_UTILS_H_
#define MIDDLEWARE_PER_INCLUDE_PER_UTILS_H_
#include <dirent.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/vfs.h>
#include <time.h>
#include <unistd.h>

#include <cstdint>
#include <ctime>
#include <deque>
#include <iostream>
#include <iterator>
#include <map>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>

#include "kvs_type.h"
#include "per_logger.h"
namespace hozon {
namespace netaos {
namespace per {

class PerUtils {
 private:
    static uint64_t GetDirectorySize(std::string dir) {
        DIR* dp;
        struct dirent* entry;
        struct stat statbuf;
        uint64_t totalSize = 0;
        if ((dp = opendir(dir.c_str())) == NULL) {
            PER_LOG_INFO << "size " << totalSize;
            return -1;
        }
        lstat(dir.c_str(), &statbuf);
        totalSize += statbuf.st_size;
        while ((entry = readdir(dp)) != NULL) {
            std::string subdir = dir + "/" + entry->d_name;
            lstat(subdir.c_str(), &statbuf);
            if (S_ISDIR(statbuf.st_mode)) {
                if (strcmp(".", entry->d_name) == 0 || strcmp("..", entry->d_name) == 0) {
                    continue;
                }
                uint64_t subDirSize = GetDirectorySize(subdir);
                totalSize += subDirSize;
            } else {
                totalSize += statbuf.st_size;
            }
        }

        closedir(dp);
        return totalSize;
    }
    static uint64_t get_system_tf_free(std::string dir) {
        struct statfs diskInfo;
        statfs(dir.c_str(), &diskInfo);
        uint64_t totalBlocks = diskInfo.f_bsize;
        uint64_t freeDisk = diskInfo.f_bavail * totalBlocks;
        return freeDisk;
    }

 public:
    static bool CheckFreeSize(std::string filePath) {
        std::string path;
        std::string::size_type poslast = filePath.find_last_of("/");
        std::string::size_type posfirst = filePath.find_first_of("/");
        if (poslast == posfirst) {
            path = "./";
        } else {
            path = filePath.substr(0, poslast);
        }
        bool res = true;
        uint64_t maxdirsize = 1000 * 1024 * 1024;
        uint64_t freedirsize = 10 * 1024 * 1024;
        uint64_t size = GetDirectorySize(path);
        uint64_t freesize = get_system_tf_free(path);
        if ((size > maxdirsize) || (freesize < freedirsize)) {
            res = false;
        }
        PER_LOG_INFO << "filePath: " << filePath << "  dirsize: " << size << "  dirfreesize: " << freesize << " res: " << res;
        return res;
    }

    static std::string VecToString(const std::vector<uint8_t>& value, const uint8_t& datatype) {
        std::string str;
        switch (datatype) {
            case JSON_PER_TYPE_BOOL: {
                bool val;
                BytesToNum(value, val);
                str = NumToString(val);
                break;
            }
            case JSON_PER_TYPE_DOUBLE: {
                double val;
                BytesToNum(value, val);
                str = NumToString(val);
                break;
            }
            case JSON_PER_TYPE_FLOAT: {
                float val;
                BytesToNum(value, val);
                str = NumToString(val);
                break;
            }
            case JSON_PER_TYPE_INT: {
                int32_t val;
                BytesToNum(value, val);
                str = NumToString(val);
                break;
            }
            case JSON_PER_TYPE_UINT64: {
                uint64_t val;
                BytesToNum(value, val);
                str = NumToString(val);
                break;
            }
            case JSON_PER_TYPE_STRING: {
                BytesToNum(value, str);
                break;
            }
            case JSON_PER_TYPE_VEC_BOOL: {
                std::vector<bool> vec;
                BytesToVec<bool>(value, vec);
                str = ChopLineStringEx<bool>(vec);
                break;
            }
            case JSON_PER_TYPE_VEC_INT: {
                std::vector<int32_t> vec;
                BytesToVec<int32_t>(value, vec);
                str = ChopLineStringEx<int32_t>(vec);
                break;
            }
            case JSON_PER_TYPE_VEC_FLOAT: {
                std::vector<float> vec;
                BytesToVec<float>(value, vec);
                str = ChopLineStringEx<float>(vec);
                break;
            }
            case JSON_PER_TYPE_VEC_DOUBLE: {
                std::vector<double> vec;
                BytesToVec<double>(value, vec);
                str = ChopLineStringEx<double>(vec);
                break;
            }
            case JSON_PER_TYPE_VEC_STRING: {
                std::vector<uint16_t> vec;
                for (size_t size = 0; size < value.size(); size++) {
                    vec.push_back(value[size]);
                }
                str = ChopLineStringEx<uint16_t>(vec);
                break;
            }
            default:
                break;
        }
        return str;
    }

    static std::vector<uint8_t> stringToVec(const std::string value, const uint8_t& datatype) {
        std::vector<uint8_t> vec;
        switch (datatype) {
            case JSON_PER_TYPE_BOOL: {
                bool val;
                stringToNum<bool>(value, val);
                vec = NumToBytes(val);
                break;
            }
            case JSON_PER_TYPE_DOUBLE: {
                double val;
                stringToNum<double>(value, val);
                vec = NumToBytes(val);
                break;
            }
            case JSON_PER_TYPE_FLOAT: {
                float val;
                stringToNum<float>(value, val);
                vec = NumToBytes(val);
                break;
            }
            case JSON_PER_TYPE_INT: {
                int32_t val;
                stringToNum<int32_t>(value, val);
                vec = NumToBytes(val);
                break;
            }
            case JSON_PER_TYPE_UINT64: {
                uint64_t val;
                stringToNum<uint64_t>(value, val);
                vec = NumToBytes(val);
                break;
            }
            case JSON_PER_TYPE_STRING: {
                vec = NumToBytes(value);
                break;
            }
            case JSON_PER_TYPE_VEC_BOOL: {
                std::vector<bool> val;
                ChopStringLineEx<bool>(value, val);
                VecToBytes<bool>(val, vec);
                break;
            }
            case JSON_PER_TYPE_VEC_INT: {
                std::vector<int32_t> val;
                ChopStringLineEx<int32_t>(value, val);
                VecToBytes<int32_t>(val, vec);
                break;
            }
            case JSON_PER_TYPE_VEC_FLOAT: {
                std::vector<float> val;
                ChopStringLineEx<float>(value, val);
                VecToBytes<float>(val, vec);
                break;
            }
            case JSON_PER_TYPE_VEC_DOUBLE: {
                std::vector<double> val;
                ChopStringLineEx<double>(value, val);
                VecToBytes<double>(val, vec);
                break;
            }
            case JSON_PER_TYPE_VEC_STRING: {
                std::vector<uint16_t> val;
                ChopStringLineEx<uint16_t>(value, val);
                for (size_t size = 0; size < val.size(); size++) {
                    vec.push_back(val[size]);
                }
                break;
            }
            default:
                break;
        }
        return vec;
    }

    template <typename T>
    static void ChopStringLineEx(std::string line, std::vector<T>& subvec) {
        std::stringstream linestream(line);
        std::string sub;
        while (linestream >> sub) {
            T val;
            stringToNum<T>(sub, val);
            subvec.push_back(val);
        }
    }
    template <typename T>
    static std::string ChopLineStringEx(const std::vector<T>& t) {
        std::stringstream ss;
        copy(t.begin(), t.end(), std::ostream_iterator<T>(ss, " "));
        return ss.str();
    }
    template <typename T>
    static std::string NumToString(const T t) {
        std::ostringstream os;
        os << t;
        return os.str();
    }
    template <typename T = uint8_t>
    static std::string NumToString(const uint8_t t) {
        return std::to_string(t);
    }
    template <typename T = uint8_t>
    static void stringToNum(const std::string str, uint8_t& num) {
        num = std::stoi(str);
    }
    template <typename T>
    static void stringToNum(const std::string str, T& num) {
        std::istringstream iss(str);
        // num = iss.get();
        iss >> std::noskipws >> num;
    }
    template <typename T>
    static void BytesToVec(const std::vector<uint8_t>& invalue, std::vector<T>& outvalue) {
        for (size_t i = 0; i < invalue.size(); i += sizeof(T)) {
            T val;
            // memcpy(&val, &invalue[i], sizeof(T));
            val = *reinterpret_cast<const T*>(&invalue[i]);
            outvalue.push_back(val);
        }
    }
    template <typename T = std::string>
    static void BytesToVec(const std::vector<uint8_t>& invalue, std::vector<std::string>& outvalue) {
        size_t size = 0;
        while (size < invalue.size()) {
            size_t nsize = *reinterpret_cast<const uint16_t*>(&invalue[size]);
            size += sizeof(uint16_t);
            std::string itemvalue(invalue.begin() + size, invalue.begin() + size + nsize);
            outvalue.push_back(itemvalue);
            size += nsize;
        }
    }

    template <typename T>
    static void VecToBytes(const std::vector<T>& invalue, std::vector<uint8_t>& outvalue) {
        for (size_t i = 0; i < invalue.size(); i++) {
            std::vector<uint8_t> vecvalue = NumToBytes<T>(invalue[i]);
            outvalue.insert(outvalue.end(), vecvalue.begin(), vecvalue.end());
        }
        PER_LOG_INFO << "invaluesize  " << invalue.size() << " outvaluesize  " << outvalue.size();
    }
    template <typename T = std::string>
    static void VecToBytes(const std::vector<std::string>& invalue, std::vector<uint8_t>& outvalue) {
        for (size_t i = 0; i < invalue.size(); i++) {
            std::vector<uint8_t> vecvalue = NumToBytes<std::string>(invalue[i]);
            std::vector<uint8_t> lenvalue(sizeof(uint16_t));
            *reinterpret_cast<uint16_t*>(lenvalue.data()) = vecvalue.size();
            outvalue.insert(outvalue.end(), lenvalue.begin(), lenvalue.end());
            outvalue.insert(outvalue.end(), vecvalue.begin(), vecvalue.end());
        }
        PER_LOG_INFO << "invaluesize  " << invalue.size() << " outvaluesize  " << outvalue.size();
    }
    template <typename T>
    static std::vector<uint8_t> NumToBytes(const T t) {
        std::vector<uint8_t> bytes(sizeof(T));
        *reinterpret_cast<T*>(bytes.data()) = t;
        return bytes;
    }

    template <typename T = std::string>
    static std::vector<uint8_t> NumToBytes(const std::string t) {
        std::vector<uint8_t> bytes(t.size());
        ::memcpy(bytes.data(), t.data(), t.size());
        return bytes;
    }
    template <typename T>
    static void BytesToNum(std::vector<uint8_t> t, T& value) {
        value = *reinterpret_cast<const T*>(t.data());
    }
    template <typename T = std::string>
    static void BytesToNum(std::vector<uint8_t> t, std::string& value) {
        value.assign(t.begin(), t.end());
    }
};

}  // namespace per
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_PER_INCLUDE_PER_UTILS_H_
