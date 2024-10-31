#include "util/string_util.h"

#include <stdio.h>
#include <string.h>

#include <bitset>

#if defined(WIN32)
#define strncasecmp _strnicmp
#define strcasecmp _stricmp
#endif

namespace advc {

    std::string &StringUtil::Trim(std::string &s) {
        if (s.empty()) {
            return s;
        }

        s.erase(0, s.find_first_not_of(" "));
        s.erase(s.find_last_not_of(" ") + 1);
        return s;
    }

    std::string StringUtil::Trim(const std::string &s,
                                 const std::string &trim_value) {
        std::string ret = StringRemovePrefix(s, trim_value);
        ret = StringRemoveSuffix(ret, trim_value);
        return ret;
    }




    std::string StringUtil::Uint64ToString(uint64_t num) {
        char buf[65];
#if __WORDSIZE == 64
        snprintf(buf, sizeof(buf), "%lu", num);
#else
        snprintf(buf, sizeof(buf), "%llu", num);
#endif
        std::string str(buf);
        return str;
    }

    std::string StringUtil::IntToString(int num) {
        char buf[65];
        snprintf(buf, sizeof(buf), "%d", num);
        std::string str(buf);
        return str;
    }

    void StringUtil::StringToUpper(std::string *s) {
        std::string::iterator end = s->end();
        for (std::string::iterator i = s->begin(); i != end; ++i)
            *i = toupper(static_cast<unsigned char>(*i));
    }

    std::string StringUtil::StringToUpper(const std::string &s) {
        std::string temp = s;
        StringUtil::StringToUpper(&temp);
        return temp;
    }

    void StringUtil::StringToLower(std::string *s) {
        std::string::iterator end = s->end();
        for (std::string::iterator i = s->begin(); i != end; ++i)
            *i = tolower(static_cast<unsigned char>(*i));
    }

    std::string StringUtil::StringToLower(const std::string &s) {
        std::string temp = s;
        StringUtil::StringToLower(&temp);
        return temp;
    }

    std::string StringUtil::JoinStrings(const std::vector<std::string> &str_vec,
                                        const std::string &delimiter) {
        std::string ret;
        for (std::vector<std::string>::const_iterator c_itr = str_vec.begin();
             c_itr != str_vec.end(); ++c_itr) {
            if (c_itr + 1 == str_vec.end()) {
                ret = ret + *c_itr;
            } else {
                ret = ret + *c_itr + delimiter;
            }
        }
        return ret;
    }

    uint64_t StringUtil::StringToUint64(const std::string &str) {
        unsigned long long temp = strtoull(str.c_str(), NULL, 10);
        return temp;
    }

    unsigned StringUtil::StringToUint32(const std::string &str) {
        unsigned temp = strtoul(str.c_str(), NULL, 10);
        return temp;
    }

    int StringUtil::StringToInt(const std::string &str) {
        std::istringstream is(str);
        int temp = 0;
        is >> temp;
        return temp;
    }

    float StringUtil::StringToFloat(const std::string &str) {
        std::istringstream is(str);
        float temp = 0;
        is >> temp;
        return temp;
    }

    bool StringUtil::StringStartsWith(const std::string &str,
                                      const std::string &prefix) {
        return (str.size() >= prefix.size()) &&
               strncmp(str.c_str(), prefix.c_str(), prefix.size()) == 0;
    }

    bool StringUtil::StringStartsWithIgnoreCase(const std::string &str,
                                                const std::string &prefix) {
        return str.size() >= prefix.size() &&
               strncasecmp(str.c_str(), prefix.c_str(), prefix.size()) == 0;
    }

    std::string StringUtil::StringRemovePrefix(const std::string &str,
                                               const std::string &prefix) {
        if (StringStartsWith(str, prefix)) {
            return str.substr(prefix.size());
        }
        return str;
    }

    bool StringUtil::StringEndsWith(const std::string &str,
                                    const std::string &suffix) {
        return (str.size() >= suffix.size()) &&
               strncmp(str.substr(str.size() - suffix.size()).c_str(), suffix.c_str(),
                       suffix.size()) == 0;
    }

    bool StringUtil::StringEndsWithIgnoreCase(const std::string &str,
                                              const std::string &suffix) {
        return (str.size() >= suffix.size()) &&
               strncasecmp(str.substr(str.size() - suffix.size()).c_str(),
                           suffix.c_str(), suffix.size()) == 0;
    }

    std::string StringUtil::StringRemoveSuffix(const std::string &str,
                                               const std::string &suffix) {
        if (StringEndsWith(str, suffix)) {
            return str.substr(0, str.size() - suffix.size());
        }
        return str;
    }

    void StringUtil::SplitString(const std::string &str, char delim,
                                 std::vector<std::string> *vec) {
        std::stringstream ss(str);
        std::string item;
        while (std::getline(ss, item, delim)) {
            if (!item.empty()) {
                vec->push_back(item);
            }
        }
    }

    void StringUtil::SplitString(const std::string &str, const std::string &sep,
                                 std::vector<std::string> *vec) {
        size_t start = 0, index = 0;

        while ((index = str.find(sep, start)) != std::string::npos) {
            if (index > start) {
                vec->push_back(str.substr(start, index - start));
            }

            start = index + sep.size();
            if (start == std::string::npos) {
                return;
            }
        }

        vec->push_back(str.substr(start));
    }



// 判断下etag的长度, V5的Etag长度是32(MD5), V4的是40(sha)
// v4的etag则跳过校验
    bool StringUtil::IsV4ETag(const std::string &etag) {
        if (etag.length() != 32) {
            return true;
        } else {
            return false;
        }
    }

    bool StringUtil::IsMultipartUploadETag(const std::string &etag) {
        if (etag.find("-") != std::string::npos) {
            return true;
        }

        return false;
    }

    uint32_t StringUtil::GetUint32FromStrWithBigEndian(const char *str) {
        uint32_t num = 0;
        std::bitset<8> bs(str[0]);
        uint32_t tmp = bs.to_ulong();
        // std::cout << "tmp " << tmp<< std::endl;
        num |= (tmp << 24);
        bs = str[1];
        tmp = bs.to_ulong();
        // std::cout << "tmp " << tmp<< std::endl;
        num |= (tmp << 16);
        bs = str[2];
        tmp = bs.to_ulong();
        // std::cout << "tmp " << tmp<< std::endl;
        num |= (tmp << 8);
        bs = str[3];
        tmp = bs.to_ulong();
        // std::cout << "tmp " << tmp<< std::endl;
        num |= (tmp);
        return num;
    }

    uint16_t StringUtil::GetUint16FromStrWithBigEndian(const char *str) {
        uint16_t num = 0;
        std::bitset<8> bs(str[0]);
        uint16_t tmp = bs.to_ulong();
        // std::cout << "tmp " << tmp<< std::endl;
        num |= (tmp << 8);
        bs = str[1];
        tmp = bs.to_ulong();
        // std::cout << "tmp " << tmp<< std::endl;
        num |= tmp;
        return num;
    }

    std::string StringUtil::subreplace(std::string resource_str, std::string sub_str, std::string new_str)
    {
        std::string::size_type pos = 0;
        while((pos = resource_str.find(sub_str)) != std::string::npos)   //替换所有指定子串
        {
             resource_str.replace(pos, sub_str.length(), new_str);
        }
        return resource_str;
    }
}  // namespace advc
