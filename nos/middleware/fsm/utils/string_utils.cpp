#include <string>
#include <cstring>
#include <stdexcept>
#include <limits>
#include "string_utils.h"

namespace hozon {
namespace fsmcore {

bool safe_stoul(const std::string& str, uint32_t& val) {
    bool ret{true};
    try {
        auto tmp = std::stoul(str);
        if (tmp > std::numeric_limits<uint32_t>::max()) {
            ret = false;
        } else {
            val = tmp;
        }
    } catch (std::out_of_range& e) {
        ret = false;
    } catch (std::invalid_argument& e) {
        ret = false;
    }
    return ret;
}

bool safe_stoull(const std::string& str, uint64_t& val) {
    bool ret{true};
    try {
        auto tmp = std::stoul(str);
        if (tmp > std::numeric_limits<uint64_t>::max()) {
            ret = false;
        } else {
            val = tmp;
        }
    } catch (std::out_of_range& e) {
        ret = false;
    } catch (std::invalid_argument& e) {
        ret = false;
    }
    return ret;
}

bool safe_stoi(const std::string& str, int32_t& val) {
    bool ret{true};
    try {
        val = std::stoi(str);
    } catch (std::out_of_range& e) {
        ret = false;
    } catch (std::invalid_argument& e) {
        ret = false;
    }
    return ret;
}

bool safe_stoll(const std::string& str, int64_t& val) {
    bool ret{true};
    try {
        val = std::stoll(str);
    } catch (std::out_of_range& e) {
        ret = false;
    } catch (std::invalid_argument& e) {
        ret = false;
    }
    // MISRA C++ 2008: 6-6-5
    return ret;
}

double safe_stod(const std::string& str, double& val) {
    bool ret{true};
    try {
        val = std::stod(str);
    } catch (std::out_of_range& e) {
        ret = false;
    } catch (std::invalid_argument& e) {
        ret = false;
    }
    // MISRA C++ 2008: 6-6-5
    return ret;
}


bool safe_stof(const std::string& str, float& val) {
    bool ret{true};
    try {
        val = std::stod(str);
    } catch (std::out_of_range& e) {
        ret = false;
    } catch (std::invalid_argument& e) {
        ret = false;
    }
    // MISRA C++ 2008: 6-6-5
    return ret;
}

size_t string_to_char(const std::string& str, char* buff, const size_t buff_size) {
    memset(buff, 0, str.length());
    return str.copy(buff, buff_size);
}

void char_to_string(const char* buff, const size_t buff_size, std::string& str) {
    str.assign(buff, buff_size);
    str.resize(buff_size);
}

std::vector<std::string> split(const std::string& str, const std::string& pattern) {
    if (str.find(pattern) == std::string::npos) {
        return {str};
    }
    std::string::size_type pos;
    std::vector<std::string> result;
    std::string tmp_str(str);
    tmp_str += pattern;
    auto size = tmp_str.size();

    for (size_t i = 0; i < size; i++) {
        pos = tmp_str.find(pattern, i);

        if (pos < size) {
            std::string s = tmp_str.substr(i, pos - i);

            if (s != "") {
                result.push_back(s);
            }

            i = pos + pattern.size() - 1;
        }
    }

    return result;
}

std::string trim(const std::string& s) {
    if (s.empty()) {
        return s;
    }
    std::string ret{s};
    ret.erase(0, ret.find_first_not_of(" "));
    ret.erase(ret.find_last_not_of(" ") + 1);
    return ret;
}

}
}
