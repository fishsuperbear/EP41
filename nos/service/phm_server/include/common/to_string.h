/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: to string
 */

#ifndef TO_STRING_H
#define TO_STRING_H

#include <string>
#include <vector>
#include <typeinfo>

namespace hozon {
namespace netaos {
namespace phm_server {

const int PRINT_DATA_MAX_NUMBER = 16;

template <class T>
static std::string ToString(T t, std::ios_base & (*f)(std::ios_base&), int n = 0)
{
    std::ostringstream oss;
    if (std::hex == f) {
        oss << "0x";
    }

    if (n) {
        oss << std::setw(n) << std::setfill('0');
    }

    int typesize = sizeof(t);
    if (1 == typesize) {
        uint8_t item = static_cast<uint8_t>(t);
        oss << f << static_cast<uint16_t>(item);
    }
    else {
        oss << f << t;
    }

    return oss.str();
}

template <class T>
static std::string ToString(T t[], std::ios_base & (*f)(std::ios_base&), uint num = 0, int n = 0)
{
    if ((nullptr == t) || (0 == num)) {
        return "";
    }

    std::ostringstream oss;
    int typesize = sizeof(t[0]);
    for (uint i = 0; i < num;) {
        if (n) {
            oss << std::setw(n) << std::setfill('0');
        }

        if (1 == typesize) {
            uint8_t item = static_cast<uint8_t>(t[i]);
            oss << f << static_cast<uint16_t>(item);
        }
        else {
            oss << f << t[i];
        }

        ++i;
        if (i > PRINT_DATA_MAX_NUMBER) {
            oss << "...";
            break;
        }

        if (i < num) {
            oss << " ";
        }
    }

    return oss.str();
}

template <class T>
static std::string ToString(std::vector<T> t, std::ios_base & (*f)(std::ios_base&), int n = 0)
{
    if (t.size() <= 0) {
        return "";
    }

    std::ostringstream oss;
    int typesize = sizeof(t[0]);
    for (uint i = 0; i < t.size();) {
        if (n) {
            oss << std::setw(n) << std::setfill('0');
        }

        if (1 == typesize) {
            uint8_t item = static_cast<uint8_t>(t[i]);
            oss << f << static_cast<uint16_t>(item);
        }
        else {
            oss << f << t[i];
        }

        ++i;
        if (i > PRINT_DATA_MAX_NUMBER) {
            oss << "...";
            break;
        }

        if (i < t.size()) {
            oss << " ";
        }
    }

    return oss.str();
}

#define UINT8_TO_STRING(type) ToString<uint8_t>(type, std::hex, 2)
#define UINT16_TO_STRING(type) ToString<uint16_t>(type, std::hex, 4)
#define UINT32_TO_STRING(type) ToString<uint32_t>(type, std::hex, 8)

#define UINT8_VEC_TO_STRING(vec) ToString<uint8_t>(vec, std::hex, 2)
#define UINT16_VEC_TO_STRING(vec) ToString<uint16_t>(vec, std::hex, 4)
#define UINT32_VEC_TO_STRING(vec) ToString<uint32_t>(vec, std::hex, 8)

#define CHAR_ARRAY_TO_STRING(array, num) ToString<char>(array, std::hex, num, 2)

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
#endif  // TO_STRING_H