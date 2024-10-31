/*
 * @Author: Z04975 liguoqiang@hozonauto.com
 * @Date: 2023-06-01 19:59:47
 * @LastEditors: Z04975 liguoqiang@hozonauto.com
 * @LastEditTime: 2023-09-13 14:04:01
 * @FilePath: /nos/middleware/per/src/value.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: value
 * Created on: Feb 7, 2023
 *
 */
#ifndef MIDDLEWARE_PER_SRC_VALUE_H_
#define MIDDLEWARE_PER_SRC_VALUE_H_

#include <string>
#include <type_traits>
#include <vector>

class Value {
 public:
    Value();
    ~Value();

    std::string Type();

    template <typename T>
    struct is_string : public std::false_type {};
    template <typename T>
    struct is_string<std::string> : public std::true_type {};

    template <typename T, std::enable_if<std::is_arithmetic<T>::value, bool> = true>
    bool as(T& value) {
        if (sizeof(T) != value_buf_.size()) {
            return false;
        }
        value = *reinterpret_cast<T*>(value_buf_.data());
        return true;
    }

    template <typename T, std::enable_if<is_string<T>::value, bool> = true>
    bool as(T& value) {
        if (value_buf_.size() > 0) {
            value.assign(reinterpret_cast<char*>(value_buf_.data(), value_buf_.size()));
        }
        return true;
    }

 private:
    std::vector<uint8_t> value_buf_;
}
#endif  // MIDDLEWARE_PER_SRC_VALUE_H_
