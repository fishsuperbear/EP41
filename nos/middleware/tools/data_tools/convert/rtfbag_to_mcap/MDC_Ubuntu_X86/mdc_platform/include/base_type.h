/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: 全局配置文件读写接口定义
 * Create: 2019-11-12
 */
#ifndef BASE_TYPE_H
#define BASE_TYPE_H

#include <string>
#include <map>
#include <vector>
#include <set>

namespace mdc {
namespace config {

using String = std::string;
template <typename T, typename Y>
using Map = std::map<T, Y>;
template <typename T, typename Allocator = std::allocator<T>>
using Vector = std::vector<T, Allocator>;
template<class KeyT, typename CompT = std::less<KeyT>, typename Allocator = std::allocator<KeyT>>
using Set = std::set<KeyT, CompT, Allocator>;

#ifndef char_t
    using char_t = char;
#endif
}
}
#endif
