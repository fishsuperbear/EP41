/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: the implementation of Map class according to AutoSAR standard core type
 * Create: 2019-07-24
 */
#ifndef ARA_CORE_MAP_H
#define ARA_CORE_MAP_H
#include <map>
namespace ara {
namespace core {
template<typename K, typename V, typename C = std::less<K>, typename Allocator = std::allocator<std::pair<const K, V>>>
using Map=std::map<K, V, C, Allocator>;

template<typename K, typename V, typename C = std::less<K>, typename Allocator = std::allocator<std::pair<const K, V>>>
using map=std::map<K, V, C, Allocator>;
}
}

#endif
