/**
 * * --------------------------------------------------------------------
 * * |                                                                  |
 * * |     _         _    _ _______ ____         _____ ____  __  __     |
 * * |    (_)   /\  | |  | |__   __/ __ \       / ____/ __ \|  \/  |    |
 * * |     _   /  \ | |  | |  | | | |  | |     | |   | |  | | \  / |    |
 * * |    | | / /\ \| |  | |  | | | |  | |     | |   | |  | | |\/| |    |
 * * |    | |/ ____ \ |__| |  | | | |__| |  _  | |___| |__| | |  | |    |
 * * |    |_/_/    \_\____/   |_|  \____/  (_)  \_____\____/|_|  |_|    |
 * * |                                                                  |
 * * --------------------------------------------------------------------
 *
 *  * Copyright @ 2020 iAuto (Shanghai) Co., Ltd.
 *  * All Rights Reserved.
 *  *
 *  * Redistribution and use in source and binary forms, with or without
 *  * modification, are NOT permitted except as agreed by
 *  * iAuto (Shanghai) Co., Ltd.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS,
 *  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 *
 * @file map.h
 * @brief
 * @date 2020-05-07
 *
 */

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#ifndef APD_ARA_CORE_MAP_H_
#define APD_ARA_CORE_MAP_H_

#include <map>
#include <memory>
#include <type_traits>

namespace ara {
namespace core {
inline namespace _19_11 {
/**
 * @brief A container that contains key-value pairs with unique keys
 *
 * @tparam K  the type of keys in this Map
 * @tparam V  the type of values in this Map
 * @tparam C  the type of comparison Callable
 * @tparam Allocator  the type of Allocator to use for this container
 *
 * @uptrace{SWS_CORE_01400}
 */
template <typename K, typename V, typename C = std::less<K>,
          typename Allocator = std::allocator<std::pair<const K, V>>>
using Map = std::map<K, V, C, Allocator>;

/**
 * @brief Add overload of std::swap for Map.
 * We actually don't need this overload at all, because our implementation is
 * just
 * a type alias and thus can simply use the overload for the std:: type.
 * However, we need this symbol in order to provide uptracing.
 *
 * @tparam K  the type of keys in the Maps
 * @tparam V  the type of values in the Maps
 * @tparam C  the type of comparison Callables in the Maps
 * @tparam Allocator  the type of Allocators in the Maps
 * @param lhs  the first argument of the swap invocation
 * @param rhs  the second argument of the swap invocation
 *
 * @uptrace{SWS_CORE_01496}
 */
template <typename K, typename V, typename C, typename Allocator>
void swap( Map<K, V, C, Allocator> &lhs, Map<K, V, C, Allocator> &rhs ) {
    lhs.swap( rhs );
}
}  // namespace _19_11
}  // namespace core
}  // namespace ara

#endif  // APD_ARA_CORE_MAP_H_
/* EOF */
