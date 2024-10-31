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
 * @file vector.h
 * @brief
 * @date 2020-05-07
 *
 */

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#ifndef APD_ARA_CORE_VECTOR_H_
#define APD_ARA_CORE_VECTOR_H_

#include <memory>
#include <type_traits>
#include <vector>

namespace ara {
namespace core {
inline namespace _19_11 {
/**
 * @brief A sequence container that encapsulates dynamically sized arrays
 *
 * @tparam T  the type of contained values
 * @tparam Allocator  the type of Allocator to use for this container
 *
 * @uptrace{SWS_CORE_01301}
 */
template <typename T, typename Allocator = std::allocator<T>>
using Vector = std::vector<T, Allocator>;

// Transitional compatibility name; should remove this before R18-10.
template <typename T>
using vector = std::vector<T>;

/**
 * @brief Global operator== for Vector instances
 *
 * @tparam T  the type of values in the Vectors
 * @tparam Allocator  the Allocator used by the Vectors
 * @tparam std::enable_if<
 * !std::is_same<Vector<T, Allocator>, std::vector<T, Allocator>>::value>::type
 * @param lhs  the first argument of the compare invocation
 * @param rhs  the second argument of the compare invocation
 * @return true lhs equal rhs
 * @return false otherwise
 *
 * @uptrace{SWS_CORE_01390}
 */
template <typename T, typename Allocator,
          typename = typename std::enable_if<
              !std::is_same<Vector<T, Allocator>, std::vector<T, Allocator>>::value>::type>
inline bool operator==( Vector<T, Allocator> const &lhs, Vector<T, Allocator> const &rhs ) {
    return std::operator==( lhs, rhs );
}

/**
 * @brief Global operator!= for Vector
 *
 * @tparam T  the type of values in the Vectors
 * @tparam Allocator  the Allocator used by the Vectors
 * @tparam std::enable_if<
 * !std::is_same<Vector<T, Allocator>, std::vector<T, Allocator>>::value>::type
 * @param lhs  the first argument of the compare invocation
 * @param rhs  the second argument of the compare invocation
 * @return true lhs not equal rhs
 * @return false otherwise
 *
 * @uptrace{SWS_CORE_01391}
 */
template <typename T, typename Allocator,
          typename = typename std::enable_if<
              !std::is_same<Vector<T, Allocator>, std::vector<T, Allocator>>::value>::type>
inline bool operator!=( Vector<T, Allocator> const &lhs, Vector<T, Allocator> const &rhs ) {
    return std::operator!=( lhs, rhs );
}

/**
 * @brief compare the two vector
 *
 * @tparam T the type of value
 * @tparam Allocator the Allocator used by the Vectors
 * @tparam std::enable_if<
 * !std::is_same<Vector<T, Allocator>, std::vector<T, Allocator>>::value>::type
 * @param lhs  the first argument of the compare invocation
 * @param rhs  the second argument of the compare invocation
 * @return true lhs smaller rhs
 * @return false otherwise
 *
 * @uptrace{SWS_CORE_01392}
 */
template <typename T, typename Allocator,
          typename = typename std::enable_if<
              !std::is_same<Vector<T, Allocator>, std::vector<T, Allocator>>::value>::type>
inline bool operator<( Vector<T, Allocator> const &lhs, Vector<T, Allocator> const &rhs ) {
    return std::operator<( lhs, rhs );
}

/**
 * @brief compare the two vector
 *
 * @tparam T the type of value
 * @tparam Allocator the Allocator used by the Vectors
 * @tparam std::enable_if<
 * !std::is_same<Vector<T, Allocator>, std::vector<T, Allocator>>::value>::type
 * @param lhs  the first argument of the compare invocation
 * @param rhs  the second argument of the compare invocation
 * @return true lhs smaller or equal rhs
 * @return false otherwise
 *
 * @uptrace{SWS_CORE_01393}
 */
template <typename T, typename Allocator,
          typename = typename std::enable_if<
              !std::is_same<Vector<T, Allocator>, std::vector<T, Allocator>>::value>::type>
inline bool operator<=( Vector<T, Allocator> const &lhs, Vector<T, Allocator> const &rhs ) {
    return std::operator<=( lhs, rhs );
}

/**
 * @brief compare the two vector
 *
 * @tparam T the type of value
 * @tparam Allocator the Allocator used by the Vectors
 * @tparam std::enable_if<
 * !std::is_same<Vector<T, Allocator>, std::vector<T, Allocator>>::value>::type
 * @param lhs  the first argument of the compare invocation
 * @param rhs  the second argument of the compare invocation
 * @return true lhs is bigger than rhs
 * @return false otherwise
 *
 * @uptrace{SWS_CORE_01394}
 */
template <typename T, typename Allocator,
          typename = typename std::enable_if<
              !std::is_same<Vector<T, Allocator>, std::vector<T, Allocator>>::value>::type>
inline bool operator>( Vector<T, Allocator> const &lhs, Vector<T, Allocator> const &rhs ) {
    return std::operator>( lhs, rhs );
}

/**
 * @brief compare the two vector
 *
 * @tparam T the type of value
 * @tparam Allocator the Allocator used by the Vectors
 * @tparam std::enable_if<
 * !std::is_same<Vector<T, Allocator>, std::vector<T, Allocator>>::value>::type
 * @param lhs  the first argument of the compare invocation
 * @param rhs  the second argument of the compare invocation
 * @return true lhs is bigger or equal
 * @return false otherwise
 *
 * @uptrace{SWS_CORE_01395}
 */
template <typename T, typename Allocator,
          typename = typename std::enable_if<
              !std::is_same<Vector<T, Allocator>, std::vector<T, Allocator>>::value>::type>
inline bool operator>=( Vector<T, Allocator> const &lhs, Vector<T, Allocator> const &rhs ) {
    return std::operator>=( lhs, rhs );
}

/**
 * @brief Add overload of swap for ara::core::Vector
 *
 * We actually don't need this overload at all, because our implementation is
 * just
 * a type alias and thus can simply use the overload for the std:: type.
 * However, we need this symbol in order to provide uptracing.
 *
 * @tparam T  the type of values in the Vectors
 * @tparam Allocator  the Allocator used by the Vectors
 * @param lhs  the first argument of the swap invocation
 * @param rhs  the second argument of the swap invocation
 *
 * @uptrace{SWS_CORE_01396}
 */
template <typename T, typename Allocator,
          typename = typename std::enable_if<
              !std::is_same<Vector<T, Allocator>, std::vector<T, Allocator>>::value>::type>
void swap( Vector<T, Allocator> &lhs, Vector<T, Allocator> &rhs ) {
    lhs.swap( rhs );
}
}  // namespace _19_11
}  // namespace core
}  // namespace ara

#endif  // APD_ARA_CORE_VECTOR_H_
/* EOF */
