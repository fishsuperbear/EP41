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
 * @file array.h
 * @brief
 * @date 2020-05-07
 *
 */

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#ifndef APD_ARA_CORE_ARRAY_H_
#define APD_ARA_CORE_ARRAY_H_

#include <array>

namespace ara {
namespace core {
inline namespace _19_11 {
/**
 * @brief A sequence container that encapsulates fixed sized arrays
 *
 * @tparam T the type of contained values
 * @tparam N the number of elements in this Array
 *
 * @uptrace{SWS_CORE_01201}
 */
template <typename T, std::size_t N>
using Array = std::array<T, N>;

/**
 * @brief Add overload of swap for ara::core::Array
 * We actually don't need this overload at all, because our implementation is just<br>
 * a type alias and thus can simply use the overload for the std:: type.<br>
 * a type alias and thus can simply use the overload for the std:: type.<br>
 * However, we need this symbol in order to provide uptracing.
 *
 * @tparam T the type of values in the Arrays
 * @tparam N the size of the Arrays
 * @param lhs the first argument of the swap invocation
 * @param rhs the second argument of the swap invocation
 *
 * @uptrace{SWS_CORE_01296}
 */
template <typename T, std::size_t N>
void swap( Array<T, N> &lhs, Array<T, N> &rhs ) {
    lhs.swap( rhs );
}
}  // namespace _19_11
}  // namespace core
}  // namespace ara

#endif  // APD_ARA_CORE_ARRAY_H_
/* EOF */
