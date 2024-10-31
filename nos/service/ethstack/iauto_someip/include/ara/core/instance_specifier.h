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
 * @file instance_specifier.h
 * @brief
 * @date 2020-05-07
 *
 */

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#ifndef APD_ARA_CORE_INSTANCE_SPECIFIER_H_
#define APD_ARA_CORE_INSTANCE_SPECIFIER_H_

#include <ara/core/result.h>
#include <ara/core/string_view.h>
#include <ara/core/string.h>

namespace ara {
namespace core {
inline namespace _19_11 {
/**
 * @class InstanceSpecifier
 *
 * @brief class representing an AUTOSAR Instance Specifier, which is basically an AUTOSAR
 * shortname-path wrapper.
 *
 * @uptrace{SWS_CORE_08001}
 */
class InstanceSpecifier final {
   public:
    /**
     * @brief throwing ctor from meta-model string.
     *
     * @param metaModelIdentifier [in] stringified meta model identifier (short name path)
     * where path separator is ’/’. Lifetime of underlying
     * string has to exceed the lifetime of the constructed
     * InstanceSpecifier.
     *
     * @uptrace{SWS_CORE_08021}
     */
    explicit InstanceSpecifier( StringView metaModelIdentifier );

    /**
     * @brief Destructor.
     *
     * @uptrace{SWS_CORE_08029}
     */
    ~InstanceSpecifier() noexcept;

    /**
     * @brief Create a new instance of this class.
     *
     * @param metaModelIdentifier [in] stringified form of InstanceSpecifier
     * @return a Result, containing either a syntactically valid InstanceSpecifier, or an ErrorCode
     *
     * @uptrace{SWS_CORE_08032}
     */
    static Result<InstanceSpecifier> Create( StringView metaModelIdentifier );

    /**
     * @brief method to return the stringified form of InstanceSpecifier.
     *
     * @return stringified form of InstanceSpecifier. Lifetime of the
     * underlying string is only guaranteed for the lifetime
     * of the underlying string of the StringView passed to
     * the constructor.
     *
     * @uptrace{SWS_CORE_08041}
     */
    StringView ToString() const noexcept;

    /**
     * @brief eq operator to compare with other InstanceSpecifier instance.
     *
     * @param other [in] InstanceSpecifier instance to compare this one with.
     * @return true in case both InstanceSpecifiers are denoting
     * exactly the same model element, false else.
     *
     * @uptrace{SWS_CORE_08042}
     */
    bool operator==( InstanceSpecifier const &other ) const noexcept;

    /**
     * @brief eq operator to compare with other InstanceSpecifier instance.
     *
     * @param other [in] string representation to compare this one with.
     * @return true in case this InstanceSpecifiers is denoting
     * exactly the same model element as other, false else.
     *
     * @uptrace{SWS_CORE_08043}
     */
    bool operator==( StringView other ) const noexcept;

    /**
     * @brief uneq operator to compare with other InstanceSpecifier instance.
     *
     * @param other [in] InstanceSpecifier instance to compare this one with.
     * @return false in case both InstanceSpecifiers are denoting
     * exactly the same model element, true else.
     *
     * @uptrace{SWS_CORE_08044}
     */
    bool operator!=( InstanceSpecifier const &other ) const noexcept;

    /**
     * @brief uneq operator to compare with other InstanceSpecifier string representation.
     *
     * @param other [in] string representation to compare this one with.
     * @return false in case this InstanceSpecifiers is denoting
     * exactly the same model element as other, true else.
     *
     * @uptrace{SWS_CORE_08045}
     */
    bool operator!=( StringView other ) const noexcept;

    /**
     * @brief lower than operator to compare with other InstanceSpecifier for ordering purposes
     * (f.i. when collecting identifiers in maps).
     *
     * @param other [in] InstanceSpecifier instance to compare this one with.
     * @return true in case this InstanceSpecifiers is lexically lower
     * than other, false else.
     *
     * @uptrace{SWS_CORE_08046}
     */
    bool operator<( InstanceSpecifier const &other ) const noexcept;

   private:
    String m_metaModelIdentifier;
};
}  // namespace _19_11
}  // namespace core
}  // namespace ara

#endif  // APD_ARA_CORE_INSTANCE_SPECIFIER_H_
/* EOF */
