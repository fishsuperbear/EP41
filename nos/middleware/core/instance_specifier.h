#ifndef NETAOS_CORE_INSTANCE_SPECIFIER_H
#define NETAOS_CORE_INSTANCE_SPECIFIER_H

#include "core/result.h"
#include "core/string.h"
#include "core/string_view.h"
#ifdef AOS_TAINT
#ifndef COVERITY_TAINT_SET_DEFINITION
#define COVERITY_TAINT_SET_DEFINITION
/**
 * @brief Function for Stain Modeling
 * @details The function is used only when the compilation macro AOS_TAINT is enabled.
 */
static void Coverity_Tainted_Set(void* buf) {}
#endif
#endif
namespace hozon {

namespace netaos {
namespace core {
/**
 * @brief class representing an AUTOSAR Instance Specifier, which is basically an AUTOSAR shortname-path wrapper.
 *
 */
class InstanceSpecifier final {
   public:
    /**
     * @brief Construct a new Instance Specifier object
     *
     * @pnetaosm metaModelIdentifier stringified meta model identifier (short name path) where path sepnetaostor is ’/’.
     */
    explicit InstanceSpecifier(StringView metaModelIdentifier) {
#ifdef AOS_TAINT
        Coverity_Tainted_Set((void*)&metaModelIdentifier);
#endif
        if (metaModelIdentifier.find("/") == String::npos) {
#ifndef NOT_SUPPORT_EXCEPTIONS
            throw Exception(ErrorCode(CoreErrc::kInvalidMetaModelPath));
#endif
        } else {
            bool validFlag = true;
            for (auto c : metaModelIdentifier) {
                bool checkRst = ((c >= '0') && (c <= '9')) || ((c >= 'a') && (c <= 'z')) || ((c >= 'A') && (c <= 'Z')) || (c == '/') || (c == '_') || (c == '-') || (c == '.');
                if (!checkRst) {
#ifndef NOT_SUPPORT_EXCEPTIONS
                    throw Exception(ErrorCode(CoreErrc::kInvalidMetaModelShortname));
#else
                    validFlag = false;
                    break;
#endif
                }
            }
            if (validFlag) {
                myIndentifier_ = metaModelIdentifier;
            }
        }
    }

    /**
     * @brief Destroy the Instance Specifier object
     *
     */
    ~InstanceSpecifier() = default;

    /**
     * @brief Create a new instance of this class.
     *
     * @pnetaosm metaModelIdentifier stringified form of InstanceSpecifier
     * @return Result<InstanceSpecifier> a Result, containing either a syntactically valid InstanceSpecifier,
     *                                   or an ErrorCode
     */
    static Result<InstanceSpecifier> Create(StringView metaModelIdentifier);

    /**
     * @brief eq operator to compare with other InstanceSpecifier instance.
     *
     * @pnetaosm other InstanceSpecifier instance to compare this one with.
     * @return bool true in case both InstanceSpecifiers are denoting exactly the same model element, false else.
     */
    bool operator==(InstanceSpecifier const& other) const noexcept;

    /**
     * @brief eq operator to compare with other InstanceSpecifier instance.
     *
     * @pnetaosm other string representation to compare this one with.
     * @return bool true in case this InstanceSpecifiers is denoting exactly the same model element as other,
     *              false else.
     */
    bool operator==(StringView other) const noexcept;

    /**
     * @brief uneq operator to compare with other InstanceSpecifier instance.
     *
     * @pnetaosm other InstanceSpecifier instance to compare this one with.
     * @return bool false in case both InstanceSpecifiers are denoting exactly the same model element, true else.
     */
    bool operator!=(InstanceSpecifier const& other) const noexcept;

    /**
     * @brief uneq operator to compare with other InstanceSpecifier string representation.
     *
     * @pnetaosm other string representation to compare this one with.
     * @return bool false in case this InstanceSpecifiers is denoting exactly the same model element as other,
     *              true else.
     */
    bool operator!=(StringView other) const noexcept;

    /**
     * @brief lower than operator to compare with other InstanceSpecifier for ordering purposes.
     *
     * @pnetaosm other InstanceSpecifier instance to compare this one with.
     * @return bool true in case this InstanceSpecifiers is lexically lower than other, false else.
     */
    bool operator<(InstanceSpecifier const& other) const noexcept;

    /**
     * @brief method to return the stringified form of InstanceSpecifier.
     *
     * @return StringView stringified form of InstanceSpecifier.
     */
    StringView ToString() const noexcept{
        return myIndentifier_;
    };

   private:
    StringView myIndentifier_;
};
}  // End of namespace core
}  // End of namespace netaos
}  // namespace hozon
#endif
