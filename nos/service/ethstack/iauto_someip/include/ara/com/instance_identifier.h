#ifndef INCLUDE_COM_INSTANCE_IDENTIFIER_H_
#define INCLUDE_COM_INSTANCE_IDENTIFIER_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <functional>
#include <ara/core/string.h>
#include <ara/core/string_view.h>

#include "ara/com/internal/definitions.h"

namespace ara {
namespace com {
inline namespace _19_11 {

/**
 * @brief Identifier of a certain instance of a service.
 *
 * @uptrace{SWS_CM_00302}
 */
class InstanceIdentifier {
   public:
    /**
     * @brief Explicit constructor to avoid implicit type conversion for instance ID
     * @param value Used for instance ID
     *
     * @uptrace{SWS_CM_00302}
     */
    explicit InstanceIdentifier(ara::core::StringView value) {
        instance_id_ = value.data();
    }

    /**
     * @brief Converts the contents of the InstanceIdentifier to an implementation-defined string
     * representation.
     * @return String representing the service instance.
     *
     * @uptrace{SWS_CM_00302}
     */
    ara::core::StringView toString() const {
        return ara::core::StringView( instance_id_.data() );
    }

    /**
     * @brief Compare the equality of two instance IDs.
     * @param other InstanceIdentifier to compare to.
     * @return true if equal else false
     *
     * @uptrace{SWS_CM_00302}
     */
    bool operator==( const InstanceIdentifier& other ) const {
        return instance_id_ == other.instance_id_;
    }

    /**
     * @brief Establishes a total order over InstanceIdentifier so that it's usable inside an
     * ara::core::Map or std::set
     * @param other InstanceIdentifier to compare to.
     * @return true if *this < other else false.
     *
     * @uptrace{SWS_CM_00302}
     */
    bool operator<( const InstanceIdentifier& other ) const {
        return instance_id_ < other.instance_id_;
    }

    /**
     * @uptrace{SWS_CM_00302}
     */
    InstanceIdentifier& operator=( const InstanceIdentifier& other ) {
        if ( this != &other ) {
            instance_id_ = other.instance_id_;
        }
        return *this;
    }

    /**
     * @brief Defines the constant that can be used a the wildcard to look for all compatible
     * services.
     *
     * This is used during service discovery for @see FindService or @see StartFindService.
     *
     * ### TOM: This one should uptrace to SWS_CM_00302 - type mismatch though
     * (SWS_CM_00302 specifies that InstanceIdentifier::Any is of type InstanceIdentifier).
     */
    static ara::core::StringView Any;

   private:
    ara::core::String instance_id_;
};

}  // inline namespace _19_11
}  // namespace com
}  // namespace ara

#endif  // INCLUDE_COM_INSTANCE_IDENTIFIER_H_
/* EOF */
