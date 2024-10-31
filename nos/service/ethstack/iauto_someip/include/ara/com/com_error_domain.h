#ifndef INCLUDE_COM_ERROR_DOMAIN_H_
#define INCLUDE_COM_ERROR_DOMAIN_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <ara/core/error_code.h>
#include <ara/core/error_domain.h>
#include <ara/core/exception.h>

namespace ara {
namespace com {
inline namespace _19_11 {

/**
 * @brief Definition of COM errors.
 *
 * @uptrace{SWS_CM_10432}
 * Ver:20-11
 * there is three kind error in ver
 * 19-11:kServiceNotAvailable\kMaxSamplesExceeded\kNetworkBindingFailure
 */
enum class ComErrc : ara::core::ErrorDomain::CodeType {
    kServiceNotAvailable            = 1,  ///< Service is not available.
    kMaxSamplesExceeded             = 2,  ///< Application holds more SamplePtrs than commited in Subscribe().
    kNetworkBindingFailure          = 3,  ///< Local failure has been detected by the network binding.
    kGrantEnforcementError          = 4,  ///< Request was refused by Grant enforcement layer.
    kPeerIsUnreachable              = 5,  ///< TLS handshake fail.
    kFieldValueIsNotValid           = 6,  ///< Field Value is not valid,.
    kSetHandlerNotSet               = 7,  ///< SetHandler has not been registered.
    kUnsetFailure                   = 8,  ///< Failure has been detected by unset operation.
    kSampleAllocationFailure        = 9,  ///< Not Sufficient memory resources can be allocated.
    kIllegalUseOfAllocate           = 10, ///< The allocation was illegally done via custom allocator (i.e., not via shared memory allocation).
    kServiceNotOffered              = 11, ///< Service not offered.
    kCommunicationLinkError         = 12, ///< Communication link is broken.
    kNoClients                      = 13, ///< No clients connected.
    kCommunicationStackError        = 14, ///< Communication Stack Error, e.g. network stack, network binding, or communication framework reports an error
    kInstanceIDCouldNotBeResolved   = 15, ///< ResolveInstanceIDs() failed to resolve InstanceID from InstanceSpecifier, i.e. is not mapped correctly.
    kMaxSampleCountNotRealizable    = 16, ///< Provided maxSampleCount not realizable.
    kWrongMethodCallProcessingMode  = 17, ///< Wrong processing mode passed to constructor method call.
    kErroneousFileHandle            = 18, ///< The FileHandle returned from FindServce is corrupt/service not available.
    kCouldNotExecute                = 19, ///< Command could not be executed in provided Execution Context.

#if 1
    // iAuto Defined
    kIAMCommunicationTimeOut         = 101, ///< IAM communication TimeOut
    kInvalidMessage                  = 102, ///< IAM communication TimeOut
    ksubscribeEventgroupSyncErr      = 103, ///< Call SomeIP 'subscribeEventgroupSync' Interface Error
    kregisterMessageHandlerSyncErr   = 104, ///< Call SomeIP 'registerMessageHandlerSync' Interface failed
    kunsubscribeEventgroupSyncErr    = 105, ///< Call SomeIP 'unsubscribeEventgroupSync' Interface Error
    kunregisterMessageHandlerSyncErr = 106, ///< Call SomeIP 'unregisterMessageHandlerSync' Interface failed
    kofferServiceSyncFailed          = 107, ///< Call SomeIP 'offerServiceSync' Interface failed
    kstopOfferServiceFailed          = 108, ///< Call SomeIP 'stopOfferService' Interface failed
    knotifySyncFailed                = 109, ///< Call SomeIP 'notifySync' Interface failed
    ksomeipinitFailed                = 110, ///< Call SomeIP 'init' Interface failed
    ksomeipstartFailed               = 111, ///< Call SomeIP 'start' Interface failed
    kEventGroupNull                  = 112, ///< the event belongs to eventgroups is null
    kSerializeError                  = 113, ///< Serialize failed
    kDeserializeError                = 114, ///< Deserialize failed
    kSerializeTypeError              = 200, ///< kSerializeTypeError
    kPayloadSizeError                = 201  ///< kPayloadSizeError
#endif
};

/**
 * @brief Definition of COM errors domain exception.
 *
 */
class ComErrorDomainException : public ara::core::Exception {
   public:
    /**
     * @brief Construct a new ComErrorDomainException from an ErrorCode.
     * @param err  the ErrorCode
     *
     */
    explicit ComErrorDomainException( ara::core::ErrorCode err ) noexcept;
};

/**
 * @brief Definition of COM errors domain.
 *
 * @uptrace{SWS_CM_11267}
 */
class ComErrorDomain final : public ara::core::ErrorDomain {
   public:
    // @brief Alias for the error code value enumeration.
    using Errc = ComErrc;

    // @brief Alias for the exception base class.
    using Exception = ComErrorDomainException;

    /**
     * @brief Default constructor.
     *
     * @uptrace{SWS_CORE_00241}
     * @uptrace{SWS_CORE_00012}
     */
    constexpr ComErrorDomain() noexcept : ara::core::ErrorDomain( kId ) {}

    /**
     * @brief Return the "shortname" of this error domain.
     *
     * @return char const* the "shortname" of this error domain.
     */
    char const* Name() const noexcept override;

    /**
     * @brief Translate an error code value into a text message.
     *
     * @param errorCode the error code value.
     * @return char const* the text message, never nullptr.
     */
    char const* Message( ara::core::ErrorDomain::CodeType errorCode ) const noexcept override;

    /**
     * @brief Throw the exception type corresponding to the given ErrorCode.
     *
     * @param errorCode the ErrorCode instance.
     */
    void ThrowAsException( ara::core::ErrorCode const& errorCode ) const noexcept( false ) override;

   private:
    /** \uptrace{SWS_CM_11267} */
    constexpr static ara::core::ErrorDomain::IdType kId = 0x8000000000001267;
};

constexpr ComErrorDomain g_ComErrorDomain;

/**
 * @brief Obtain the reference to the single global ComErrorDomain instance.
 * @returns reference to the ComErrorDomain instance
 *
 */
inline constexpr ara::core::ErrorDomain const& GetComErrorDomain() noexcept {
    return g_ComErrorDomain;
}

/**
 * @brief Create a new ErrorCode for ComErrorDomain with the given support data type and message.
 *
 * @param code an enumeration value from ComErrc
 * @param data a vendor-defined supplementary value
 * @return constexpr ara::core::ErrorCode
 */
inline constexpr ara::core::ErrorCode MakeErrorCode(
    ComErrc code, ara::core::ErrorDomain::SupportDataType data ) noexcept {
    return ara::core::ErrorCode( static_cast<ara::core::ErrorDomain::CodeType>( code ),
                                 GetComErrorDomain(), data );
}

}  // inline namespace _19_11
}  // namespace com
}  // namespace ara

#endif  // INCLUDE_COM_ERROR_DOMAIN_H_
/* EOF */
