#ifndef ARA_CRYPTO_COMMON_CRYPTO_ERROR_DOMAIN_H_
#define ARA_CRYPTO_COMMON_CRYPTO_ERROR_DOMAIN_H_

#include "core/error_domain.h"
#include "core/error_code.h"
#include "core/exception.h"
#include "core/map.h"
#include <cstdint>


namespace hozon {
namespace netaos {
namespace crypto {
// enum class CryptoErrc : std::uint32_t {

// using namespace netaos::core;
enum class CryptoErrc : netaos::core::ErrorDomain::CodeType {
    kSuccess = 0x0,
    kErrorClass = 0x1000000,
    kErrorSubClass = 0x10000,
    kErrorSubSubClass = 0x100,
    kResourceFault = 1 * kErrorClass,
    kBusyResource = kResourceFault + 1,
    kInsufficientResource = kResourceFault + 2,
    kUnreservedResource = kResourceFault + 3,
    kBadAlloc = kResourceFault + 1 * kErrorSubClass,

    kLogicFault = 2 * kErrorClass,
    kInvalidArgument = kLogicFault + 1 * kErrorSubClass,
    kUnknownIdentifier = kInvalidArgument + 1,
    kInsufficientCapacity = kInvalidArgument + 2,
    kInvalidInputSize = kInvalidArgument + 3,
    kIncompatibleArguments = kInvalidArgument + 4,
    kInOutBuffersIntersect = kInvalidArgument + 5,
    kBelowBoundary = kInvalidArgument + 6,
    kAboveBoundary = kInvalidArgument + 7,
    kUnsupported = kInvalidArgument + 1 * kErrorSubSubClass,
    kInvalidUsageOrder = kLogicFault + 2 * kErrorSubClass,
    kUninitializedContext = kInvalidUsageOrder + 1,
    kProcessingNotStarted = kInvalidUsageOrder + 2,
    kProcessingNotFinished = kInvalidUsageOrder + 3,

    kRuntimeFault = 3 * kErrorClass,
    kUnsupportedFormat = kRuntimeFault + 1,
    kBruteForceRisk = kRuntimeFault + 2,
    kContentRestrictions = kRuntimeFault + 3,
    kBadObjectReference = kRuntimeFault + 4,
    kLockedByReference = kRuntimeFault + 5,
    kContentDuplication = kRuntimeFault + 6,
    kUnexpectedValue = kRuntimeFault + 1 * kErrorSubClass,
    kIncompatibleObject = kUnexpectedValue + 1,
    kIncompleteArgState = kUnexpectedValue + 2,
    kEmptyContainer = kUnexpectedValue + 3,
    kBadObjectType = kUnexpectedValue + 1 * kErrorSubSubClass,
    kUsageViolation = kRuntimeFault + 2 * kErrorSubClass,
    kAccessViolation = kRuntimeFault + 3 * kErrorSubClass,

    kCommonErr = 4 * kErrorClass,
    kCommunicationError = kCommonErr + 1,
};

class CryptoException:public  netaos::core::Exception{
public:
    explicit CryptoException (netaos::core::ErrorCode err) noexcept;

private:
 
};

class CryptoErrorDomain final:public netaos::core::ErrorDomain{
public:
    using Errc = CryptoErrc;
    using Exception = CryptoException;
    CryptoErrorDomain() noexcept : ErrorDomain(kId) {}
    ~CryptoErrorDomain() = default;
    const char* Name () const noexcept override{ 
        return "Crypto";
    }
    const char* Message(netaos::core::ErrorDomain::CodeType errorCode) const noexcept override{
        if (errorMsgMap_.find(static_cast<Errc>(errorCode)) != errorMsgMap_.end()) {
            return errorMsgMap_.at(static_cast<Errc>(errorCode)).c_str();
        }
        return "Crypto error message is not defined.";
    }
    
    void ThrowAsException(const netaos::core::ErrorCode& errorCode) const override{
        (void)errorCode;
    }
   private:
    const netaos::core::Map<Errc, netaos::core::String> errorMsgMap_ = {
        {Errc::kResourceFault, "Generic resource fault!"},
        {Errc::kBusyResource, "Specified resource is busy!"},
        {Errc::kInsufficientResource, "Insufficient capacity of specified resource!"},
        {Errc::kUnreservedResource, "Specified resource was not reserved!"},
        {Errc::kBadAlloc, "Cannot allocate requested resources!"},

        {Errc::kLogicFault, "Generic logic fault!"},
        {Errc::kInvalidArgument, "An invalid argument value is provided!"},
        {Errc::kUnknownIdentifier, "Unknown identifier is provided!"},
        {Errc::kInsufficientCapacity, "Insufficient capacity of the output buffer!"},
        {Errc::kInvalidInputSize, "Invalid size of an input buffer!"},
        {Errc::kIncompatibleArguments, "Provided values of arguments are incompatible!"},
        {Errc::kInOutBuffersIntersect, "Input and output buffers are intersect!"},
        {Errc::kBelowBoundary, "Provided value is below the lower boundary!"},
        {Errc::kAboveBoundary, "Provided value is above the upper boundary!"},
        {Errc::kUnsupported, "Unsupported request (due to limitations of the implementation)!"},
        {Errc::kInvalidUsageOrder, "Invalid usage order of the interface!"},
        {Errc::kUninitializedContext, "Context of the interface was not initialized!"},
        {Errc::kProcessingNotStarted, "Data processing was not started yet!"},
        {Errc::kProcessingNotFinished, "Data processing was not finished yet!"},

        {Errc::kRuntimeFault, "Generic runtime fault!"},
        {Errc::kUnsupportedFormat, "Unsupported serialization format for this object type!"},
        {Errc::kBruteForceRisk, "Operation is prohibitted due to a risk of a brute force attack!"},
        {Errc::kContentRestrictions, "The operation violates content restrictions of the target container!"},
        {Errc::kBadObjectReference, "Incorrect reference between objects!"},
        {Errc::kLockedByReference, "An object stored in the container is locked due to a reference from another one!"},
        {Errc::kContentDuplication, "Provided content already exists in the target storage!"},
        {Errc::kUnexpectedValue, "Unexpected value of an argument is provided!"},
        {Errc::kIncompatibleObject, "The provided object is incompatible with requested operation or configuration!"},
        {Errc::kIncompleteArgState, "Incomplete state of an argument!"},
        {Errc::kEmptyContainer, "Specified container is empty!"},
        {Errc::kBadObjectType, "Provided object has unexpected type!"},
        {Errc::kUsageViolation, "Violation of allowed usage for the object!"},
        {Errc::kAccessViolation, "Access rights violation!"},

        {Errc::kCommonErr, "Common error!"},
        {Errc::kCommunicationError, "Ipc communication error!"}
    };
   private:
    constexpr static ErrorDomain::IdType kId = 0x8000000000000021;
};

// namespace internal {
// hozon::netaos::crypto::CryptoErrorDomain CRYPTO_ERROR_DOMAIN;
// }

hozon::netaos::crypto::CryptoErrorDomain const& GetCryptoErrorDomain() noexcept;
hozon::netaos::core::ErrorCode MakeErrorCode(CryptoErrorDomain::Errc code, netaos::core::ErrorDomain::SupportDataType data) noexcept;

}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_COMMON_CRYPTO_ERROR_DOMAIN_H_