#ifndef NETAOS_CORE_CORE_ERROR_DOMAIN_H
#define NETAOS_CORE_CORE_ERROR_DOMAIN_H
#include "core/abort.h"
#include "core/error_domain.h"
#include "core/exception.h"
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
enum class CoreErrc : ErrorDomain::CodeType { kInvalidArgument = 22, kInvalidMetaModelShortname = 137, kInvalidMetaModelPath = 138 };

class CoreException : public Exception {
   public:
    explicit CoreException(ErrorCode err) noexcept : Exception(std::move(err)) {
#ifdef AOS_TAINT
        Coverity_Tainted_Set((void*)&err);
#endif
    }
    ~CoreException() = default;
};

class CoreErrorDomain final : public ErrorDomain {
   public:
    using Errc = CoreErrc;
    using Exception = CoreException;
    constexpr CoreErrorDomain() noexcept : ErrorDomain(kId) {}
    ~CoreErrorDomain() = default;
    char const* Name() const noexcept override { return "Core"; }
    char const* Message(ErrorDomain::CodeType errorCode) const noexcept override {
#ifdef AOS_TAINT
        Coverity_Tainted_Set((void*)&errorCode);
#endif
        switch (static_cast<Errc>(errorCode)) {
            case Errc::kInvalidArgument:
                return "Invalid argument";
            case Errc::kInvalidMetaModelShortname:
                return "Invalid meta model short name";
            case Errc::kInvalidMetaModelPath:
                return "Invalid meta model path";
            default:
                hozon::netaos::core::Abort("Unknown core error");
                return "Unknown core error";
        }
    }
    void ThrowAsException(ErrorCode const& code) const noexcept(false) override {
#ifdef AOS_TAINT
        Coverity_Tainted_Set((void*)&code);
#endif
        ThrowOrTerminate<Exception>(code);
    }

   private:
    constexpr static ErrorDomain::IdType kId = 0x8000000000000014;
};

namespace internal {
constexpr CoreErrorDomain CORE_ERROR_DOMAIN;
}

constexpr ErrorDomain const& GetCoreDomain() noexcept { return internal::CORE_ERROR_DOMAIN; }

constexpr ErrorCode MakeErrorCode(CoreErrc code, ErrorDomain::SupportDataType data) noexcept { return ErrorCode(static_cast<ErrorDomain::CodeType>(code), GetCoreDomain(), data); }
}  // namespace core
}  // namespace netaos
}  // namespace hozon

#endif
