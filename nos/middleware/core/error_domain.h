#ifndef NETAOS_CORE_ERROR_DOMAIN_H
#define NETAOS_CORE_ERROR_DOMAIN_H
#include <cstdint>
namespace hozon {

namespace netaos {
namespace core {
class ErrorCode;

/**
 * @brief Encapsulation of an error domain [SWS_CORE_00110].
 *
 */
class ErrorDomain {
   public:
    /**
     * @brief Alias type for a unique ErrorDomain identifier type [SWS_CORE_00121].
     *
     */
    using IdType = std::uint64_t;

    /**
     * @brief Alias type for a domain-specific error code value [SWS_CORE_00122].
     *
     */
    using CodeType = std::int32_t;

    /**
     * @brief Alias type for vendor-specific supplementary data [SWS_CORE_00123].
     *
     */
    using SupportDataType = std::int32_t;

    /**
     * @brief Copy construction shall be disabled [SWS_CORE_00131].
     *
     */
    ErrorDomain(ErrorDomain const&) = delete;

    /**
     * @brief Move construction shall be disabled [SWS_CORE_00132].
     *
     */
    ErrorDomain(ErrorDomain&&) = delete;

    /**
     * @brief Copy assignment shall be disabled [SWS_CORE_00133].
     *
     */
    ErrorDomain& operator=(ErrorDomain const&) = delete;

    /**
     * @brief Move assignment shall be disabled [SWS_CORE_00134].
     *
     */
    ErrorDomain& operator=(ErrorDomain&&) = delete;

    /**
     * @brief Compare for equality with another ErrorDomain instance [SWS_CORE_00137].
     *
     * @pnetaosm[in]  other    the other instance
     * @return     bool     true if other is equal to *this, false otherwise
     */
    constexpr bool operator==(ErrorDomain const& other) const noexcept { return uniqueId_ == other.uniqueId_; }

    /**
     * @brief Compare for non-equality with another ErrorDomain instance [SWS_CORE_00138].
     *
     * @pnetaosm[in]   other   the other instance
     * @return      bool    true if other is not equal to *this, false otherwise
     */
    constexpr bool operator!=(ErrorDomain const& other) const noexcept { return uniqueId_ != other.uniqueId_; }

    /**
     * @brief Return the unique domain identifier [SWS_CORE_00151].
     *
     * @return IdType the identifier
     */
    constexpr IdType Id() const noexcept { return uniqueId_; }

    virtual char const* Name() const noexcept = 0;
    virtual char const* Message(CodeType errorCode) const noexcept = 0;
    virtual void ThrowAsException(ErrorCode const& errorCode) const noexcept(false) = 0;

   protected:
    /**
     * @brief Construct a new instance with the given identifier [SWS_CORE_00135].
     *
     * @pnetaosm[in] id the unique identifier
     */
    explicit constexpr ErrorDomain(IdType id) noexcept : uniqueId_(id) {}

    /**
     * @brief Destructor [SWS_CORE_00136].
     *
     */
    ~ErrorDomain() = default;

   private:
    IdType uniqueId_;
};
}  // namespace core
}  // namespace netaos
}  // namespace hozon
#endif
