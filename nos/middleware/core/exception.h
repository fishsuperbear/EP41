#ifndef NETAOS_CORE_EXPETION_H
#define NETAOS_CORE_EXPETION_H
#include <exception>

#include "core/error_code.h"

namespace hozon {

namespace netaos {
namespace core {
/**
 * @brief Base type for all AUTOSAR exception types [SWS_CORE_00601].
 *
 */
class Exception : public std::exception {
   public:
    /**
     * @brief Construct a new Exception object [SWS_CORE_00611]
     *
     * @pnetaosm[in] err   the ErrorCode
     */
    explicit Exception(ErrorCode err) noexcept : theErrorCode(err) {
#ifdef AOS_TAINT
        Coverity_Tainted_Set((void*)&err);
#endif
    }

    /**
     * @brief Destroy the Exception object
     *
     */
    ~Exception() override {}

    /**
     * @brief Return the explanatory string [SWS_CORE_00612].
     *
     * @return char const*   a null-terminated string
     */
    char const* what() const noexcept override { return theErrorCode.Message().data(); }

    /**
     * @brief Return the embedded ErrorCode that was given to the constructor [SWS_CORE_00613].
     *
     * @return ErrorCode const&   reference to the embedded ErrorCode
     */
    ErrorCode const& Error() const noexcept { return theErrorCode; }

   private:
    ErrorCode const theErrorCode;
};
}  // namespace core
}  // namespace netaos
}  // namespace hozon
#endif
