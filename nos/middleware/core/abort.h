#ifndef NETAOS_CORE_ABORT_H
#define NETAOS_CORE_ABORT_H
#include <iostream>
#include <string>

namespace hozon {

namespace netaos {
namespace core {
/**
 * @brief Terminate the current process abnormally [SWS_CORE_00052].
 *
 * @pnetaosm[in]   text a custom text to include in the log message being output
 */
static void Abort(char const* text) noexcept { std::cout << std::string(text) << std::endl; }
}  // namespace core
}  // namespace netaos
}  // namespace hozon
#endif
