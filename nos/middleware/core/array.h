#ifndef NETAOS_CORE_ARRAY_H
#define NETAOS_CORE_ARRAY_H
#include <array>
namespace hozon {

namespace netaos {
namespace core {
template <class T, std::size_t N>
using Array = std::array<T, N>;
}
}  // namespace netaos
}  // namespace hozon
#endif
