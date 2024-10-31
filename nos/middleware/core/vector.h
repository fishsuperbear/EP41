#ifndef NETAOS_CORE_VECTOR_H
#define NETAOS_CORE_VECTOR_H
#include <vector>
namespace hozon {

namespace netaos {
namespace core {
template <typename T>
using vector = std::vector<T>;

template <typename T, typename Allocator = std::allocator<T>>
using Vector = std::vector<T, Allocator>;

template <typename T, typename Allocator>
void swap(Vector<T, Allocator> const& lhs, Vector<T, Allocator> const& rhs) {
    return std::swap(lhs, rhs);
}
}  // namespace core
}  // namespace netaos
}  // namespace hozon
#endif
