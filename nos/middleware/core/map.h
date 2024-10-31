#ifndef NETAOS_CORE_MAP_H
#define NETAOS_CORE_MAP_H
#include <map>
namespace hozon {

namespace netaos {
namespace core {
template <typename K, typename V, typename C = std::less<K>, typename Allocator = std::allocator<std::pair<const K, V>>>
using Map = std::map<K, V, C, Allocator>;

template <typename K, typename V, typename C = std::less<K>, typename Allocator = std::allocator<std::pair<const K, V>>>
using map = std::map<K, V, C, Allocator>;
}  // namespace core
}  // namespace netaos
}  // namespace hozon
#endif
