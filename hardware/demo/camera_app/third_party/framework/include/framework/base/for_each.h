#ifndef CYBER_BASE_FOR_EACH_H_
#define CYBER_BASE_FOR_EACH_H_

#include <type_traits>

#include "framework/base/macros.h"

namespace netaos {
namespace framework {
namespace base {

DEFINE_TYPE_TRAIT(HasLess, operator<)  // NOLINT

template <class Value, class End>
typename std::enable_if<HasLess<Value>::value && HasLess<End>::value,
                        bool>::type
LessThan(const Value& val, const End& end) {
  return val < end;
}

template <class Value, class End>
typename std::enable_if<!HasLess<Value>::value || !HasLess<End>::value,
                        bool>::type
LessThan(const Value& val, const End& end) {
  return val != end;
}

#define FOR_EACH(i, begin, end)           \
  for (auto i = (true ? (begin) : (end)); \
       netaos::framework::base::LessThan(i, (end)); ++i)

}  // namespace base
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_BASE_FOR_EACH_H_
