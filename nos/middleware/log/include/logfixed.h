#ifndef __LOGFIXED_H__
#define __LOGFIXED_H__

#include <cstring>
#include <iostream>
#include <limits>

namespace hozon {
namespace netaos {
namespace log {

// 相当于 std::fixed 输出浮点数
class Fixed {
   public:
    Fixed() {}

    ~Fixed() {}
};

#define FIXED Fixed()

}  // namespace log
}  // namespace netaos
}  // namespace hozon
#endif  // __LOGFIXED_H__
