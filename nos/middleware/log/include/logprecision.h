#ifndef __LOGPRECISION_H__
#define __LOGPRECISION_H__

#include <cstring>
#include <iostream>
#include <limits>

namespace hozon {
namespace netaos {
namespace log {

// 小数点后面的精度，设置多少即保留多少位有效数字
class DataPrecison {
   public:
    DataPrecison(int num) {
        if (num > kMaxFileNameSize)
            precision = kMaxFileNameSize;
        else
            precision = num;
    }

    ~DataPrecison() {}
    int precision;

   private:
    // double类型最大精度
    static constexpr int kMaxFileNameSize {std::numeric_limits<double>::digits10 + 1};
};

#define SET_PRECISION(num) DataPrecison(num)

}  // namespace log
}  // namespace netaos
}  // namespace hozon
#endif  // __LOGPRECISION_H__
