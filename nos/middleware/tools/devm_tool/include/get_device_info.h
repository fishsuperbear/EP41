
#include <iostream>
#include <vector>

namespace hozon {
namespace netaos {
namespace tools {


class DeviceInfo {
public:
    DeviceInfo(std::vector<std::string> arguments)
        :arguments_(arguments)
    {}
    ~DeviceInfo(){}
    int32_t StartGetDeviceInfo();

private:

    std::vector<std::string> arguments_;
};


}
}
}
