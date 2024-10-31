
#include <iostream>
#include <vector>

namespace hozon {
namespace netaos {
namespace tools {


class CpuInfo {
public:
    CpuInfo(std::vector<std::string> arguments)
        :arguments_(arguments)
    {}
    ~CpuInfo(){}
    int32_t StartGetCpuInfo();

private:

    std::vector<std::string> arguments_;
};


}
}
}
