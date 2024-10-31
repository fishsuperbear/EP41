
#include <iostream>
#include <vector>

namespace hozon {
namespace netaos {
namespace tools {

class IostatInfo {
public:
    IostatInfo(std::vector<std::string> arguments)
        :arguments_(arguments)
    {}
    ~IostatInfo(){}
    void PrintUsage();
    int32_t StartGetIostat();

private:
    std::vector<std::string> arguments_;
};


}
}
}
