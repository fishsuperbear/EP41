

namespace hozon {
namespace netaos {
namespace tools {


class IfDataInfo {
public:
    IfDataInfo(std::vector<std::string> arguments)
        :arguments_(arguments)
    {}
    ~IfDataInfo(){}
    void PrintUsage();
    int32_t StartGetIfdata();

private:

    std::vector<std::string> arguments_;
};


}
}
}
