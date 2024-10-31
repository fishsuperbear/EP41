

namespace hozon {
namespace netaos {
namespace tools {

class ReadDidInfo {
public:
    ReadDidInfo(std::vector<std::string> arguments)
        :arguments_(arguments)
    {}
    ~ReadDidInfo(){}
    int32_t StartReadDid();

private:
    
    std::vector<std::string> arguments_;
};


}
}
}
