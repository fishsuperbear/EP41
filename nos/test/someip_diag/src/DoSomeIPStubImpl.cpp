// DoSomeIPStubImpl.cpp
#include "DoSomeIPStubImpl.hpp"

DoSomeIPStubImpl::DoSomeIPStubImpl() { }
DoSomeIPStubImpl::~DoSomeIPStubImpl() { }

void DoSomeIPStubImpl::udsMessageRequest(const std::shared_ptr<CommonAPI::ClientId> _client, v1::commonapi::DoSomeIP::DoSomeIPReqUdsMessage _req, udsMessageRequestReply_t _reply)
{

	std::cout << "Hello , I am UDS server."<< "'\n";

    // 打印
	std::cout << "UDSReq from client is : "<< "'\n";
    std::cout << "udsSa: " << _req.getUdsSa() << std::endl;
    std::cout << "udsTa: " << _req.getUdsTa() << std::endl;
    std::cout << "taType: " << _req.getTaType().toString() << std::endl;
    std::vector<uint8_t> tmp = _req.getUdsData();
    std::cout << "udsData size is: " << tmp.size() << std::endl;
    std::cout << "udsData: ";
    for (const auto& value : tmp) {
        std::cout << std::hex << static_cast<int>(value) << " ";
    }
    std::cout << std::endl;

	v1::commonapi::DoSomeIP::DoSomeIPRespUdsMessage resp{};
    uint16_t source_addr = 8888;
    uint16_t target_addr = 6666;
    uint32_t result = 0;
    v1::commonapi::DoSomeIP::TargetAddressType type = v1::commonapi::DoSomeIP::TargetAddressType::kFunctional;
    v1::commonapi::DoSomeIP::my_array array = {3,1,0,1,0,1,2,3,4,5,6,7,8,9};
    resp.setUdsSa(source_addr);
    resp.setUdsTa(target_addr);
    resp.setResult(result);
    resp.setTaType(type);
    resp.setUdsData(array);

    v1::commonapi::DoSomeIP::stdErrorTypeEnum  methodError = v1::commonapi::DoSomeIP::stdErrorTypeEnum::OK;

    std::cout << "resp end !!!" << std::endl;
    _reply(methodError, resp);
}