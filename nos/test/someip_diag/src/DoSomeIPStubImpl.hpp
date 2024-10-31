// DoSomeIPStubImpl.hpp
#ifndef DOSOMEIPSTUBIMPL_H_
#define DOSOMEIPSTUBIMPL_H_
#include <CommonAPI/CommonAPI.hpp>
#include <v1/commonapi/DoSomeIPStubDefault.hpp>

class DoSomeIPStubImpl: public v1_0::commonapi::DoSomeIPStubDefault {
public:
    DoSomeIPStubImpl();
    virtual ~DoSomeIPStubImpl();
    virtual void udsMessageRequest(const std::shared_ptr<CommonAPI::ClientId> _client, v1::commonapi::DoSomeIP::DoSomeIPReqUdsMessage _req, udsMessageRequestReply_t _reply);
};
#endif /* DOSOMEIPSTUBIMPL_H_ */