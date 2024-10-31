
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include "cm_proxy.h"
#include "idl/generated/cm_protobuf.h"
#include "logger.h"
#include "third_party/orin/fast-dds/include/fastdds/dds/topic/TopicDataType.hpp"
#include "idl/generated/cm_protobufPubSubTypes.h"


namespace hozon {
namespace netaos {
namespace sensor {

CmProxy::CmProxy() {
    _proxy = std::make_shared<AdfProxy>();
    // _proxy = std::make_shared<hozon::netaos::cm::Proxy>(std::make_shared<CmProtoBufPubSubType>()); 
     
    // SENSOR_LOG_INFO << "Create cm " << name << " Proxy. " << " func " << &func;
    // if(0 == _proxy->Init(domainID, topic)) {
    //     _proxy->Listen(std::bind(&CmProxy::Receive, this));
    // }  
    // else {
    //     SENSOR_LOG_WARN << "Init domain ( " << domainID << " ), Topic ( " 
    //         << topic << " ) fail.";
    // }
}

int CmProxy::Register(std::string trigger, AdfProxy::AlgProcessFunc func) {
    if(_proxy != nullptr) {
        _proxy->RegistAlgProcessFunc(trigger, func);
    }
    return 0;
} 

int CmProxy::Start(std::string &config) {
    if(_proxy != nullptr) {
        _proxy->Start(config);
        return 0;
    }
    return -1;
} 
int CmProxy::Pause(std::string &name) {
    if(_proxy != nullptr) {
        _proxy->PauseTrigger(name);
        return 0;
    }
    return -1;
} 

int CmProxy::Resume(std::string &name) {
    if(_proxy != nullptr) {
        _proxy->ResumeTrigger(name);
        return 0;
    }
    return -1;
}

void CmProxy::WaitStop(void) {
    if (_proxy != nullptr) {
        _proxy->NeedStopBlocking();
    }
}   
void CmProxy::Deinit() {
    if (_proxy != nullptr) {
        _proxy->Stop();
    }
}
int CmProxy::Run() {
    // std::shared_ptr<CmProtoBuf> idl_msg = std::make_shared<CmProtoBuf>();
    // std::lock_guard<std::mutex> lock(_proxy_mutex);
    // SENSOR_LOG_INFO << "Cm proxy write data " << _name << " _func " ;
    // if(_func != nullptr) {
    //     auto funcPtr = 
    //             _func.target<int (*)(std::string, std::shared_ptr<void>)>();
    //     if (funcPtr != nullptr) {
    //         std::cout << "Function address: " << funcPtr << std::endl;
    //     } else {
    //         std::cout << "Function pointer is null." << std::endl;
    //     }
    //     _func(_name , idl_msg);
    // }
    // else {
    //     SENSOR_LOG_WARN << " Func is nullptr.";
    // }
    return 0;
}

}
} 
}