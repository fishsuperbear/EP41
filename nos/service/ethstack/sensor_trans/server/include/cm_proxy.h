#pragma once
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include "adf/include/node_proto_register.h"
#include "adf/include/node_base.h"
#include "logger.h"
#include "proto/planning/planning.pb.h"
#include "proto/soc/apa2mcu_chassis.pb.h"
#include "proto/perception/transport_element.pb.h"
#include "proto/localization/localization.pb.h"
#include "proto/perception/perception_obstacle.pb.h"
#include "proto/statemachine/state_machine.pb.h"


namespace hozon {
namespace netaos {
namespace sensor {
class AdfProxy : public adf::NodeBase {
public:
    AdfProxy() {
        REGISTER_PROTO_MESSAGE_TYPE("apa2chassis", hozon::soc::Apa2Chassis)
        REGISTER_PROTO_MESSAGE_TYPE("ego2chassis", hozon::planning::ADCTrajectory)
        REGISTER_PROTO_MESSAGE_TYPE("nnplane", hozon::perception::TransportElement)
        REGISTER_PROTO_MESSAGE_TYPE("hpplane", hozon::perception::TransportElement)
        REGISTER_PROTO_MESSAGE_TYPE("nnplocation", hozon::localization::Localization)
        REGISTER_PROTO_MESSAGE_TYPE("hpplocation", hozon::localization::Localization)
        REGISTER_PROTO_MESSAGE_TYPE("nnpobject", hozon::perception::PerceptionObstacles)
        REGISTER_PROTO_MESSAGE_TYPE("hppobject", hozon::perception::PerceptionObstacles)
        REGISTER_PROTO_MESSAGE_TYPE("sm2mcu", hozon::state::StateMachine)
	REGISTER_PROTO_MESSAGE_TYPE("parkinglot2hmi_2", hozon::soc::Apa2Chassis)
        REGISTER_PROTO_MESSAGE_TYPE("ihbc", hozon::soc::Apa2Chassis)
        REGISTER_PROTO_MESSAGE_TYPE("guard_mode", hozon::soc::Apa2Chassis)
        REGISTER_PROTO_MESSAGE_TYPE("mod", hozon::soc::Apa2Chassis)
        REGISTER_PROTO_MESSAGE_TYPE("tsrtlr", hozon::soc::Apa2Chassis)
    }
    ~AdfProxy(){
        SENSOR_EARLY_LOG << "AdfProxy deinit.";
    }
    virtual int32_t AlgInit() {
        return 0;
    }
    virtual void AlgRelease() {  }
};

class CmProxy {
private:
    
    // using Func = std::function<int(std::string, std::shared_ptr<void>)>;
    std::shared_ptr<AdfProxy> _proxy;
    // AdfProxy::AlgProcessFunc _func;
    // std::string _name;
    // std::mutex _proxy_mutex;

public:
    CmProxy();
    ~CmProxy() {
        SENSOR_EARLY_LOG << "CmProxy deinit.";
    };
    void WaitStop(void);
    int Register(std::string trigger, AdfProxy::AlgProcessFunc func);
    int Start(std::string &config);
    int Resume(std::string &name);
    int Pause(std::string &name);
    int Run();
    void Deinit();

};

}
} 
}
