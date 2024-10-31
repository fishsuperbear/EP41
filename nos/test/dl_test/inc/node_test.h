#pragma once
#include <iostream>
#include "adf/include/node_base.h"
#include "idl/generated/chassis.h"
#include "idl/generated/chassisPubSubTypes.h"
using namespace hozon::netaos::adf;
using namespace hozon::netaos::log;

class Derived : public hozon::netaos::adf::NodeBase {
public: 
    Derived() {
        REGISTER_CM_TYPE_CLASS("chassis", AlgChassisInfoPubSubType);

        RegistAlgProcessWithProfilerFunc("main", 
            std::bind(&Derived::AlgProcess1, this, std::placeholders::_1, std::placeholders::_2));
    }
    ~Derived() {
    }
    virtual int32_t AlgInit() { return 0; }
    virtual int32_t AlgProcess1(hozon::netaos::adf::NodeBundle* input,
                               const hozon::netaos::adf::ProfileToken& token);
    virtual void AlgRelease() { }

};
