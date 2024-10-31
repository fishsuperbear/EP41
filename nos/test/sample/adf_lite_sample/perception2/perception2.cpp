#include "perception2.h"
#include "proto/test/soc/for_test.pb.h"
#include "adf/include/node_proto_register.h"

Perception2::Perception2(){

}

Perception2::~Perception2() {

}

int32_t Perception2::AlgInit() {
    NODE_LOG_INFO << "AlgInit ========================== ";
    REGISTER_PROTO_MESSAGE_TYPE("link_sample_topic4", ::adf::lite::dbg::WorkflowResult)
    RegistAlgProcessFunc("receive_workflow1", std::bind(&Perception2::ReceiveWorkFlow1, this, std::placeholders::_1));
    RegistAlgProcessWithProfilerFunc("receive_link_sample4", std::bind(&Perception2::ReceiveLinkSample4, this, std::placeholders::_1, std::placeholders::_2));
    return 0;
}

void Perception2::AlgRelease() {
    NODE_LOG_INFO << "Release Perception2";
}

int32_t Perception2::ReceiveWorkFlow1(Bundle* input) {
    BaseDataTypePtr workresult1 = input->GetOne("workresult1");
    if (!workresult1) {
        NODE_LOG_INFO << "ReceiveWorkFlow1 recv workresult1 null";
        return -1;
    }

    std::shared_ptr<adf::lite::dbg::WorkflowResult> workresult1_proto = std::static_pointer_cast<adf::lite::dbg::WorkflowResult>(workresult1->proto_msg);
    NODE_LOG_INFO << "ReceiveWorkFlow1 recv workresult1 " << workresult1_proto->val1() << ", " << workresult1_proto->val2() << ", " << workresult1_proto->val3() << ", " << workresult1_proto->val4();

    return 0;
}

int32_t Perception2::ReceiveLinkSample4(Bundle* input, const ProfileToken& token) {
    BaseDataTypePtr workresult = input->GetOne("link_sample_topic4");
    if (!workresult) {
        NODE_LOG_INFO << "Perception2::ReceiveLinkSample4 None";
        return -1;
    }
    NODE_LOG_INFO << "ReceiveLinkSample4 link_sample_topic4 timestamp is: " << workresult->__header.timestamp_real_us;
    NODE_LOG_INFO << "Perception2::ReceiveLinkSample4 OK";
    std::shared_ptr<adf::lite::dbg::WorkflowResult> workresult_proto =
        std::static_pointer_cast<adf::lite::dbg::WorkflowResult>(workresult->proto_msg);
    NODE_LOG_INFO << "ReceiveLinkSample4 link_sample_topic4 recv " << workresult_proto->val1() << ", "
                  << workresult_proto->val2() << ", " << workresult_proto->val3() << ", " << workresult_proto->val4();
    if (token.latency_info.data.size() > 0) {
        NODE_LOG_INFO << "link_sample_topic4 latency size:" << token.latency_info.data.size();
        for (auto iter = token.latency_info.data.begin(); iter != token.latency_info.data.end(); iter++) {
            NODE_LOG_INFO << "link_sample_topic4 latency link:" << iter->first << " sec: " << iter->second.sec
                          << " nsec: " << iter->second.nsec;
        }
    }
    return 0;
}
