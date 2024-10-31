#include "perception1.h"
#include "proto/test/soc/for_test.pb.h"
#include "adf/include/node_proto_register.h"

Perception1::Perception1(){

}

Perception1::~Perception1() {

}

int32_t Perception1::AlgInit() {
    REGISTER_PROTO_MESSAGE_TYPE("link_sample_topic2", ::adf::lite::dbg::WorkflowResult);
    RegistAlgProcessFunc("receive_workflow1", std::bind(&Perception1::ReceiveWorkFlow1, this, std::placeholders::_1));
    RegistAlgProcessWithProfilerFunc("receive_link_sample2", std::bind(&Perception1::ReceiveLinkSample2, this, std::placeholders::_1, std::placeholders::_2));
    return 0;
}

void Perception1::AlgRelease() {
    NODE_LOG_INFO << "Release Perception1";
}

int32_t Perception1::ReceiveWorkFlow1(Bundle* input) {
    BaseDataTypePtr workresult1 = input->GetOne("workresult1");
    if (!workresult1) {
        NODE_LOG_INFO << "ReceiveWorkFlow1 recv workresult1 null";
        return -1;
    }

    std::shared_ptr<adf::lite::dbg::WorkflowResult> workresult1_proto = std::static_pointer_cast<adf::lite::dbg::WorkflowResult>(workresult1->proto_msg);
    NODE_LOG_INFO << "ReceiveWorkFlow1 recv workresult1 " << workresult1_proto->val1() << ", " << workresult1_proto->val2() << ", " << workresult1_proto->val3() << ", " << workresult1_proto->val4();

    return 0;
}
BaseDataTypePtr GenWorkResult(int64_t i) {
    BaseDataTypePtr workflow_result = std::make_shared<BaseData>();
    std::shared_ptr<adf::lite::dbg::WorkflowResult> workflow_result_msg(new adf::lite::dbg::WorkflowResult);
    workflow_result_msg->set_val1(i);
    workflow_result_msg->set_val2(i + 100);
    workflow_result_msg->set_val3(i + 200);
    workflow_result_msg->set_val4(i + 300);

    workflow_result_msg->mutable_header()->set_seq(i);
    workflow_result_msg->mutable_header()->set_publish_stamp(GetRealTimestamp());
    workflow_result->proto_msg = workflow_result_msg;

    return workflow_result;
}

int32_t Perception1::ReceiveLinkSample2(Bundle* input, const ProfileToken& token) {
    BaseDataTypePtr workresult = input->GetOne("link_sample_topic2");
    if (!workresult) {
        NODE_LOG_INFO << "Perception1::ReceiveLinkSample2 None";
        return -1;
    }
    NODE_LOG_INFO << "Perception1::ReceiveLinkSample2 OK";
    NODE_LOG_INFO << "ReceiveLinkSample2 link_sample_topic2 timestamp is: " << workresult->__header.timestamp_real_us;
    std::shared_ptr<adf::lite::dbg::WorkflowResult> workresult_proto =
        std::static_pointer_cast<adf::lite::dbg::WorkflowResult>(workresult->proto_msg);
    NODE_LOG_INFO << "ReceiveLinkSample2 link_sample_topic2 recv " << workresult_proto->val1() << ", "
                  << workresult_proto->val2() << ", " << workresult_proto->val3() << ", " << workresult_proto->val4();
    if (token.latency_info.data.size() > 0) {
        NODE_LOG_INFO << "link_sample_topic2 latency size:" << token.latency_info.data.size();
        for (auto iter = token.latency_info.data.begin(); iter != token.latency_info.data.end(); iter++) {
            NODE_LOG_INFO << "link_sample_topic2 latency link:" << iter->first << " sec: " << iter->second.sec
                          << " nsec: " << iter->second.nsec;
        }
    }
    BaseDataTypePtr link_sample_topic3_data = GenWorkResult(workresult_proto->val1() + 10000);
    NODE_LOG_INFO << "send link_sample_topic3 timestamp is: " << link_sample_topic3_data->__header.timestamp_real_us;
    SendOutput("link_sample_topic3", link_sample_topic3_data, token);
    return 0;
}
