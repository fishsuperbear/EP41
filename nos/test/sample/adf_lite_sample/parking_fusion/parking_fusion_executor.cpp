#include "parking_fusion_executor.h"
#include "adf-lite/include/ds/builtin_types.h"
#include "proto/test/soc/for_test.pb.h"
#include "adf/include/node_proto_register.h"
#include "adf-lite/service/rpc/lite_rpc.h"
#include "adf-lite/include/reader.h"

ParkingFusionExecutor::ParkingFusionExecutor() : _recv_status(true) {

}

ParkingFusionExecutor::~ParkingFusionExecutor() {

}

int32_t ParkingFusionExecutor::AlgInit() {
    REGISTER_PROTO_MESSAGE_TYPE("proto_sample_topic", ::adf::lite::dbg::WorkflowResult)
    DsLogger::GetInstance()._logger.Init("PAFU", ADF_LOG_LEVEL_INFO);
    PAFU_LOG_INFO << "Init parking fusion.";
    RegistAlgProcessFunc("receive_cm_topic", std::bind(&ParkingFusionExecutor::ReceiveCmTopic, this, std::placeholders::_1));
    RegistAlgProcessFunc("show_latest", std::bind(&ParkingFusionExecutor::ShowLatest, this));
    RegistAlgProcessFunc("receive_workflow1", std::bind(&ParkingFusionExecutor::ReceiveWorkFlow1, this, std::placeholders::_1));
    RegistAlgProcessFunc("receive_workflow2", std::bind(&ParkingFusionExecutor::ReceiveWorkFlow2, this, std::placeholders::_1));
    RegistAlgProcessFunc("recv_status_change", std::bind(&ParkingFusionExecutor::ReceiveStatusChange, this, std::placeholders::_1));
    return 0;
}

void ParkingFusionExecutor::AlgRelease() {
    PAFU_LOG_INFO << "Release parking fusion.";
}

int32_t ParkingFusionExecutor::ReceiveCmTopic(Bundle* input) {
    BaseDataTypePtr workresult = input->GetOne("proto_sample_topic");
    if (!workresult) {
        return -1;
    }

    std::shared_ptr<adf::lite::dbg::WorkflowResult> workresult_proto = 
        std::static_pointer_cast<adf::lite::dbg::WorkflowResult>(workresult->proto_msg);
    PAFU_LOG_INFO << "ParkingFusionExecutor:CmTopic proto_sample_topic recv " 
        << workresult_proto->val1() << ", "
        << workresult_proto->val2() << ", "
        << workresult_proto->val3() << ", "
        << workresult_proto->val4();

    return 0;
}

int32_t ParkingFusionExecutor::ReceiveFreeData(Bundle* input) {
    std::shared_ptr<Reader> reader(new Reader);

    reader->Init("free_data_topic", 5);
    BaseDataTypePtr workresult = reader->GetLatestOneBlocking(1000, true);
    if (workresult != nullptr) {
        std::shared_ptr<adf::lite::dbg::WorkflowResult> workresult_proto = 
        std::static_pointer_cast<adf::lite::dbg::WorkflowResult>(workresult->proto_msg);
    PAFU_LOG_INFO << "ParkingFusionExecutor:ReceiveFreeData " 
        << workresult_proto->val1() << ", "
        << workresult_proto->val2() << ", "
        << workresult_proto->val3() << ", "
        << workresult_proto->val4();
    }
    return 0;
}

int32_t ParkingFusionExecutor::ShowLatest() {
    auto func = LiteRpc::GetInstance().GetServiceFunc("GetLatestData");
    if (func == nullptr) {
        PAFU_LOG_INFO << "GetLatestData Func maybe not beed registed";
        return -1;
    } else {
        PAFU_LOG_INFO << "ParkingFusionExecutor call GetLatestData";
        BaseDataTypePtr workresult{nullptr};
        int32_t res = func(workresult);
        if (res < 0) {
            PAFU_LOG_INFO << "GetLatestData is Has Error";
            return -1;
        }

        if (workresult == nullptr) {
            PAFU_LOG_INFO << "workresult pointer is nullptr";
            return -1;
        }
        std::shared_ptr<adf::lite::dbg::WorkflowResult> workresult_proto = 
            std::static_pointer_cast<adf::lite::dbg::WorkflowResult>(workresult->proto_msg);
        PAFU_LOG_INFO << "ShowLatest: " 
            << workresult_proto->val1() << ", "
            << workresult_proto->val2() << ", "
            << workresult_proto->val3() << ", "
            << workresult_proto->val4();

    }

    return 0;
}

int32_t ParkingFusionExecutor::ReceiveWorkFlow1(Bundle* input) {
    BaseDataTypePtr workresult1 = input->GetOne("workresult1");
    if (!workresult1) {
        PAFU_LOG_INFO << "ReceiveWorkFlow1 recv workresult1 null";
        return -1;
    }

    std::shared_ptr<adf::lite::dbg::WorkflowResult> workresult1_proto = std::static_pointer_cast<adf::lite::dbg::WorkflowResult>(workresult1->proto_msg);
    PAFU_LOG_INFO << "ReceiveWorkFlow1 recv workresult1 " << workresult1_proto->val1() << ", " << workresult1_proto->val2() << ", " << workresult1_proto->val3() << ", " << workresult1_proto->val4();

    return 0;
}

int32_t ParkingFusionExecutor::ReceiveWorkFlow2(Bundle* input) {
    BaseDataTypePtr workresult2 = input->GetOne("workresult2");
    if (!workresult2) {
        PAFU_LOG_INFO << "ReceiveWorkFlow2 recv workresult2 null";
        return -1;
    }

    std::shared_ptr<adf::lite::dbg::WorkflowResult> workresult2_proto = std::static_pointer_cast<adf::lite::dbg::WorkflowResult>(workresult2->proto_msg);
    PAFU_LOG_INFO << "ReceiveWorkFlow2 recv workresult2 " << workresult2_proto->val1() << ", " << workresult2_proto->val2() << ", " << workresult2_proto->val3() << ", " << workresult2_proto->val4();

    return 0;
}

/* 暂停/恢复接收trigger receive_cm_topic */
int32_t ParkingFusionExecutor::ReceiveStatusChange(Bundle* input) {
    _recv_status = !_recv_status;
    if (_recv_status) {
        PAFU_LOG_INFO << "ParkingFusionExecutor ResumeTrigger receive_cm_topic";
        ResumeTrigger("receive_cm_topic");
    } else {
        PAFU_LOG_INFO << "ParkingFusionExecutor PauseTrigger receive_cm_topic";
        PauseTrigger("receive_cm_topic");
    }
    return 0;
}

