#include "fisheye_mock_executor.h"
#include "adf-lite/include/writer.h"
#include "proto/test/soc/for_test.pb.h"

FisheyeMockExecutor::FisheyeMockExecutor() {}

FisheyeMockExecutor::~FisheyeMockExecutor() {}

int32_t FisheyeMockExecutor::AlgInit() {
    NODE_LOG_INFO << "Init fisheye mock.";
    RegistAlgProcessFunc("fisheye_front_yield", std::bind(&FisheyeMockExecutor::FisheyeFrontYield, this, std::placeholders::_1));
    RegistAlgProcessFunc("fisheye_left_yield", std::bind(&FisheyeMockExecutor::FisheyeLeftYield, this, std::placeholders::_1));
    RegistAlgProcessFunc("fisheye_right_yield", std::bind(&FisheyeMockExecutor::FisheyeRightYield, this, std::placeholders::_1));
    RegistAlgProcessFunc("fisheye_rear_yield", std::bind(&FisheyeMockExecutor::FisheyeRearYield, this, std::placeholders::_1));
    return 0;
}

void FisheyeMockExecutor::AlgRelease() {
}

BaseDataTypePtr GenWorkResult(int64_t i) {
    BaseDataTypePtr workflow_result = std::make_shared<BaseData>();
    std::shared_ptr<adf::lite::dbg::WorkflowResult> workflow_result_msg(new adf::lite::dbg::WorkflowResult);
    workflow_result_msg->set_val1(i);
    workflow_result_msg->set_val2(i + 100);
    workflow_result_msg->set_val3(i + 200);
    workflow_result_msg->set_val4(i + 300);

    workflow_result_msg->mutable_header()->set_seq(i);
    double current_time = GetRealTimestamp();
    workflow_result_msg->mutable_header()->set_publish_stamp(current_time);
    workflow_result->proto_msg = workflow_result_msg;

    return workflow_result;
}

int32_t FisheyeMockExecutor::FisheyeFrontYield(Bundle* input) {
    static int64_t i_front = 1000000;
    ++i_front;
    BaseDataTypePtr workflow_result = GenWorkResult(i_front);
    NODE_LOG_INFO << "FisheyeFront send i = " << i_front;
    SendOutput("fisheye_front", workflow_result);
    return 0;
}

int32_t FisheyeMockExecutor::FisheyeLeftYield(Bundle* input) {
    static int64_t i_left = 2000000;
    ++i_left;
    BaseDataTypePtr workflow_result = GenWorkResult(i_left);
    NODE_LOG_INFO << "FisheyeLeft send i = " << i_left;
    SendOutput("fisheye_left", workflow_result);

    return 0;
}

int32_t FisheyeMockExecutor::FisheyeRightYield(Bundle* input) {
    static int64_t i_right = 3000000;
    ++i_right;
    BaseDataTypePtr workflow_result = GenWorkResult(i_right);
    NODE_LOG_INFO << "FisheyeRight send i = " << i_right;
    SendOutput("fisheye_right", workflow_result);

    return 0;
}

int32_t FisheyeMockExecutor::FisheyeRearYield(Bundle* input) {
    static int64_t i_rear = 4000000;
    ++i_rear;
    BaseDataTypePtr workflow_result = GenWorkResult(i_rear);
    NODE_LOG_INFO << "FisheyeRear send i = " << i_rear;
    SendOutput("fisheye_rear", workflow_result);

    return 0;
}
