
#include <iostream>
#include "gflags/gflags.h"
#include "fisheye_perception_executor.h"
#include "adf-lite/include/ds/builtin_types.h"
#include "adf-lite/include/writer.h"
#include "adf-lite/service/rpc/lite_rpc.h"
#include "adf/include/node_proto_register.h"
#include "fisheye_datatype.h"
#include "adf-lite/include/struct_register.h"
#include "proto/test/soc/for_test.pb.h"

DEFINE_string(config_file, "/app/runtime_service/adf-lite-process/conf/config.ini", "Path to the configuration file");
DEFINE_bool(async_state, true, "async_state");
DEFINE_string(perception_config, "/app/conf/perception_common_onboard_config.yaml", "perception_config");
DEFINE_string(phm_config, "/app/conf/phm_config.yaml", "phm_config");

FisheyePerceptionExecutor::FisheyePerceptionExecutor() : _recv_status(0){}

FisheyePerceptionExecutor::~FisheyePerceptionExecutor() {}
std::string key = "test";
std::string key1 = "~test";

void uint8func(const std::string& clientname, const std::string& key, const int32_t& value) {
    NODE_LOG_INFO << "uint8func func receive the event that the value of param:test"
                  << " is set to " << clientname << "  key: " << key << "  value: " << (int32_t)value;
}
int32_t FisheyePerceptionExecutor::AlgInit() {
    REGISTER_PROTO_MESSAGE_TYPE("proto_sample_topic", ::adf::lite::dbg::WorkflowResult)
    REGISTER_PROTO_MESSAGE_TYPE("link_sample_topic", ::adf::lite::dbg::WorkflowResult)
    REGISTER_STRUCT_TYPE("plain_struct_topic_test", TestPlainStruct)
    REGISTER_STRUCT_TYPE("notplain_struct_topic_test", TestNotPlainStruct)

    CfgResultCode res = ConfigParam::Instance()->MonitorParam<int32_t>(key, uint8func);
    NODE_LOG_INFO << "MonitorParam  key: " << key << "  res:" << res;
    res = ConfigParam::Instance()->MonitorParam<int32_t>(key1, uint8func);
    NODE_LOG_INFO << "MonitorParam  key: " << key1 << "  res:" << res;

    NODE_LOG_INFO << "Init fisheye perception.";
    RegistAlgProcessFunc("recv_fisheye_img", std::bind(&FisheyePerceptionExecutor::ImageProcess, this, std::placeholders::_1));
    RegistAlgProcessFunc("fisheye_event_check1", std::bind(&FisheyePerceptionExecutor::FisheyeEventCheck1, this, std::placeholders::_1));
    RegistAlgProcessFunc("fisheye_event_check2", std::bind(&FisheyePerceptionExecutor::FisheyeEventCheck2, this, std::placeholders::_1));
    RegistAlgProcessFunc("workflow1", std::bind(&FisheyePerceptionExecutor::Workflow1, this, std::placeholders::_1));
    RegistAlgProcessFunc("workflow2", std::bind(&FisheyePerceptionExecutor::Workflow2, this, std::placeholders::_1));
    RegistAlgProcessFunc("workflow3", std::bind(&FisheyePerceptionExecutor::Workflow3, this, std::placeholders::_1));
    RegistAlgProcessFunc("receive_struct_topic", std::bind(&FisheyePerceptionExecutor::ReceiveStructTopic, this, std::placeholders::_1));
    RegistAlgProcessFunc("receive_cm_topic", std::bind(&FisheyePerceptionExecutor::ReceiveCmTopic, this, std::placeholders::_1));
    RegistAlgProcessWithProfilerFunc("receive_link_sample", std::bind(&FisheyePerceptionExecutor::ReceiveLinkSample, this, std::placeholders::_1, std::placeholders::_2));
    RegistAlgProcessFunc("receive_plain_struct_topic", std::bind(&FisheyePerceptionExecutor::ReceivePlainStructTopic, this, std::placeholders::_1));
    RegistAlgProcessFunc("receive_notplain_struct_topic", std::bind(&FisheyePerceptionExecutor::ReceiveNotPlainStructTopic, this, std::placeholders::_1));
    RegistAlgProcessFunc("recv_status_change", std::bind(&FisheyePerceptionExecutor::ReceiveStatusChange, this, std::placeholders::_1));

    LiteRpc::GetInstance().RegisterServiceFunc("GetLatestData", std::bind(&FisheyePerceptionExecutor::GetLatestData, this, std::placeholders::_1));
    char* argv1[] = {""};
    char** argv2=argv1;
    char*** argv=&argv2;
    google::ReadFromFlagsFile(FLAGS_config_file, *argv[0], true);
    if (FLAGS_async_state) {
        NODE_LOG_INFO << "async_state is ture";
    } else {
        NODE_LOG_INFO << "async_state is false";
    }
    NODE_LOG_INFO << "perception_config is " << FLAGS_perception_config;
    NODE_LOG_INFO << "phm_config is " << FLAGS_phm_config;

    return 0;
}

void FisheyePerceptionExecutor::AlgRelease() {
    NODE_LOG_INFO << "Release fisheye perception.";
    CfgResultCode res = ConfigParam::Instance()->UnMonitorParam(key);
    NODE_LOG_INFO << "UnMonitorParam  key: " << key << "  res:" << res;
    res = ConfigParam::Instance()->UnMonitorParam(key1);
    NODE_LOG_INFO << "UnMonitorParam  key: " << key1 << "  res:" << res;
}

int32_t FisheyePerceptionExecutor::ImageProcess(Bundle* input) {
    NODE_LOG_INFO << "=========================== Enter ImageProcess ===========================";

    BaseDataTypePtr fisheye_front = input->GetOne("fisheye_front");
    if (!fisheye_front) {
        return -1;
    }

    BaseDataTypePtr fisheye_left = input->GetOne("fisheye_left");
    if (!fisheye_left) {
        return -1;
    }
    BaseDataTypePtr fisheye_right = input->GetOne("fisheye_right");
    if (!fisheye_right) {
        return -1;
    }
    BaseDataTypePtr fisheye_rear = input->GetOne("fisheye_rear");
    if (!fisheye_rear) {
        return -1;
    }

    std::shared_ptr<adf::lite::dbg::WorkflowResult> fisheye_front_proto = std::static_pointer_cast<adf::lite::dbg::WorkflowResult>(fisheye_front->proto_msg);
    NODE_LOG_INFO << "fisheye_front recv " << fisheye_front_proto->val1() << ", " << fisheye_front_proto->val2() << ", " << fisheye_front_proto->val3() << ", " << fisheye_front_proto->val4();

    std::shared_ptr<adf::lite::dbg::WorkflowResult> fisheye_left_proto = std::static_pointer_cast<adf::lite::dbg::WorkflowResult>(fisheye_left->proto_msg);
    NODE_LOG_INFO << "fisheye_left recv " << fisheye_left_proto->val1() << ", " << fisheye_left_proto->val2() << ", " << fisheye_left_proto->val3() << ", " << fisheye_left_proto->val4();

    std::shared_ptr<adf::lite::dbg::WorkflowResult> fisheye_right_proto = std::static_pointer_cast<adf::lite::dbg::WorkflowResult>(fisheye_right->proto_msg);
    NODE_LOG_INFO << "fisheye_right recv " << fisheye_right_proto->val1() << ", " << fisheye_right_proto->val2() << ", " << fisheye_right_proto->val3() << ", " << fisheye_right_proto->val4();

    std::shared_ptr<adf::lite::dbg::WorkflowResult> fisheye_rear_proto = std::static_pointer_cast<adf::lite::dbg::WorkflowResult>(fisheye_rear->proto_msg);
    NODE_LOG_INFO << "fisheye_rear recv " << fisheye_rear_proto->val1() << ", " << fisheye_rear_proto->val2() << ", " << fisheye_rear_proto->val3() << ", " << fisheye_rear_proto->val4();
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

int32_t FisheyePerceptionExecutor::Workflow1(Bundle* input) {
    static int64_t i = 0;
    ++i;
    NODE_LOG_INFO << "perception_config is " << FLAGS_perception_config;

    CfgResultCode res = ConfigParam::Instance()->SetParam<int32_t>(key, (int32_t)i);
    NODE_LOG_INFO << "SetParam  key: " << key << "  value: " << (int32_t)i << "  res:" << res;
    int32_t val = 0;
    res = ConfigParam::Instance()->GetParam<int32_t>(key, val);
    NODE_LOG_INFO << "GetParam  key: " << key << "  value: " << (int32_t)val << "  res:" << res;

    res = ConfigParam::Instance()->SetParam<int32_t>(key1, (int32_t)i);
    NODE_LOG_INFO << "SetParam  key: " << key1 << "  value: " << (int32_t)i << "  res:" << res;
    res = ConfigParam::Instance()->GetParam<int32_t>(key1, val);
    NODE_LOG_INFO << "GetParam  key: " << key1 << "  value: " << (int32_t)val << "  res:" << res;

    APP_OP_LOG_INFO << "Send workresult1, i = " << i;
    BaseDataTypePtr workflow_result = GenWorkResult(i);
    SendOutput("workresult1", workflow_result);

    return 0;
}

int32_t FisheyePerceptionExecutor::Workflow2(Bundle* input) {
    BaseDataTypePtr workresult1 = input->GetOne("workresult1");
    if (!workresult1) {
        return -1;
    }

    std::shared_ptr<adf::lite::dbg::WorkflowResult> workresult1_proto = std::static_pointer_cast<adf::lite::dbg::WorkflowResult>(workresult1->proto_msg);
    NODE_LOG_INFO << "Workflow2 recv workresult1 " << workresult1_proto->val1() << ", " << workresult1_proto->val2() << ", " << workresult1_proto->val3() << ", " << workresult1_proto->val4();
    int64_t i = workresult1_proto->val1();
    BaseDataTypePtr workflow_result = GenWorkResult(i);
    SendOutput("workresult2", workflow_result);

    return 0;
}

int32_t FisheyePerceptionExecutor::Workflow3(Bundle* input) {
    BaseDataTypePtr workresult2 = input->GetOne("workresult2");
    _lastet_data = workresult2;
    if (!workresult2) {
        return -1;
    }

    std::shared_ptr<adf::lite::dbg::WorkflowResult> workresult2_proto = std::static_pointer_cast<adf::lite::dbg::WorkflowResult>(workresult2->proto_msg);
    NODE_LOG_INFO << "Workflow3 recv workresult2 " << workresult2_proto->val1() << ", " << workresult2_proto->val2() << ", " << workresult2_proto->val3() << ", " << workresult2_proto->val4();

    int64_t i = workresult2_proto->val1();
    BaseDataTypePtr workflow_result = GenWorkResult(i);
    SendOutput("workresult3", workflow_result);

    std::shared_ptr<hozon::netaos::adf_lite::TestStruct> test_struct = std::make_shared<hozon::netaos::adf_lite::TestStruct>();
    test_struct->isValid = true;
    test_struct->info.push_back(100);
    test_struct->info.push_back(200);
    test_struct->info.push_back(300);
    test_struct->info.push_back(400);
    SendOutput("struct_topic_test", test_struct);
    NODE_LOG_INFO << "StructTopic struct_topic_test Send";

    std::shared_ptr<hozon::netaos::adf_lite::TestPlainStruct> test_plain_struct = std::make_shared<hozon::netaos::adf_lite::TestPlainStruct>();
    test_plain_struct->data = 111;
    test_plain_struct->data2 = 2222;

    SendOutput("plain_struct_topic_test", test_plain_struct);
    NODE_LOG_INFO << "StructTopic plain_struct_topic_test Send";

    std::shared_ptr<hozon::netaos::adf_lite::TestNotPlainStruct> test_notplain_struct = std::make_shared<hozon::netaos::adf_lite::TestNotPlainStruct>();
    test_notplain_struct->data = 111;
    test_notplain_struct->data2 = 2222;
    test_notplain_struct->data_str = "Hello World!!!";
    test_notplain_struct->data_vec.push_back(100);
    test_notplain_struct->data_vec.push_back(200);
    test_notplain_struct->data_vec.push_back(300);
    test_notplain_struct->data_vec.push_back(400);
    test_notplain_struct->data_map.emplace(1111, "AAAA");
    test_notplain_struct->data_map.emplace(2222, "BBBBBB");
    test_notplain_struct->data_map.emplace(3333, "CCCCCCCC");

    SendOutput("notplain_struct_topic_test", test_notplain_struct);
    NODE_LOG_INFO << "StructTopic notplain_struct_topic_test Send";

    return 0;
}

/* 接收Struct 数据类型的Topic，
 */
int32_t FisheyePerceptionExecutor::ReceiveStructTopic(Bundle* input) {
    BaseDataTypePtr workresult = input->GetOne("struct_topic_test");
    if (!workresult) {
        return -1;
    }
    std::shared_ptr<hozon::netaos::adf_lite::TestStruct> test_struct = std::static_pointer_cast<hozon::netaos::adf_lite::TestStruct>(workresult);

    if (test_struct->isValid) {
        NODE_LOG_INFO << "StructTopic struct_topic_test recv Valid: " << test_struct->info[0] << ", " << test_struct->info[1] << ", " << test_struct->info[2] << ", " << test_struct->info[3];
    } else {
        NODE_LOG_INFO << "StructTopic struct_topic_test recv inValid";
    }

    return 0;
}

/* 接收Plain Struct 数据类型的Topic */
int32_t FisheyePerceptionExecutor::ReceivePlainStructTopic(Bundle* input) {
    BaseDataTypePtr workresult2 = input->GetOne("plain_struct_topic_test");
    if (workresult2 == nullptr) {
        NODE_LOG_INFO << "ReceivePlainStructTopic recv inValid";
        return -1;
    }
    std::shared_ptr<hozon::netaos::adf_lite::TestPlainStruct> test_plain_struct = std::static_pointer_cast<hozon::netaos::adf_lite::TestPlainStruct>(workresult2);

    NODE_LOG_INFO << "StructTopic plain_struct_topic_test recv: " << test_plain_struct->data << ", " << test_plain_struct->data2;
    return 0;
}

/* 接收NotPlain Struct 数据类型的Topic */
int32_t FisheyePerceptionExecutor::ReceiveNotPlainStructTopic(Bundle* input) {
    BaseDataTypePtr workresult2 = input->GetOne("notplain_struct_topic_test");
    if (workresult2 == nullptr) {
        NODE_LOG_INFO << "ReceiveNotPlainStructTopic recv inValid";
        return -1;
    }
    std::shared_ptr<hozon::netaos::adf_lite::TestNotPlainStruct> test_notplain_struct = std::static_pointer_cast<hozon::netaos::adf_lite::TestNotPlainStruct>(workresult2);

    NODE_LOG_TRACE << "StructTopic notplain_struct_topic_test recv: " << test_notplain_struct->data << ", " << test_notplain_struct->data2  << ", " << test_notplain_struct->data_str;
    for (uint32_t i = 0; i < test_notplain_struct->data_vec.size(); i++) {
        NODE_LOG_DEBUG << "vector : " << test_notplain_struct->data_vec[i];
    }

    for (auto it = test_notplain_struct->data_map.begin(); it != test_notplain_struct->data_map.end(); it++) {
        NODE_LOG_INFO << "key: " << it->first << " value: " << it->second;
    }
    return 0;
}

/* 暂停/恢复接收trigger receive_cm_topic */
int32_t FisheyePerceptionExecutor::ReceiveStatusChange(Bundle* input) {
    _recv_status = (_recv_status + 1) % 4;
    NODE_LOG_INFO << "FisheyePerceptionExecutor _recv_status: " << _recv_status;
    switch (_recv_status) {
        case 0:
            NODE_LOG_INFO << "FisheyePerceptionExecutor PauseTrigger receive_cm_topic";
            PauseTrigger("receive_cm_topic");
            break;
        case 1:
            NODE_LOG_INFO << "FisheyePerceptionExecutor ResumeTrigger receive_cm_topic";
            ResumeTrigger("receive_cm_topic");
            break;
        case 2:
            NODE_LOG_INFO << "FisheyePerceptionExecutor PauseTrigger All";
            PauseTrigger();
            break;
        case 3:
            NODE_LOG_INFO << "FisheyePerceptionExecutor ResumeTrigger All";
            ResumeTrigger();
            break;
        default:
        break;
    }
    return 0;
}

/* 接收CM Topic，需要运行proto_send_sample，示例如下：
    cd ./nos/output/x86_2004/test/adf-proto-sample
    ./bin/proto_send_sample ./conf/proto_send_sample_conf.yaml
*/
int32_t FisheyePerceptionExecutor::ReceiveCmTopic(Bundle* input) {
    BaseDataTypePtr workresult = input->GetOne("proto_sample_topic");
    if (!workresult) {
        NODE_LOG_INFO << "FisheyePerceptionExecutor::ReceiveCmTopic None";
        return -1;
    }
    NODE_LOG_INFO << "FisheyePerceptionExecutor::ReceiveCmTopic OK";
    std::shared_ptr<adf::lite::dbg::WorkflowResult> workresult_proto = std::static_pointer_cast<adf::lite::dbg::WorkflowResult>(workresult->proto_msg);
    NODE_LOG_INFO << "CmTopic proto_sample_topic recv " << workresult_proto->val1() << ", " << workresult_proto->val2() << ", " << workresult_proto->val3() << ", " << workresult_proto->val4();
    return 0;
}

int32_t FisheyePerceptionExecutor::ReceiveLinkSample(Bundle* input, const ProfileToken& token) {
    BaseDataTypePtr workresult = input->GetOne("link_sample_topic");
    if (!workresult) {
        NODE_LOG_INFO << "FisheyePerceptionExecutor::ReceiveLinkSample None";
        return -1;
    }
    NODE_LOG_INFO << "ReceiveLinkSample link_sample_topic timestamp is: " << workresult->__header.timestamp_real_us;
    std::shared_ptr<adf::lite::dbg::WorkflowResult> workresult_proto =
        std::static_pointer_cast<adf::lite::dbg::WorkflowResult>(workresult->proto_msg);
    NODE_LOG_INFO << "ReceiveLinkSample proto_sample_topic recv " << workresult_proto->val1() << ", "
                  << workresult_proto->val2() << ", " << workresult_proto->val3() << ", " << workresult_proto->val4();
    // 再将workresult传送给下一个executor
    if (token.latency_info.data.size() > 0) {
        NODE_LOG_INFO << "link_sample_topic latency size:" << token.latency_info.data.size();
        for (auto iter = token.latency_info.data.begin(); iter != token.latency_info.data.end(); iter++) {
            NODE_LOG_INFO << "link_sample_topic latency link:" << iter->first << " sec: " << iter->second.sec
                          << " nsec: " << iter->second.nsec;
        }
    }

    BaseDataTypePtr link_sample_topic2_data = GenWorkResult(workresult_proto->val1() + 10000);
    NODE_LOG_INFO << "send link_sample_topic2_data timestamp is: "
                  << link_sample_topic2_data->__header.timestamp_real_us;
    SendOutput("link_sample_topic2", link_sample_topic2_data, token);
    return 0;
}

int32_t FisheyePerceptionExecutor::FreeDataTopic(Bundle* input) {
    Writer _writer;

    _writer.Init("free_data_topic");

    BaseDataTypePtr base_ptr = GenWorkResult(1111);
    int32_t ret = _writer.Write(base_ptr);
    if (ret < 0) {
        NODE_LOG_ERROR << "Fail to write "
                       << "free_data_topic";
        return -1;
    }

    return 0;
}

int32_t FisheyePerceptionExecutor::GetLatestData(BaseDataTypePtr& ptr) {
    ptr = _lastet_data;

    return 0;
}

int32_t FisheyePerceptionExecutor::FisheyeEventCheck1(Bundle* input) {
    BaseDataTypePtr fisheye_front = input->GetOne("fisheye_front");
    if (!fisheye_front) {
        return -1;
    }

    BaseDataTypePtr fisheye_left = input->GetOne("fisheye_rear");
    if (!fisheye_left) {
        return -1;
    }

    NODE_LOG_INFO << "FisheyePerceptionExecutor::FisheyeEventCheck1 Receive fisheye_front and fisheye_rear";
    static int64_t i = 10000;
    ++i;
    NODE_LOG_INFO << "Send fisheye_result1, i = " << i;
    BaseDataTypePtr workflow_result = GenWorkResult(i);
    SendOutput("fisheye_result1", workflow_result);
    return 0;
}

int32_t FisheyePerceptionExecutor::FisheyeEventCheck2(Bundle* input) {
    BaseDataTypePtr fisheye_front = input->GetOne("fisheye_front");
    if (!fisheye_front) {
        return -1;
    }

    BaseDataTypePtr fisheye_left = input->GetOne("fisheye_rear");
    if (!fisheye_left) {
        return -1;
    }

    NODE_LOG_INFO << "FisheyePerceptionExecutor::FisheyeEventCheck2 Receive fisheye_front and fisheye_rear";
    static int64_t i = 20000;
    ++i;
    NODE_LOG_INFO << "Send fisheye_result2, i = " << i;
    BaseDataTypePtr workflow_result = GenWorkResult(i);
    SendOutput("fisheye_result2", workflow_result);
    return 0;
}