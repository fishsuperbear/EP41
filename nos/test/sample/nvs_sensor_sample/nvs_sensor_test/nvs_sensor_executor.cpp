#include <iostream>
#include <fstream>

#include "nvs_sensor_executor.h"
#include "adf-lite/include/ds/builtin_types.h"
#include "adf/include/node_proto_register.h"
#include "fisheye_datatype.h"
#include "adf-lite/service/rpc/lite_rpc.h"
#include "proto/test/soc/for_test.pb.h"
#include "adf-lite/include/writer.h"

NvsSensorExecutor::NvsSensorExecutor() {}

NvsSensorExecutor::~NvsSensorExecutor() {}
std::string key = "test";
std::string key1 = "~test";

void uint8func(const std::string& clientname, const std::string& key, const int32_t& value) {
    NODE_LOG_INFO << "uint8func func receive the event that the value of param:test"
                 << " is set to " << clientname << "  key: " << key << "  value: " << (int32_t)value;
}
int32_t NvsSensorExecutor::AlgInit() {
    REGISTER_PROTO_MESSAGE_TYPE("proto_sample_topic", ::adf::lite::dbg::WorkflowResult)

    CfgResultCode res = ConfigParam::Instance()->MonitorParam<int32_t>(key, uint8func);
    NODE_LOG_INFO << "MonitorParam  key: " << key << "  res:" << res;
    res = ConfigParam::Instance()->MonitorParam<int32_t>(key1, uint8func);
    NODE_LOG_INFO << "MonitorParam  key: " << key1 << "  res:" << res;

    NODE_LOG_INFO << "Init NvsSensorExecutor.";
    // RegistAlgProcessFunc("workflow1", std::bind(&NvsSensorExecutor::Workflow1, this, std::placeholders::_1));
    // RegistAlgProcessFunc("workflow2", std::bind(&NvsSensorExecutor::Workflow2, this, std::placeholders::_1));
    // RegistAlgProcessFunc("workflow3", std::bind(&NvsSensorExecutor::Workflow3, this, std::placeholders::_1));
    // RegistAlgProcessFunc("receive_struct_topic", std::bind(&NvsSensorExecutor::ReceiveStructTopic, this, std::placeholders::_1));
    // RegistAlgProcessFunc("receive_cm_topic", std::bind(&NvsSensorExecutor::ReceiveCmTopic, this, std::placeholders::_1));
    RegistAlgProcessFunc("nvs_cam", std::bind(&NvsSensorExecutor::NVSCamProcess, this, std::placeholders::_1));
    RegistAlgProcessFunc("front_nvs_cam", std::bind(&NvsSensorExecutor::FrontNVSCamProcess, this, std::placeholders::_1));
    RegistAlgProcessFunc("avm_nvs_cam", std::bind(&NvsSensorExecutor::AVMNVSCamProcess, this, std::placeholders::_1));
    RegistAlgProcessFunc("side_nvs_cam", std::bind(&NvsSensorExecutor::SideNVSCamProcess, this, std::placeholders::_1));
    RegistAlgProcessFunc("dump_nvs_cam", std::bind(&NvsSensorExecutor::DumpNVSCamProcess, this, std::placeholders::_1));

    LiteRpc::GetInstance().RegisterServiceFunc("GetLatestData", std::bind(&NvsSensorExecutor::GetLatestData, this, std::placeholders::_1));
    return 0;
}

void NvsSensorExecutor::AlgRelease() {
    NODE_LOG_INFO << "Release NvsSensorExecutor.";
    CfgResultCode res = ConfigParam::Instance()->UnMonitorParam(key);
    NODE_LOG_INFO << "UnMonitorParam  key: " << key << "  res:" << res;
    res = ConfigParam::Instance()->UnMonitorParam(key1);
    NODE_LOG_INFO << "UnMonitorParam  key: " << key1 << "  res:" << res;
}

#ifdef BUILD_FOR_ORIN
int32_t NvsSensorExecutor::DumpNVSCamProcess(Bundle* input) {
    cam_idx ++;

    std::shared_ptr<NvsImageCUDA> camera_0_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_0"));
    if (camera_0_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_0.";
        return -1;
    }

    if (dump_file == true) {
        ImageDumpFile(std::string("camera_0"), cam_idx, camera_0_ptr);
    }

    // NODE_LOG_INFO << " -----camera_0_ptr : " <<"  _packet->post_fence : "<<  camera_0_ptr->_packet->post_fence 
    //         << " camera_0_ptr->cuda_dev_ptr : " << camera_0_ptr->cuda_dev_ptr << " packet->cuda_dev_ptr : " << camera_0_ptr->_packet->cuda_dev_ptr 
    //         << " need_user_free " << camera_0_ptr->_packet->need_user_free << " data_time_sec " << camera_0_ptr->_packet->capture_start_us
    //         << " width " << camera_0_ptr->_packet->width << " height " << camera_0_ptr->_packet->height;

    freq_checker.say("camera_0");
    std::shared_ptr<NvsImageCUDA> camera_1_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_1"));
    if (camera_1_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_1.";
        return -1;
    }
    freq_checker.say("camera_1");

    if (dump_file == true) {
        ImageDumpFile(std::string("camera_1"), cam_idx, camera_1_ptr);
    }

    // NODE_LOG_INFO << " -----camera_1_ptr : " <<" camera_1_ptr->post_fence : "<<  camera_1_ptr->post_fence <<"  _packet->post_fence : "<<  camera_1_ptr->_packet->post_fence 
    //         << " camera_1_ptr->cuda_dev_ptr : " << camera_1_ptr->cuda_dev_ptr << " packet->cuda_dev_ptr : " << camera_1_ptr->_packet->cuda_dev_ptr 
    //         << " need_user_free " << camera_1_ptr->_packet->need_user_free << " data_time_sec " << camera_1_ptr->_packet->capture_start_us
    //         << " width " << camera_1_ptr->_packet->width << " height " << camera_1_ptr->_packet->height;

    std::shared_ptr<NvsImageCUDA> camera_3_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_3"));
    if (camera_3_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_3.";
        return -1;
    }
    freq_checker.say("camera_3");

    if (dump_file == true) {
        ImageDumpFile(std::string("camera_3"), cam_idx, camera_3_ptr);
    }

    std::shared_ptr<NvsImageCUDA> camera_4_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_4"));
    if (camera_4_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_4.";
        return -1;
    }
    freq_checker.say("camera_4");

    if (dump_file == true) {
        ImageDumpFile(std::string("camera_4"), cam_idx, camera_4_ptr);
    }

    std::shared_ptr<NvsImageCUDA> camera_5_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_5"));
    if (camera_5_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_5.";
        return -1;
    }
    freq_checker.say("camera_5");

    if (dump_file == true) {
        ImageDumpFile(std::string("camera_5"), cam_idx, camera_5_ptr);
    }

    std::shared_ptr<NvsImageCUDA> camera_6_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_6"));
    if (camera_6_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_6.";
        return -1;
    }
    freq_checker.say("camera_6");

    if (dump_file == true) {
        ImageDumpFile(std::string("camera_6"), cam_idx, camera_6_ptr);
    }

    std::shared_ptr<NvsImageCUDA> camera_7_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_7"));
    if (camera_7_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_7.";
        return -1;
    }
    freq_checker.say("camera_7");

    if (dump_file == true) {
        ImageDumpFile(std::string("camera_7"), cam_idx, camera_7_ptr);
    }

    std::shared_ptr<NvsImageCUDA> camera_8_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_8"));
    if (camera_8_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_8.";
        return -1;
    }
    freq_checker.say("camera_8");

    if (dump_file == true) {
        ImageDumpFile(std::string("camera_8"), cam_idx, camera_8_ptr);
    }

    std::shared_ptr<NvsImageCUDA> camera_9_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_9"));
    if (camera_9_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_9.";
        return -1;
    }
    freq_checker.say("camera_9");

    if (dump_file == true) {
        ImageDumpFile(std::string("camera_9"), cam_idx, camera_9_ptr);
    }

    std::shared_ptr<NvsImageCUDA> camera_10_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_10"));
    if (camera_10_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_10.";
        return -1;
    }
    freq_checker.say("camera_10");

    if (dump_file == true) {
        ImageDumpFile(std::string("camera_10"), cam_idx, camera_10_ptr);
    }

    std::shared_ptr<NvsImageCUDA> camera_11_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_11"));
    if (camera_11_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_11.";
        return -1;
    }
    freq_checker.say("camera_11");

    if (dump_file == true) {
        ImageDumpFile(std::string("camera_11"), cam_idx, camera_11_ptr);
    }

    return 0;
}

int32_t NvsSensorExecutor::NVSCamProcess(Bundle* input) {
    std::shared_ptr<NvsImageCUDA> camera_0_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_0"));
    if (camera_0_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_0.";
        return -1;
    }

    freq_checker.say("camera_0");
    std::shared_ptr<NvsImageCUDA> camera_1_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_1"));
    if (camera_1_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_1.";
        return -1;
    }
    freq_checker.say("camera_1");

    std::shared_ptr<NvsImageCUDA> camera_3_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_3"));
    if (camera_3_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_3.";
        return -1;
    }
    freq_checker.say("camera_3");

    std::shared_ptr<NvsImageCUDA> camera_4_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_4"));
    if (camera_4_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_4.";
        return -1;
    }
    freq_checker.say("camera_4");

    std::shared_ptr<NvsImageCUDA> camera_5_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_5"));
    if (camera_5_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_5.";
        return -1;
    }
    freq_checker.say("camera_5");

    std::shared_ptr<NvsImageCUDA> camera_6_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_6"));
    if (camera_6_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_6.";
        return -1;
    }
    freq_checker.say("camera_6");

    std::shared_ptr<NvsImageCUDA> camera_7_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_7"));
    if (camera_7_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_7.";
        return -1;
    }
    freq_checker.say("camera_7");

    std::shared_ptr<NvsImageCUDA> camera_8_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_8"));
    if (camera_8_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_8.";
        return -1;
    }
    freq_checker.say("camera_8");

    std::shared_ptr<NvsImageCUDA> camera_9_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_9"));
    if (camera_9_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_9.";
        return -1;
    }
    freq_checker.say("camera_9");

    std::shared_ptr<NvsImageCUDA> camera_10_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_10"));
    if (camera_10_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_10.";
        return -1;
    }
    freq_checker.say("camera_10");

    std::shared_ptr<NvsImageCUDA> camera_11_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_11"));
    if (camera_11_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_11.";
        return -1;
    }
    freq_checker.say("camera_11");

    return 0;
}

int32_t NvsSensorExecutor::FrontNVSCamProcess(Bundle* input) {
    std::shared_ptr<NvsImageCUDA> camera_0_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_0"));
    if (camera_0_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_0.";
        return -1;
    }

    freq_checker.say("camera_0");
    std::shared_ptr<NvsImageCUDA> camera_1_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_1"));
    if (camera_1_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_1.";
        return -1;
    }
    freq_checker.say("camera_1");

    std::shared_ptr<NvsImageCUDA> camera_3_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_3"));
    if (camera_3_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_3.";
        return -1;
    }
    freq_checker.say("camera_3");

    return 0;
}

int32_t NvsSensorExecutor::AVMNVSCamProcess(Bundle* input) {
    std::shared_ptr<NvsImageCUDA> camera_8_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_8"));
    if (camera_8_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_8.";
        return -1;
    }
    freq_checker.say("camera_8");

    std::shared_ptr<NvsImageCUDA> camera_9_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_9"));
    if (camera_9_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_9.";
        return -1;
    }
    freq_checker.say("camera_9");

    std::shared_ptr<NvsImageCUDA> camera_10_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_10"));
    if (camera_10_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_10.";
        return -1;
    }
    freq_checker.say("camera_10");

    std::shared_ptr<NvsImageCUDA> camera_11_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_11"));
    if (camera_11_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_11.";
        return -1;
    }
    freq_checker.say("camera_11");

    return 0;
}

int32_t NvsSensorExecutor::SideNVSCamProcess(Bundle* input) {
    std::shared_ptr<NvsImageCUDA> camera_4_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_4"));
    if (camera_4_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_4.";
        return -1;
    }
    freq_checker.say("camera_4");

    std::shared_ptr<NvsImageCUDA> camera_5_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_5"));
    if (camera_5_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_5.";
        return -1;
    }
    freq_checker.say("camera_5");

    std::shared_ptr<NvsImageCUDA> camera_6_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_6"));
    if (camera_6_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_6.";
        return -1;
    }
    freq_checker.say("camera_6");

    std::shared_ptr<NvsImageCUDA> camera_7_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_7"));
    if (camera_7_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_7.";
        return -1;
    }
    freq_checker.say("camera_7");

    return 0;
}

void NvsSensorExecutor::WriteFile(const std::string& name, uint8_t* data, uint32_t size) {
    std::ofstream of(name);

    if (!of) {
        NODE_LOG_ERROR << "Fail to open " << name;
        return;
    }

    of.write((const char*)data, size);
    of.close();
    NODE_LOG_INFO << "Succ to write " << name;

    return;
}

void NvsSensorExecutor::ImageDumpFile(const std::string& file_name, int index, std::shared_ptr<NvsImageCUDA> packet) {
    if (index % 10 != 0) {
        return ;
    }

    uint8_t* local_ptr = (uint8_t*)malloc(packet->size);

    /* Instruct CUDA to copy the packet data buffer to the target buffer */
    uint32_t cuda_rt_err = cudaMemcpy(local_ptr,
                                packet->cuda_dev_ptr,
                                packet->size,
                                cudaMemcpyDeviceToHost);
    if (cudaSuccess != cuda_rt_err) {
        NODE_LOG_ERROR << "Failed to issue copy command, ret " << packet->cuda_dev_ptr;
        return;
    }

    std::string dump_name = file_name + "_" + std::to_string(index) + ".data";
    WriteFile(dump_name, local_ptr, packet->size);

    free(local_ptr);
}

#endif

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

int32_t NvsSensorExecutor::Workflow1(Bundle* input) {
    static int64_t i = 0;
    ++i;

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

int32_t NvsSensorExecutor::Workflow2(Bundle* input) {
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

int32_t NvsSensorExecutor::Workflow3(Bundle* input) {
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
    return 0;
}

/* 接收Struct 数据类型的Topic，
 */
int32_t NvsSensorExecutor::ReceiveStructTopic(Bundle* input) {
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

/* 接收CM Topic，需要运行proto_send_sample，示例如下：
    cd ./nos/output/x86_2004/test/adf-proto-sample
    ./bin/proto_send_sample ./conf/proto_send_sample_conf.yaml
*/
int32_t NvsSensorExecutor::ReceiveCmTopic(Bundle* input) {
    BaseDataTypePtr workresult = input->GetOne("proto_sample_topic");
    if (!workresult) {
        return -1;
    }

    std::shared_ptr<adf::lite::dbg::WorkflowResult> workresult_proto = std::static_pointer_cast<adf::lite::dbg::WorkflowResult>(workresult->proto_msg);
    NODE_LOG_INFO << "CmTopic proto_sample_topic recv " << workresult_proto->val1() << ", " << workresult_proto->val2() << ", " << workresult_proto->val3() << ", " << workresult_proto->val4();

    return 0;
}

int32_t NvsSensorExecutor::FreeDataTopic(Bundle* input) {
    Writer _writer;

    _writer.Init("free_data_topic");

    BaseDataTypePtr base_ptr = GenWorkResult(1111);
    int32_t ret = _writer.Write(base_ptr);
    if (ret < 0) {
        NODE_LOG_ERROR << "Fail to write " << "free_data_topic";
        return -1;
    }

    return 0;
}


int32_t NvsSensorExecutor::GetLatestData(BaseDataTypePtr& ptr) {
    ptr = _lastet_data;

    return 0;
}