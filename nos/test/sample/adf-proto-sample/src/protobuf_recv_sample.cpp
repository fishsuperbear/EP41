#include <memory>
#include <map>
#include <unistd.h>
#include <fstream>
#include "adf/include/log.h"
#include "proto/test/soc/for_test.pb.h"
#include "adf/include/node_base.h"
#include "proto/soc/sensor_image.pb.h"
#include "cm/include/proto_method.h"
#include "adf/include/data_types/image/orin_image.h"

class FreqChecker {
  using checker_time = std::chrono::time_point<std::chrono::system_clock>;
 public:
  FreqChecker() = default;
  void say(const std::string& unique_name, uint64_t sample_cnt = 100);
 private:
  std::unordered_map<std::string, std::pair<uint64_t, checker_time>> freq_map_;
};
void FreqChecker::say(const std::string& unique_name, uint64_t sample_cnt) {
  if (freq_map_.find(unique_name) == freq_map_.end()) {
    freq_map_[unique_name] = std::make_pair(1, std::chrono::system_clock::now());
  } else {
    freq_map_[unique_name].first++;
  }

  if (freq_map_[unique_name].first == sample_cnt) {
    auto now = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = now - freq_map_[unique_name].second;
    NODE_LOG_INFO << "check " << unique_name << " frequency: " << sample_cnt / diff.count() << " Hz";
    freq_map_[unique_name].second = now;
    freq_map_[unique_name].first = 0;
  }
}


using namespace hozon::netaos::log;
using namespace hozon::netaos::adf;

class ProtobufRecvSample : public hozon::netaos::adf::NodeBase {
public:
    ProtobufRecvSample() {

    }

    ~ProtobufRecvSample() {}

    virtual int32_t AlgInit() {
        REGISTER_PROTO_MESSAGE_TYPE("workresult", adf::lite::dbg::WorkflowResult)
        REGISTER_PROTO_MESSAGE_TYPE("fisheye_front", hozon::soc::Image)
        REGISTER_PROTO_MESSAGE_TYPE("fisheye_left", hozon::soc::Image)
        REGISTER_PROTO_MESSAGE_TYPE("fisheye_right", hozon::soc::Image)
        REGISTER_PROTO_MESSAGE_TYPE("fisheye_rear", hozon::soc::Image)

        REGISTER_PROTO_MESSAGE_TYPE("camera_0", hozon::soc::Image)
        REGISTER_PROTO_MESSAGE_TYPE("camera_1", hozon::soc::Image)
        REGISTER_PROTO_MESSAGE_TYPE("camera_3", hozon::soc::Image)
        REGISTER_PROTO_MESSAGE_TYPE("camera_4", hozon::soc::Image)

        REGISTER_PROTO_MESSAGE_TYPE("camera_5", hozon::soc::Image)
        REGISTER_PROTO_MESSAGE_TYPE("camera_6", hozon::soc::Image)
        REGISTER_PROTO_MESSAGE_TYPE("camera_7", hozon::soc::Image)
        REGISTER_PROTO_MESSAGE_TYPE("camera_8", hozon::soc::Image)

        REGISTER_PROTO_MESSAGE_TYPE("camera_9", hozon::soc::Image)
        REGISTER_PROTO_MESSAGE_TYPE("camera_10", hozon::soc::Image)
        REGISTER_PROTO_MESSAGE_TYPE("camera_11", hozon::soc::Image)
        // RegistAlgProcessWithProfilerFunc("main", std::bind(&ProtobufRecvSample::AlgProcess, this, std::placeholders::_1, std::placeholders::_2));
        // RegistAlgProcessWithProfilerFunc("image", std::bind(&ProtobufRecvSample::ImageProcess, this, std::placeholders::_1, std::placeholders::_2));
        // RegistAlgProcessWithProfilerFunc("methodtest", std::bind(&ProtobufRecvSample::MethodProcess, this, std::placeholders::_1, std::placeholders::_2))
        RegistAlgProcessWithProfilerFunc("nvs_cam", std::bind(&ProtobufRecvSample::NVSCamProcess, this, std::placeholders::_1, std::placeholders::_2));
        int32_t ret = _proto_client.Init(0, "/method_test");
        if (ret < 0) {
            ReportFault(111111, 111111);
            NODE_LOG_CRITICAL << "Fail to init method client, " << ret;
            return -1;
        }
        return 0;
    }

    int32_t AlgProcess(hozon::netaos::adf::NodeBundle* input,
                            const hozon::netaos::adf::ProfileToken& token) {
        BaseDataTypePtr base_ptr = input->GetOne("workresult");
        if (base_ptr == nullptr) {
            NODE_LOG_ERROR << "Fail to recv data.";
            return -1;
        }

        std::shared_ptr<adf::lite::dbg::WorkflowResult> result = std::static_pointer_cast<adf::lite::dbg::WorkflowResult>(base_ptr->proto_msg);
        if (result == nullptr) {
            NODE_LOG_ERROR << "Fail to get proto data.";
            return -1;
        }

        NODE_LOG_INFO << "Recv result " << result->val1() 
            << ", time sec " << result->mutable_header()->publish_stamp();

        return 0;
    }

    int32_t NVSCamProcess(hozon::netaos::adf::NodeBundle* input,
                            const hozon::netaos::adf::ProfileToken& token) {
        std::shared_ptr<NvsImageCUDA> camera_0_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_0"));
        if (camera_0_ptr == nullptr) {
            NODE_LOG_ERROR << "Fail to recv camera_0.";
            return -1;
        }
        checker.say("camera_0");

        std::shared_ptr<NvsImageCUDA> camera_1_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_1"));
        if (camera_1_ptr == nullptr) {
            NODE_LOG_ERROR << "Fail to recv camera_1.";
            return -1;
        }
        checker.say("camera_1");

        std::shared_ptr<NvsImageCUDA> camera_3_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_3"));
        if (camera_3_ptr == nullptr) {
            NODE_LOG_ERROR << "Fail to recv camera_3.";
            return -1;
        }
        checker.say("camera_3");

        std::shared_ptr<NvsImageCUDA> camera_4_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_4"));
        if (camera_4_ptr == nullptr) {
            NODE_LOG_ERROR << "Fail to recv camera_4.";
            return -1;
        }
        checker.say("camera_4");

        std::shared_ptr<NvsImageCUDA> camera_5_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_5"));
        if (camera_5_ptr == nullptr) {
            NODE_LOG_ERROR << "Fail to recv camera_5.";
            return -1;
        }
        checker.say("camera_5");

        std::shared_ptr<NvsImageCUDA> camera_6_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_6"));
        if (camera_6_ptr == nullptr) {
            NODE_LOG_ERROR << "Fail to recv camera_6.";
            return -1;
        }
        checker.say("camera_6");

        std::shared_ptr<NvsImageCUDA> camera_7_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_7"));
        if (camera_7_ptr == nullptr) {
            NODE_LOG_ERROR << "Fail to recv camera_7.";
            return -1;
        }
        checker.say("camera_7");

        std::shared_ptr<NvsImageCUDA> camera_8_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_8"));
        if (camera_8_ptr == nullptr) {
            NODE_LOG_ERROR << "Fail to recv camera_8.";
            return -1;
        }
        checker.say("camera_8");

        std::shared_ptr<NvsImageCUDA> camera_9_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_9"));
        if (camera_9_ptr == nullptr) {
            NODE_LOG_ERROR << "Fail to recv camera_9.";
            return -1;
        }
        checker.say("camera_9");

        std::shared_ptr<NvsImageCUDA> camera_10_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_10"));
        if (camera_10_ptr == nullptr) {
            NODE_LOG_ERROR << "Fail to recv camera_10.";
            return -1;
        }
        checker.say("camera_10");

        std::shared_ptr<NvsImageCUDA> camera_11_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("camera_11"));
        if (camera_11_ptr == nullptr) {
            NODE_LOG_ERROR << "Fail to recv camera_11.";
            return -1;
        }
        checker.say("camera_11");

        return 0;
    }

    int32_t MethodProcess(hozon::netaos::adf::NodeBundle* input,
                            const hozon::netaos::adf::ProfileToken& token) {
        static int i = 0;
        ++i;

        std::shared_ptr<adf::lite::dbg::WorkflowResult> req(new adf::lite::dbg::WorkflowResult);
        req->set_val1(i);
        req->set_val2(2 * i);
        std::shared_ptr<adf::lite::dbg::WorkflowResult> resp(new adf::lite::dbg::WorkflowResult);
        int32_t ret = _proto_client.Request(req, resp, 100);
        if (ret < 0) {
            NODE_LOG_ERROR << "Fail to request, ret " << ret;
            return -1;
        }

        NODE_LOG_INFO << "Method: " << req->val1() << " + " << req->val2() << " = " << resp->val3();

        return 0;
    }

    virtual void AlgRelease() { 
        _proto_client.DeInit();
    }

private:
    FreqChecker checker;
    hozon::netaos::cm::ProtoMethodClient<adf::lite::dbg::WorkflowResult, adf::lite::dbg::WorkflowResult> _proto_client;
};

int main(int argc, char* argv[]) {
    ProtobufRecvSample recv_node;

    recv_node.InitLoggerStandAlone(std::string(argv[1]));
    recv_node.Start(std::string(argv[1]), true);
    recv_node.NeedStopBlocking();
    recv_node.Stop();

    return 0;
}