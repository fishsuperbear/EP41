#include <unistd.h>
#include <fstream>
#include <map>
#include <memory>
#include "adf/include/data_types/image/orin_image.h"
#include "adf/include/log.h"
#include "adf/include/node_base.h"
#include "proto/soc/sensor_image.pb.h"

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

class IdlRecvSample : public hozon::netaos::adf::NodeBase {
   public:
    IdlRecvSample() {}

    ~IdlRecvSample() {}

    virtual int32_t AlgInit() {

        // RegistAlgProcessWithProfilerFunc("main", std::bind(&IdlRecvSample::AlgProcess, this, std::placeholders::_1, std::placeholders::_2));
        // RegistAlgProcessWithProfilerFunc("image", std::bind(&IdlRecvSample::ImageProcess, this, std::placeholders::_1, std::placeholders::_2));
        // RegistAlgProcessWithProfilerFunc("methodtest", std::bind(&IdlRecvSample::MethodProcess, this, std::placeholders::_1, std::placeholders::_2))
        RegistAlgProcessWithProfilerFunc("nvs_cam", std::bind(&IdlRecvSample::NVSCamProcess, this, std::placeholders::_1, std::placeholders::_2));

        return 0;
    }

    int32_t NVSCamProcess(hozon::netaos::adf::NodeBundle* input, const hozon::netaos::adf::ProfileToken& token) {
        std::shared_ptr<BaseData> camera_0_ptr = std::static_pointer_cast<BaseData>(input->GetOne("camera_0"));
        if (camera_0_ptr == nullptr) {
            NODE_LOG_ERROR << "Fail to recv camera_0.";
            return -1;
        }
        checker.say("camera_0");

        std::shared_ptr<BaseData> camera_1_ptr = std::static_pointer_cast<BaseData>(input->GetOne("camera_1"));
        if (camera_1_ptr == nullptr) {
            NODE_LOG_ERROR << "Fail to recv camera_1.";
            return -1;
        }
        checker.say("camera_1");

        std::shared_ptr<BaseData> camera_2_ptr = std::static_pointer_cast<BaseData>(input->GetOne("camera_2"));
        if (camera_2_ptr == nullptr) {
            NODE_LOG_ERROR << "Fail to recv camera_2.";
            return -1;
        }
        checker.say("camera_2");

        std::shared_ptr<BaseData> camera_4_ptr = std::static_pointer_cast<BaseData>(input->GetOne("camera_4"));
        if (camera_4_ptr == nullptr) {
            NODE_LOG_ERROR << "Fail to recv camera_4.";
            return -1;
        }
        checker.say("camera_4");

        std::shared_ptr<BaseData> camera_5_ptr = std::static_pointer_cast<BaseData>(input->GetOne("camera_5"));
        if (camera_5_ptr == nullptr) {
            NODE_LOG_ERROR << "Fail to recv camera_5.";
            return -1;
        }
        checker.say("camera_5");

        std::shared_ptr<BaseData> camera_6_ptr = std::static_pointer_cast<BaseData>(input->GetOne("camera_6"));
        if (camera_6_ptr == nullptr) {
            NODE_LOG_ERROR << "Fail to recv camera_6.";
            return -1;
        }
        checker.say("camera_6");

        std::shared_ptr<BaseData> camera_7_ptr = std::static_pointer_cast<BaseData>(input->GetOne("camera_7"));
        if (camera_7_ptr == nullptr) {
            NODE_LOG_ERROR << "Fail to recv camera_7.";
            return -1;
        }
        checker.say("camera_7");

        std::shared_ptr<BaseData> camera_8_ptr = std::static_pointer_cast<BaseData>(input->GetOne("camera_8"));
        if (camera_8_ptr == nullptr) {
            NODE_LOG_ERROR << "Fail to recv camera_8.";
            return -1;
        }
        checker.say("camera_8");

        std::shared_ptr<BaseData> camera_9_ptr = std::static_pointer_cast<BaseData>(input->GetOne("camera_9"));
        if (camera_9_ptr == nullptr) {
            NODE_LOG_ERROR << "Fail to recv camera_9.";
            return -1;
        }
        checker.say("camera_9");

        std::shared_ptr<BaseData> camera_10_ptr = std::static_pointer_cast<BaseData>(input->GetOne("camera_10"));
        if (camera_10_ptr == nullptr) {
            NODE_LOG_ERROR << "Fail to recv camera_10.";
            return -1;
        }
        checker.say("camera_10");

        std::shared_ptr<BaseData> camera_11_ptr = std::static_pointer_cast<BaseData>(input->GetOne("camera_11"));
        if (camera_11_ptr == nullptr) {
            NODE_LOG_ERROR << "Fail to recv camera_11.";
            return -1;
        }
        checker.say("camera_11");

        return 0;
    }

    virtual void AlgRelease() {}

   private:
    FreqChecker checker;
};

int main(int argc, char* argv[]) {
    IdlRecvSample recv_node;

    recv_node.InitLoggerStandAlone(std::string(argv[1]));
    recv_node.Start(std::string(argv[1]), true);
    recv_node.NeedStopBlocking();
    recv_node.Stop();

    return 0;
}