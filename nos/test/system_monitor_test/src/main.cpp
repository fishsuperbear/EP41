#include <thread>
#include <chrono>
#include <signal.h>
#include <memory>

#include "cm/include/proxy.h"
#include "cm/include/skeleton.h"
#include "idl/generated/monitorPubSubTypes.h"
#include "log/include/logging.h"
#include "cm/include/proto_cm_reader.h"
#include "cm/include/proto_cm_writer.h"
#include "cfg/include/config_param.h"
#include "proto/test/soc/system_monitor.pb.h"

using namespace hozon::netaos::cm;
using namespace hozon::netaos::cfg;

bool stopFlag = false;

// std::shared_ptr<Proxy> notify_proxy_ = nullptr;
// std::shared_ptr<Proxy> alarm_proxy_ = nullptr;
// std::shared_ptr<Skeleton> control_skeleton_ = nullptr;

std::shared_ptr<ProtoCMReader<hozon::system::monitor::NotifyMessage>> notify_reader_;
std::shared_ptr<ProtoCMReader<hozon::system::monitor::AlarmMessage>> alarm_reader_;
std::shared_ptr<ProtoCMWriter<hozon::system::monitor::ControlMessage>> control_writer_;

void SigHandler(int signum)
{
    std::cout << "--- system monitor test sigHandler enter, signum [" << signum << "] ---" << std::endl;
    stopFlag = true;
}

// void NotifyEventCallback()
// {
//     if (nullptr == notify_proxy_) {
//         return;
//     }

//     if (notify_proxy_->IsMatched()) {
//         std::shared_ptr<monitor_notify_event> data = std::make_shared<monitor_notify_event>();
//         notify_proxy_->Take(data);
//         std::cout << "NotifyEventCallback monitor_id: " << static_cast<uint16_t>(data->monitor_id()) << " info: " << data->notify_info() << std::endl;
//     }
// }

// void AlarmEventCallback()
// {
//     if (nullptr == alarm_proxy_) {
//         return;
//     }

//     if (alarm_proxy_->IsMatched()) {
//         std::shared_ptr<monitor_alarm_event> data = std::make_shared<monitor_alarm_event>();
//         alarm_proxy_->Take(data);
//         std::cout << "AlarmEventCallback monitor_id: " << static_cast<uint16_t>(data->monitor_id()) << " info: " << data->alarm_info() << std::endl;
//     }
// }

void NotifyProtoCallback(std::shared_ptr<hozon::system::monitor::NotifyMessage> msg)
{
    std::cout << "NotifyProtoCallback monitor_id: " << static_cast<uint16_t>(msg->id()) << " info: " << msg->info() << std::endl;
}

void AlarmProtoCallback(std::shared_ptr<hozon::system::monitor::AlarmMessage> msg)
{
    std::cout << "AlarmProtoCallback monitor_id: " << static_cast<uint16_t>(msg->id()) << " info: " << msg->info() << std::endl;
}

int main(int argc, char* argv[])
{
	signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);

    hozon::netaos::log::InitLogging(
        "system_monitor_test",
        "system_monitor_test",
        hozon::netaos::log::LogLevel::kDebug,
        hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE,
        "./",
        10,
        100
    );

    // get vin number
    std::string vin = "";
    ConfigParam::Instance()->Init(1000);
    std::vector<uint8_t> data;
    ConfigParam::Instance()->GetParam<std::vector<uint8_t>>("dids/F190", data);
    if (17 == data.size()) {
        vin.assign(data.begin(), data.end());
    }

    // // notify proxy init
    // std::shared_ptr<monitor_notify_eventPubSubType> notifyPubsubtype = std::make_shared<monitor_notify_eventPubSubType>();
    // notify_proxy_ = std::make_shared<Proxy>(notifyPubsubtype);
    // notify_proxy_->Init(0, "monitor_notify_eventTopic_" + vin);
    // notify_proxy_->Listen(NotifyEventCallback);

    // // alarm proxy init
    // std::shared_ptr<monitor_alarm_eventPubSubType> alarmPubsubtype = std::make_shared<monitor_alarm_eventPubSubType>();
    // alarm_proxy_ = std::make_shared<Proxy>(alarmPubsubtype);
    // alarm_proxy_->Init(0, "monitor_alarm_eventTopic_" + vin);
    // alarm_proxy_->Listen(AlarmEventCallback);

    // // control skeleton init
    // std::shared_ptr<monitor_control_eventPubSubType> controlPubsubtype = std::make_shared<monitor_control_eventPubSubType>();
    // control_skeleton_ = std::make_shared<Skeleton>(controlPubsubtype);
    // control_skeleton_->Init(0, "monitor_control_eventTopic_" + vin);

    // notify reader init
    notify_reader_ = std::make_shared<ProtoCMReader<hozon::system::monitor::NotifyMessage>>();
    notify_reader_->Init(0, "monitor_notify_pb_" + vin, NotifyProtoCallback);

    // alarm reader init
    alarm_reader_ = std::make_shared<ProtoCMReader<hozon::system::monitor::AlarmMessage>>();
    alarm_reader_->Init(0, "monitor_alarm_pb_" + vin, AlarmProtoCallback);

    // control writer init
    control_writer_ = std::make_shared<ProtoCMWriter<hozon::system::monitor::ControlMessage>>();
    control_writer_->Init(0, "monitor_control_pb_" + vin);

    // 这里只演示开关控制，其他同理
    // id: kCpuMonitor = 0,kMemMonitor = 1,kDiskMonitor = 2,kEmmcMonitor = 3,kTemperatureMonitor = 4,kVoltageMonitor = 5,kFileSystemMonitor = 6,kProcessMonitor = 7,kNetworkMonitor = 8,kAllMonitor = 9
    // type: kMonitorSwitch = 0,kMonitorCycle = 1,kIsRecordFile = 2,kRecordFilePath = 3,kIsAlarm = 4,kAlarmValue = 5,kPostProcessingSwitch = 6
    // value: std::string（根据需要控制的类型给定对应的值就好，比如开关控制（打开："on", 关闭："off"）, 监控周期控制（3s："3000", 10s："10000"），文件记录周期控制（不记录："0", 5s记录一次："5"），记录文件路径（路径："绝对路径"），是否告警（告警："1"，不告警："0"），告警值（CPU90%："90"）,后处理开关控制（打开："on", 关闭："off"））
    // std::shared_ptr<monitor_control_event> controlData = std::make_shared<monitor_control_event>();
    // controlData->monitor_id(0);
    // controlData->control_type(0);
    // controlData->control_value("off");

    hozon::system::monitor::ControlMessage controlMsg;
    controlMsg.set_id(0);
    controlMsg.set_type(0);
    controlMsg.set_value("off");

    std::cout << "chassis test start." << std::endl;
    while(!stopFlag) {
        // if (!control_skeleton_->IsMatched()) {
        //     continue;
        // }

        // if (0 != control_skeleton_->Write(controlData)) {
        //     continue;
        // }

        if (0 != control_writer_->Write(controlMsg)) {
            continue;
        }

        // if ("on" == controlData->control_value()) {
        //     controlData->control_value("off");
        // }
        // else {
        //     controlData->control_value("on");
        // }

        if ("on" == controlMsg.value()) {
            controlMsg.set_value("off");
        }
        else {
            controlMsg.set_value("on");
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    }

    // // control skeleton deinit
    // if (nullptr != control_skeleton_) {
    //     control_skeleton_->Deinit();
    //     control_skeleton_ = nullptr;
    // }

    // // alarm proxy deinit
    // if (nullptr != alarm_proxy_) {
    //     alarm_proxy_->Deinit();
    //     alarm_proxy_ = nullptr;
    // }

    // // notify proxy deinit
    // if (nullptr != notify_proxy_) {
    //     notify_proxy_->Deinit();
    //     notify_proxy_ = nullptr;
    // }

    // control writer deinit
    if (nullptr != control_writer_) {
        control_writer_->Deinit();
        control_writer_ = nullptr;
    }

    // alarm reader deinit
    if (nullptr != alarm_reader_) {
        alarm_reader_->Deinit();
        alarm_reader_ = nullptr;
    }

    // notify reader deinit
    if (nullptr != notify_reader_) {
        notify_reader_->Deinit();
        notify_reader_ = nullptr;
    }

    ConfigParam::Instance()->DeInit();
    std::cout << "system monitor test end." << std::endl;
	return 0;
}