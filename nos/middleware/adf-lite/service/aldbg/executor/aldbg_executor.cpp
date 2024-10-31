
#include <iostream>
#include "adf-lite/service/aldbg/utility/utility.h"
#include "proto/test/soc/dbg_msg.pb.h"
#include "adf-lite/include/cm_writer.h"
#include "cm/include/proto_cm_writer.h"
//#include "sm/include/state_client.h"
#include "executor/aldbg_executor.h"
#include "adf-lite/service/rpc/lite_rpc.h"
#include "adf-lite/include/base.h"
#include "adf-lite/include/struct_register.h"

//using namespace hozon::netaos::sm;
using namespace hozon::netaos::adf_lite;
namespace hozon {
namespace netaos {
namespace adf_lite {

AldbgExecutor::AldbgExecutor()
:_server(std::bind(&AldbgExecutor::AldbgCmdHandle, this, std::placeholders::_1, std::placeholders::_2))
 {
}

AldbgExecutor::~AldbgExecutor() {

}

int32_t AldbgExecutor::AlgInit() {
    AldbgLogger::GetInstance()._logger.Init("ALDB", ADF_LOG_LEVEL_INFO);

    auto func = LiteRpc::GetInstance().GetStringServiceFunc("GetProcessName");
    if (func == nullptr) {
        ALDBG_LOG_WARN << "GetProcessName Func maybe not beed registed";
    } else {
        int32_t res = func(_process_name);
        if (res < 0) {
            ALDBG_LOG_WARN << "GetProcessName is Has Error";
        }
    }
    _routing_attrs = Topology::GetInstance().AppendQueueForeachTopic(5);
    RegistAlgProcessFunc("ReadFreq", std::bind(&AldbgExecutor::FreqSendRoutine, this, std::placeholders::_1));

    _need_stop = false;
    _server.Start(0, "/lite/" + _process_name + "/method/cmd");

    std::string freq_topic = "/lite/" + _process_name + "/info";
    int32_t ret = _writer.Init(0, freq_topic);
    if (ret < 0) {
        ALDBG_LOG_ERROR << "Fail to init writer " << ret;
        _writer.Deinit();
        return -1;
    }
    return 0;
}

void AldbgExecutor::AlgRelease() {
    _need_stop = true;
    _server.Stop();
    _lite_player.End();
    _writer.Deinit();
}

int32_t AldbgExecutor::AldbgCmdHandle(const std::shared_ptr<hozon::adf::lite::dbg::EchoRequest>& req, std::shared_ptr<hozon::adf::lite::dbg::GeneralResponse>& resp) {

    std::string cmd = req->cmd();
    ALDBG_LOG_INFO << "Received cmd: [" << cmd << "], status is [" << req->status() << "]";
    if (cmd == "echo" || cmd == "record") {
        if (req->status()) {
            _lite_player.End();
            StartRecord(req);
        } else {
            EndRecord();
        }
    } else if (cmd == "play") {
        if (req->status()) {
            EndRecord();
            _lite_player.Start(_routing_attrs);
        } else {
            _lite_player.End();
        }
    }
    else if (cmd == "executors") {
        ALDBG_LOG_INFO << "executors";
        auto result = DbgInfo::GetInstance().GetExecutors();
        for (auto &it: result) {
            resp->add_data(it);
        }
    }
    else if (cmd == "topics") {
        ALDBG_LOG_INFO << "topics";
        auto result = DbgInfo::GetInstance().GetExecutorTopics(req->other());
        for (auto &it: result) {
            resp->add_data(it);
        }
    }

    resp->set_res(true);
    resp->set_err(0);
    return 0;
}

int32_t AldbgExecutor::FreqSendRoutine(Bundle* input) {
    std::string topic_prefix = "/lite/" + _process_name + "/event/";
    hozon::adf::lite::dbg::FreqDebugMessage freq_msg;

    for (auto attr : _routing_attrs) {
        hozon::adf::lite::dbg::FreqDebugMessage::Element* ele = freq_msg.add_elements();

        FreqInfo info = attr.queue->ReadInfo();

        ele->set_topic(topic_prefix + attr.topic);
        ele->set_samples(info.samples);
        ele->set_freq(info.freq);
        ele->set_max_delta_us(info.max_delta_us);
        ele->set_min_delta_us(info.min_delta_us);
        ele->set_std_dev_us(info.std_dev_us);
        ele->set_duration_us(info.duration_us);
    }

    int32_t ret = _writer.Write(freq_msg);
    if (ret < 0) {
        ALDBG_LOG_ERROR << "Fail to write " << ret;
        return -1;
    }
    return 0;
}

void AldbgExecutor::StartRecord(const std::shared_ptr<hozon::adf::lite::dbg::EchoRequest>& req)
{
    if (_pub_thread.size() > 0) {
        ALDBG_LOG_INFO << "PubTopics is running";
    } else {
        _stop_record = false;
        std::vector<std::string> topics;
        for (int i = 0; i < req->elements().size(); ++i) {
            auto ele = req->elements().at(i).topic();
            topics.push_back(ele);
        }
        PubTopics(topics);
    }
}

void AldbgExecutor::EndRecord()
{
    _stop_record = true;
    for (auto &t : _pub_thread) {
        // 需要立即结束线程。
        t->join();
        t = nullptr;
    }

    _pub_thread.clear();
}

void AldbgExecutor::PubTopics(const std::vector<std::string>& topics) {
    if (topics.size() == 0) {
        for (auto& attr : _routing_attrs) {
            std::shared_ptr<std::thread> t1 = std::make_shared<std::thread>(&AldbgExecutor::PubTopic, this, attr.topic);
            _pub_thread.push_back(t1);
        }
    } else {
        for (std::string topic: topics) {
             std::shared_ptr<std::thread> t1 = std::make_shared<std::thread>(&AldbgExecutor::PubTopic, this, topic);
            _pub_thread.push_back(t1);
        }
    }
}

void AldbgExecutor::PubTopic(const std::string topic)
{
    std::string adf_topic;
    int32_t ret = GetInnerTopicFromCmTopic(topic, adf_topic);
    if (ret != 0) {
        ALDBG_LOG_INFO << "topic [" << topic << "], is illegal lite topic";
        return;
    }

    RoutingAttr* target_attr = nullptr;
    for (auto& attr : _routing_attrs) {
        if (attr.topic == adf_topic) {
            target_attr = &attr;
            break;
        }
    }
    if (target_attr == nullptr) {
        ALDBG_LOG_INFO << "topic [" << topic << "], not found in lite topics";
        return;
    }

    hozon::netaos::adf_lite::CMWriter writer;
    ret = writer.Init(0, topic);
    if (ret < 0) {
        ALDBG_LOG_ERROR << "Fail to init writer " << ret;
        writer.Deinit();
        return;
    }

    while (!_stop_record) {
        if (target_attr->queue == nullptr) {
            break;
        }
        std::shared_ptr<BaseDataTypePtr> ptr = target_attr->queue->GetLatestOneBlocking(true, 500);
        if (ptr) {
            if ((*ptr)->proto_msg != nullptr) {
                ALDBG_LOG_VERBOSE << "Serialize as proto message, " << ret;
                ret = writer.Write((*ptr)->proto_msg);
                if (ret < 0) {
                    ALDBG_LOG_ERROR << "Fail to write " << ret;
                }
            } else {
                if (BaseDataTypeMgr::GetInstance().GetSize(adf_topic) != 0) {
                    ALDBG_LOG_VERBOSE << "Serialize as Registered struct!";
                    std::string data;
                    (*ptr)->SerializeAsString(data, BaseDataTypeMgr::GetInstance().GetSize(adf_topic));

                    ret = writer.Write(topic, data);
                    if (ret < 0) {
                        ALDBG_LOG_ERROR << "Fail to write " << ret;
                    }
                } else {
                    ALDBG_LOG_VERBOSE << "Don't Serialize!";
                }
            }
        }
    }
    writer.Deinit();
}

}    
}
}
