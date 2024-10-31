#pragma once

#include "proto/test/soc/dbg_msg.pb.h"
#include "cm/include/proto_method.h"
#include "cm/include/method.h"
#include "adf-lite/include/executor.h"
#include "adf-lite/include/topology.h"
#include "adf-lite/include/dbg_info.h"
#include "adf-lite/include/adf_lite_logger.h"
#include "adf-lite/service/aldbg/player/lite_player.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

class AldbgLogger {
public:
    static AldbgLogger& GetInstance() {
        static AldbgLogger instance;
        return instance;
    }

    CtxLogger _logger;
};

#define ALDBG_LOG_FATAL          CTX_LOG_FATAL(AldbgLogger::GetInstance()._logger)
#define ALDBG_LOG_ERROR          CTX_LOG_ERROR(AldbgLogger::GetInstance()._logger)
#define ALDBG_LOG_WARN           CTX_LOG_WARN(AldbgLogger::GetInstance()._logger)
#define ALDBG_LOG_INFO           CTX_LOG_INFO(AldbgLogger::GetInstance()._logger)
#define ALDBG_LOG_DEBUG          CTX_LOG_DEBUG(AldbgLogger::GetInstance()._logger)
#define ALDBG_LOG_VERBOSE        CTX_LOG_VERBOSE(AldbgLogger::GetInstance()._logger)

class AldbgExecutor : public Executor {
public:
    AldbgExecutor();
    ~AldbgExecutor();

    int32_t AlgInit();
    void AlgRelease();

private:
    int32_t AldbgCmdHandle(const std::shared_ptr<hozon::adf::lite::dbg::EchoRequest>& req, std::shared_ptr<hozon::adf::lite::dbg::GeneralResponse>& resp);
    int32_t FreqSendRoutine(Bundle* input);
    void StartRecord(const std::shared_ptr<hozon::adf::lite::dbg::EchoRequest>& req);
    void EndRecord();
    void PubTopics(const std::vector<std::string>& topics);
    void PubTopic(const std::string topic);

    std::vector<RoutingAttr> _routing_attrs;
    bool _need_stop = false;
    bool _stop_record = true;
    std::string _process_name;
    std::vector<std::shared_ptr<std::thread>> _pub_thread;
    hozon::netaos::cm::ProtoMethodServer<hozon::adf::lite::dbg::EchoRequest, hozon::adf::lite::dbg::GeneralResponse> _server;
    LitePlayer _lite_player;
    hozon::netaos::cm::ProtoCMWriter<hozon::adf::lite::dbg::FreqDebugMessage> _writer;
};

REGISTER_ADF_CLASS(Aldbg, AldbgExecutor)

}
}
}
