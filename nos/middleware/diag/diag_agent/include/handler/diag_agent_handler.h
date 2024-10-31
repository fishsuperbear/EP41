
#ifndef DIAG_AGENT_HANDLER_H
#define DIAG_AGENT_HANDLER_H

#include <mutex>
#include <memory>
#include "diag/diag_agent/include/service/diag_agent_data_identifier.h"
#include "diag/diag_agent/include/service/diag_agent_routine_control.h"

namespace hozon {
namespace netaos {
namespace diag {
namespace diag_agent{

class DiagAgentHandlerImpl;

class DiagAgentHandler {

public:
    DiagAgentHandler();
    ~DiagAgentHandler();

    /**  初始化
     @param[in]  configPath 配置文件路径
     @param[in]  dataIdentifier Did服务实现类
     @param[in]  routineControl Rid服务实现类
     @param[out] none
     @return     DiagAgentInitResultCode 0：初始化成功 负值：初始化失败
     @warning    none
     @note       建立通信通道，初始化数据
    */
    DiagAgentInitResultCode Init(const std::string& configPath,
                 std::shared_ptr<DiagAgentDataIdentifier> dataIdentifier = nullptr,
                 std::shared_ptr<DiagAgentRoutineControl> routineControl = nullptr);

    /**  资源释放
     @param[in]  none
     @param[out] none
     @return     void
     @warning    none
     @note       主动释放资源
    */
    void DeInit();

private:
    DiagAgentHandler(const DiagAgentHandler &);
    DiagAgentHandler & operator = (const DiagAgentHandler &);

private:
    std::unique_ptr<DiagAgentHandlerImpl> handler_impl_;
};

}  // namespace diag_agent
}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_AGENT_HANDLER_H