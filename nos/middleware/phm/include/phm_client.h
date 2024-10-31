
#ifndef PHM_CLIENT_H
#define PHM_CLIENT_H

#include <memory>
#include <vector>
#include <functional>

#include "phm_def.h"

namespace hozon {
namespace netaos {
namespace phm {


class PHMClientImpl;

class PHMClient {
public:
    PHMClient();
    ~PHMClient();

    /**  PHM模块资源初始化
     @param[in]  phmConfigPath phm配置文件路径，该配置文件配置了监控任务及故障订阅等选项
     @param[in]  service_available_callback phm服务是否可用的回调函数注册
     @param[in]  fault_receive_callback 注册订阅故障接收的回调函数
     @param[in]  processName 注册订阅故障的进程名
     @param[out] none
     @return     int32_t 0：初始化成功 负值：初始化失败
     @warning    none
     @note       phm配置文件可不配，则不提供监控及故障接收功能，默认提供故障上报功能
    */
    int32_t Init(const std::string phmConfigPath = "",
                 std::function<void(bool)> service_available_callback = nullptr,
                 std::function<void(ReceiveFault_t)> fault_receive_callback = nullptr,
                 const std::string processName = "");

    /**  PHM模块功能启动
     @param[in]  delayTime 延时启动phm 监控功能（故障功能暂不受延时影响）
     @param[out] none
     @return     int32_t 0：启动成功 负值：启动失败
     @warning    none
     @note       延时启动针对 如alive监控，由于上报检查点不及时导致的故障产生问题，交由使用方来控制监控使能的逻辑。
    */
    int32_t Start(uint32_t delayTime = 0);

    /**  PHM模块功能关闭
     @param[in]  none
     @param[out] none
     @return     void
     @warning    none
     @note       主动停止监控功能
    */
    void Stop();

    /**  PHM模块资源释放
     @param[in]  none
     @param[out] none
     @return     void
     @warning    none
     @note       主动释放phm资源
    */
    void Deinit();

    /**  监控任务上报检查点
     @param[in]  checkPointId 使用phm配置文件定义的监控任务的checkPointId字段
     @param[out] none
     @return     int32_t 0：上报成功 负值：上报失败
     @warning    配置文件中不存在的检查点会上报失败
     @note       不同监控任务可共用检查点
    */
    int32_t ReportCheckPoint(const uint32_t checkPointId);

    /**  故障上报
     @param[in]  faultInfo 用户填充好基本的故障信息，具体定义在phm_def.h文件中
     @param[out] none
     @return     int32_t 0：上报成功 负值：上报失败
     @warning    none
     @note       默认不使用faultDebounce，如果使用消抖机制，填充该字段
    */
    int32_t ReportFault(const SendFault_t& faultInfo);

    /**  故障抑制，提供给用户根据自身条件主动抑制故障的能力
     @param[in]  faultKeys 故障列表，faultKey 由 [faultId*100+faultObj] 构成
     @param[out] none
     @return     void
     @warning    none
     @note       none
    */
    void InhibitFault(const std::vector<uint32_t>& faultKeys);

    /**  故障抑制的恢复，提供给用户根据自身条件主动恢复抑制故障的能力
     @param[in]  faultKeys 故障列表，faultKey 由 [faultId*100+faultObj] 构成
     @param[out] none
     @return     void
     @warning    none
     @note       none
    */
    void RecoverInhibitFault(const std::vector<uint32_t>& faultKeys);

    /**  抑制所有故障，提供给用户根据自身条件主动抑制故障的能力
     @param[in]  none
     @param[out] none
     @return     void
     @warning    none
     @note       none
    */
    void InhibitAllFault();

    /**  所有故障抑制的恢复，提供给用户根据自身条件主动恢复抑制故障的能力
     @param[in]  none
     @param[out] none
     @return     void
     @warning    none
     @note       none
    */
    void RecoverInhibitAllFault();

    /**  大数据上传接口
     @param[in]  none
     @param[out] outResult 文件列表
     @return     int32_t 0：上报成功 负值：获取失败
     @warning    仅限大数据使用，其他进程请不要使用此接口
     @note       none
    */
    int32_t GetDataCollectionFile(std::vector<std::string>& outResult); // abandon
    int32_t GetDataCollectionFile(std::function<void(std::vector<std::string>&)> collectionFileCb);

private:
    PHMClient(const PHMClient &);
    PHMClient & operator = (const PHMClient &);

    std::unique_ptr<PHMClientImpl> phm_impl_;

};

}  // namespace phm
}  // namespace netaos
}  // namespace hozon
#endif  // PHM_CLIENT_H
