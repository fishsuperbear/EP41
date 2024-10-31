/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: dc_client.h
 * @Date: 2023/10/27
 * @Author: cheng
 * @Desc: --
 */

#pragma once
#ifndef NOS_COMMIT_SERVICE_DATA_COLLECTION_INCLUDE_DC_CLIENT_H__
#define NOS_COMMIT_SERVICE_DATA_COLLECTION_INCLUDE_DC_CLIENT_H__

#include "cm/include/method.h"
#include "idl/generated/data_collection_infoPubSubTypes.h"

using namespace hozon::netaos::cm;
using namespace std;

namespace hozon {
namespace netaos {
namespace dc {

enum DcResultCode {
    DC_OK = 0,
    DC_TIMEOUT=50,
    DC_INNER_ERROR=100,
    DC_SERVICE_NO_READY=150,
    DC_INVALID_PARAM=200,
    DC_NO_PARAM=250,
    DC_PATH_NOT_FOUND=300,
    DC_UNSUPPORT = 400
};

enum UploadDataType {
    file,
    folder,
    fileAndFolder,
    memory,
    null,
};


class DcClient {
   public:
    DcClient();

    ~DcClient();

    /**
     * 初始化neta_dc，CM通信建立。调用其他接口前调用一次。
     * @param client_name 调用方名称, 自定义。长度限制：[2, 100]
     * @param max_wait_millis 服务发现超时时间。單位毫秒，取值范围：[100, 10000]
     * @return DcResultCode枚举
     */
    DcResultCode Init(const std::string client_name, const uint32_t max_wait_millis = 1000);

    /**
     * 资源释放，CM通信销毁。
     * @return DcResultCode枚举
     */
    DcResultCode DeInit();

    /**
     * 触发采集
     * @param trigger_id 数据回传触发器ID，有效值：参见MDC数据上云收集
     * @return DcResultCode枚举
     */
    DcResultCode CollectTrigger(uint32_t trigger_id) ;

    /**  触发采集, 在后台进行数据采集
     @param[in]  trigger_id: trigger id.
     @param[in]  time: trigger time.
     @param[out] none
     @return     DcResultCode: result that indicates whether trigger description is sent successfully.
     @warning    None
     @note       None
    */
    DcResultCode CollectTriggerDesc(uint32_t trigger_id, uint64_t time);

    /**  Send collect trigger description.
     @param[in]  trigger_id: trigger id.
     @param[in]  time: trigger time.
     @param[in]  desc: trigger description.
     @param[out] none
     @return     DcResultCode: result that indicates whether trigger description is sent successfully.
     @warning    None
     @note       None
    */
    DcResultCode CollectTriggerDesc(uint32_t trigger_id, uint64_t time, std::string desc);

    /**
     * 上传指定的文件夹/文件，尽量避免自行上传数据场景. 需要自行上传文件的场景, 请和SOC 同事确认场景, 确认可以使用后记录业务的进程名, 否则上传可能失败.
     * @param path_list 上传数据路径列表.
     * @param file_type 文件的类型。有效值：["CAN","TRIGGER","FAULT","ETH","CALIBRATION","LOG","PLANNING","MCU"]
     * @param file_name 文件压缩包名称。
     * @param cache_file_num 要缓存的文件数量。
     * @return DcResultCode枚举
     */
    DcResultCode Upload(std::vector<std::string> &path_list, std::string file_type, std::string file_name, uint16_t cache_file_num);

    /**
     * 上传指定的自定义数据，尽量避免自行上传数据场景. 需要自行上传文件的场景, 请和SOC 同事确认场景, 确认可以使用后记录业务的进程名, 否则上传可能失败.
     * @param data 要上传的自定义数据.自定义数据大小不得大于2M
     * @param file_type 文件的类型。有效值：["CAN","TRIGGER","FAULT","ETH","CALIBRATION","LOG","PLANNING","MCU"]
     * @param file_name 文件压缩包名称。
     * @param cache_file_num 要缓存的文件数量。
     * @return DcResultCode枚举
     */
    DcResultCode Upload(std::vector<char> &data, std::string file_type, std::string file_name, uint16_t cache_file_num);
   private:
    void *m_client;
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // NOS_COMMIT_SERVICE_DATA_COLLECTION_INCLUDE_DC_CLIENT_H__
