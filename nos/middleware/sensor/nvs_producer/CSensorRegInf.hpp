/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#pragma once
/* STL Headers */
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <string>

#include "NvSIPLCamera.hpp"
#include "NvSIPLPipelineMgr.hpp"
#include "CUtils.hpp"
#include "CChannel.hpp"
#include "CIpcProducerChannel.hpp"

#include "nvscibuf.h"
#include "nvscisync.h"
#include "nvscistream.h"
#include "CustomInterface.hpp"
#include "cm/include/method.h"
#include "idl/generated/camera_internal_dataPubSubTypes.h"

using namespace std;
using namespace nvsipl;

typedef enum
{
    CAMERA_REG_SENSOR = 0,
    CAMERA_REG_DESER,
    CAMERA_REG_SER,
    CAMERA_REG_EEPROM,
    CAMERA_REG_CAMPWR
} CAMERA_REG_TYPE;

#pragma pack(push, 1)
struct camera_interal_data_x3f_x8b {
    uint8_t time_year;
    uint8_t time_month;
    uint8_t time_day;
    uint8_t version;
    uint8_t model;
    double fx;
    double fy;
    double cx;
    double cy;
    double k1;
    double k2;
    double k3;
    double k4;
    double k5;
    double k6;
    double p1;
    double p2;
    double averang;
    double maximum;
};

struct camera_interal_data_x031 {
    float fx;
    float fy;
    float cx;
    float cy;
    float k5;
    float k6;
    float xx;
    float k1;
    float k2;
    float p1;
    float p2;
    float k3;
    float k4;
};
#pragma pack(pop)

struct CameraInternalData {
    bool isValid;
    std::uint8_t sensor_id;
    std::string module_name;
    std::vector<uint8_t> data;
};

class CameraInternalDataServer : public hozon::netaos::cm::Server<camera_internal_data_request, camera_internal_data_reply> {
public:
    CameraInternalDataServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data)
        : Server(req_data, resp_data) {}
        
    int32_t Process(const std::shared_ptr<camera_internal_data_request> req, std::shared_ptr<camera_internal_data_reply> resp) {
        uint8_t sensor_id = req->sensor_id();
        std::lock_guard<std::mutex> lck(mtx);
        if(camera_internal_data.find(sensor_id) != camera_internal_data.end()) {
            resp->isvalid(camera_internal_data[sensor_id].isValid);
            resp->sensor_id(camera_internal_data[sensor_id].sensor_id);
            resp->data(camera_internal_data[sensor_id].data);
            resp->module_name(camera_internal_data[sensor_id].module_name);
        }
        return 0;
    }

    int32_t CameraDataInsertMap(uint8_t sensor_id, CameraInternalData data) {
        std::lock_guard<std::mutex> lck(mtx);
        camera_internal_data[sensor_id] = data;
        return 0;
    }

private:
    std::map<std::uint8_t, CameraInternalData> camera_internal_data;
    std::mutex mtx;
};

class CSensorRegInf
{
public:
    CSensorRegInf()
        : camera_data_server(std::make_shared<camera_internal_data_requestPubSubType>(), std::make_shared<camera_internal_data_replyPubSubType>())
    {
        camera_data_server.Start(2, "camera_interal_data");
    }

    ~CSensorRegInf(){}

    typedef struct {
        Sensor_CustomInterface* sensor[16];
        SER_CustomInterface* serializer[16];
        DeSER_CustomInterface* Deserializer[16];
        EEPROM_CustomInterface* eeprom[16];
        CAMPWR_CustomInterface* campwr[16];
    } InterfaceRegister;

    InterfaceRegister interfaceregister {
        .sensor = {nullptr},
        .serializer = {nullptr},
        .Deserializer = {nullptr},
        .eeprom = {nullptr},
        .campwr = {nullptr},
    };

    void InterfaceInitRegister(uint32_t uSensor, InterfaceRegister* interfaceregister, INvSIPLCamera* m_upCamera);
    SIPLStatus InterfaceReadRegister(uint8_t uSensor, CAMERA_REG_TYPE dev, uint16_t address, uint8_t *data, uint16_t length);
    SIPLStatus InterfaceWriteRegister(uint8_t uSensor, CAMERA_REG_TYPE dev, uint16_t address, uint8_t *data, uint16_t length);
    void GetRegisterInfo(std::string uSensorName, uint8_t uSensor, uint16_t length, CameraInternalData& camera_data);
    void Get_ISX031_RegisterInfo(std::string uSensorName, uint8_t uSensor, uint16_t length, CameraInternalData& camera_data);
    SIPLStatus GetSensorRegisterInfo(const std::string uSensorName, const uint8_t uSensor);

    void Deinit() {
        camera_data_server.Stop();
    }

private:
    CameraInternalDataServer camera_data_server;
};

