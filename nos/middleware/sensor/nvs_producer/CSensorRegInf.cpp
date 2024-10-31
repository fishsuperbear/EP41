/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "CSensorRegInf.hpp"

void CSensorRegInf::InterfaceInitRegister(uint32_t uSensor, InterfaceRegister* interfaceregister, INvSIPLCamera* m_upCamera) {
    IInterfaceProvider* moduleInterfaceProvider = nullptr;
    SIPLStatus status = m_upCamera->GetModuleInterfaceProvider(uSensor, moduleInterfaceProvider);

    if (status != NVSIPL_STATUS_OK) {
        LOG_ERR("Error %d while getting module interface provider for sensor ID: %d\n",status, uSensor);
    } else if (moduleInterfaceProvider != nullptr) {
        interfaceregister->Deserializer[uSensor] = static_cast<DeSER_CustomInterface*>
                    (moduleInterfaceProvider->GetInterface(DeSER_CUSTOM_INTERFACE_ID));
        if (interfaceregister->Deserializer[uSensor] != nullptr) {
            // Verify that the ID matches expected - we have the correct custom interface
            if (interfaceregister->Deserializer[uSensor]->getInstanceInterfaceID() == DeSER_CUSTOM_INTERFACE_ID) {
                LOG_DBG("deserializer interface found\n");
            } else {
                LOG_ERR("Incorrect interface obtained from module\n");
                interfaceregister->Deserializer[uSensor] = nullptr;
            }
        }

        interfaceregister->serializer[uSensor] = static_cast<SER_CustomInterface*>
                    (moduleInterfaceProvider->GetInterface(SER_CUSTOM_INTERFACE_ID));
        if (interfaceregister->serializer[uSensor] != nullptr) {
            // Verify that the ID matches expected - we have the correct custom interface
            if (interfaceregister->serializer[uSensor]->getInstanceInterfaceID() == SER_CUSTOM_INTERFACE_ID) {
                LOG_DBG("serializer interface found\n");
            } else {
                LOG_ERR("Incorrect interface obtained from module\n");
                interfaceregister->serializer[uSensor] = nullptr;
            }
        }

        interfaceregister->sensor[uSensor] = static_cast<Sensor_CustomInterface*>
                    (moduleInterfaceProvider->GetInterface(SENSOR_CUSTOM_INTERFACE_ID));
        if (interfaceregister->sensor[uSensor] != nullptr) {
            // Verify that the ID matches expected - we have the correct custom interface
            if (interfaceregister->sensor[uSensor]->getInstanceInterfaceID() == SENSOR_CUSTOM_INTERFACE_ID) {
                LOG_DBG("sensor interface found\n");
            } else {
                LOG_ERR("Incorrect interface obtained from module\n");
                interfaceregister->sensor[uSensor] = nullptr;
            }
        }

        interfaceregister->eeprom[uSensor] = static_cast<EEPROM_CustomInterface*>
                    (moduleInterfaceProvider->GetInterface(EEPROM__CUSTOM_INTERFACE_ID));
        if (interfaceregister->eeprom[uSensor] != nullptr) {
            // Verify that the ID matches expected - we have the correct custom interface
            if (interfaceregister->eeprom[uSensor]->getInstanceInterfaceID() == EEPROM__CUSTOM_INTERFACE_ID) {
                LOG_DBG("eeprom interface found\n");
            } else {
                LOG_ERR("Incorrect interface obtained from module\n");
                interfaceregister->eeprom[uSensor] = nullptr;
            }
        }

        interfaceregister->campwr[uSensor] = static_cast<CAMPWR_CustomInterface*>
                    (moduleInterfaceProvider->GetInterface(CAMPWR__CUSTOM_INTERFACE_ID));
        if (interfaceregister->campwr[uSensor] != nullptr) {
            // Verify that the ID matches expected - we have the correct custom interface
            if (interfaceregister->campwr[uSensor]->getInstanceInterfaceID() == CAMPWR__CUSTOM_INTERFACE_ID){
                LOG_DBG("campwr interface found\n");
            } else {
                LOG_ERR("Incorrect interface obtained from module\n");
                interfaceregister->campwr[uSensor] = nullptr;
            }
        }
    }
}

SIPLStatus CSensorRegInf::InterfaceReadRegister(uint8_t uSensor, CAMERA_REG_TYPE dev, uint16_t address, uint8_t *data, uint16_t length) {
    SIPLStatus status = NVSIPL_STATUS_OK;

    if (uSensor >= 16) {
        LOG_ERR("[%s:%s():L%d],bad uSensor\n", __FILE__, __func__, __LINE__);
        status = NVSIPL_STATUS_BAD_ARGUMENT;
    } else {
        if (dev == CAMERA_REG_SENSOR) {
            if (interfaceregister.sensor[uSensor] == nullptr) {
                LOG_ERR("[%s:%s():L%d],id %d sensor interface not found\n", __FILE__, __func__, __LINE__, uSensor);
                status = NVSIPL_STATUS_INVALID_STATE;
            } else {
                status = interfaceregister.sensor[uSensor]->IReadRegisterSensor(address, data, length);
            }
        } else if (dev == CAMERA_REG_DESER) {
            if (interfaceregister.Deserializer[uSensor] == nullptr) {
                LOG_ERR("[%s:%s():L%d],id %d deserializer interface not found\n", __FILE__, __func__, __LINE__, uSensor);
                status = NVSIPL_STATUS_INVALID_STATE;
            } else {
                status = interfaceregister.Deserializer[uSensor]->IReadRegisterDeSER(address, data, length);
            }
        } else if (dev == CAMERA_REG_SER) {
            if (interfaceregister.serializer[uSensor] == nullptr) {
                LOG_ERR("[%s:%s():L%d],id %d serializer interface not found\n", __FILE__, __func__, __LINE__, uSensor);
                status = NVSIPL_STATUS_INVALID_STATE;
            } else {
                status = interfaceregister.serializer[uSensor]->IReadRegisterSER(address, data, length);
            }
        } else if (dev == CAMERA_REG_EEPROM) {
            if (interfaceregister.eeprom[uSensor] == nullptr) {
                LOG_ERR("[%s:%s():L%d],id %d eeprom interface not found\n", __FILE__, __func__, __LINE__, uSensor);
                status = NVSIPL_STATUS_INVALID_STATE;
            } else {
                status = interfaceregister.eeprom[uSensor]->IReadRegisterEEPROM(address, data, length);
            }
        } else if(dev == CAMERA_REG_CAMPWR) {
            if(interfaceregister.campwr[uSensor] == nullptr){
                LOG_ERR("[%s:%s():L%d],sensor id %d,campwr interface not found\n", __FILE__, __func__, __LINE__, uSensor);
                status = NVSIPL_STATUS_INVALID_STATE;
            } else {
                status = interfaceregister.campwr[uSensor]->IReadRegisterCAMPWR(address, data, length);
            }
        } else {
            LOG_ERR("[%s:%s():L%d],id %d no this dev\n", __FILE__, __func__, __LINE__, uSensor);
            status = NVSIPL_STATUS_BAD_ARGUMENT;
        }
    }

    return status;
}

SIPLStatus CSensorRegInf::InterfaceWriteRegister(uint8_t uSensor, CAMERA_REG_TYPE dev, uint16_t address, uint8_t *data, uint16_t length) {
    SIPLStatus status = NVSIPL_STATUS_OK;

    if (uSensor >= 16) {
        LOG_ERR("[%s:%s():L%d],bad uSensor\n", __FILE__, __func__, __LINE__);
        status = NVSIPL_STATUS_BAD_ARGUMENT;
    } else {
        if (dev == CAMERA_REG_SENSOR) {
            if (interfaceregister.sensor[uSensor] == nullptr) {
                LOG_ERR("[%s:%s():L%d],sensor interface not found\n", __FILE__, __func__, __LINE__);
                status = NVSIPL_STATUS_INVALID_STATE;
            } else {
                status = interfaceregister.sensor[uSensor]->IWriteRegisterSensor(address, data, length);
            }
        } else if (dev == CAMERA_REG_DESER) {
            if (interfaceregister.Deserializer[uSensor] == nullptr) {
                LOG_ERR("[%s:%s():L%d],deserializer interface not found\n", __FILE__, __func__, __LINE__);
                status = NVSIPL_STATUS_INVALID_STATE;
            } else {
                status = interfaceregister.Deserializer[uSensor]->IWriteRegisterDeSER(address, data, length);
            }
        } else if (dev == CAMERA_REG_SER) {
            if (interfaceregister.serializer[uSensor] == nullptr) {
                LOG_ERR("[%s:%s():L%d],serializer interface not found\n", __FILE__, __func__, __LINE__);
                status = NVSIPL_STATUS_INVALID_STATE;
            } else {
                status = interfaceregister.serializer[uSensor]->IWriteRegisterSER(address, data, length);
            }
        } else if (dev == CAMERA_REG_EEPROM) {
            if (interfaceregister.eeprom[uSensor] == nullptr) {
                LOG_ERR("[%s:%s():L%d],eeprom interface not found\n", __FILE__, __func__, __LINE__);
                status = NVSIPL_STATUS_INVALID_STATE;
            } else {
                status = interfaceregister.eeprom[uSensor]->IWriteRegisterEEPROM(address, data, length);
            }
        }  else if(dev == CAMERA_REG_CAMPWR) {
            if(interfaceregister.campwr[uSensor] == nullptr){
                LOG_ERR("[%s:%s():L%d],campwr interface not found\n", __FILE__, __func__, __LINE__);
                status = NVSIPL_STATUS_INVALID_STATE;
            } else {
                    status = interfaceregister.campwr[uSensor]->IWriteRegisterCAMPWR(address, data, length);
            }
        } else {
            LOG_ERR("[%s:%s():L%d],no this dev\n", __FILE__, __func__, __LINE__);
            status = NVSIPL_STATUS_BAD_ARGUMENT;
        }
    }

    return status;
}

void CSensorRegInf::GetRegisterInfo(std::string uSensorName, uint8_t uSensor, uint16_t length, CameraInternalData& camera_data) {
    uint16_t start_address = 0;
    camera_data.data.resize(length);
    InterfaceReadRegister(uSensor, CAMERA_REG_EEPROM, start_address, camera_data.data.data(), length);
    camera_data.isValid = true;
    camera_data.sensor_id = uSensor;
    camera_data.module_name = uSensorName;
}

void CSensorRegInf::Get_ISX031_RegisterInfo(std::string uSensorName, uint8_t uSensor, uint16_t length, CameraInternalData& camera_data) {
    uint8_t read_byte = 4;
    for (uint16_t i = 0; i < length / read_byte; i++) {
        uint8_t data_tmp[read_byte] = {0};
        InterfaceReadRegister(uSensor, CAMERA_REG_EEPROM, read_byte * i, data_tmp, read_byte);
        camera_data.data.push_back(data_tmp[3]);
        camera_data.data.push_back(data_tmp[2]);
        camera_data.data.push_back(data_tmp[1]);
        camera_data.data.push_back(data_tmp[0]);
    }
    camera_data.isValid = true;
    camera_data.sensor_id = uSensor;
    camera_data.module_name = uSensorName;
}

SIPLStatus CSensorRegInf::GetSensorRegisterInfo(const std::string uSensorName, const uint8_t uSensor) {
    SIPLStatus status = NVSIPL_STATUS_OK;
    CameraInternalData camera_data;

    if (uSensorName == SENSOR_0X8B40) {
        uint16_t length = 0x75;
        GetRegisterInfo(uSensorName, uSensor, length, camera_data);
        // test print
        camera_interal_data_x3f_x8b *data = 
                reinterpret_cast<camera_interal_data_x3f_x8b*>(camera_data.data.data());
        LOG_INFO("Camera [%s], fx: %lf fy: %lf cx: %lf cy: %lf k1: %lf k2: %lf k3: %lf k4: %lf k5: %lf k6: %lf p1: %lf p2: %lf \n",
            camera_data.module_name.c_str(),
            data->fx, data->fy, data->cx, data->cy,
            data->k1, data->k2, data->k3, data->k4, data->k5, data->k6,
            data->p1, data->p2);
    } else if (uSensorName == SENSOR_ISX021) {
        uint16_t length = 0x34;
        GetRegisterInfo(uSensorName, uSensor, length, camera_data);

        // test print
        camera_interal_data_x031 *data = 
                reinterpret_cast<camera_interal_data_x031*>(camera_data.data.data());
        LOG_INFO("Camera [%s], fx: %lf fy: %lf cx: %lf cy: %lf k1: %lf k2: %lf k3: %lf k4: %lf k5: %lf k6: %lf p1: %lf p2: %lf \n",
            camera_data.module_name.c_str(),
            data->fx, data->fy, data->cx, data->cy,
            data->k1, data->k2, data->k3, data->k4, data->k5, data->k6,
            data->p1, data->p2);
    } else if (uSensorName == SENSOR_ISX031) {
        uint16_t length = 0x34;
        Get_ISX031_RegisterInfo(uSensorName, uSensor, length, camera_data);

        // test print
        camera_interal_data_x031 *data = 
                reinterpret_cast<camera_interal_data_x031*>(camera_data.data.data());
        LOG_INFO("Camera [%s], fx: %lf fy: %lf cx: %lf cy: %lf k1: %lf k2: %lf k3: %lf k4: %lf k5: %lf k6: %lf p1: %lf p2: %lf \n",
            camera_data.module_name.c_str(),
            data->fx, data->fy, data->cx, data->cy,
            data->k1, data->k2, data->k3, data->k4, data->k5, data->k6,
            data->p1, data->p2);
    }

    if (camera_data.isValid == true) {
        camera_data_server.CameraDataInsertMap(camera_data.sensor_id, camera_data);
    }

    return status;
}
