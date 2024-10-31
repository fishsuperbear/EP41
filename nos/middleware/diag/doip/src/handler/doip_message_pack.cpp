/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip message packer
 */
#include <stdio.h>
#include <stdlib.h>

#include "diag/common/include/timer_manager.h"
#include "diag/doip/include/handler/doip_message_pack.h"
#include "diag/doip/include/base/doip_util.h"
#include "diag/doip/include/base/doip_logger.h"


namespace hozon {
namespace netaos {
namespace diag {


std::function<uint32_t(doip_link_data_t*, std::string, uint32_t)> DoIPMessagePack::messageTaskCallback_;
DoIPMessageHandler* DoIPMessagePack::handler_;

void
DoIPMessagePack::RegistMessageTaskCallback(std::function<uint32_t(doip_link_data_t*, std::string, uint32_t)> callback,
                                           DoIPMessageHandler* handler) {
    messageTaskCallback_ = callback;
    handler_ = handler;
}

void
DoIPMessagePack::DoipPackDiagnostic2Cache(doip_link_data_t *link_data, doip_equip_tcp_table_t* equip_tcp_table) {
    DOIP_INFO << "<DoIPMessagePack> DoipPackDiagnostic2Cache run.";
    uint32_t payload_length = link_data->data_size + DOIP_LOGICAL_ADDRESS_LENGTH * 2;
    uint32_t header_length = DOIP_HEADER_COMMON_LENGTH;
    uint8_t protocol_version = DoIPConfig::Instance()->GetProtocalVersion();
    uint32_t pos = 0;

    char* send_data = new char[header_length + payload_length];
    char header[DOIP_HEADER_COMMON_LENGTH] =
        { static_cast<char>(protocol_version), static_cast<char>(~protocol_version),
          static_cast<char>(0x80), 0x01, 0x00, 0x00, 0x00, 0x00 };
    pos += DOIP_PROTOCOL_VERSION_LENGTH + DOIP_INVERSE_PROTOCOL_VERSION_LENGTH + DOIP_PAYLOAD_TYPE_LENGTH;
    payload_length = DoipUtil::Instance().DoipBswap32(payload_length);
    memcpy(header + pos, &payload_length, DOIP_PAYLOAD_LENGTH_LENGTH);
    payload_length = DoipUtil::Instance().DoipBswap32(payload_length);

    pos = 0;
    memcpy(send_data + pos, header, header_length);
    pos += header_length;
    uint16_t equip_logical_address = DoipUtil::Instance().DoipBswap16(equip_tcp_table->equip_logical_address);
    memcpy(send_data + pos, &equip_logical_address, DOIP_LOGICAL_ADDRESS_LENGTH);
    pos += DOIP_LOGICAL_ADDRESS_LENGTH;
    uint16_t entity_logical_address = DoipUtil::Instance().DoipBswap16(equip_tcp_table->entity_logical_address);
    memcpy(send_data + pos, &entity_logical_address, DOIP_LOGICAL_ADDRESS_LENGTH);
    pos += DOIP_LOGICAL_ADDRESS_LENGTH;
    memcpy(send_data + pos, link_data->data, link_data->data_size);

    doip_cache_data_t* cache_data = new doip_cache_data_t;
    cache_data->fd = link_data->fd;
    cache_data->data = send_data;
    cache_data->data_size = header_length + payload_length;
    if (equip_tcp_table) {
        equip_tcp_table->doip_data_queue->DoipInsertQueue(cache_data);
    }
}

void
DoIPMessagePack::DoipPackDiagnostic2Cache(doip_link_data_t *link_data, doip_equip_tcp_table_t* equip_tcp_table, uint16_t replace_address) {
    DOIP_INFO << "<DoIPMessagePack> DoipPackDiagnostic2Cache run.";
    uint32_t payload_length = link_data->data_size + DOIP_LOGICAL_ADDRESS_LENGTH * 2;
    uint32_t header_length = DOIP_HEADER_COMMON_LENGTH;
    uint8_t protocol_version = DoIPConfig::Instance()->GetProtocalVersion();
    uint32_t pos = 0;

    char* send_data = new char[header_length + payload_length];
    char header[DOIP_HEADER_COMMON_LENGTH] =
        { static_cast<char>(protocol_version), static_cast<char>(~protocol_version),
          static_cast<char>(0x80), 0x01, 0x00, 0x00, 0x00, 0x00 };
    pos += DOIP_PROTOCOL_VERSION_LENGTH + DOIP_INVERSE_PROTOCOL_VERSION_LENGTH + DOIP_PAYLOAD_TYPE_LENGTH;
    payload_length = DoipUtil::Instance().DoipBswap32(payload_length);
    memcpy(header + pos, &payload_length, DOIP_PAYLOAD_LENGTH_LENGTH);
    payload_length = DoipUtil::Instance().DoipBswap32(payload_length);

    pos = 0;
    memcpy(send_data + pos, header, header_length);
    pos += header_length;
    uint16_t equip_logical_address = DoipUtil::Instance().DoipBswap16(equip_tcp_table->equip_logical_address);
    memcpy(send_data + pos, &equip_logical_address, DOIP_LOGICAL_ADDRESS_LENGTH);
    pos += DOIP_LOGICAL_ADDRESS_LENGTH;
    uint16_t entity_logical_address = DoipUtil::Instance().DoipBswap16(replace_address);
    memcpy(send_data + pos, &entity_logical_address, DOIP_LOGICAL_ADDRESS_LENGTH);
    pos += DOIP_LOGICAL_ADDRESS_LENGTH;
    memcpy(send_data + pos, link_data->data, link_data->data_size);

    doip_cache_data_t* cache_data = new doip_cache_data_t;
    cache_data->fd = link_data->fd;
    cache_data->data = send_data;
    cache_data->data_size = header_length + payload_length;
    if (equip_tcp_table) {
        equip_tcp_table->doip_data_queue->DoipInsertQueue(cache_data);
    }
}

void
DoIPMessagePack::DoipPackVehicleIdentifyReq2Udp(doip_link_data_t *link_data) {
    DOIP_INFO << "<DoIPMessagePack> DoipPackVehicleIdentifyReq2Udp run.";
    doip_link_data_t* link_data_ptr = new doip_link_data_t;
    memset(link_data_ptr, 0, sizeof(doip_link_data_t));

    uint32_t pos = 0;
    uint32_t payload_length = 0;
    uint32_t header_length = DOIP_HEADER_COMMON_LENGTH;
    uint32_t data_size = payload_length + header_length;
    uint8_t protocol_version = DoIPConfig::Instance()->GetProtocalVersion();
    char* send_data = new char[DOIP_HEADER_COMMON_LENGTH];
    char header[DOIP_HEADER_COMMON_LENGTH] =
        { static_cast<char>(protocol_version), static_cast<char>(~protocol_version),
          0x00, 0x01, 0x00, 0x00, 0x00, 0x00 };

    memcpy(send_data + pos, header, header_length);

    link_data_ptr->fd = link_data->fd;
    link_data_ptr->comm_type = link_data->comm_type;
    memcpy(link_data_ptr->ip, link_data->ip, strlen(link_data->ip) + 1);
    link_data_ptr->port = link_data->port;
    link_data_ptr->data = send_data;
    link_data_ptr->data_size = data_size;

    messageTaskCallback_(link_data_ptr, "DOIP_TASK_TYPE_SEND", 0);
}

void
DoIPMessagePack::DoipPackVehicleIdentifyReqWithEID2Udp(doip_link_data_t *link_data, char *eid) {
    DOIP_INFO << "<DoIPMessagePack> DoipPackVehicleIdentifyReqWithEID2Udp run.";
    doip_link_data_t* link_data_ptr = new doip_link_data_t;
    memset(link_data_ptr, 0, sizeof(doip_link_data_t));

    uint32_t pos = 0;
    uint32_t payload_length = DOIP_EID_SIZE;
    uint32_t header_length = DOIP_HEADER_COMMON_LENGTH;
    uint32_t data_size = payload_length + header_length;
    uint8_t protocol_version = DoIPConfig::Instance()->GetProtocalVersion();
    char* send_data = new char[DOIP_HEADER_COMMON_LENGTH + DOIP_EID_SIZE];
    char header[DOIP_HEADER_COMMON_LENGTH] =
        { static_cast<char>(protocol_version), static_cast<char>(~protocol_version),
          0x00, 0x02, 0x00, 0x00, 0x00, 0x06 };

    memcpy(send_data + pos, header, header_length);
    pos += header_length;
    memcpy(send_data + pos, eid, payload_length);

    link_data_ptr->fd = link_data->fd;
    link_data_ptr->comm_type = link_data->comm_type;
    memcpy(link_data_ptr->ip, link_data->ip, strlen(link_data->ip) + 1);
    link_data_ptr->port = link_data->port;
    link_data_ptr->data = send_data;
    link_data_ptr->data_size = data_size;

    messageTaskCallback_(link_data_ptr, "DOIP_TASK_TYPE_SEND", 0);
}

void
DoIPMessagePack::DoipPackVehicleIdentifyReqWithVIN2Udp(doip_link_data_t *link_data, char *vin) {
    DOIP_INFO << "<DoIPMessagePack> DoipPackVehicleIdentifyReqWithVIN2Udp run.";
    doip_link_data_t* link_data_ptr = new doip_link_data_t;
    memset(link_data_ptr, 0, sizeof(doip_link_data_t));

    uint32_t pos = 0;
    uint32_t payload_length = DOIP_VIN_SIZE;
    uint32_t header_length = DOIP_HEADER_COMMON_LENGTH;
    uint32_t data_size = payload_length + header_length;
    uint8_t protocol_version = DoIPConfig::Instance()->GetProtocalVersion();
    char* send_data = new char[DOIP_HEADER_COMMON_LENGTH + DOIP_VIN_SIZE];
    char header[DOIP_HEADER_COMMON_LENGTH] =
        { static_cast<char>(protocol_version), static_cast<char>(~protocol_version),
          0x00, 0x03, 0x00, 0x00, 0x00, 0x11 };

    memcpy(send_data + pos, header, header_length);
    pos += header_length;
    memcpy(send_data + pos, vin, payload_length);

    link_data_ptr->fd = link_data->fd;
    link_data_ptr->comm_type = link_data->comm_type;
    memcpy(link_data_ptr->ip, link_data->ip, strlen(link_data->ip) + 1);
    link_data_ptr->port = link_data->port;
    link_data_ptr->data = send_data;
    link_data_ptr->data_size = data_size;

    messageTaskCallback_(link_data_ptr, "DOIP_TASK_TYPE_SEND", 0);
}

void
DoIPMessagePack::DoipPackAnnouceOrIdentifyResponse2Udp(doip_link_data_t *link_data,
                                                       uint8_t further_action,
                                                       uint8_t vin_gid_sync_status) {
    DOIP_INFO << "<DoIPMessagePack> DoipPackAnnouceOrIdentifyResponse2Udp run.";

    uint32_t pos = 0;
    uint32_t payload_length = link_data->data_size;
    uint32_t header_length = DOIP_HEADER_COMMON_LENGTH;
    uint32_t data_size = payload_length + header_length;
    uint8_t protocol_version = DoIPConfig::Instance()->GetProtocalVersion();
    uint16_t logical_addr = DoIPConfig::Instance()->GetEntityConfig().logical_address;
    char* vin = DoIPConfig::Instance()->GetVIN();
    char* gid = DoIPConfig::Instance()->GetGID();
    char* eid = DoIPConfig::Instance()->GetEID();
    char* send_data = new char[DOIP_HEADER_COMMON_LENGTH + DOIP_ANNOUNCE_OR_IDENTITYRES_ALL_LENGTH];
    char header[DOIP_HEADER_COMMON_LENGTH] =
        { static_cast<char>(protocol_version), static_cast<char>(~protocol_version),
         0x00, 0x04, 0x00, 0x00, 0x00, 0x20 };

    memcpy(send_data + pos, header, header_length);
    pos += header_length;
    memcpy(send_data + pos, vin, DOIP_VIN_SIZE);
    pos += DOIP_VIN_SIZE;
    logical_addr = DoipUtil::Instance().DoipBswap16(logical_addr);
    memcpy(send_data + pos, &logical_addr, DOIP_LOGICAL_ADDRESS_LENGTH);
    pos += DOIP_LOGICAL_ADDRESS_LENGTH;
    memcpy(send_data + pos, eid, DOIP_EID_SIZE);
    pos += DOIP_EID_SIZE;
    memcpy(send_data + pos, gid, DOIP_GID_SIZE);
    pos += DOIP_GID_SIZE;
    memcpy(send_data + pos, &further_action, DOIP_FURTHER_ACTION_LENGTH);
    pos += DOIP_FURTHER_ACTION_LENGTH;
    if (payload_length == DOIP_ANNOUNCE_OR_IDENTITYRES_ALL_LENGTH) {
        payload_length = DoipUtil::Instance().DoipBswap32(payload_length);
        memcpy(send_data + pos, &vin_gid_sync_status, DOIP_VIN_GID_SYNC_LENGTH);
        memcpy(send_data + 4, &payload_length, DOIP_PAYLOAD_LENGTH_LENGTH);
    }

    doip_link_data_t* link_data_ptr = new doip_link_data_t;
    memset(link_data_ptr, 0, sizeof(doip_link_data_t));
    link_data_ptr->fd = link_data->fd;
    link_data_ptr->comm_type = link_data->comm_type;
    memcpy(link_data_ptr->ip, link_data->ip, strlen(link_data->ip) + 1);
    link_data_ptr->port = link_data->port;
    link_data_ptr->data = send_data;
    link_data_ptr->data_size = data_size;

    messageTaskCallback_(link_data_ptr, "DOIP_TASK_TYPE_SEND", 0);
}

void
DoIPMessagePack::DoipPackHeaderNegativeAck2Udp(doip_link_data_t *link_data, uint8_t code) {
    DOIP_INFO << "<DoIPMessagePack> DoipPackHeaderNegativeAck2Udp run.";
    doip_link_data_t* link_data_ptr = new doip_link_data_t;
    memset(link_data_ptr, 0, sizeof(doip_link_data_t));

    uint32_t pos = 0;
    uint32_t payload_length = DOIP_HEADER_NEGATIVE_ACK_LENGTH;
    uint32_t header_length = DOIP_HEADER_COMMON_LENGTH;
    uint32_t data_size = payload_length + header_length;
    uint8_t protocol_version = DoIPConfig::Instance()->GetProtocalVersion();
    char* send_data = new char[DOIP_HEADER_COMMON_LENGTH + DOIP_HEADER_NEGATIVE_ACK_LENGTH];
    char header[DOIP_HEADER_COMMON_LENGTH] =
        { static_cast<char>(protocol_version), static_cast<char>(~protocol_version),
          0x00, 0x00, 0x00, 0x00, 0x00, 0x01 };

    memcpy(send_data + pos, header, header_length);
    pos += header_length;
    memcpy(send_data + pos, &code, DOIP_NACK_CODE_LENGTH);

    link_data_ptr->fd = link_data->fd;
    link_data_ptr->comm_type = link_data->comm_type;
    memcpy(link_data_ptr->ip, link_data->ip, strlen(link_data->ip) + 1);
    link_data_ptr->port = link_data->port;
    link_data_ptr->data = send_data;
    link_data_ptr->data_size = data_size;

    messageTaskCallback_(link_data_ptr, "DOIP_TASK_TYPE_SEND", 0);
}

void
DoIPMessagePack::DoipPackEntityStatusReq2Udp(doip_link_data_t *link_data) {
    DOIP_INFO << "<DoIPMessagePack> DoipPackEntityStatusReq2Udp run.";
    doip_link_data_t* link_data_ptr = new doip_link_data_t;
    memset(link_data_ptr, 0, sizeof(doip_link_data_t));

    uint32_t pos = 0;
    uint32_t payload_length = 0;
    uint32_t header_length = DOIP_HEADER_COMMON_LENGTH;
    uint32_t data_size = payload_length + header_length;
    uint8_t protocol_version = DoIPConfig::Instance()->GetProtocalVersion();
    char* send_data = new char[DOIP_HEADER_COMMON_LENGTH];
    char header[DOIP_HEADER_COMMON_LENGTH] =
        { static_cast<char>(protocol_version), static_cast<char>(~protocol_version),
          0x40, 0x01, 0x00, 0x00, 0x00, 0x00 };

    memcpy(send_data + pos, header, header_length);

    link_data_ptr->fd = link_data->fd;
    link_data_ptr->comm_type = link_data->comm_type;
    memcpy(link_data_ptr->ip, link_data->ip, strlen(link_data->ip) + 1);
    link_data_ptr->port = link_data->port;
    link_data_ptr->data = send_data;
    link_data_ptr->data_size = data_size;

    messageTaskCallback_(link_data_ptr, "DOIP_TASK_TYPE_SEND", 0);
}

void
DoIPMessagePack::DoipPackEntityStatusRes2Udp(doip_link_data_t *link_data,
                                             const doip_entity_config_t& entity_config,
                                             uint8_t ncts) {
    DOIP_INFO << "<DoIPMessagePack> DoipPackEntityStatusRes2Udp run.";
    doip_node_type_t node_type = DOIP_NT_UNKOWN;
    if (DOIP_ENTITY_TYPE_EDGE_GATEWAY == entity_config.entity_type || DOIP_ENTITY_TYPE_GATEWAY == entity_config.entity_type) {
        node_type = DOIP_NT_GATEWAY;
    } else if (DOIP_ENTITY_TYPE_NODE == entity_config.entity_type) {
        node_type = DOIP_NT_NODE;
    } else {
        DOIP_WARN << "<DoIPMessagePack> DoipPackEntityStatusRes2Udp config->entity_type is unknown!.";
    }
    doip_link_data_t* link_data_ptr = new doip_link_data_t;
    memset(link_data_ptr, 0, sizeof(doip_link_data_t));

    uint32_t pos = 0;
    uint32_t payload_length = link_data->data_size;
    uint32_t header_length = DOIP_HEADER_COMMON_LENGTH;
    uint32_t data_size = payload_length + header_length;
    uint8_t protocol_version = DoIPConfig::Instance()->GetProtocalVersion();
    char* send_data = new char[DOIP_HEADER_COMMON_LENGTH + DOIP_ENTITY_STATUS_RES_ALL_LENGTH];
    char header[DOIP_HEADER_COMMON_LENGTH] =
        { static_cast<char>(protocol_version), static_cast<char>(~protocol_version),
          0x40, 0x02, 0x00, 0x00, 0x00, 0x03 };

    memcpy(send_data + pos, header, header_length);
    pos += header_length;
    memcpy(send_data + pos, &node_type, DOIP_NODE_TYPE_LENGTH);
    pos += DOIP_NODE_TYPE_LENGTH;
    memcpy(send_data + pos, &entity_config.mcts, DOIP_MCTS_LENGTH);
    pos += DOIP_MCTS_LENGTH;
    memcpy(send_data + pos, &ncts, DOIP_NCTS_LENGTH);
    pos += DOIP_NCTS_LENGTH;
    if (payload_length == DOIP_ENTITY_STATUS_RES_ALL_LENGTH) {
        uint32_t mds = DoipUtil::Instance().DoipBswap32(entity_config.mds + 8); // fix qibiao test
        payload_length = DoipUtil::Instance().DoipBswap32(payload_length);
        memcpy(send_data + pos, &mds, DOIP_MDS_LENGTH);
        memcpy(send_data + 4, &payload_length, DOIP_PAYLOAD_LENGTH_LENGTH);
    }

    link_data_ptr->fd = link_data->fd;
    link_data_ptr->comm_type = link_data->comm_type;
    memcpy(link_data_ptr->ip, link_data->ip, strlen(link_data->ip) + 1);
    link_data_ptr->port = link_data->port;
    link_data_ptr->data = send_data;
    link_data_ptr->data_size = data_size;

    messageTaskCallback_(link_data_ptr, "DOIP_TASK_TYPE_SEND", 0);
}

void
DoIPMessagePack::DoipPackPowerModeReq2Udp(doip_link_data_t *link_data) {
    DOIP_INFO << "<DoIPMessagePack> DoipPackPowerModeReq2Udp run.";
    doip_link_data_t* link_data_ptr = new doip_link_data_t;
    memset(link_data_ptr, 0, sizeof(doip_link_data_t));

    uint32_t pos = 0;
    uint32_t payload_length = 0;
    uint32_t header_length = DOIP_HEADER_COMMON_LENGTH;
    uint32_t data_size = payload_length + header_length;
    uint8_t protocol_version = DoIPConfig::Instance()->GetProtocalVersion();
    char* send_data = new char[DOIP_HEADER_COMMON_LENGTH];
    char header[DOIP_HEADER_COMMON_LENGTH] =
        { static_cast<char>(protocol_version), static_cast<char>(~protocol_version),
          0x40, 0x03, 0x00, 0x00, 0x00, 0x00 };

    memcpy(send_data + pos, header, header_length);

    link_data_ptr->fd = link_data->fd;
    link_data_ptr->comm_type = link_data->comm_type;
    memcpy(link_data_ptr->ip, link_data->ip, strlen(link_data->ip) + 1);
    link_data_ptr->port = link_data->port;
    link_data_ptr->data = send_data;
    link_data_ptr->data_size = data_size;

    messageTaskCallback_(link_data_ptr, "DOIP_TASK_TYPE_SEND", 0);
}

void
DoIPMessagePack::DoipPackPowerModeRes2Udp(doip_link_data_t *link_data, uint8_t diag_power_mode) {
    DOIP_INFO << "<DoIPMessagePack> DoipPackPowerModeRes2Udp run.";
    doip_link_data_t* link_data_ptr = new doip_link_data_t;
    memset(link_data_ptr, 0, sizeof(doip_link_data_t));

    uint32_t pos = 0;
    uint32_t payload_length = DOIP_POWERMODE_INFO_RES_LENGTH;
    uint32_t header_length = DOIP_HEADER_COMMON_LENGTH;
    uint32_t data_size = payload_length + header_length;
    uint8_t protocol_version = DoIPConfig::Instance()->GetProtocalVersion();
    char* send_data = new char[DOIP_HEADER_COMMON_LENGTH + DOIP_POWERMODE_INFO_RES_LENGTH];
    char header[DOIP_HEADER_COMMON_LENGTH] =
        { static_cast<char>(protocol_version), static_cast<char>(~protocol_version),
          0x40, 0x04, 0x00, 0x00, 0x00, 0x01 };

    memcpy(send_data + pos, header, header_length);
    pos += header_length;
    memcpy(send_data + pos, &diag_power_mode, DOIP_POWERMODE_LENGTH);

    link_data_ptr->fd = link_data->fd;
    link_data_ptr->comm_type = link_data->comm_type;
    memcpy(link_data_ptr->ip, link_data->ip, strlen(link_data->ip) + 1);
    link_data_ptr->port = link_data->port;
    link_data_ptr->data = send_data;
    link_data_ptr->data_size = data_size;

    messageTaskCallback_(link_data_ptr, "DOIP_TASK_TYPE_SEND", 0);
}

uint32_t
DoIPMessagePack::DoipPackDiagPositiveAck2Tcp(doip_link_data_t *link_data,
                                             uint16_t source_logical_address,
                                             uint16_t target_logical_address) {
    DOIP_INFO << "<DoIPMessagePack> DoipPackDiagPositiveAck2Tcp run.";
    doip_link_data_t* link_data_ptr = new doip_link_data_t;
    memset(link_data_ptr, 0, sizeof(doip_link_data_t));

    uint32_t pos = 0;
    uint32_t payload_length = DOIP_DIAG_POSITIVE_ACK_LENGTH;
    uint32_t header_length = DOIP_HEADER_COMMON_LENGTH;
    uint32_t data_size = payload_length + header_length;
     uint8_t protocol_version = DoIPConfig::Instance()->GetProtocalVersion();
    char* send_data = new char[DOIP_HEADER_COMMON_LENGTH + DOIP_DIAG_POSITIVE_ACK_LENGTH];
    char header[DOIP_HEADER_COMMON_LENGTH] =
        { static_cast<char>(protocol_version), static_cast<char>(~protocol_version),
          static_cast<char>(0x80), 0x02, 0x00, 0x00, 0x00, 0x05 };

    memcpy(send_data + pos, header, header_length);
    pos += header_length;
    source_logical_address = DoipUtil::Instance().DoipBswap16(source_logical_address);
    memcpy(send_data + pos, &source_logical_address, DOIP_LOGICAL_ADDRESS_LENGTH);
    pos += DOIP_LOGICAL_ADDRESS_LENGTH;
    target_logical_address = DoipUtil::Instance().DoipBswap16(target_logical_address);
    memcpy(send_data + pos, &target_logical_address, DOIP_LOGICAL_ADDRESS_LENGTH);
    pos += DOIP_LOGICAL_ADDRESS_LENGTH;
    uint8_t code = 0x00;
    memcpy(send_data + pos, &code, DOIP_ACK_CODE_LENGTH);

    link_data_ptr->fd = link_data->fd;
    link_data_ptr->comm_type = link_data->comm_type;
    link_data_ptr->data = send_data;
    link_data_ptr->data_size = data_size;

    return messageTaskCallback_(link_data_ptr, "DOIP_TASK_TYPE_SEND", 0);
}

uint32_t
DoIPMessagePack::DoipPackDiagNegativeAck2Tcp(doip_link_data_t *link_data,
                                             uint16_t source_logical_address,
                                             uint16_t target_logical_address,
                                             uint8_t code) {
    DOIP_INFO << "<DoIPMessagePack> DoipPackDiagNegativeAck2Tcp run.";
    doip_link_data_t* link_data_ptr = new doip_link_data_t;
    memset(link_data_ptr, 0, sizeof(doip_link_data_t));

    uint32_t pos = 0;
    uint32_t payload_length = DOIP_DIAG_POSITIVE_ACK_LENGTH;
    uint32_t header_length = DOIP_HEADER_COMMON_LENGTH;
    uint32_t data_size = payload_length + header_length;
    uint8_t protocol_version = DoIPConfig::Instance()->GetProtocalVersion();
    char* send_data = new char[DOIP_HEADER_COMMON_LENGTH + DOIP_DIAG_POSITIVE_ACK_LENGTH];
    char header[DOIP_HEADER_COMMON_LENGTH] =
        { static_cast<char>(protocol_version), static_cast<char>(~protocol_version),
          static_cast<char>(0x80), 0x03, 0x00, 0x00, 0x00, 0x05 };

    memcpy(send_data + pos, header, header_length);
    pos += header_length;
    source_logical_address = DoipUtil::Instance().DoipBswap16(source_logical_address);
    memcpy(send_data + pos, &source_logical_address, DOIP_LOGICAL_ADDRESS_LENGTH);
    pos += DOIP_LOGICAL_ADDRESS_LENGTH;
    target_logical_address = DoipUtil::Instance().DoipBswap16(target_logical_address);
    memcpy(send_data + pos, &target_logical_address, DOIP_LOGICAL_ADDRESS_LENGTH);
    pos += DOIP_LOGICAL_ADDRESS_LENGTH;
    memcpy(send_data + pos, &code, DOIP_ACK_CODE_LENGTH);

    link_data_ptr->fd = link_data->fd;
    link_data_ptr->comm_type = link_data->comm_type;
    link_data_ptr->data = send_data;
    link_data_ptr->data_size = data_size;

    return messageTaskCallback_(link_data_ptr, "DOIP_TASK_TYPE_SEND", 0);
}

void
DoIPMessagePack::DoipPackHeaderNegativeAck2Tcp(doip_link_data_t *link_data, uint8_t code) {
    DOIP_INFO << "<DoIPMessagePack> DoipPackHeaderNegativeAck2Tcp run. code:" << code;
    doip_link_data_t* link_data_ptr = new doip_link_data_t;
    memset(link_data_ptr, 0, sizeof(doip_link_data_t));

    uint32_t pos = 0;
    uint32_t payload_length = DOIP_HEADER_NEGATIVE_ACK_LENGTH;
    uint32_t header_length = DOIP_HEADER_COMMON_LENGTH;
    uint32_t data_size = payload_length + header_length;
    uint8_t protocol_version = DoIPConfig::Instance()->GetProtocalVersion();
    char* send_data = new char[DOIP_HEADER_COMMON_LENGTH + DOIP_HEADER_NEGATIVE_ACK_LENGTH];
    char header[DOIP_HEADER_COMMON_LENGTH] =
        { static_cast<char>(protocol_version), static_cast<char>(~protocol_version),
          0x00, 0x00, 0x00, 0x00, 0x00, 0x01 };

    memcpy(send_data + pos, header, header_length);
    pos += header_length;
    memcpy(send_data + pos, &code, DOIP_NACK_CODE_LENGTH);

    link_data_ptr->fd = link_data->fd;
    link_data_ptr->comm_type = link_data->comm_type;
    link_data_ptr->data = send_data;
    link_data_ptr->data_size = data_size;

    uint32_t cid = 0;
    if (link_data->comm_type == DOIP_SOCKET_TYPE_TCP_CLIENT) {
        DOIP_DEBUG << "<DoIPMessagePack> DoipPackHeaderNegativeAck2Tcp skip pack header nack to tcp client.";
    } else {
        cid = messageTaskCallback_(link_data_ptr, "DOIP_TASK_TYPE_SEND", 0);
    }

    if (DOIP_HEADER_NACK_INCORRECT_PATTERN_FORMAT == code
        || DOIP_HEADER_NACK_INVALID_PAYLOAD_LENGTH == code) {
        doip_link_data_t* link_data_ptr2 = new doip_link_data_t;
        memset(link_data_ptr2, 0, sizeof(doip_link_data_t));
        link_data_ptr2->fd = link_data->fd;
        link_data_ptr2->comm_type = link_data->comm_type;

        messageTaskCallback_(link_data_ptr2, "DOIP_TASK_TYPE_CLOSE", cid);
    }
}

uint32_t
DoIPMessagePack::DoipPackRoutingActivationRes2Tcp(doip_link_data_t *link_data,
                                                  uint16_t equip_logical_address,
                                                  uint16_t entity_logical_address,
                                                  uint8_t res_code,
                                                  uint32_t oem_specific_use) {
    DOIP_INFO << "<DoIPMessagePack> DoipPackRoutingActivationRes2Tcp run.";
    doip_link_data_t* link_data_ptr = new doip_link_data_t;
    memset(link_data_ptr, 0, sizeof(doip_link_data_t));

    uint32_t pos = 0;
    uint32_t payload_length = DOIP_ROUTING_ACTIVATION_RES_MAND_LENGTH;
    uint32_t header_length = DOIP_HEADER_COMMON_LENGTH;
    uint32_t data_size = payload_length + header_length;
    uint8_t protocol_version = DoIPConfig::Instance()->GetProtocalVersion();
    char* send_data = new char[DOIP_HEADER_COMMON_LENGTH + DOIP_ROUTING_ACTIVATION_RES_ALL_LENGTH];
    char header[DOIP_HEADER_COMMON_LENGTH] =
        { static_cast<char>(protocol_version), static_cast<char>(~protocol_version),
          0x00, 0x06, 0x00, 0x00, 0x00, 0x09 };

    memcpy(send_data + pos, header, header_length);
    pos += header_length;
    equip_logical_address = DoipUtil::Instance().DoipBswap16(equip_logical_address);
    memcpy(send_data + pos, &equip_logical_address, DOIP_LOGICAL_ADDRESS_LENGTH);
    pos += DOIP_LOGICAL_ADDRESS_LENGTH;
    entity_logical_address = DoipUtil::Instance().DoipBswap16(entity_logical_address);
    memcpy(send_data + pos, &entity_logical_address, DOIP_LOGICAL_ADDRESS_LENGTH);
    pos += DOIP_LOGICAL_ADDRESS_LENGTH;
    memcpy(send_data + pos, &res_code, DOIP_RA_RES_CODE_LENGTH);
    pos += DOIP_RA_RES_CODE_LENGTH;
    uint32_t iso_reserved = 0x00000000;
    iso_reserved = DoipUtil::Instance().DoipBswap32(iso_reserved);
    memcpy(send_data + pos, &iso_reserved, DOIP_RESERVED_LENGTH);
    pos += DOIP_RESERVED_LENGTH;
    oem_specific_use = DoipUtil::Instance().DoipBswap32(oem_specific_use);
    memcpy(send_data + pos, &oem_specific_use, DOIP_OEM_SPECIFIC_LENGTH);

    link_data_ptr->fd = link_data->fd;
    link_data_ptr->comm_type = DOIP_SOCKET_TYPE_TCP_SERVER;
    link_data_ptr->data = send_data;
    link_data_ptr->data_size = data_size;

    return messageTaskCallback_(link_data_ptr, "DOIP_TASK_TYPE_SEND", 0);
}

void
DoIPMessagePack::DoipPackAliveCheckReq2Tcp(doip_link_data_t *link_data) {
    DOIP_INFO << "<DoIPMessagePack> DoipPackAliveCheckReq2Tcp run.";
    doip_link_data_t* link_data_ptr = new doip_link_data_t;
    memset(link_data_ptr, 0, sizeof(doip_link_data_t));

    uint32_t pos = 0;
    uint32_t payload_length = 0;
    uint32_t header_length = DOIP_HEADER_COMMON_LENGTH;
    uint32_t data_size = payload_length + header_length;
    uint8_t protocol_version = DoIPConfig::Instance()->GetProtocalVersion();
    char* send_data = new char[DOIP_HEADER_COMMON_LENGTH];
    char header[DOIP_HEADER_COMMON_LENGTH] =
        { static_cast<char>(protocol_version), static_cast<char>(~protocol_version),
          0x00, 0x07, 0x00, 0x00, 0x00, 0x00 };

    memcpy(send_data + pos, header, header_length);

    link_data_ptr->fd = link_data->fd;
    link_data_ptr->comm_type = DOIP_SOCKET_TYPE_TCP_SERVER;
    link_data_ptr->data = send_data;
    link_data_ptr->data_size = data_size;

    messageTaskCallback_(link_data_ptr, "DOIP_TASK_TYPE_SEND", 0);
}

void
DoIPMessagePack::DoipPackAliveCheckRes2Tcp(doip_link_data_t *link_data, uint16_t equip_logical_address) {
    DOIP_INFO << "<DoIPMessagePack> DoipPackAliveCheckRes2Tcp run.";
    doip_link_data_t* link_data_ptr = new doip_link_data_t;
    memset(link_data_ptr, 0, sizeof(doip_link_data_t));

    uint32_t pos = 0;
    uint32_t payload_length = DOIP_LOGICAL_ADDRESS_LENGTH;
    uint32_t header_length = DOIP_HEADER_COMMON_LENGTH;
    uint32_t data_size = payload_length + header_length;
    uint8_t protocol_version = DoIPConfig::Instance()->GetProtocalVersion();
    char* send_data = new char[DOIP_HEADER_COMMON_LENGTH + DOIP_LOGICAL_ADDRESS_LENGTH];
    char header[DOIP_HEADER_COMMON_LENGTH] =
        { static_cast<char>(protocol_version), static_cast<char>(~protocol_version),
          0x00, 0x08, 0x00, 0x00, 0x00, 0x02 };

    memcpy(send_data + pos, header, header_length);
    pos += header_length;
    equip_logical_address = DoipUtil::Instance().DoipBswap16(equip_logical_address);
    memcpy(send_data + pos, &equip_logical_address, DOIP_LOGICAL_ADDRESS_LENGTH);

    link_data_ptr->fd = link_data->fd;
    link_data_ptr->comm_type = DOIP_SOCKET_TYPE_TCP_CLIENT;
    link_data_ptr->data = send_data;
    link_data_ptr->data_size = data_size;

    messageTaskCallback_(link_data_ptr, "DOIP_TASK_TYPE_SEND", 0);
}

void
DoIPMessagePack::DoipPackDiagnostic2Tcp(doip_link_data_t *link_data,
                                        uint16_t logical_source_address,
                                        uint16_t logical_target_address) {
    DOIP_INFO << "<DoIPMessagePack> DoipPackDiagnostic2Tcp run.";
    doip_link_data_t* link_data_ptr = new doip_link_data_t;
    memset(link_data_ptr, 0, sizeof(doip_link_data_t));

    uint32_t pos = 0;
    uint32_t payload_length = link_data->data_size + DOIP_LOGICAL_ADDRESS_LENGTH * 2;
    uint32_t header_length = DOIP_HEADER_COMMON_LENGTH;
    uint32_t data_size = payload_length + header_length;
    uint8_t protocol_version = DoIPConfig::Instance()->GetProtocalVersion();
    char* send_data = new char[data_size];
    char header[DOIP_HEADER_COMMON_LENGTH] =
        { static_cast<char>(protocol_version), static_cast<char>(~protocol_version),
          static_cast<char>(0x80), 0x01, 0x00, 0x00, 0x00, 0x00 };
    payload_length = DoipUtil::Instance().DoipBswap32(payload_length);
    memcpy(header + DOIP_PROTOCOL_VERSION_LENGTH + DOIP_INVERSE_PROTOCOL_VERSION_LENGTH + DOIP_PAYLOAD_TYPE_LENGTH,
           &payload_length, DOIP_PAYLOAD_LENGTH_LENGTH);
    payload_length = DoipUtil::Instance().DoipBswap32(payload_length);

    memcpy(send_data + pos, header, header_length);
    pos += header_length;
    logical_source_address = DoipUtil::Instance().DoipBswap16(logical_source_address);
    memcpy(send_data + pos, &logical_source_address, DOIP_LOGICAL_ADDRESS_LENGTH);
    pos += DOIP_LOGICAL_ADDRESS_LENGTH;
    logical_target_address = DoipUtil::Instance().DoipBswap16(logical_target_address);
    memcpy(send_data + pos, &logical_target_address, DOIP_LOGICAL_ADDRESS_LENGTH);
    pos += DOIP_LOGICAL_ADDRESS_LENGTH;
    memcpy(send_data + pos, link_data->data, link_data->data_size);

    link_data_ptr->fd = link_data->fd;
    link_data_ptr->comm_type = DOIP_SOCKET_TYPE_TCP_SERVER;
    link_data_ptr->data = send_data;
    link_data_ptr->data_size = data_size;

    messageTaskCallback_(link_data_ptr, "DOIP_TASK_TYPE_SEND", 0);
}

void
DoIPMessagePack::DoipPackRoutingActivationReq2Tcp(doip_link_data_t *link_data, uint16_t logical_source_address,
                                                  uint8_t activation_type, uint32_t oem_specific_use) {
    DOIP_INFO << "<DoIPMessagePack> DoipPackRoutingActivationReq2Tcp run.";
    doip_link_data_t* link_data_ptr = new doip_link_data_t;
    memset(link_data_ptr, 0, sizeof(doip_link_data_t));

    uint32_t pos = 0;
    uint32_t payload_length = DOIP_ROUTING_ACTIVATION_REQ_MAND_LENGTH;
    uint32_t header_length = DOIP_HEADER_COMMON_LENGTH;
    uint32_t data_size = payload_length + header_length;
    uint8_t protocol_version = DoIPConfig::Instance()->GetProtocalVersion();
    char* send_data = new char[data_size];
    char header[DOIP_HEADER_COMMON_LENGTH] =
        { static_cast<char>(protocol_version), static_cast<char>(~protocol_version),
          0x00, 0x05, 0x00, 0x00, 0x00, 0x07 };

    memcpy(send_data + pos, header, header_length);
    pos += header_length;
    logical_source_address = DoipUtil::Instance().DoipBswap16(logical_source_address);
    memcpy(send_data + pos, &logical_source_address, DOIP_LOGICAL_ADDRESS_LENGTH);
    pos += DOIP_LOGICAL_ADDRESS_LENGTH;
    memcpy(send_data + pos, &activation_type, DOIP_ACTIVATION_TYPE_LENGTH);
    pos += DOIP_ACTIVATION_TYPE_LENGTH;
    uint32_t iso_reserved = 0x00000000;
    iso_reserved = DoipUtil::Instance().DoipBswap32(iso_reserved);
    memcpy(send_data + pos, &iso_reserved, DOIP_RESERVED_LENGTH);

    link_data_ptr->fd = link_data->fd;
    link_data_ptr->comm_type = DOIP_SOCKET_TYPE_TCP_CLIENT;
    link_data_ptr->data = send_data;
    link_data_ptr->data_size = data_size;

    messageTaskCallback_(link_data_ptr, "DOIP_TASK_TYPE_SEND", 0);
}


}  // namespace diag
}  // namespace netaos
}  // namespace hozon
/* EOF */
