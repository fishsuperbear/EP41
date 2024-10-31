/**

* Copyright @ 2020 - 2027 iAuto Software(Shanghai) Co., Ltd.

* All Rights Reserved.

*

* Redistribution and use in source and binary forms, with or without

* modification, are NOT permitted except as agreed by

* iAuto Software(Shanghai) Co., Ltd.

*

* Unless required by applicable law or agreed to in writing, software

* distributed under the License is distributed on an "AS IS" BASIS,

* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

*/
#include <sys/uio.h>
#include "ne_someip_endpoint_tool.h"
#include "ne_someip_required_service_instance.h"
#include "ne_someip_provided_service_instance.h"
#include "ne_someip_internal_define.h"
#include "ne_someip_transmit.h"
#include "ne_someip_endpoint_define.h"
#include "ne_someip_tp.h"

/***********************************compare function for hashMap*************************************/
uint32_t ne_someip_ep_tool_unix_key_hash_func(const void* key)
{
    if (NULL == key) {
        return 0;
    }

    ne_someip_endpoint_unix_addr_t* temp_key = (ne_someip_endpoint_unix_addr_t*)key;
    return ne_someip_map_string_hash_func(temp_key->unix_path);
}

uint32_t ne_someip_ep_tool_net_key_hash_func(const void* key)
{
    if (NULL == key) {
        return 0;
    }

    ne_someip_endpoint_net_addr_t* temp_key = (ne_someip_endpoint_net_addr_t*)key;
    return (uint32_t)(temp_key->ip_addr + temp_key->port);
}

uint32_t ne_someip_ep_tool_inst_spec_key_hash_func(const void* key)
{
    if (NULL == key) {
        return 0;
    }

    ne_someip_service_instance_spec_t* temp_key = (ne_someip_service_instance_spec_t*)key;
    return (uint32_t)(temp_key->service_id + temp_key->instance_id + temp_key->major_version);
}

uint32_t ne_someip_ep_tool_client_inst_spec_key_hash_func(const void* key)
{
    if (NULL == key) {
        return 0;
    }

    ne_someip_endpoint_client_instance_spec_t* temp_key = (ne_someip_endpoint_client_instance_spec_t*)key;
    return (uint32_t)(temp_key->client_id + temp_key->inst_spec.service_id +
        temp_key->inst_spec.instance_id + temp_key->inst_spec.major_version);
}

uint32_t ne_someip_ep_tool_find_offer_spec_key_hash_func(const void* key)
{
    if (NULL == key) {
        return 0;
    }

    ne_someip_find_offer_service_spec_t* tmp_key = (ne_someip_find_offer_service_spec_t*)key;
    return (uint32_t)(tmp_key->ins_spec.service_id + tmp_key->ins_spec.instance_id +
        tmp_key->ins_spec.major_version + tmp_key->minor_version);
}

uint32_t ne_someip_ep_tool_uint16_hash_func(const void* key)
{
    if (!key) {
        return 0;
    }
    return (uint32_t)(*((uint16_t*)key));
}

int ne_someip_ep_tool_net_addr_compare(const void* k1, const void* k2)
{
    if (NULL == k1 && NULL == k2) {
        return -1;
    }
    if ((NULL == k1 && NULL != k2) || (NULL != k1 && NULL == k2)) {
        return -1;
    }

    ne_someip_endpoint_net_addr_t* k1_tmp = (ne_someip_endpoint_net_addr_t*)k1;
    ne_someip_endpoint_net_addr_t* k2_tmp = (ne_someip_endpoint_net_addr_t*)k2;

    if (k1_tmp->ip_addr != k2_tmp->ip_addr) {
        return -1;
    }
    if (k1_tmp->port != k2_tmp->port) {
        return -1;
    }
    if (k1_tmp->type != k2_tmp->type) {
        return -1;
    }
    return 0;
}

int ne_someip_ep_tool_unix_addr_compare(const void* k1, const void* k2)
{
    if (NULL == k1 && NULL == k2) {
        return -1;
    }
    if ((NULL == k1 && NULL != k2) || (NULL != k1 && NULL == k2)) {
        return -1;
    }

    ne_someip_endpoint_unix_addr_t* k1_tmp = (ne_someip_endpoint_unix_addr_t*)k1;
    ne_someip_endpoint_unix_addr_t* k2_tmp = (ne_someip_endpoint_unix_addr_t*)k2;

    return strcmp(k1_tmp->unix_path, k2_tmp->unix_path);
}

int ne_someip_ep_tool_service_instance_spec_compare(const void* k1, const void* k2)
{
    if (NULL == k1 && NULL == k2) {
        return -1;
    }
    if ((NULL == k1 && NULL != k2) || (NULL != k1 && NULL == k2)) {
        return -1;
    }

    ne_someip_service_instance_spec_t* k1_tmp = (ne_someip_service_instance_spec_t*)k1;
    ne_someip_service_instance_spec_t* k2_tmp = (ne_someip_service_instance_spec_t*)k2;

    if (k1_tmp->service_id != k2_tmp->service_id) {
        return -1;
    }

    if (k1_tmp->instance_id != k2_tmp->instance_id) {
        return -1;
    }

    if (k1_tmp->major_version != k2_tmp->major_version) {
        return -1;
    }
    return 0;
}

int ne_someip_ep_tool_client_instance_spec_compare(const void* k1, const void* k2)
{
    if (NULL == k1 && NULL == k2) {
        return -1;
    }
    if ((NULL == k1 && NULL != k2) || (NULL != k1 && NULL == k2)) {
        return -1;
    }

    ne_someip_endpoint_client_instance_spec_t* k1_tmp = (ne_someip_endpoint_client_instance_spec_t*)k1;
    ne_someip_endpoint_client_instance_spec_t* k2_tmp = (ne_someip_endpoint_client_instance_spec_t*)k2;

    if (k1_tmp->client_id != k2_tmp->client_id) {
        return -1;
    }

    if (k1_tmp->inst_spec.service_id != k2_tmp->inst_spec.service_id) {
        return -1;
    }

    if (k1_tmp->inst_spec.instance_id != k2_tmp->inst_spec.instance_id) {
        return -1;
    }

    if (k1_tmp->inst_spec.major_version != k2_tmp->inst_spec.major_version) {
        return -1;
    }
    return 0;
}

int ne_someip_ep_tool_int32_compare(const void* k1, const void* k2)
{
    if (NULL == k1 && NULL == k2) {
        return -1;
    }
    if ((NULL == k1 && NULL != k2) || (NULL != k1 && NULL == k2)) {
        return -1;
    }

    int32_t* k1_tmp = (int32_t*)k1;
    int32_t* k2_tmp = (int32_t*)k2;

    if (*k1_tmp != *k2_tmp) {
        return -1;
    }
    return 0;
}

int ne_someip_ep_tool_uint16_compare(const void* k1, const void* k2)
{
    if (NULL == k1 && NULL == k2) {
        return -1;
    }
    if ((NULL == k1 && NULL != k2) || (NULL != k1 && NULL == k2)) {
        return -1;
    }

    uint16_t* k1_tmp = (uint16_t*)k1;
    uint16_t* k2_tmp = (uint16_t*)k2;

    if (*k1_tmp != *k2_tmp) {
        return -1;
    }
    return 0;
}


/***********************************free function for hashmap*************************************/
void ne_someip_ep_tool_net_addr_free(void* data)
{
    if (NULL == data) {
        return;
    }

    ne_someip_endpoint_net_addr_t* addr = (ne_someip_endpoint_net_addr_t*)data;
    free(addr);
}

void ne_someip_ep_tool_unix_addr_free(void* data)
{
    if (NULL == data) {
        return;
    }

    ne_someip_endpoint_unix_addr_t* addr = (ne_someip_endpoint_unix_addr_t*)data;
    free(addr);
}

void ne_someip_ep_tool_udp_receive_buff_list_free(void* data)
{
    if (NULL != data) {
        free(data);
    }
}

void ne_someip_ep_tool_service_instance_info_free(void* data)
{
    if (NULL == data) {
        return;
    }

    ne_someip_endpoint_instance_info_t* info = (ne_someip_endpoint_instance_info_t*)data;
    // if (ne_someip_endpoint_instance_type_client == info->instance_type) {
    //     // ne_someip_required_service_instance_unref((ne_someip_required_service_instance_t*)(info->service_instance));
    // }
    // else if (ne_someip_endpoint_instance_type_server == info->instance_type ) {
    //     // TODO: wait provided instance interface
    //     // ne_someip_provided_service_instance_unref((ne_someip_provided_service_instance_t*)(info->service_instance));
    // }
    // else {

    // }

    free(info);
}

void ne_someip_ep_tool_transmit_link_info_free(void* data)
{
    if (NULL == data) {
        return;
    }

    ne_someip_endpoint_transmit_link_info_t* link_info = (ne_someip_endpoint_transmit_link_info_t*)data;
    link_info->transmit_link = NULL;
    free(link_info);
    data = NULL;
}

void ne_someip_ep_tool_int32_free(void* data)
{
    if (NULL == data) {
        return;
    }

    free((int32_t*)data);
    data = NULL;
}

void ne_someip_ep_tool_uint16_free(void* data)
{
    if (NULL == data) {
        return;
    }

    free((uint16_t*)data);
    data = NULL;
}

void ne_someip_ep_tool_error_code_free(void* data)
{
    if (NULL == data) {
        return;
    }

    free((ne_someip_error_code_t*)data);
    data = NULL;
}

void ne_someip_ep_tool_service_instance_spec_free(void* data)
{
    if (NULL == data) {
        return;
    }

    free((ne_someip_service_instance_spec_t*)data);
    data = NULL;
}

void ne_someip_ep_tool_client_instance_spec_free(void* data)
{
    if (NULL == data) {
        return;
    }

    free((ne_someip_endpoint_client_instance_spec_t*)data);
    data = NULL;
}

/***********************************free function for list*************************************/

void ne_someip_ep_tool_service_instance_info_list_free(void* data)
{
    if (NULL == data) {
        return;
    }

    ne_someip_list_destroy((ne_someip_list_t*)data, ne_someip_ep_tool_service_instance_info_free);
}

// send cache list free (unix/udp/tcp)。 两种情况：1.发送成功；2.socket出错或link断开。
void ne_someip_ep_tool_send_cache_list_free(void* data)
{
    if (NULL == data) {
        return;
    }

    ne_someip_list_destroy((ne_someip_list_t*)data, ne_someip_ep_tool_ep_send_cache_free);
}

void ne_someip_ep_tool_ep_send_cache_free(void* data)
{
    if (NULL == data) {
        return;
    }

    ne_someip_endpoint_send_cache_t* tmp_data = (ne_someip_endpoint_send_cache_t*)data;
    ne_someip_ep_tool_trans_buffer_free((ne_someip_trans_buffer_struct_t*)(tmp_data->buffer));
    tmp_data->buffer = NULL;
    tmp_data->seq_data = NULL;

    free(tmp_data);
}

void ne_someip_ep_tool_ep_send_cache_tp_free(void* data)
{
    if (NULL == data) {
        return;
    }

    ne_someip_endpoint_send_cache_t* tmp_data = (ne_someip_endpoint_send_cache_t*)data;
    ne_someip_tp_free_buffer_struct_header_data(tmp_data->buffer);
    tmp_data->buffer = NULL;
    tmp_data->seq_data = NULL;

    free(tmp_data);
}

void ne_someip_ep_tool_tp_data_list_free(void* data)
{
    if (NULL == data) {
        return;
    }

    ne_someip_list_destroy((ne_someip_list_t*)data, NULL);
}

void ne_someip_ep_tool_tp_data_list_all_free(void* data)
{
    if (NULL == data) {
        return;
    }

    ne_someip_list_destroy((ne_someip_list_t*)data, ne_someip_tp_free_buffer_struct);
}

void ne_someip_ep_tool_tp_iov_data_list_free(void* data)
{
    if (NULL == data) {
        return;
    }

    ne_someip_list_destroy((ne_someip_list_t*)data, ne_someip_ep_tool_trans_iov_buf_tp_free);
}

void ne_someip_ep_tool_trans_buffer_free(ne_someip_trans_buffer_struct_t* trans_buffer)
{
    if (NULL == trans_buffer) {
        return;
    }

    if (NULL != trans_buffer->ipc_data) {
        ne_someip_ep_tool_ep_buf_list_free(trans_buffer->ipc_data);
    }
    if (NULL != trans_buffer->someip_header) {
        ne_someip_ep_tool_ep_buf_list_free(trans_buffer->someip_header);
    }
    if (NULL != trans_buffer->payload) {
        ne_someip_payload_unref(trans_buffer->payload);
    }
    
    free(trans_buffer);
    trans_buffer = NULL;
}

void ne_someip_ep_tool_trans_buffer_no_payload_free(ne_someip_trans_buffer_struct_t* trans_buffer)
{
    if (NULL == trans_buffer) {
        return;
    }

    if (NULL != trans_buffer->ipc_data) {
        ne_someip_ep_tool_ep_buf_list_free(trans_buffer->ipc_data);
    }
    if (NULL != trans_buffer->someip_header) {
        ne_someip_ep_tool_ep_buf_list_free(trans_buffer->someip_header);
    }
    
    free(trans_buffer);
    trans_buffer = NULL;
}

// 正常接收数据时（unix/tcp），buffer数据传给upper，删除接收数据的ne_someip_transmit_normal_buffer_t
void ne_someip_ep_tool_trans_normal_buf_free(void* data)  // normal case
{
    if (NULL != data) {
        ne_someip_transmit_normal_buffer_t* temp_data = (ne_someip_transmit_normal_buffer_t*)data;
        temp_data->buffer = NULL;
        temp_data->user_data = NULL;

        free(temp_data);
    }
}

// 异常接收数据时（unix/tcp），删除接收数据的ne_someip_transmit_normal_buffer_t，包括实际接收数据的iov_buf
void ne_someip_ep_tool_trans_normal_buf_complete_free(void* data)  // abnormal case, free all buffer
{
    if (NULL != data) {
        ne_someip_transmit_normal_buffer_t* temp_data = (ne_someip_transmit_normal_buffer_t*)data;
        if (NULL != temp_data->buffer) {
            free(temp_data->buffer);
        }
        temp_data->user_data = NULL;

        free(temp_data);
    }
}

void ne_someip_ep_tool_trans_normal_buf_list_free(void* data)
{
    if (NULL != data) {
        ne_someip_list_destroy(data, ne_someip_ep_tool_trans_normal_buf_free);
    }
}

void ne_someip_ep_tool_udp_iov_cache_free(void* data)
{
    if (NULL == data) {
        return;
    }

    ne_someip_endpoint_udp_iov_cache_t* tmp_cache = (ne_someip_endpoint_udp_iov_cache_t*)data;
    ne_someip_ep_tool_trans_iov_buf_free(tmp_cache->buffer);
    ne_someip_ep_tool_net_addr_free(tmp_cache->peer_addr);
    ne_someip_ep_tool_tp_iov_data_list_free(tmp_cache->tp_buffer_list);
    ne_someip_ep_tool_ep_send_cache_free(tmp_cache->orig_data);
    free(tmp_cache);
    tmp_cache = NULL;
}

void ne_someip_ep_tool_udp_iov_cache_fail_free(void* data)
{
    if (NULL == data) {
        return;
    }

    ne_someip_endpoint_udp_iov_cache_t* tmp_cache = (ne_someip_endpoint_udp_iov_cache_t*)data;
    ne_someip_ep_tool_trans_iov_buf_trig_fail_free(tmp_cache->buffer);
    ne_someip_ep_tool_net_addr_free(tmp_cache->peer_addr);
    tmp_cache->tp_buffer_list = NULL;
    tmp_cache->orig_data = NULL;
    if (NULL != tmp_cache->orig_data) {
        free(tmp_cache->orig_data);
    }
    free(tmp_cache);
    tmp_cache = NULL;
}

void ne_someip_ep_tool_udp_iov_cache_list_free(void* data)
{
    if (NULL != data) {
        ne_someip_list_destroy(data, ne_someip_ep_tool_udp_iov_cache_free);
    }
}

// trigger成功后，发送数据（unix/tcp/udp）成功/失败时，数据清除处理；接收数据（udp）成功时，数据清除处理
void ne_someip_ep_tool_trans_iov_buf_free(void* data)
{
    if (NULL == data) {
        return;
    }

    ne_someip_transmit_iov_buffer_t* tmp_buffer = (ne_someip_transmit_iov_buffer_t*)data;
    if (NULL != tmp_buffer->iovBuffer) {
        free(tmp_buffer->iovBuffer);
        tmp_buffer->iovBuffer = NULL;
    }
    if (NULL != tmp_buffer->user_data) {  // 发送数据时的缓存，如果是接收数据，此时应该为NULL（TCP、UNIX间断收数据需再考虑）
        ne_someip_ep_tool_ep_send_cache_free(tmp_buffer->user_data);
        tmp_buffer->user_data = NULL;
    }

    free(tmp_buffer);
    tmp_buffer = NULL;
}

void ne_someip_ep_tool_trans_iov_buf_tp_free(void* data)
{
    if (NULL == data) {
        return;
    }

    ne_someip_transmit_iov_buffer_t* tmp_buffer = (ne_someip_transmit_iov_buffer_t*)data;
    if (NULL != tmp_buffer->iovBuffer) {
        free(tmp_buffer->iovBuffer);
        tmp_buffer->iovBuffer = NULL;
    }
    if (NULL != tmp_buffer->user_data) {
        ne_someip_ep_tool_ep_send_cache_tp_free(tmp_buffer->user_data);
        tmp_buffer->user_data = NULL;
    }

    free(tmp_buffer);
    tmp_buffer = NULL;
}

// 发送数据时，trigger之前及trigger时 失败，清除保存的相应数据
void ne_someip_ep_tool_trans_iov_buf_trig_fail_free(void* data)
{
    if (NULL == data) {
        return;
    }

    ne_someip_transmit_iov_buffer_t* tmp_buffer = (ne_someip_transmit_iov_buffer_t*)data;
    if (NULL != tmp_buffer->iovBuffer) {
        free(tmp_buffer->iovBuffer);
        tmp_buffer->iovBuffer = NULL;
    }
    tmp_buffer->user_data = NULL;
    // if (NULL != tmp_buffer->user_data) {  // 发送数据时的缓存，如果是接收数据，此时应该为NULL（TCP、UNIX间断收数据需再考虑）
    //     free(tmp_buffer->user_data);
    //     tmp_buffer->user_data = NULL;
    // }

    free(tmp_buffer);
    tmp_buffer = NULL;
}

// 接收数据（udp）失败时，清除所有iov buffer
void ne_someip_ep_tool_trans_iov_buf_complete_free(void* data)
{
    if (NULL == data) {
        return;
    }

    ne_someip_transmit_iov_buffer_t* tmp_buffer = (ne_someip_transmit_iov_buffer_t*)data;
    if (NULL != tmp_buffer->iovBuffer) {
        for (int i = 0; i < tmp_buffer->length; ++i) {
            free(((struct iovec*)(tmp_buffer->iovBuffer))[i].iov_base);
            ((struct iovec*)(tmp_buffer->iovBuffer))[i].iov_base = NULL;
        }
        free(tmp_buffer->iovBuffer);
        tmp_buffer->iovBuffer = NULL;
    }
    tmp_buffer->user_data = NULL;

    free(tmp_buffer);
    tmp_buffer = NULL;
}

void ne_someip_ep_tool_trans_iov_buf_list_free(void* data)
{
    if (NULL != data) {
        ne_someip_list_destroy(data, ne_someip_ep_tool_trans_iov_buf_free);
    }
}

void ne_someip_ep_tool_ep_buf_list_free(void* data)
{
    if (NULL == data) {
        return;
    }

    ne_someip_endpoint_buffer_t* ep_buffer = (ne_someip_endpoint_buffer_t*)data;
    if (NULL != ep_buffer->iov_buffer) {
        free(ep_buffer->iov_buffer);
    }

    free(ep_buffer);
    data = NULL;
}

void ne_someip_ep_tool_ep_buf_list_self_free(void* data)
{
    if (NULL == data) {
        return;
    }

    ne_someip_endpoint_buffer_t* ep_buffer = (ne_someip_endpoint_buffer_t*)data;
    free(ep_buffer);
    data = NULL;
}

void ne_someip_ep_tool_ep_group_addr_free(void* data)
{
    if (NULL == data) {
        return;
    }

    ne_someip_endpoint_add_multicast_addr_info_t* group_addr = (ne_someip_endpoint_add_multicast_addr_info_t*)data;
    if (NULL != group_addr->addr) {
        free(group_addr->addr);
    }

    free(group_addr);
    data = NULL;
}

void ne_someip_ep_tool_ep_tp_cache_free(void* data)
{
    if (NULL == data) {
        return;
    }

    ne_someip_endpoint_tp_cache_t* tp_cache = (ne_someip_endpoint_tp_cache_t*)data;
    ne_someip_ep_tool_udp_iov_cache_list_free(tp_cache->tp_list);
    tp_cache->tp_list = NULL;
    free(tp_cache);
}

uint32_t ne_someip_ep_tool_get_total_len(ne_someip_transmit_iov_buffer_t* buffer)
{
    if (NULL == buffer || NULL == buffer->iovBuffer) {
        return 0;
    }

    uint32_t total_len = 0;
    for (int i = buffer->offset; i < buffer->offset + buffer->length; ++i) {
        total_len += ((struct iovec*)(buffer->iovBuffer))[i].iov_len;
    }

    return total_len;
}

uint32_t ne_someip_ep_tool_get_trans_total_len(ne_someip_trans_buffer_struct_t* ep_trans_buffer)
{
    if (NULL == ep_trans_buffer) {
        return 0;
    }

    uint32_t total_len = 0;
    if (NULL != ep_trans_buffer->ipc_data) {
        total_len += ep_trans_buffer->ipc_data->size;
    }
    if (NULL != ep_trans_buffer->someip_header) {
        total_len += ep_trans_buffer->someip_header->size;
    }
    if (NULL != ep_trans_buffer->payload) {
        for (int i = 0; i < ep_trans_buffer->payload->num; ++i) {
            if (NULL == ep_trans_buffer->payload->buffer_list) {
                break;
            }
            if (NULL != ep_trans_buffer->payload->buffer_list[i]) {
                total_len += ep_trans_buffer->payload->buffer_list[i]->length;
            }
        }
    }

    return total_len;
}
