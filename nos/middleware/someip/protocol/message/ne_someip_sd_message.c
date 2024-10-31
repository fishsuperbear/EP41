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
#include <stddef.h>
#include "ne_someip_sd_message.h"
#include "ne_someip_serializer.h"
#include "ne_someip_deserializer.h"
#include "ne_someip_message.h"
#include "ne_someip_list.h"
#include "ne_someip_sd_define.h"
#include "ne_someip_log.h"
#include "ne_someip_sd_msg.h"

bool ne_someip_sd_find_offer_entry_ser(const ne_someip_sd_offer_find_entry_t* entry, uint8_t** out_data);
bool ne_someip_sd_subscribe_entry_ser(const ne_someip_sd_subscribe_entry_t* entry, uint8_t** out_data);
bool ne_someip_sd_ip_option_ser(const ne_someip_sd_ip_option_t* option, uint8_t** out_data);
bool ne_someip_sd_find_offer_entry_deser(ne_someip_sd_offer_find_entry_t* entry, const uint8_t** in_data);
bool ne_someip_sd_subscribe_entry_deser(ne_someip_sd_subscribe_entry_t* entry, const uint8_t** in_data);
bool ne_someip_sd_ip_option_deser(ne_someip_sd_ip_option_t* option, const uint8_t** in_data);

bool ne_someip_sd_msg_ser(const ne_someip_sd_msg_t* in_data, uint8_t* out_data)
{
	if (NULL == in_data) {
		ne_someip_log_error("msg is null");
		return false;
	}

	uint8_t** tmp_data = &out_data;

	// bool ret = ne_someip_msg_header_ser(tmp_data, &in_data->header);
	// if (!ret) {
	// 	ne_someip_log_error("header serialize error");
	// 	return false;
	// }

	uint32_t flag_reser
		= ((in_data->reboot_flag) << 31 | (in_data->unicast_flag) << 30 | in_data->reserved);

	bool ret = ne_someip_ser_uint32(tmp_data, flag_reser, 0);
	if (!ret) {
		ne_someip_log_error("flag and reserved serialize error");
		return false;
	}

	ret = ne_someip_ser_uint32(tmp_data, in_data->entry_length, 0);
	if (!ret) {
		ne_someip_log_error("entry_length serialize error");
		return false;
	}

	ne_someip_list_element_t* entry_element = ne_someip_list_first(in_data->entry_list);
	while (entry_element) {
		ne_someip_sd_base_entry_t* base_entry = (ne_someip_sd_base_entry_t*)(entry_element->data);
		if (NULL == base_entry) {
			ne_someip_log_error("type conversion error");
			return false;
		}

		if (ne_someip_sd_entry_type_find == base_entry->type
			|| ne_someip_sd_entry_type_offer == base_entry->type) {
			ne_someip_sd_offer_find_entry_t* entry = (ne_someip_sd_offer_find_entry_t*)(entry_element->data);
			if (NULL == entry) {
				ne_someip_log_error("type conversion error");
				return false;
			}
			ret = ne_someip_sd_find_offer_entry_ser(entry, tmp_data);
			if (!ret) {
				ne_someip_log_error("entry serialize error");
				return false;
			}
		}
		else if (ne_someip_sd_entry_type_subscribe == base_entry->type
			|| ne_someip_sd_entry_type_subscribe_ack == base_entry->type) {
			ne_someip_sd_subscribe_entry_t* entry = (ne_someip_sd_subscribe_entry_t*)(entry_element->data);
			if (NULL == entry) {
				ne_someip_log_error("type conversion error");
				return false;
			}
			ret = ne_someip_sd_subscribe_entry_ser(entry, tmp_data);
			if (!ret) {
				ne_someip_log_error("entry serialize error");
				return false;
			}
		}

		entry_element = entry_element->next;
	}

	ret = ne_someip_ser_uint32(tmp_data, in_data->option_length, 0);
	if (!ret) {
		ne_someip_log_error("option_length serialize error");
		return false;
	}

	if (0 < in_data->option_length) {
		ne_someip_list_element_t* option_element = ne_someip_list_first(in_data->option_list);
		while (option_element) {
			ne_someip_sd_base_option_t* option_base = (ne_someip_sd_base_option_t*)(option_element->data);
			if (NULL == option_base) {
				ne_someip_log_error("type conversion error");
				return false;
			}

			if (ne_someip_sd_option_type_ipv4_endpoint == option_base->type
				|| ne_someip_sd_option_type_ipv4_multicast == option_base->type) {
				ne_someip_sd_ip_option_t* option = (ne_someip_sd_ip_option_t*)(option_element->data);
				if (NULL == option) {
					ne_someip_log_error("type conversion error");
					return false;
				}
				ret = ne_someip_sd_ip_option_ser(option, tmp_data);
				if (!ret) {
					ne_someip_log_error("option serialize error");
					return false;
				}
			}

			option_element = option_element->next;
		}
	}

	return true;
}

bool ne_someip_sd_msg_deser(ne_someip_sd_msg_t* out_data, const uint8_t* in_header_data, const uint8_t* in_sd_data,
	uint32_t header_length, uint32_t payload_length)
{
	if (NULL == in_header_data || NULL == in_sd_data) {
		ne_someip_log_error("header or sd data is null");
		return false;
	}

	if (NULL == out_data) {
		ne_someip_log_error("out_data is null");
		return false;
	}

	if (NE_SOMEIP_SD_LEAST_LENGTH > header_length + payload_length
		|| payload_length < sizeof(uint32_t) + sizeof(uint32_t)) {
		ne_someip_log_error("sd message length is error");
		return false;
	}

	uint8_t** tmp_data = &in_header_data;
	bool ret = ne_someip_msg_header_deser(tmp_data, &out_data->header);
	if (!ret) {
		ne_someip_log_error("header deserilize error");
		return false;
	}

	uint8_t** temp_data = &in_sd_data;
	uint32_t flag_reser;
	ret = ne_someip_deser_uint32(temp_data, &flag_reser, 0);
	if (!ret) {
		ne_someip_log_error("flag_reser error");
		return false;
	}
	uint32_t remainder_len = payload_length - sizeof(uint32_t);

	out_data->reboot_flag = (flag_reser & 0x80000000) >> 31;
	out_data->unicast_flag = (flag_reser & 0x40000000) >> 30;
	out_data->reserved = flag_reser & 0x00FFFFFF;

	ret = ne_someip_deser_uint32(temp_data, &out_data->entry_length, 0);
	if (!ret) {
		ne_someip_log_error("entry_length error");
		return false;
	}
	remainder_len = remainder_len - sizeof(uint32_t);
	ne_someip_log_debug("entry length [%d],  remainder len [%d]", out_data->entry_length, remainder_len);

	if (out_data->entry_length < NE_SOMEIP_SD_ENTRY_LEAST_LENGTH
		|| remainder_len < out_data->entry_length
		|| out_data->entry_length % NE_SOMEIP_SD_ENTRY_LEAST_LENGTH != 0) {
		ne_someip_log_error("entry length [%d] error, remainder length [%d]", out_data->entry_length, remainder_len);
		uint8_t i = 0;
		while (i < ((out_data->entry_length % NE_SOMEIP_SD_ENTRY_LEAST_LENGTH) + 1)
			&& (remainder_len >= NE_SOMEIP_SD_ENTRY_LEAST_LENGTH)) {
			ne_someip_log_error("error type [%d]", **temp_data);
			if (ne_someip_sd_entry_type_subscribe == **temp_data) {
				ne_someip_sd_subscribe_entry_t* entry = (ne_someip_sd_subscribe_entry_t*)malloc(sizeof(ne_someip_sd_subscribe_entry_t));
				if (NULL == entry) {
					ne_someip_log_error("malloc error");
					return false;
				}
				memset(entry, 0, sizeof(ne_someip_sd_subscribe_entry_t));
				ret = ne_someip_sd_subscribe_entry_deser(entry, temp_data);
				ne_someip_log_debug("subscribe ttl [%d]", entry->ttl);
				if (ret) {
					ne_someip_list_append(out_data->entry_list, entry);
				}
				remainder_len = remainder_len - NE_SOMEIP_SD_ENTRY_LEAST_LENGTH;
			} else {
				*temp_data = *temp_data + NE_SOMEIP_SD_ENTRY_LEAST_LENGTH;
				remainder_len = remainder_len - NE_SOMEIP_SD_ENTRY_LEAST_LENGTH;
			}

			++i;
		}

		return false;
	}

	uint32_t entry_num = (out_data->entry_length) / NE_SOMEIP_SD_ENTRY_LEAST_LENGTH;
	while (entry_num) {
		if (ne_someip_sd_entry_type_find == **temp_data
			|| ne_someip_sd_entry_type_offer == **temp_data) {
			//free when msg is not be used
			ne_someip_sd_offer_find_entry_t* entry = (ne_someip_sd_offer_find_entry_t*)malloc(sizeof(ne_someip_sd_offer_find_entry_t));
			if (NULL == entry) {
				ne_someip_log_error("malloc error");
				return false;
			}
			memset(entry, 0, sizeof(ne_someip_sd_offer_find_entry_t));
			ret = ne_someip_sd_find_offer_entry_deser(entry, temp_data);
			if (!ret) {
				ne_someip_log_error("entry deserialize error");
				if (NULL != entry) {
					free(entry);
				}
				return false;
			} else {
				if (NULL != ne_someip_sd_find_entry(out_data, entry->type, entry->service_id, entry->instance_id,
					entry->major_version, entry->minor_version, entry->ttl, 0, 0)) {
					ne_someip_log_error("the entry is already exist");
					if (NULL != entry) {
						free(entry);
					}
                                	return false;
                                }
				ne_someip_list_append(out_data->entry_list, entry);
			}
			remainder_len = remainder_len - NE_SOMEIP_SD_ENTRY_LEAST_LENGTH;
		}
		else if (ne_someip_sd_entry_type_subscribe == **temp_data
				|| ne_someip_sd_entry_type_subscribe_ack == **temp_data){
			//free when msg is not be used
			ne_someip_sd_subscribe_entry_t* entry = (ne_someip_sd_subscribe_entry_t*)malloc(sizeof(ne_someip_sd_subscribe_entry_t));
			if (NULL == entry) {
				ne_someip_log_error("malloc error");
				return false;
			}
			memset(entry, 0, sizeof(ne_someip_sd_subscribe_entry_t));
			ret = ne_someip_sd_subscribe_entry_deser(entry, temp_data);
			if (!ret) {
				ne_someip_log_error("entry deserialize error");
				if (NULL != entry) {
					free(entry);
				}
				return false;
			} else {
				if (NULL != ne_someip_sd_find_entry(out_data, entry->type, entry->service_id, entry->instance_id,
					entry->major_version, 0, entry->ttl, entry->counter, entry->eventgroup_id)) {
					ne_someip_log_error("the entry is already exist");
					if (NULL != entry) {
						free(entry);
					}
					return false;
				}
				ne_someip_list_append(out_data->entry_list, entry);
			}
			remainder_len = remainder_len - NE_SOMEIP_SD_ENTRY_LEAST_LENGTH;
		}
		if (!ret) {
			ne_someip_log_error("entry deserialize error");
			return false;
		}

		entry_num--;
	}

	if (0 == ne_someip_list_length(out_data->entry_list)) {
		ne_someip_log_error("entry list is null");
		return false;
	}

    if(remainder_len < sizeof(uint32_t)) {
        ne_someip_log_error("option length less");
        return false;
    }

	ret = ne_someip_deser_uint32(temp_data, &out_data->option_length, 0);
	if (!ret) {
		ne_someip_log_error("option_length error");
		return false;
	}
	remainder_len = remainder_len - sizeof(uint32_t);

	if (0 == out_data->option_length) {
		if (0 == remainder_len) {
			ne_someip_log_debug("remainder length is 0");
			return true;
		} else {
			ne_someip_log_error("remainder length error");
			return false;
		}
	}
	ne_someip_log_debug("option length [%d], remainder len [%d]", out_data->option_length, remainder_len);

	uint32_t option_total_len = out_data->option_length;
	while ((option_total_len > NESOMEIP_SD_BASE_OPTION_LENGTH)
		&& (remainder_len > NESOMEIP_SD_BASE_OPTION_LENGTH)) {
		uint8_t option_type = *(*temp_data + 2);
		uint16_t tmp_option_length = *(*temp_data) << 8 | *(*temp_data + 1);
		if (ne_someip_sd_option_type_ipv4_endpoint == option_type
			|| ne_someip_sd_option_type_ipv4_multicast == option_type
			|| ne_someip_sd_option_type_ipv4_sd_endpoint == option_type) {
			if (tmp_option_length != NE_SOMEIP_SD_IP_OPTION_DEFAULT_LENGTH
				|| option_total_len < NE_SOMEIP_SD_IP_OPTION_LENGTH
				|| remainder_len < NE_SOMEIP_SD_IP_OPTION_LENGTH) {
				ne_someip_log_error("remainder length [%d] is error", remainder_len);
				return false;
			}

			ne_someip_sd_ip_option_t* option = (ne_someip_sd_ip_option_t*)malloc(sizeof(ne_someip_sd_ip_option_t));
			if (NULL == option) {
				ne_someip_log_error("malloc error");
				return false;
			}
			memset(option, 0, sizeof(ne_someip_sd_ip_option_t));
			ret = ne_someip_sd_ip_option_deser(option, temp_data);
			if (!ret) {
				ne_someip_log_error("option deserialize error");
				if (NULL != option) {
					free(option);
				}
				return false;
			} else {
				if (ne_someip_sd_find_option(out_data, option->type, option->protocol, option->ip_addr, option->port)) {
					ne_someip_log_warn("the option is already exist error");
					// if (NULL != option) {
					// 	free(option);
					// }
					// return false;
				}
				ne_someip_list_append(out_data->option_list, option);
			}
			option_total_len = option_total_len - NE_SOMEIP_SD_IP_OPTION_LENGTH;
			remainder_len = remainder_len - NE_SOMEIP_SD_IP_OPTION_LENGTH;
		} else if (ne_someip_sd_option_type_configuration == option_type) {
			ne_someip_log_warn("this option type is not support now");
			ne_someip_sd_configuration_option_t* option = (ne_someip_sd_configuration_option_t*)malloc(sizeof(ne_someip_sd_configuration_option_t));
			if (NULL == option) {
				ne_someip_log_error("malloc error");
				return false;
			}
			memset(option, 0, sizeof(ne_someip_sd_configuration_option_t));
			option->type = ne_someip_sd_option_type_configuration;
			ne_someip_list_append(out_data->option_list, option);
			option_total_len = option_total_len - tmp_option_length;
			remainder_len = remainder_len - tmp_option_length;
			//TODO
		} else if (ne_someip_sd_option_type_load_balancing == option_type) {
			ne_someip_log_warn("this option type is not support now");
			option_total_len = option_total_len - NE_SOMEIP_SD_LOAD_BALANCING_LEN;
			remainder_len = remainder_len - NE_SOMEIP_SD_LOAD_BALANCING_LEN;
			//TODO
		} else if (ne_someip_sd_option_type_ipv6_endpoint == option_type
			|| ne_someip_sd_option_type_ipv6_multicast == option_type
			|| ne_someip_sd_option_type_ipv6_sd_endpoint == option_type) {
			ne_someip_log_warn("this option type is not support now");
			option_total_len = option_total_len - NE_SOMEIP_SD_IPV6_LEN;
			remainder_len = remainder_len - NE_SOMEIP_SD_IPV6_LEN;
			//TODO
		} else {
			ne_someip_log_error("option type error");
			return false;
		}
	}

	return true;
}

bool ne_someip_sd_find_offer_entry_ser(const ne_someip_sd_offer_find_entry_t* entry, uint8_t** out_data)
{
	if (NULL == entry) {
		ne_someip_log_error("entry is null");
		return false;
	}

	// int8_t* tmp_data = out_data;

	bool ret = ne_someip_ser_uint8(out_data, entry->type);
	if (!ret) {
		ne_someip_log_error("type serialize error");
		return false;
	}

	ret = ne_someip_ser_uint8(out_data, entry->index1);
	if (!ret) {
		ne_someip_log_error("index1 serialize error");
		return false;
	}

	ret = ne_someip_ser_uint8(out_data, entry->index2);
	if (!ret) {
		ne_someip_log_error("index2 serialize error");
		return false;
	}

	uint8_t option_num = (entry->option_number1) << 4 | entry->option_number2;
	ret = ne_someip_ser_uint8(out_data, option_num);
	if (!ret) {
		ne_someip_log_error("option num serialize error");
		return false;
	}

	ret = ne_someip_ser_uint16(out_data, entry->service_id);
	if (!ret) {
		ne_someip_log_error("service id num serialize error");
		return false;
	}

	ret = ne_someip_ser_uint16(out_data, entry->instance_id);
	if (!ret) {
		ne_someip_log_error("instance id num serialize error");
		return false;
	}

	uint32_t major_ttl = (entry->major_version) << 24 | entry->ttl;
	ret = ne_someip_ser_uint32(out_data, major_ttl, 0);
	if (!ret) {
		ne_someip_log_error("major version and ttl serialize error");
		return false;
	}

	ret = ne_someip_ser_uint32(out_data, entry->minor_version, 0);

	return ret;
}

bool ne_someip_sd_subscribe_entry_ser(const ne_someip_sd_subscribe_entry_t* entry, uint8_t** out_data)
{
	if (NULL == entry) {
		ne_someip_log_error("entry is null");
		return false;
	}

	bool ret = ne_someip_ser_uint8(out_data, entry->type);
	if (!ret) {
		ne_someip_log_error("type serialize error");
		return false;
	}

	ret = ne_someip_ser_uint8(out_data, entry->index1);
	if (!ret) {
		ne_someip_log_error("index1 serialize error");
		return false;
	}

	ret = ne_someip_ser_uint8(out_data, entry->index2);
	if (!ret) {
		ne_someip_log_error("index2 serialize error");
		return false;
	}

	uint8_t option_num = (entry->option_number1) << 4 | entry->option_number2;
	ret = ne_someip_ser_uint8(out_data, option_num);
	if (!ret) {
		ne_someip_log_error("option num serialize error");
		return false;
	}

	ret = ne_someip_ser_uint16(out_data, entry->service_id);
	if (!ret) {
		ne_someip_log_error("service id num serialize error");
		return false;
	}

	ret = ne_someip_ser_uint16(out_data, entry->instance_id);
	if (!ret) {
		ne_someip_log_error("instance id num serialize error");
		return false;
	}

	uint32_t major_ttl = (entry->major_version) << 24 | entry->ttl;
	ret = ne_someip_ser_uint32(out_data, major_ttl, 0);
	if (!ret) {
		ne_someip_log_error("major version and ttl serialize error");
		return false;
	}

	ret = ne_someip_ser_uint8(out_data, 0);
	if (!ret) {
		ne_someip_log_error("reserved serialize error");
		return false;
	}

	uint8_t intial_counter = (entry->counter) & 0x0F;
	ret = ne_someip_ser_uint8(out_data, intial_counter);
	if (!ret) {
		ne_someip_log_error("intial_counter serialize error");
		return false;
	}

	ret = ne_someip_ser_uint16(out_data, entry->eventgroup_id);

	return ret;
}

bool ne_someip_sd_ip_option_ser(const ne_someip_sd_ip_option_t* option, uint8_t** out_data)
{
	if (NULL == option) {
		ne_someip_log_error("option is null");
		return false;
	}

	bool ret = ne_someip_ser_uint16(out_data, option->length);
	if (!ret) {
		ne_someip_log_error("length serialize error");
		return false;
	}

	ret = ne_someip_ser_uint8(out_data, option->type);
	if (!ret) {
		ne_someip_log_error("type serialize error");
		return false;
	}

	ret = ne_someip_ser_uint8(out_data, option->reserved);
	if (!ret) {
		ne_someip_log_error("reserved2 serialize error");
		return false;
	}

	uint32_t tmp_ip_addr = (option->ip_addr & 0xFF) << 24 | (option->ip_addr & 0xFF00) << 8
		| (option->ip_addr & 0xFF0000) >> 8 | (option->ip_addr & 0xFF000000) >> 24;
	ret = ne_someip_ser_uint32(out_data, tmp_ip_addr, 0);
	if (!ret) {
		ne_someip_log_error("ip_addr serialize error");
		return false;
	}

	ret = ne_someip_ser_uint8(out_data, option->reserved);
	if (!ret) {
		ne_someip_log_error("reserved2 serialize error");
		return false;
	}

	ret = ne_someip_ser_uint8(out_data, option->protocol);
	if (!ret) {
		ne_someip_log_error("protocol serialize error");
		return false;
	}

	ret = ne_someip_ser_uint16(out_data, option->port);

	return ret;
}

bool ne_someip_sd_find_offer_entry_deser(ne_someip_sd_offer_find_entry_t* entry, const uint8_t** in_data)
{
	if (NULL == *in_data || NULL == entry) {
		ne_someip_log_error("in data or entry is null");
		return false;
	}

	bool ret = ne_someip_deser_uint8(in_data, &entry->type);
	if (!ret) {
		ne_someip_log_error("type deserailize error");
		return false;
	}

	ret = ne_someip_deser_uint8(in_data, &entry->index1);
	if (!ret) {
		ne_someip_log_error("index1 deserailize error");
		return false;
	}

	ret = ne_someip_deser_uint8(in_data, &entry->index2);
	if (!ret) {
		ne_someip_log_error("index2 deserailize error");
		return false;
	}

	uint8_t option_num;
	ret = ne_someip_deser_uint8(in_data, &option_num);
	if (!ret) {
		ne_someip_log_error("option_num deserailize error");
		return false;
	}

	entry->option_number1 = (option_num & 0xF0) >> 4;
	entry->option_number2 = option_num & 0x0F;

	ret = ne_someip_deser_uint16(in_data, &entry->service_id);
	if (!ret) {
		ne_someip_log_error("service_id deserailize error");
		return false;
	}

	ret = ne_someip_deser_uint16(in_data, &entry->instance_id);
	if (!ret) {
		ne_someip_log_error("instance_id deserailize error");
		return false;
	}

	uint32_t major_ttl;
	ret = ne_someip_deser_uint32(in_data, &major_ttl, 0);
	if (!ret) {
		ne_someip_log_error("major_ttl deserailize error");
		return false;
	}

	entry->major_version = (major_ttl & 0xFF000000) >> 24;
	entry->ttl = major_ttl & 0x00FFFFFF;

	ret = ne_someip_deser_uint32(in_data, &entry->minor_version, 0);

	return ret;
}

bool ne_someip_sd_subscribe_entry_deser(ne_someip_sd_subscribe_entry_t* entry, const uint8_t** in_data)
{
	if (NULL == *in_data || NULL == entry) {
		ne_someip_log_error("in data or entry is null");
		return false;
	}

	bool ret = ne_someip_deser_uint8(in_data, &entry->type);
	if (!ret) {
		ne_someip_log_error("type deserailize error");
		return false;
	}

	ret = ne_someip_deser_uint8(in_data, &entry->index1);
	if (!ret) {
		ne_someip_log_error("index1 deserailize error");
		return false;
	}

	ret = ne_someip_deser_uint8(in_data, &entry->index2);
	if (!ret) {
		ne_someip_log_error("index2 deserailize error");
		return false;
	}

	uint8_t option_num;
	ret = ne_someip_deser_uint8(in_data, &option_num);
	if (!ret) {
		ne_someip_log_error("option_num deserailize error");
		return false;
	}

	entry->option_number1 = (option_num & 0xF0) >> 4;
	entry->option_number2 = option_num & 0x0F;

	ret = ne_someip_deser_uint16(in_data, &entry->service_id);
	if (!ret) {
		ne_someip_log_error("service_id deserailize error");
		return false;
	}

	ret = ne_someip_deser_uint16(in_data, &entry->instance_id);
	if (!ret) {
		ne_someip_log_error("instance_id deserailize error");
		return false;
	}

	uint32_t major_ttl;
	ret = ne_someip_deser_uint32(in_data, &major_ttl, 0);
	if (!ret) {
		ne_someip_log_error("major_ttl deserailize error");
		return false;
	}

	entry->major_version = (major_ttl & 0xFF000000) >> 24;
	entry->ttl = major_ttl & 0x00FFFFFF;

	uint8_t reserved;
	ret = ne_someip_deser_uint8(in_data, &reserved);
	if (!ret) {
		ne_someip_log_error("reserved deserailize error");
		return false;
	}

	uint8_t initial_counter;
	ret = ne_someip_deser_uint8(in_data, &initial_counter);
	if (!ret) {
		ne_someip_log_error("intial_counter deserailize error");
		return false;
	}
	entry->counter = initial_counter & 0x0F;

	ret = ne_someip_deser_uint16(in_data, &entry->eventgroup_id);

	return ret;
}

bool ne_someip_sd_ip_option_deser(ne_someip_sd_ip_option_t* option, const uint8_t** in_data)
{
	if (NULL == *in_data || NULL == option) {
		ne_someip_log_error("in data or option null");
		return false;
	}

	bool ret = ne_someip_deser_uint16(in_data, &option->length);
	if (!ret) {
		ne_someip_log_error("length deserailize error");
		return false;
	}

	ret = ne_someip_deser_uint8(in_data, &option->type);
	if (!ret) {
		ne_someip_log_error("type deserailize error");
		return false;
	}

	ret = ne_someip_deser_uint8(in_data, &option->reserved);
	if (!ret) {
		ne_someip_log_error("reserved deserailize error");
		return false;
	}

	ret = ne_someip_deser_uint32(in_data, &option->ip_addr, 0);
	if (!ret) {
		ne_someip_log_error("ip_addr deserailize error");
		return false;
	}
	option->ip_addr = (option->ip_addr & 0xFF) << 24 | (option->ip_addr & 0xFF00) << 8
		| (option->ip_addr & 0xFF0000) >> 8 | (option->ip_addr & 0xFF000000) >> 24;

	ret = ne_someip_deser_uint8(in_data, &option->reserved);
	if (!ret) {
		ne_someip_log_error("reserved deserailize error");
		return false;
	}

	ret = ne_someip_deser_uint8(in_data, &option->protocol);
	if (!ret) {
		ne_someip_log_error("protocol deserailize error");
		return false;
	}

	ret = ne_someip_deser_uint16(in_data, &option->port);

	return ret;
}
/* EOF */
