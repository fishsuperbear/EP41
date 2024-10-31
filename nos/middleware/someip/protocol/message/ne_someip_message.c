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
#include "ne_someip_message.h"
#include "ne_someip_serializer.h"
#include "ne_someip_deserializer.h"
#include "ne_someip_endpoint_define.h"
#include "ne_someip_log.h"
#include <stddef.h>

bool ne_someip_msg_ser(ne_someip_list_t* buffer_list, const ne_someip_header_t* header,
    const ne_someip_payload_t* payload)
{
	bool ret = false;
    if (NULL == buffer_list || NULL == header) {
        ne_someip_log_error("buffer_list or header is NULL.");
        return ret;
    }

    uint32_t payload_len = 0;
    if (NULL != payload && NULL != payload->buffer_list) {
        for (int i = 0; i< payload->num; ++i) {
            if (NULL == payload->buffer_list[i]) {
                ne_someip_log_error("payload->buffer_list[%d] is NULL.", i);
                return ret;
            }
            payload_len = payload->buffer_list[i]->length;
        }
    }
    
    if (header->message_length != payload_len + NE_SOMEIP_HEADER_LEN_IN_MSG_LEN) {
    	ne_someip_log_error("header->message_length is error.");
    	return ret;
    }

    uint8_t* data = (uint8_t*)malloc(NE_SOMEIP_HEADER_LENGTH);
    if (NULL == data) {
        ne_someip_log_error("malloc data error.");
        return false;
    }
    memset(data, 0, NE_SOMEIP_HEADER_LENGTH);
    uint32_t length = NE_SOMEIP_HEADER_LENGTH;
    uint8_t* tmp_data = data;
    ret = ne_someip_msg_header_ser(&tmp_data, header);
    if (!ret) {
        ne_someip_log_error("ne_someip_msg_header_ser failed.");
        free(data);
        return ret;
    }

    // serialize header
    ne_someip_endpoint_buffer_t* header_buffer = (ne_someip_endpoint_buffer_t*)malloc(sizeof(ne_someip_endpoint_buffer_t));
    header_buffer->iov_buffer = (char*)data;
    header_buffer->size = length;
    buffer_list = ne_someip_list_append(buffer_list, (void*)header_buffer);

    // serialize payload
    if (NULL != payload && NULL != payload->buffer_list) {
        for (int i = 0; i< payload->num; ++i) {
            ne_someip_endpoint_buffer_t* payload_buffer = (ne_someip_endpoint_buffer_t*)malloc(sizeof(ne_someip_endpoint_buffer_t));
            payload_buffer->iov_buffer = (char*)(payload->buffer_list[i]->data);
            payload_buffer->size = payload->buffer_list[i]->length;
            buffer_list = ne_someip_list_append(buffer_list, (void*)payload_buffer);
        }
    }
    
    ret = true;
    return ret;
}

// bool ne_someip_msg_header_ser_new(uint8_t** data, uint32_t* length, const ne_someip_header_t* message)
// {
//     if (NULL == data || NULL == length || NULL == message) {
//     	ne_someip_log_error("data or length or message is NULL.");
//         return false;
//     }

//     *data = (uint8_t*)malloc(NE_SOMEIP_HEADER_LENGTH);
//     if (NULL == *data) {
//         ne_someip_log_error("malloc data error.");
//         return false;
//     }
//     memset(data, 0, NE_SOMEIP_HEADER_LENGTH);
//     *length = NE_SOMEIP_HEADER_LENGTH;
//     uint8_t* tmp_data = *data;

//     bool ret = ne_someip_ser_uint16(&tmp_data, message->service_id);
//     if (!ret) {
//     	ne_someip_log_error("service_id serialize error.");
//         return ret;
//     }

//     ret = ne_someip_ser_uint16(&tmp_data, message->method_id);
//     if (!ret) {
//     	ne_someip_log_error("method_id serialize error.");
//         return ret;
//     }

//     ret = ne_someip_ser_uint32(&tmp_data, message->message_length, 0);
//     ne_someip_log_info("message len = %d", message->message_length);
//     if (!ret) {
//     	ne_someip_log_error("message_length serialize error.");
//         return ret;
//     }

//     ret = ne_someip_ser_uint16(&tmp_data, message->client_id);
//     if (!ret) {
//     	ne_someip_log_error("client_id serialize error.");
//         return ret;
//     }

//     ret = ne_someip_ser_uint16(&tmp_data, message->session_id);
//     if (!ret) {
//     	ne_someip_log_error("session_id serialize error.");
//         return ret;
//     }

//     ret = ne_someip_ser_uint8(&tmp_data, message->protocol_version);
//     if (!ret) {
//     	ne_someip_log_error("protocol_version serialize error.");
//         return ret;
//     }

//     ret = ne_someip_ser_uint8(&tmp_data, message->interface_version);
//     if (!ret) {
//     	ne_someip_log_error("interface_version serialize error.");
//         return ret;
//     }

//     ret = ne_someip_ser_uint8(&tmp_data, message->message_type);
//     if (!ret) {
//     	ne_someip_log_error("message_type serialize error.");
//         return ret;
//     }

//     ret = ne_someip_ser_uint8(&tmp_data, message->return_code);
//     if (!ret) {
//     	ne_someip_log_error("return_code serialize error.");
//         return ret;
//     }

//     return ret;
// }

bool ne_someip_msg_header_ser(uint8_t** data, const ne_someip_header_t* message)
{
    if (NULL == message) {
    	ne_someip_log_error("message is NULL.");
        return false;
    }

    bool ret = ne_someip_ser_uint16(data, message->service_id);
    if (!ret) {
    	ne_someip_log_error("service_id serialize error.");
        return ret;
    }

    ret = ne_someip_ser_uint16(data, message->method_id);
    if (!ret) {
    	ne_someip_log_error("method_id serialize error.");
        return ret;
    }

    ret = ne_someip_ser_uint32(data, message->message_length, 0);
    if (!ret) {
    	ne_someip_log_error("message_length serialize error.");
        return ret;
    }

    ret = ne_someip_ser_uint16(data, message->client_id);
    if (!ret) {
    	ne_someip_log_error("client_id serialize error.");
        return ret;
    }

    ret = ne_someip_ser_uint16(data, message->session_id);
    if (!ret) {
    	ne_someip_log_error("session_id serialize error.");
        return ret;
    }

    ret = ne_someip_ser_uint8(data, message->protocol_version);
    if (!ret) {
    	ne_someip_log_error("protocol_version serialize error.");
        return ret;
    }

    ret = ne_someip_ser_uint8(data, message->interface_version);
    if (!ret) {
    	ne_someip_log_error("interface_version serialize error.");
        return ret;
    }

    ret = ne_someip_ser_uint8(data, message->message_type);
    if (!ret) {
    	ne_someip_log_error("message_type serialize error.");
        return ret;
    }

    ret = ne_someip_ser_uint8(data, message->return_code);
    if (!ret) {
    	ne_someip_log_error("return_code serialize error.");
        return ret;
    }

    return ret;
}

bool ne_someip_msg_header_deser(const uint8_t** data, ne_someip_header_t* message)
{
    bool ret = false;
    if (NULL == *data || NULL == message) {
    	ne_someip_log_error("data or message is NULL.");
        return ret;
    }

    ret = ne_someip_deser_uint16(data, &message->service_id);
	if (!ret) {
		ne_someip_log_error("service_id deserialize error.");
		return ret;
	}

	ret = ne_someip_deser_uint16(data, &message->method_id);
	if (!ret) {
		ne_someip_log_error("method_id deserialize error.");
		return ret;
	}

	ret = ne_someip_deser_uint32(data, &message->message_length, 0);
	if (!ret) {
		ne_someip_log_error("message_length deserialize error.");
		return ret;
	}

	ret = ne_someip_deser_uint16(data, &message->client_id);
	if (!ret) {
		ne_someip_log_error("client_id deserialize error.");
		return ret;
	}

	ret = ne_someip_deser_uint16(data, &message->session_id);
	if (!ret) {
		ne_someip_log_error("session_id deserialize error.");
		return ret;
	}

	ret = ne_someip_deser_uint8(data, &message->protocol_version);
	if (!ret) {
		ne_someip_log_error("protocol_version deserialize error.");
		return ret;
	}

	ret = ne_someip_deser_uint8(data, &message->interface_version);
	if (!ret) {
		ne_someip_log_error("interface_version deserialize error.");
		return ret;
	}

	ret = ne_someip_deser_uint8(data, &message->message_type);
	if (!ret) {
		ne_someip_log_error("message_type deserialize error.");
		return ret;
	}

	ret = ne_someip_deser_uint8(data, &message->return_code);
	if (!ret) {
		ne_someip_log_error("return_code deserialize error.");
		return ret;
	}

	return ret;
}

bool ne_someip_msg_deser(const ne_someip_list_t* buffer_list, ne_someip_header_t* header,
    ne_someip_payload_t* payload)
{
	bool ret = false;
    if (NULL == buffer_list || NULL == header || NULL == payload) {
    	ne_someip_log_error("buffer_list or header or payload is NULL.");
        return ret;
    }

    ne_someip_list_iterator_t* iter = ne_someip_list_iterator_create((ne_someip_list_t*)buffer_list);
    // deserialize someip header
    if (!ne_someip_list_iterator_valid(iter)) {
        ne_someip_log_error("buffer_list data is error.");
        ne_someip_list_iterator_destroy(iter);
    	return ret;
    }
    ne_someip_endpoint_buffer_t* header_buffer = (ne_someip_endpoint_buffer_t*)ne_someip_list_iterator_data(iter);
    if (NULL == header_buffer) {
        ne_someip_log_error("header_buffer is NULL.");
        ne_someip_list_iterator_destroy(iter);
    	return ret;
    }

    ret = ne_someip_msg_header_deser_new((uint8_t*)(header_buffer->iov_buffer), header_buffer->size, header);
    if (!ret) {
    	ne_someip_log_error("deserialize someip header error.");
    	ne_someip_list_iterator_destroy(iter);
        return ret;
    }

    // deserialize payload (TODO: 这里假定payload是完整的buffer，后续看是否要改成buffer list)
    int payload_buffer_num = ne_someip_list_length(buffer_list) - 1;
    payload->buffer_list = malloc(sizeof(ne_someip_payload_slice_t*)*payload_buffer_num);
    payload->num = payload_buffer_num;
    ne_someip_list_iterator_next(iter);
    int i = 0;
    while (ne_someip_list_iterator_valid(iter)) {
    	ne_someip_endpoint_buffer_t* payload_buffer = (ne_someip_endpoint_buffer_t*)ne_someip_list_iterator_data(iter);
    	if (NULL == payload_buffer) {
            ne_someip_log_error("payload_buffer in buffer_list is NULL.");
            ne_someip_payload_unref(payload);
            return ret;
    	}
    	else {
            ne_someip_payload_slice_t* payload_slice = malloc(sizeof(ne_someip_payload_slice_t));
            if (NULL == payload_slice) {
                ne_someip_log_error("malloc error.");
                ne_someip_payload_unref(payload);
		    	ne_someip_list_iterator_destroy(iter);
                return ret;
            }
            payload_slice->data = (uint8_t*)payload_buffer->iov_buffer;
            payload_slice->length = payload_buffer->size;
            (payload->buffer_list)[i++] = payload_slice;
            ret = true;
    	}
        ne_someip_list_iterator_next(iter);
    }
    ne_someip_list_iterator_destroy(iter);
    return ret;
}

bool ne_someip_msg_header_deser_new(uint8_t* data, uint32_t length, ne_someip_header_t* message)
{
	bool ret = false;
    if (NULL == data || NULL == message) {
    	ne_someip_log_error("data or message is NULL.");
        return ret;
    }

    if (length < NE_SOMEIP_HEADER_LENGTH) {
        ne_someip_log_error("someip header is smaller than NE_SOMEIP_HEADER_LENGTH.");
        return ret;
    }

    uint8_t* tmp_data = data;
    ret = ne_someip_deser_uint16(&tmp_data, &message->service_id);
	if (!ret) {
		ne_someip_log_error("service_id deserialize error.");
		return ret;
	}

	ret = ne_someip_deser_uint16(&tmp_data, &message->method_id);
	if (!ret) {
		ne_someip_log_error("method_id deserialize error.");
		return ret;
	}

	ret = ne_someip_deser_uint32(&tmp_data, &message->message_length, 0);
	if (!ret) {
		ne_someip_log_error("message_length deserialize error.");
		return ret;
	}

	ret = ne_someip_deser_uint16(&tmp_data, &message->client_id);
	if (!ret) {
		ne_someip_log_error("client_id deserialize error.");
		return ret;
	}

	ret = ne_someip_deser_uint16(&tmp_data, &message->session_id);
	if (!ret) {
		ne_someip_log_error("session_id deserialize error.");
		return ret;
	}

	ret = ne_someip_deser_uint8(&tmp_data, &message->protocol_version);
	if (!ret) {
		ne_someip_log_error("protocol_version deserialize error.");
		return ret;
	}

	ret = ne_someip_deser_uint8(&tmp_data, &message->interface_version);
	if (!ret) {
		ne_someip_log_error("interface_version deserialize error.");
		return ret;
	}

    uint8_t type = 0;
	ret = ne_someip_deser_uint8(&tmp_data, &type);
	if (!ret) {
		ne_someip_log_error("message_type deserialize error.");
		return ret;
	}
    message->message_type = (ne_someip_message_type_t)type;

	ret = ne_someip_deser_uint8(&tmp_data, &message->return_code);
	if (!ret) {
		ne_someip_log_error("return_code deserialize error.");
		return ret;
	}

	return ret;
}
