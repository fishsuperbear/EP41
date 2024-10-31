#ifndef ARA_COM_INTERNAL_PROXY_H_
#define ARA_COM_INTERNAL_PROXY_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <deque>
#include <vector>
#include <mutex>
#include "ara/core/promise.h"
#include "ara/core/instance_specifier.h"
#include "ara/com/types.h"
#include "ara/com/com_error_domain.h"
#include "ara/com/instance_identifier.h"
#include "ara/com/internal/internal_types.h"
#include "ara/com/internal/instance_base.h"
#include "ara/com/internal/proxy_instance.h"
#include "ara/com/internal/runtime.h"
#include "ara/com/internal/com_error_domains_registry.h"
#include "ara/com/serializer/error_code_transformation.h"
#include "ara/com/serializer/transformation_props.h"

#ifndef PT_TRACE_TAG_COM
#define PT_TRACE_THIS_TAG 0x01 << 8
#else
#define PT_TRACE_THIS_TAG PT_TRACE_THIS_COM
#endif
#include "osal/perftoolkit/pt_trace_api.h"

/* =============== method =================================================================== */
// Non-fire-and-forget Method
// response Output(non-void)
template<typename Output>
void setMethodResponseParameters(const std::shared_ptr<ara::com::runtime::ProxyInstance>& instance,
    uint32_t idx,
    const ara::com::SomeipTransformationProps& transformation_props)
{
    // rr
    instance->setMethodResponseCallback(idx,
        [transformation_props](int32_t return_code, const std::shared_ptr<BufferList>& payload, std::shared_ptr<TagBase>& tag) {
            if (!tag) {
                // Never happen
                printf("internal error!!!!!!\n");
                return;
            }

            // check serialization type
            int32_t serialization_type = tag->serialization_type;
            if ((serialization_type & SerializationType_SOMEIP) == 0) {
                // TODO: Set Error serialization type error
                ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
                return;
            }

            if (0 == return_code) { // normal payload
                if (payload == nullptr || payload->size() == 0) {
                    ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
                    return;
                }
                else {
                    std::vector<uint8_t> data_serialized = (*payload)[0];
                    Output output;
                    SomeIPDeSerializer someip_deserializer(&transformation_props);
                    //SomeIPDeSerializer someip_deserializer;
                    someip_deserializer.from_buffer(data_serialized);
                    InputStream<SomeIPDeSerializer> input_stream(someip_deserializer);
                    input_stream.read_struct_as_args(output);
                    ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.set_value(output);
                }
            }
            else if (1 == return_code) { // error payload
                if (payload == nullptr || payload->size() == 0) {
                    ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
                    return;
                }
                else {
                    std::vector<uint8_t> data_serialized = (*payload)[0];
                    ara::core::ErrorCode error_code = ara::com::runtime::deserializeErrorCode(data_serialized);
                    ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(error_code);
                    return;
                }
            }
            else { // no payload
                // according return_code, setError
                ara::core::ErrorCode error_code{static_cast<ara::com::ComErrc>(~return_code+1)};
                ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(error_code);
                return;
            }
            return;
    });
}

// Non-fire-and-forget Method
// return (future<void>)
static void setMethodResponseParameters(const std::shared_ptr<ara::com::runtime::ProxyInstance>& instance,
    uint32_t idx)
{
    // rr
    instance->setMethodResponseCallback(idx,
        [](int32_t return_code, const std::shared_ptr<BufferList>& payload, std::shared_ptr<TagBase>& tag){
        if (!tag) {
            // Never happen
            printf("internal error!!!!!!\n");
            return;
        }

        // check serialization type
        int32_t serialization_type = tag->serialization_type;
        if ((serialization_type & SerializationType_SOMEIP) == 0) {
            // TODO: Set Error serialization type error
            ((ara::com::runtime::PromiseTag<void> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
            return;
        }

        if (0 == return_code) { // normal payload (no payload)
            ((ara::com::runtime::PromiseTag<void> *)(tag.get()))->promise_.set_value();
        }
        else if (1 == return_code) { // error payload
            if (payload == nullptr || payload->size() == 0) {
                ((ara::com::runtime::PromiseTag<void> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
                return;
            }
            else {
                std::vector<uint8_t> data_serialized = (*payload)[0];
                ara::core::ErrorCode error_code = ara::com::runtime::deserializeErrorCode(data_serialized);
                ((ara::com::runtime::PromiseTag<void> *)(tag.get()))->promise_.SetError(error_code);
                return;
            }
        }
        else { // no payload
            // according return_code, setError
            ara::core::ErrorCode error_code{static_cast<ara::com::ComErrc>(~return_code+1)};
            ((ara::com::runtime::PromiseTag<void> *)(tag.get()))->promise_.SetError(error_code);
            return;
        }

        return;
    });
}

// Non-fire-and-forget Method
// request Input(non-void)
// response Output(void and non-void)
template<typename Output, typename ...InputArgs>
ara::core::Future<Output> sendMethodAsyncRequest(const std::shared_ptr<ara::com::runtime::ProxyInstance>& instance,
    uint32_t method_idx,
    const ara::com::SomeipTransformationProps& transformation_props,
    InputArgs& ...args)
{
    PT_TRACE_FUNCTION_AUTO("proxy_serialize::send_method_request(method,idx:%d)", method_idx);
    int32_t serialization_type = instance->methodSerializationType(method_idx);

    std::shared_ptr<TagBase> tag = std::make_shared<ara::com::runtime::PromiseTag<Output>>();
    if (!tag) {
        ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
        return ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.get_future();
    }
    tag->serialization_type = serialization_type;

    // check serialization type
    if ((serialization_type & SerializationType_SOMEIP) == 0) {
        // TODO: Set Error serialization type error
        ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
        return ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.get_future();
    }

    SomeIPSerializer someip_serializer(&transformation_props);
    OutputStream<SomeIPSerializer> output_stream(someip_serializer);
    int ret = output_stream.serialize(args...);    // TODO check ret
    if (ret) {
        ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
        return ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.get_future();
    }
    const std::vector<uint8_t>& data_serialized = someip_serializer.getData();

    std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
    if (!payload) {
        ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
        return ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.get_future();
    }
    payload->push_back(std::move(data_serialized));

    int32_t return_ret = instance->sendMethodAsyncRequest(method_idx, payload, tag);
    if (-1 == return_ret) {
        ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
        return ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.get_future();
    }

    return ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.get_future();
}

// Non-fire-and-forget Method
// request Input(void)
// response Output(void and non-void)
template<typename Output>
ara::core::Future<Output> sendMethodAsyncRequest(const std::shared_ptr<ara::com::runtime::ProxyInstance>& instance,
    uint32_t method_idx)
{
    PT_TRACE_FUNCTION_AUTO("proxy_serialize::send_method_request(method,idx:%d)", method_idx);
    int32_t serialization_type = instance->methodSerializationType(method_idx);

    std::shared_ptr<TagBase> tag = std::make_shared<ara::com::runtime::PromiseTag<Output>>();
    if (!tag) {
        ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
        return ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.get_future();
    }
    tag->serialization_type = serialization_type;

    // check serialization type
    if ((serialization_type & SerializationType_SOMEIP) == 0) {
        // TODO: Set Error serialization type error
        ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
        return ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.get_future();
    }

    std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
    if (!payload) {
        ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
        return ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.get_future();
    }
    int32_t return_ret = instance->sendMethodAsyncRequest(method_idx, payload, tag);
    if (-1 == return_ret) {
        ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
        return ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.get_future();
    }
    return ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.get_future();
}

// fire-and-forget Method
// request Input(non-void)
// response Output(void)
template<typename ...InputArgs>
void sendMethodAsyncRequest(const std::shared_ptr<ara::com::runtime::ProxyInstance>& instance,
    uint32_t method_idx,
    const ara::com::SomeipTransformationProps& transformation_props,
    InputArgs& ...args)
{
    PT_TRACE_FUNCTION_AUTO("proxy_serialize::send_method_request(method,idx:%d)", method_idx);
    int32_t serialization_type = instance->methodSerializationType(method_idx);
    // check serialization type
    if ((serialization_type & SerializationType_SOMEIP) == 0) {
        return;
    }

    SomeIPSerializer someip_serializer(&transformation_props);
    OutputStream<SomeIPSerializer> output_stream(someip_serializer);
    int ret = output_stream.serialize(args...);    // TODO check ret
    if (ret) {
        return;
    }
    std::vector<uint8_t> data_serialized = someip_serializer.getData();

    std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
    if (!payload) {
        return;
    }
    payload->push_back(std::move(data_serialized));

    std::shared_ptr<TagBase> tag = nullptr;
    instance->sendMethodAsyncRequest(method_idx, payload, tag);
}

// fire-and-forget Method
// request Input(void)
// response Output(void)
static void sendMethodAsyncRequest(const std::shared_ptr<ara::com::runtime::ProxyInstance>& instance,
    uint32_t method_idx)
{
    PT_TRACE_FUNCTION_AUTO("proxy_serialize::send_method_request(method,idx:%d)", method_idx);
    int32_t serialization_type = instance->methodSerializationType(method_idx);
    // check serialization type
    if ((serialization_type & SerializationType_SOMEIP) == 0) {
        return;
    }

    std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
    if (!payload) {
        return;
    }
    std::shared_ptr<TagBase> tag = nullptr;
    instance->sendMethodAsyncRequest(method_idx, payload, tag);
}

/* =============== method =================================================================== */

/* =============== event =================================================================== */
template<typename Output>
size_t getEventNewSamples(const std::shared_ptr<ara::com::runtime::ProxyInstance>& instance,
    uint32_t event_idx,
    const ara::com::SomeipTransformationProps& transformation_props,
    std::function<void(ara::com::SamplePtr<Output const>)>&& f,
    size_t maxNumberOfSamples)
{
    PT_TRACE_FUNCTION_AUTO("proxy_serialize::get_event_new_samples(event,idx:%d)", event_idx);
    int32_t serialization_type = instance->eventSerializationType(event_idx);
    // check serialization type
    if ((serialization_type & SerializationType_SOMEIP) == 0) {
        return 0;
    }

    std::vector<std::shared_ptr<BufferList>> payloads;
    size_t number = instance->getEventNewSamples(event_idx, maxNumberOfSamples, payloads);
    for (auto& payload : payloads) {

        if (payload != nullptr && payload->size() != 0) {
            std::vector<uint8_t> data_serialized = (*payload)[0];
            std::shared_ptr<Output> radar_objects = std::make_shared<Output>();
            if (NULL == radar_objects) {
                number--;
                continue;
            }

            SomeIPDeSerializer someip_deserializer(&transformation_props);
            int ret = someip_deserializer.from_buffer(data_serialized);
            if (ret) {
                number--;
                continue;
            }

            InputStream<SomeIPDeSerializer> input_stream(someip_deserializer);
            ret = input_stream.deserialize(*radar_objects);
            if (ret) {
                number--;
                continue;
            }

            ara::com::SamplePtr<Output const> sampleptr_radar_objects(radar_objects);
            PT_TRACE_BEGINF("proxy_serialize::get_event_new_samples user callback(event,idx:%d)", event_idx);
            f(sampleptr_radar_objects);
            PT_TRACE_END();
        }
    }

    return number;
}
/* =============== event =================================================================== */

/* =============== field =================================================================== */
template<typename Output>
void setFieldSetResponseParameters(const std::shared_ptr<ara::com::runtime::ProxyInstance>& instance,
    uint32_t idx,
    const ara::com::SomeipTransformationProps& transformation_props)
{
    instance->setFieldSetResponseCallback(idx,
        [transformation_props](int return_code, const std::shared_ptr<BufferList>& payload, std::shared_ptr<TagBase>& tag){
//            std::vector<uint8_t> data_serialized = (*payload)[0];
//            Output output;
//
//            int32_t serialization_type = tag->serialization_type;
//            if (serialization_type & SerializationType_SOMEIP) {
//                SomeIPDeSerializer someip_deserializer(&transformation_props);
//                someip_deserializer.from_buffer(data_serialized);
//                InputStream<SomeIPDeSerializer> input_stream(someip_deserializer);
//                int ret = input_stream.deserialize(output);
//            }
//            else if (0) {
//
//            }
//            else {
//                // Set Error
//            }
//
//            ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.set_value(output);


            if (!tag) {
                // Never happen
                printf("internal error!!!!!!\n");
                return;
            }

            // check serialization type
            int32_t serialization_type = tag->serialization_type;
            if ((serialization_type & SerializationType_SOMEIP) == 0) {
                // TODO: Set Error serialization type error
                ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
                return;
            }

            if (0 == return_code) { // normal payload
                if (payload == nullptr || payload->size() == 0) {
                    ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
                    return;
                }
                else {
                    std::vector<uint8_t> data_serialized = (*payload)[0];
                    Output output;
                    SomeIPDeSerializer someip_deserializer(&transformation_props);
                    //SomeIPDeSerializer someip_deserializer;
                    int ret = someip_deserializer.from_buffer(data_serialized);
                    if (ret) {
                        ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
                        return;
                    }
                    InputStream<SomeIPDeSerializer> input_stream(someip_deserializer);
                    ret = input_stream.deserialize(output);
                    if (ret) {
                        ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
                        return;
                    }
                    ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.set_value(output);
                }
            }
            else if (1 == return_code) { // error payload
                if (payload == nullptr || payload->size() == 0) {
                    ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
                    return;
                }
                else {
                    std::vector<uint8_t> data_serialized = (*payload)[0];
                    ara::core::ErrorCode error_code = ara::com::runtime::deserializeErrorCode(data_serialized);
                    ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(error_code);
                    return;
                }
            }
            else { // no payload
                // according return_code, setError
                ara::core::ErrorCode error_code{static_cast<ara::com::ComErrc>(~return_code+1)};
                ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(error_code);
                return;
            }
            return;
    });
}

template<typename Output>
void setFieldGetResponseParameters(const std::shared_ptr<ara::com::runtime::ProxyInstance>& instance,
    uint32_t idx,
    const ara::com::SomeipTransformationProps& transformation_props)
{
    instance->setFieldGetResponseCallback(idx,
        [transformation_props](int return_code, const std::shared_ptr<BufferList>& payload, std::shared_ptr<TagBase>& tag){

//            std::vector<uint8_t> data_serialized = (*payload)[0];
//            Output output;
//            int32_t serialization_type = tag->serialization_type;
//            if (serialization_type & SerializationType_SOMEIP) {
//                SomeIPDeSerializer someip_deserializer(&transformation_props);
//                someip_deserializer.from_buffer(data_serialized);
//                InputStream<SomeIPDeSerializer> input_stream(someip_deserializer);
//                int ret = input_stream.deserialize(output);
//            }
//            else if (0) {
//
//            }
//            else {
//                // Set Error
//            }
//
//            ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.set_value(output);
        if (!tag) {
            // Never happen
            printf("internal error!!!!!!\n");
            return;
        }

        // check serialization type
        int32_t serialization_type = tag->serialization_type;
        if ((serialization_type & SerializationType_SOMEIP) == 0) {
            // TODO: Set Error serialization type error
            ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
            return;
        }

        if (0 == return_code) { // normal payload
            if (payload == nullptr || payload->size() == 0) {
                ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
                return;
            }
            else {
                std::vector<uint8_t> data_serialized = (*payload)[0];
                Output output;
                SomeIPDeSerializer someip_deserializer(&transformation_props);
                //SomeIPDeSerializer someip_deserializer;
                int ret = someip_deserializer.from_buffer(data_serialized);
                if (ret) {
                    ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
                    return;
                }
                InputStream<SomeIPDeSerializer> input_stream(someip_deserializer);
                ret = input_stream.deserialize(output);
                if (ret) {
                    ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
                    return;
                }
                ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.set_value(output);
            }
        }
        else if (1 == return_code) { // error payload
            if (payload == nullptr || payload->size() == 0) {
                ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
                return;
            }
            else {
                std::vector<uint8_t> data_serialized = (*payload)[0];
                ara::core::ErrorCode error_code = ara::com::runtime::deserializeErrorCode(data_serialized);
                ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(error_code);
                return;
            }
        }
        else { // no payload
            // according return_code, setError
            ara::core::ErrorCode error_code{static_cast<ara::com::ComErrc>(~return_code+1)};
            ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(error_code);
            return;
        }
        return;
    });
}

template<typename Output, typename Input>
ara::core::Future<Output> sendFieldSetAsyncRequest(const std::shared_ptr<ara::com::runtime::ProxyInstance>& instance,
    uint32_t field_idx,
    const ara::com::SomeipTransformationProps& transformation_props,
    const Input& value)
{
    PT_TRACE_FUNCTION_AUTO("proxy_serialize::send_field_set_request(field,idx:%d)", field_idx);
    int32_t serialization_type = instance->fieldSerializationType(field_idx);

    std::shared_ptr<TagBase> tag = std::make_shared<ara::com::runtime::PromiseTag<Output>>();
    if (!tag) {
        ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
        return ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.get_future();
    }
    tag->serialization_type = serialization_type;

    // check serialization type
    if ((serialization_type & SerializationType_SOMEIP) == 0) {
        // TODO: Set Error serialization type error
        ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
        return ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.get_future();
    }

    SomeIPSerializer someip_serializer(&transformation_props);
    OutputStream<SomeIPSerializer> output_stream(someip_serializer);
    int ret = output_stream.serialize(value);    // TODO check ret
    if (ret) {
        ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
        return ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.get_future();
    }
    std::vector<uint8_t> data_serialized = someip_serializer.getData();

    std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
    if (!payload) {
        ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
        return ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.get_future();
    }
    payload->push_back(std::move(data_serialized));

    int32_t return_ret = instance->sendFieldSetAsyncRequest(field_idx, payload, tag);
    if (return_ret) {
        ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
        return ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.get_future();
    }
    return ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.get_future();
}

template<typename Output>
ara::core::Future<Output> sendFieldGetAsyncRequest(const std::shared_ptr<ara::com::runtime::ProxyInstance>& instance,
    uint32_t field_idx)
{
    PT_TRACE_FUNCTION_AUTO("proxy_serialize::send_field_get_request(field,idx:%d)", field_idx);
    int32_t serialization_type = instance->fieldSerializationType(field_idx);

    std::shared_ptr<TagBase> tag = std::make_shared<ara::com::runtime::PromiseTag<Output>>();
    if (!tag) {
        ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
        return ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.get_future();
    }
    tag->serialization_type = serialization_type;

    // check serialization type
     if ((serialization_type & SerializationType_SOMEIP) == 0) {
         // TODO: Set Error serialization type error
         ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
         return ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.get_future();
     }

    std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
    if (!payload) {
        ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.SetError(ara::core::ErrorCode{ara::com::ComErrc::kCommunicationStackError});
        return ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.get_future();
    }
    instance->sendFieldGetAsyncRequest(field_idx, payload, tag);
    return ((ara::com::runtime::PromiseTag<Output> *)(tag.get()))->promise_.get_future();
}

template<typename Output>
size_t getFieldNewSamples(const std::shared_ptr<ara::com::runtime::ProxyInstance>& instance,
    uint32_t field_idx,
    const ara::com::SomeipTransformationProps& transformation_props,
    std::function<void(ara::com::SamplePtr<Output const>)>&& f,
    size_t maxNumberOfSamples)
{
    PT_TRACE_FUNCTION_AUTO("proxy_serialize::get_field_new_samples(field,idx:%d)", field_idx);
    int32_t serialization_type = instance->fieldSerializationType(field_idx);
    // check serialization type
    if ((serialization_type & SerializationType_SOMEIP) == 0) {
        return 0;
    }

    std::vector<std::shared_ptr<BufferList>> payloads;
    size_t number = instance->getFieldNewSamples(field_idx, maxNumberOfSamples, payloads);
    for (auto& payload : payloads) {
        if (payload != nullptr && payload->size() != 0) {
            std::vector<uint8_t> data_serialized = (*payload)[0];
            std::shared_ptr<Output> rate = std::make_shared<Output>();
            if (!rate) {
                number--;
                continue;
            }

            SomeIPDeSerializer someip_deserializer(&transformation_props);
            someip_deserializer.from_buffer(data_serialized);
            InputStream<SomeIPDeSerializer> input_stream(someip_deserializer);
            int ret = input_stream.deserialize(*rate);
            if (ret) {
                number--;
                continue;
            }

            ara::com::SamplePtr<Output const> sampleptr_rate(rate);
            PT_TRACE_BEGINF("proxy_serialize::get_field_new_samples user callback(field,idx:%d)", field_idx);
            f(sampleptr_rate);
            PT_TRACE_END();
        }
    }

    return number;
}
/* =============== field =================================================================== */

#endif // ARA_COM_INTERNAL_PROXY_H_
/* EOF */
