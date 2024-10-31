#ifndef ARA_COM_INTERNAL_SKELETON_H_
#define ARA_COM_INTERNAL_SKELETON_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <vector>
#include <map>
#include <memory>
#include "ara/core/promise.h"
#include "ara/core/instance_specifier.h"
#include "ara/com/types.h"
#include "ara/com/internal/internal_types.h"
#include "ara/com/internal/instance_base.h"
#include "ara/com/internal/skeleton_instance.h"
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

// Non-fire-and-forget Method
// Input(non-void)
// Output(non-void)
template<typename Output, typename ...InputArgs>
std::enable_if_t<!std::is_void<Output>::value, void>  executeMethodAsyncRequest(const std::shared_ptr<ara::com::runtime::SkeletonInstance>& instance,
    uint32_t idx,
    const std::shared_ptr<BufferList>& payload,
    std::shared_ptr<TagBase>& tag,
    int e2e_result,
    std::function<ara::core::Future<Output>(const InputArgs&...)> callback,
    const ara::com::SomeipTransformationProps& transformation_props,
    InputArgs ...args)
{
    if (!tag) {
        // Never happen
        printf("internal error!!!!!!\n");
        return;
    }

    // e2e check failed
    if (e2e_result != 0) {
        // TODO e2e Error
        ara::core::ErrorCode error_code{ara::com::ComErrc::kCommunicationStackError};
        std::vector<uint8_t> data;
        ara::com::runtime::serializeErrorCode(error_code, data);

        // send errro response
        int code = 1; // 1:error
        std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
        if (!payload) {
            instance->sendMethodResponse(idx, code, nullptr, tag);
            return;
        }
        payload->push_back(std::move(data));

        instance->sendMethodResponse(idx, code, payload, tag);
        return;
    }

    int32_t serialization_type = tag->serialization_type;
    if ((serialization_type & SerializationType_SOMEIP) == 0) {
        // TODO Error
        ara::core::ErrorCode error_code{ara::com::ComErrc::kCommunicationStackError};
        std::vector<uint8_t> data;
        ara::com::runtime::serializeErrorCode(error_code, data);

        // send errro response
        int code = 1; // 1:error
        std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
        if (!payload) {
            instance->sendMethodResponse(idx, code, nullptr, tag);
            return;
        }

        payload->push_back(std::move(data));

        instance->sendMethodResponse(idx, code, payload, tag);
        return;
    }

    if (payload == nullptr || payload->size() == 0) {
        // TODO Error
        ara::core::ErrorCode error_code{ara::com::ComErrc::kCommunicationStackError};
        std::vector<uint8_t> data;
        ara::com::runtime::serializeErrorCode(error_code, data);

        // send errro response
        int code = 1; // 1:error
        std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
        if (!payload) {
            instance->sendMethodResponse(idx, code, nullptr, tag);
            return;
        }

        payload->push_back(std::move(data));

        instance->sendMethodResponse(idx, code, payload, tag);
        return;
    }

    // deserialize payload to structural input parameters
    std::vector<uint8_t> data_serialized = (*payload)[0];
    SomeIPDeSerializer someip_deserializer(&transformation_props);
    int error_code = 0;
    int ret = someip_deserializer.from_buffer(data_serialized);
    if (ret) {
        error_code = 1;
        instance->sendMethodResponse(idx, error_code, nullptr, tag);
        return;
    }
    InputStream<SomeIPDeSerializer> input_stream(someip_deserializer);
    ret = input_stream.deserialize(args...);
    if (ret) {
        error_code = 1;
        instance->sendMethodResponse(idx, error_code, nullptr, tag);
        return;
    }

    // call implementation of method
    PT_TRACE_BEGINF("skeleton_serialize::on_method_request user callback(method,idx:%d)", idx);
    ara::core::Future<Output> f = callback(args...);
    PT_TRACE_END();

    f.then([=](auto f) mutable {
        PT_TRACE_FUNCTION_AUTO("skeleton_serialize::send_method_response(method,idx:%d)", idx);
        auto result = f.GetResult();
        if (result.HasValue()) {
            // NO Error
            auto out = result.Value();
            int code = 0; // 0:OK
            std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
            if (!payload) {
                code = 1;
                instance->sendMethodResponse(idx, code, nullptr, tag);
                return;
            }

            SomeIPSerializer someip_serializer(&transformation_props);
            OutputStream<SomeIPSerializer> output_stream(someip_serializer);
            int ret = output_stream.write_struct_as_args(out);                                // TODO check ret
            if (ret) {
                code = 1;
                instance->sendMethodResponse(idx, code, payload, tag);
                return;
            }
            std::vector<uint8_t> data_serialized = someip_serializer.getData();
            payload->push_back(std::move(data_serialized));

            instance->sendMethodResponse(idx, code, payload, tag);
        }
        else {
            // Error
            auto error_code = result.Error();
            // TODO:
            // std::int32_t error_code_value = error_code.Value();

            std::vector<uint8_t> data;
            ara::com::runtime::serializeErrorCode(error_code, data);

            int code = 1; // 1:error domain payload
            std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
            if (!payload) {
                instance->sendMethodResponse(idx, code, nullptr, tag);
                return;
            }
            payload->push_back(std::move(data));

            instance->sendMethodResponse(idx, code, payload, tag);
        }
    });
    return;

}

// fire-and-forget Method
// Input(non-void)
// Output(none)
template<typename ...InputArgs>
void executeMethodAsyncRequest(const std::shared_ptr<ara::com::runtime::SkeletonInstance>& instance,
    uint32_t idx,
    const std::shared_ptr<BufferList>& payload,
    std::shared_ptr<TagBase>& tag,
    int e2e_result,
    std::function<void(const InputArgs&...)> callback,
    const ara::com::SomeipTransformationProps& transformation_props,
    InputArgs ...args)
{
    // TODO:
    (void)instance;
    (void)idx;
    if (!tag) {
        // Never happen
        printf("internal error!!!!!!\n");
        return;
    }

    // e2e check failed
    if (e2e_result != 0) {
        return;
    }

    int32_t serialization_type = tag->serialization_type;
    if ((serialization_type & SerializationType_SOMEIP) == 0) {
        return;
    }

    if (payload == nullptr || payload->size() == 0) {
        return;
    }

    // deserialize payload to structural input parameters
    std::vector<uint8_t> data_serialized = (*payload)[0];
    SomeIPDeSerializer someip_deserializer(&transformation_props);
    int ret = someip_deserializer.from_buffer(data_serialized);
    if (ret) {
        return;
    }
    InputStream<SomeIPDeSerializer> input_stream(someip_deserializer);
    ret = input_stream.deserialize(args...);
    if (ret) {
        return;
    }

    // call implementation of method
    PT_TRACE_BEGINF("skeleton_serialize::on_method_request user callback(method,idx:%d)", idx);
    callback(args...);
    PT_TRACE_END();

    return;

}

// Non-fire-and-forget Method
// Input(non-void)
// Output(void)
template<typename Output, typename ...InputArgs>
std::enable_if_t<std::is_void<Output>::value, void> executeMethodAsyncRequest(const std::shared_ptr<ara::com::runtime::SkeletonInstance>& instance,
    uint32_t idx,
    const std::shared_ptr<BufferList>& payload,
    std::shared_ptr<TagBase>& tag,
    int e2e_result,
    std::function<ara::core::Future<void>(const InputArgs&...)> callback,
    const ara::com::SomeipTransformationProps& transformation_props,
    InputArgs ...args)
{
    if (!tag) {
        // Never happen
        printf("internal error!!!!!!\n");
        return;
    }

    // e2e check failed
    if (e2e_result != 0) {
        // TODO e2e Error
        ara::core::ErrorCode error_code{ara::com::ComErrc::kCommunicationStackError};
        std::vector<uint8_t> data;
        ara::com::runtime::serializeErrorCode(error_code, data);

        // send errro response
        int code = 1; // 1:error
        std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
        if (!payload) {
            instance->sendMethodResponse(idx, code, nullptr, tag);
            return;
        }
        payload->push_back(std::move(data));

        instance->sendMethodResponse(idx, code, payload, tag);
        return;
    }

    int32_t serialization_type = tag->serialization_type;
    if ((serialization_type & SerializationType_SOMEIP) == 0) {
        // TODO Error
        ara::core::ErrorCode error_code{ara::com::ComErrc::kCommunicationStackError};
        std::vector<uint8_t> data;
        ara::com::runtime::serializeErrorCode(error_code, data);

        // send errro response
        int code = 1; // 1:error
        std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
        if (!payload) {
            instance->sendMethodResponse(idx, code, nullptr, tag);
            return;
        }
        payload->push_back(std::move(data));

        instance->sendMethodResponse(idx, code, payload, tag);
        return;
    }

    if (payload == nullptr || payload->size() == 0) {
        // TODO Error
        ara::core::ErrorCode error_code{ara::com::ComErrc::kCommunicationStackError};
        std::vector<uint8_t> data;
        ara::com::runtime::serializeErrorCode(error_code, data);

        // send errro response
        int code = 1; // 1:error
        std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
        if (!payload) {
            instance->sendMethodResponse(idx, code, nullptr, tag);
            return;
        }
        payload->push_back(std::move(data));

        instance->sendMethodResponse(idx, code, payload, tag);
        return;
    }

    // deserialize payload to structural input parameters
    std::vector<uint8_t> data_serialized = (*payload)[0];
    SomeIPDeSerializer someip_deserializer(&transformation_props);
    int error_code = 0;
    int ret = someip_deserializer.from_buffer(data_serialized);
    if (ret) {
        error_code = 1;
        instance->sendMethodResponse(idx, error_code, nullptr, tag);
        return;
    }
    InputStream<SomeIPDeSerializer> input_stream(someip_deserializer);
    ret = input_stream.deserialize(args...);
    if (ret) {
        error_code = 1;
        instance->sendMethodResponse(idx, error_code, nullptr, tag);
        return;
    }

    // call implementation of method
    PT_TRACE_BEGINF("skeleton_serialize::on_method_request user callback(method,idx:%d)", idx);
    ara::core::Future<void> f = callback(args...);
    PT_TRACE_END();

    f.then([=](auto f) mutable {
        PT_TRACE_FUNCTION_AUTO("skeleton_serialize::send_method_response(method,idx:%d)", idx);
        auto result = f.GetResult();
        if (result.HasValue()) {
            // NO Error
            // auto out = result.Value();
            int code = 0; // 0:OK
            std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
            if (!payload) {
                code = 1;
                instance->sendMethodResponse(idx, code, nullptr, tag);
                return;
            }
            instance->sendMethodResponse(idx, code, payload, tag);
        }
        else {
            // Error
            auto error_code = result.Error();
            // TODO:
            // std::int32_t error_code_value = error_code.Value();

            std::vector<uint8_t> data;
            ara::com::runtime::serializeErrorCode(error_code, data);

            int code = 1; // 1:error domain payload
            std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
            if (!payload) {
                instance->sendMethodResponse(idx, code, nullptr, tag);
                return;
            }
            payload->push_back(std::move(data));

            instance->sendMethodResponse(idx, code, payload, tag);
        }

    });
    return;

}

// Non-fire-and-forget Method
// Input(void)
// Output(non-void)
template<typename Output>
void executeMethodAsyncRequest(const std::shared_ptr<ara::com::runtime::SkeletonInstance>& instance,
    uint32_t idx,
    const std::shared_ptr<BufferList>& payload,
    std::shared_ptr<TagBase>& tag,
    int e2e_result,
    std::function<ara::core::Future<Output>()> callback,
    const ara::com::SomeipTransformationProps& transformation_props)
{
    // TODO:
    (void)payload;
    if (!tag) {
        // Never happen
        printf("internal error!!!!!!\n");
        return;
    }

    if (e2e_result != 0) {
        // TODO e2e Error
        ara::core::ErrorCode error_code{ara::com::ComErrc::kCommunicationStackError};
        std::vector<uint8_t> data;
        ara::com::runtime::serializeErrorCode(error_code, data);

        // send errro response
        int code = 1; // 1:error
        std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
        if (!payload) {
            instance->sendMethodResponse(idx, code, payload, tag);
            return;
        }
        payload->push_back(std::move(data));

        instance->sendMethodResponse(idx, code, payload, tag);
        return;
    }

    int32_t serialization_type = tag->serialization_type;
    if ((serialization_type & SerializationType_SOMEIP) == 0) {
        // TODO Error
        ara::core::ErrorCode error_code{ara::com::ComErrc::kCommunicationStackError};
        std::vector<uint8_t> data;
        ara::com::runtime::serializeErrorCode(error_code, data);

        // send errro response
        int code = 1; // 1:error
        std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
        if (!payload) {
            instance->sendMethodResponse(idx, code, payload, tag);
            return;
        }
        payload->push_back(std::move(data));

        instance->sendMethodResponse(idx, code, payload, tag);
        return;
    }

    // call implementation of method
    PT_TRACE_BEGINF("skeleton_serialize::on_method_request user callback(method,idx:%d)", idx);
    ara::core::Future<Output> f = callback();
    PT_TRACE_END();

    f.then([=](auto f) mutable {

        PT_TRACE_FUNCTION_AUTO("skeleton_serialize::send_method_response(method,idx:%d)", idx);
        auto result = f.GetResult();
        if (result.HasValue()) {
            // NO Error
            auto out = result.Value();
            int code = 0; // 0:OK
            std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
            if (!payload) {
                code = 1;
                instance->sendMethodResponse(idx, code, nullptr, tag);
                return;
            }

            SomeIPSerializer someip_serializer(&transformation_props);
            OutputStream<SomeIPSerializer> output_stream(someip_serializer);
            int ret = output_stream.write_struct_as_args(out);                                // TODO check ret
            if (ret) {
                code = 1;
                instance->sendMethodResponse(idx, code, payload, tag);
                return;
            }
            std::vector<uint8_t> data_serialized = someip_serializer.getData();
            payload->push_back(std::move(data_serialized));

            instance->sendMethodResponse(idx, code, payload, tag);
        }
        else {
            // Error
            auto error_code = result.Error();
            // TODO:
            // std::int32_t error_code_value = error_code.Value();

            std::vector<uint8_t> data;
            ara::com::runtime::serializeErrorCode(error_code, data);

            int code = 1; // 1:error domain payload
            std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
            if (!payload) {
                instance->sendMethodResponse(idx, code, payload, tag);
                return;
            }
            payload->push_back(std::move(data));

            instance->sendMethodResponse(idx, code, payload, tag);
        }

    });

    return;
}

// fire-and-forget Method
// Input(void)
// Output(none)
static void executeMethodAsyncRequest(const std::shared_ptr<ara::com::runtime::SkeletonInstance>& instance,
    uint32_t idx,
    const std::shared_ptr<BufferList>& payload,
    std::shared_ptr<TagBase>& tag,
    int e2e_result,
    std::function<void()> callback)
{
    // TODO:
    (void)instance;
    (void)idx;
    (void)payload;
    (void)tag;
    (void)e2e_result;
    // call implementation of method
    PT_TRACE_BEGINF("skeleton_serialize::on_method_request user callback(method,idx:%d)", idx);
    callback();
    PT_TRACE_END();

    return;
}

// Non-fire-and-forget Method
// Input(void)
// Output(void)
static void executeMethodAsyncRequest(const std::shared_ptr<ara::com::runtime::SkeletonInstance>& instance,
    uint32_t idx,
    const std::shared_ptr<BufferList>& payload,
    std::shared_ptr<TagBase>& tag,
    int e2e_result,
    std::function<ara::core::Future<void>()> callback)
{
    // TODO:
    (void)payload;
    (void)e2e_result;
    if (!tag) {
        // Never happen
        printf("internal error!!!!!!\n");
        return;
    }

    // call implementation of method
    PT_TRACE_BEGINF("skeleton_serialize::on_method_request user callback(method,idx:%d)", idx);
    ara::core::Future<void> f = callback();
    PT_TRACE_END();

    f.then([=](auto f) mutable {
        PT_TRACE_FUNCTION_AUTO("skeleton_serialize::send_method_response(method,idx:%d)", idx);
        auto result = f.GetResult();
        if (result.HasValue()) {
            // NO Error
            // auto out = result.Value();
            int code = 0; // 0:OK
            std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
            if (!payload) {
                code = 1;
                instance->sendMethodResponse(idx, code, nullptr, tag);
                return;
            }

            instance->sendMethodResponse(idx, code, payload, tag);
        }
        else {
            // Error
            auto error_code = result.Error();
            // TODO:
            // std::int32_t error_code_value = error_code.Value();

            std::vector<uint8_t> data;
            ara::com::runtime::serializeErrorCode(error_code, data);

            int code = 1; // 1:error domain payload
            std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
            if (!payload) {
                instance->sendMethodResponse(idx, code, nullptr, tag);
                return;
            }
            payload->push_back(std::move(data));

            instance->sendMethodResponse(idx, code, payload, tag);
        }


    });

    return;

}

template<typename InOutput>
void executeFieldSetAsyncRequest(const std::shared_ptr<ara::com::runtime::SkeletonInstance>& instance,
    uint32_t idx,
    const std::shared_ptr<BufferList>& payload,
    std::shared_ptr<TagBase>& tag,
    int e2e_result,
    std::function<ara::core::Future<InOutput>(const InOutput&)> callback,
    const ara::com::SomeipTransformationProps& transformation_props)
{
    if (!tag) {
        // Never happen
        printf("internal error!!!!!!\n");
        return;
    }

    // e2e check failed
    if (e2e_result != 0) {
        // TODO e2e Error
        ara::core::ErrorCode error_code{ara::com::ComErrc::kCommunicationStackError};
        std::vector<uint8_t> data;
        ara::com::runtime::serializeErrorCode(error_code, data);

        // send errro response
        int code = 1; // 1:error
        std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
        if (!payload) {
            instance->sendFieldSetResponse(idx, code, payload, tag);
            return;
        }
        payload->push_back(std::move(data));

        instance->sendFieldSetResponse(idx, code, payload, tag);

        return;
    }

    int32_t serialization_type = tag->serialization_type;
    if ((serialization_type & SerializationType_SOMEIP) == 0) {
        // TODO Error
        ara::core::ErrorCode error_code{ara::com::ComErrc::kCommunicationStackError};
        std::vector<uint8_t> data;
        ara::com::runtime::serializeErrorCode(error_code, data);

        // send errro response
        int code = 1; // 1:error
        std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
        if (!payload) {
            instance->sendFieldSetResponse(idx, code, payload, tag);
            return;
        }
        payload->push_back(std::move(data));

        instance->sendFieldSetResponse(idx, code, payload, tag);

        return;
    }

    if (payload == nullptr || payload->size() == 0) {
        // TODO Error
        ara::core::ErrorCode error_code{ara::com::ComErrc::kCommunicationStackError};
        std::vector<uint8_t> data;
        ara::com::runtime::serializeErrorCode(error_code, data);

        // send errro response
        int code = 1; // 1:error
        std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
        if (!payload) {
            instance->sendFieldSetResponse(idx, code, nullptr, tag);
            return;
        }
        payload->push_back(std::move(data));

        instance->sendFieldSetResponse(idx, code, payload, tag);

        return;
    }

    InOutput input_value;

    // deserialize payload to structural input parameters
    std::vector<uint8_t> data_serialized = (*payload)[0];
    SomeIPDeSerializer someip_deserializer(&transformation_props);
    int error_code = 0;
    int ret = someip_deserializer.from_buffer(data_serialized);
    if (ret) {
        error_code = 1;
        instance->sendFieldSetResponse(idx, error_code, nullptr, tag);
        return;
    }
    InputStream<SomeIPDeSerializer> input_stream(someip_deserializer);
    ret = input_stream.deserialize(input_value);
    if (ret) {
        error_code = 1;
        instance->sendFieldSetResponse(idx, error_code, nullptr, tag);
        return;
    }

    // call implementation of field set by user registered
    PT_TRACE_BEGINF("skeleton_serialize::on_field_set_request user callback(field,idx:%d)", idx);
    ara::core::Future<InOutput> f = callback(input_value);
    PT_TRACE_END();

    f.then([=](auto f) mutable {
        PT_TRACE_FUNCTION_AUTO("skeleton_serialize::send_field_set_response(field,idx:%d)", idx);
        auto result = f.GetResult();
        if (result.HasValue()) {
            // NO Error
            InOutput out = result.Value();
            int code = 0; // 0:OK
            std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
            if (!payload) {
                code = 1;
                instance->sendFieldSetResponse(idx, code, nullptr, tag);
                return;
            }

            SomeIPSerializer someip_serializer(&transformation_props);
            OutputStream<SomeIPSerializer> output_stream(someip_serializer);
            int ret = output_stream.serialize(out);                                // TODO check ret
            if (ret) {
                code = 1;
                instance->sendFieldSetResponse(idx, code, payload, tag);
                return;
            }
            std::vector<uint8_t> data_serialized = someip_serializer.getData();
            payload->push_back(std::move(data_serialized));

            instance->sendFieldSetResponse(idx, code, payload, tag);
        }
        else {
            // Error
            auto error_code = result.Error();
            // TODO:
            // std::int32_t error_code_value = error_code.Value();

            std::vector<uint8_t> data;
            ara::com::runtime::serializeErrorCode(error_code, data);

            int code = 1; // 1:error domain payload
            std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
            if (!payload) {
                instance->sendFieldSetResponse(idx, code, nullptr, tag);
                return;
            }
            payload->push_back(std::move(data));

            instance->sendFieldSetResponse(idx, code, payload, tag);
        }

    });

    return;
}

template<typename Output>
void executeFieldGetAsyncRequest(const std::shared_ptr<ara::com::runtime::SkeletonInstance>& instance,
    uint32_t idx,
    const std::shared_ptr<BufferList>& payload,
    std::shared_ptr<TagBase>& tag,
    int32_t e2e_result,
    std::function<ara::core::Future<Output>()> callback,
    const ara::com::SomeipTransformationProps& transformation_props)
{
    // TODO:
    (void)payload;
    (void)e2e_result;
    if (!tag) {
        // Never happen
        printf("internal error!!!!!!\n");
        return;
    }


    int32_t serialization_type = tag->serialization_type;
    if ((serialization_type & SerializationType_SOMEIP) == 0) {
        // TODO Error
        ara::core::ErrorCode error_code{ara::com::ComErrc::kCommunicationStackError};
        std::vector<uint8_t> data;
        ara::com::runtime::serializeErrorCode(error_code, data);

        // send errro response
        int code = 1; // 1:error
        std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
        if (!payload) {
            instance->sendFieldGetResponse(idx, code, nullptr, tag);
            return;
        }
        payload->push_back(std::move(data));

        instance->sendFieldGetResponse(idx, code, payload, tag);
        return;
    }

    // call implementation of field get by user registered
    PT_TRACE_BEGINF("skeleton_serialize::on_field_get_request user callback(field,idx:%d)", idx);
    ara::core::Future<Output> f = callback();
    PT_TRACE_END();

    f.then([=](auto f) mutable {
        PT_TRACE_FUNCTION_AUTO("skeleton_serialize::send_field_get_response(field,idx:%d)", idx);

        auto result = f.GetResult();
        if (result.HasValue()) {
            // NO Error
            Output out = result.Value();
            int code = 0; // 0:OK
            std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
            if (!payload) {
                code = 1;
                instance->sendFieldGetResponse(idx, code, nullptr, tag);
                return;
            }

            SomeIPSerializer someip_serializer(&transformation_props);
            OutputStream<SomeIPSerializer> output_stream(someip_serializer);
            int ret = output_stream.serialize(out);                                // TODO check ret
            if (ret) {
                code = 1;
                instance->sendFieldGetResponse(idx, code, nullptr, tag);
                return;
            }
            std::vector<uint8_t> data_serialized = someip_serializer.getData();
            payload->push_back(std::move(data_serialized));

            instance->sendFieldGetResponse(idx, code, payload, tag);
        }
        else {
            // Error
            auto error_code = result.Error();
            // TODO:
            // std::int32_t error_code_value = error_code.Value();

            std::vector<uint8_t> data;
            ara::com::runtime::serializeErrorCode(error_code, data);

            int code = 1; // 1:error domain payload
            std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
            if (!payload) {
                instance->sendFieldGetResponse(idx, code, nullptr, tag);
                return;
            }
            payload->push_back(std::move(data));

            instance->sendFieldGetResponse(idx, code, payload, tag);
        }
    });

    return;
}

template<typename Input>
void sendEventNotify(const std::shared_ptr<ara::com::runtime::SkeletonInstance>& instance,
    uint32_t event_idx,
    const ara::com::SomeipTransformationProps& transformation_props,
    const Input& data)
{
    PT_TRACE_FUNCTION_AUTO("skeleton_serialize::send_event_notify(event,idx:%d)", event_idx);
    int32_t serialization_type = instance->eventSerializationType(event_idx);
    if ((serialization_type & SerializationType_SOMEIP) == 0) {
        return;
    }

    std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
    if (!payload) {
        return;
    }

    SomeIPSerializer someip_serializer(&transformation_props);
    OutputStream<SomeIPSerializer> output_stream(someip_serializer);
    int ret = output_stream.serialize(data);    // TODO check ret
    if (ret) {
        return;
    }
    std::vector<uint8_t> data_serialized = someip_serializer.getData();
    payload->push_back(std::move(data_serialized));

    std::shared_ptr<TagBase> tag = std::make_shared<ara::com::runtime::OneWayTag>();
    if (!tag) {
        return;
    }
    tag->serialization_type = SerializationType_SOMEIP;
    instance->sendEventNotify(event_idx, payload, tag);
}

template<typename Input>
void sendEventNotify(const std::shared_ptr<ara::com::runtime::SkeletonInstance>& instance,
    uint32_t event_idx,
    const ara::com::SomeipTransformationProps& transformation_props,
    ara::com::SampleAllocateePtr<Input> data)
{
    PT_TRACE_FUNCTION_AUTO("skeleton_serialize::send_event_notify(event,idx:%d)", event_idx);
    int32_t serialization_type = instance->eventSerializationType(event_idx);
    if ((serialization_type & SerializationType_SOMEIP) == 0) {
        return;
    }

    std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
    if (!payload) {
        return;
    }

    SomeIPSerializer someip_serializer(&transformation_props);
    OutputStream<SomeIPSerializer> output_stream(someip_serializer);
    int ret = output_stream.serialize(*(data.get()));    // TODO check ret
    if (ret) {
        return;
    }
    std::vector<uint8_t> data_serialized = someip_serializer.getData();
    payload->push_back(std::move(data_serialized));

    std::shared_ptr<TagBase> tag = std::make_shared<ara::com::runtime::OneWayTag>();
    if (!tag) {
        return;
    }
    tag->serialization_type = SerializationType_SOMEIP;
    instance->sendEventNotify(event_idx, payload, tag);
}

template<typename Input>
void sendFieldNotify(const std::shared_ptr<ara::com::runtime::SkeletonInstance>& instance,
    uint32_t field_idx,
    const ara::com::SomeipTransformationProps& transformation_props,
    const Input& data)
{
    PT_TRACE_FUNCTION_AUTO("skeleton_serialize::send_field_notify(event,idx:%d)", field_idx);
    int32_t serialization_type = instance->fieldSerializationType(field_idx);
    if ((serialization_type & SerializationType_SOMEIP) == 0) {
        return;
    }

    std::shared_ptr<BufferList> payload = std::make_shared<BufferList>();
    if (!payload) {
        return;
    }

    SomeIPSerializer someip_serializer(&transformation_props);
    OutputStream<SomeIPSerializer> output_stream(someip_serializer);
    int ret = output_stream.serialize(data);    // TODO check ret
    if (ret) {
        return;
    }
    std::vector<uint8_t> data_serialized = someip_serializer.getData();
    payload->push_back(std::move(data_serialized));

    std::shared_ptr<TagBase> tag = std::make_shared<ara::com::runtime::OneWayTag>();
    if (!tag) {
        return;
    }
    tag->serialization_type = SerializationType_SOMEIP;
    instance->sendFieldNotify(field_idx, payload, tag);
}

#endif // ARA_COM_INTERNAL_SKELETON_H_
/* EOF */
