#ifndef E2EXF_IMPL_H_
#define E2EXF_IMPL_H_

#include "e2e/e2exf_cpp/include/e2exf_mapping.h"
namespace hozon {
namespace netaos {
namespace e2e {

bool AddE2EXfConfig(const E2EXf_Index& Index, const E2EXf_Config& Config);

ProtectResult E2EXf_Protect(const E2EXf_Index& Index, Payload& Buffer, const std::uint32_t InputBufferLength);

ProtectResult E2EXf_Protect_OutOfPlace(const E2EXf_Index& Index, const Payload& InputBuffer, Payload& Buffer, const std::uint32_t InputBufferLength, std::uint32_t& BufferLength);

ProtectResult E2EXf_Forward(const E2EXf_Index& Index, Payload& Buffer, const std::uint32_t InputBufferLength, Std_TransformerForwardCode ForwardCode);

ProtectResult E2EXf_Forward_OutOfPlace(const E2EXf_Index& Index, const Payload& InputBuffer, Payload& Buffer, const std::uint32_t InputBufferLength, std::uint32_t& BufferLength,
                                       Std_TransformerForwardCode ForwardCode);

CheckResult E2EXf_Check(const E2EXf_Index& Index, Payload& Buffer, const std::uint32_t InputBufferLength);

CheckResult E2EXf_Check_OutOfPlace(const E2EXf_Index& Index, const Payload& InputBuffer, Payload& Buffer, const std::uint32_t InputBufferLength, std::uint32_t& BufferLength);

void DeInit();

}  // namespace e2e
}  // namespace netaos
}  // namespace hozon
#endif