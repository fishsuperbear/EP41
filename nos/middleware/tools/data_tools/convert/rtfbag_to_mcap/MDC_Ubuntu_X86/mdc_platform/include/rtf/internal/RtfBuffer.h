/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description:
 *      This file is the implement of class RtfBuffer.
 *      RtfBuffer is for store messages
 * Create: 2019-11-30
 * Notes: NA
 */
#ifndef RTF_BUFFER_H
#define RTF_BUFFER_H

#include <cstdint>

namespace rtf {
namespace rtfbag {
class RtfBuffer {
public:
    RtfBuffer();
    ~RtfBuffer();

    bool SetSize(const uint32_t& size);
    uint32_t GetSize() const;

    const uint8_t* GetData() const;
    bool IsValid() const { return isValid_; }
    void SetValid(bool isValid) { isValid_ = isValid; }
private:
    uint8_t* buffer_;
    uint32_t size_;
    bool isValid_ = true;
};
}  // namespace rtfbag
}  // namespace rtf
#endif
