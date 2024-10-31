/**
 * @file desen_process.h
 * @author 
 * @brief 
 * @version 0.1
 * @date 2023-11-08
 * 
 * Copyright Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * 
 */
#ifndef NOS_MIDDLEWARE_DESEN_PROCESS_DESEN_PROCESS_H_
#define NOS_MIDDLEWARE_DESEN_PROCESS_DESEN_PROCESS_H_

#include <memory>
#include <string>

namespace hozon {
namespace netaos {
namespace desen {

class DesenProcess {
   public:
    DesenProcess(uint16_t width, uint16_t height);
    ~DesenProcess();
    uint32_t Process(const std::string& input, std::string& output);

   private:
    class DesenProcessImpl;
    std::unique_ptr<DesenProcessImpl> impl_;
};
}  // namespace desen
}  // namespace netaos
}  // namespace hozon

#endif  // NOS_MIDDLEWARE_DESEN_PROCESS_DESEN_PROCESS_H_
