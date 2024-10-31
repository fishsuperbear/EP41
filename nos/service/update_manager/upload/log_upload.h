/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: log upload
 */
#ifndef LOG_UPLOAD_H
#define LOG_UPLOAD_H

namespace hozon {
namespace netaos {
namespace update {


class LogUploader {
public:
    LogUploader();
    ~LogUploader();

private:
    LogUploader(const LogUploader &);
    LogUploader & operator = (const LogUploader &);

};

}  // namespace update
}  // namespace netaos
}  // namespace hozon
#endif  // LOG_UPLOAD_H