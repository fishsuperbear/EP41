#ifndef ROSBAG2_STORAGE__ATTACHMENT_HPP_
#define ROSBAG2_STORAGE__ATTACHMENT_HPP_

#include <memory>
#include <string>

#include "rcutils/time_ros.h"

namespace rosbag2_storage
{

struct Attachment
{
  rcutils_time_point_value_t logTime;
  rcutils_time_point_value_t createTime;
  std::string name;
  std::string mediaType;
  uint64_t dataSize;
  std::string data;
  uint32_t crc;
};

typedef std::shared_ptr<Attachment> AttachmentSharedPtr;

}  // namespace rosbag2_storage

#endif  // ROSBAG2_STORAGE__ATTACHMENT_HPP_
