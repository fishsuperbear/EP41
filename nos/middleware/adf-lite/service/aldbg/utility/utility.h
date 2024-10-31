
#ifndef __UTILITY_H_
#define __UTILITY_H_
#include <string>

namespace hozon {
namespace netaos {
namespace adf_lite {

int32_t GetPartFromCmTopic(const std::string& topic, const int32_t index, std::string& value);
int32_t GetProcessFromCmTopic(const std::string& topic, std::string& process);
int32_t GetInnerTopicFromCmTopic(const std::string& topic, std::string& process);

} // namespace adf_lite
} // namespace netaos {
} // namespace  hozon

#endif /* __UTILITY_H_ */