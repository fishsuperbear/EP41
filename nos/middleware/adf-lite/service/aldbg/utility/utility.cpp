
#include <string>
#include "adf-lite/service/aldbg/utility/utility.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

//例   /lite/lite1/event/workresult2
int32_t GetPartFromCmTopic(const std::string& topic, const int32_t index, std::string& value)
{
    value = "";
    if (index > 3) {
        return -1;
    }
    if (topic.substr(0, 6) != "/lite/") return -1;
    std::size_t start = 1;
    std::size_t end = 1;
    for (int32_t i = 0; i <= index; i++) {
        end = topic.find_first_of('/', start);
        if (i != 3 && end == std::string::npos) {
            return -1;
        }
        //判断第1个字段，是否是lite
        if (i == 0 && topic.substr(start, end - start) != "lite") {
            return -1;
        } //判断第3个字段，是否是event
        // else if (i == 2 && topic.substr(start, end - start) != "event") {
        //     return -1;
        // }
        else if (i == 3 && index == 3) {
            value = topic.substr(start);
        } else {
            value = topic.substr(start, end - start);
        }
        if (i == index) break;
        start = end + 1;
    }
    return 0;
}

int32_t GetProcessFromCmTopic(const std::string& topic, std::string& process)
{
    return GetPartFromCmTopic(topic, 1, process);
}

int32_t GetInnerTopicFromCmTopic(const std::string& topic, std::string& process)
{
    return GetPartFromCmTopic(topic, 3, process);
}
} // namespace adf_lite
} // namespace netaos {
} // namespace  hozon