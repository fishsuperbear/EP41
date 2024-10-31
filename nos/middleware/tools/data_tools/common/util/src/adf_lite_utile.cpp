
#include "adf_lite_utile.h"

namespace hozon {
namespace netaos {
namespace data_tool_common {

//例   /lite/lite1/info
bool IsLiteInfoTopic(const std::string& topic) {
    // 判断/字符数是否为3
    if (3 != std::count(topic.begin(), topic.end(), '/')) {
        return false;
    }

    // 判断是否以/lite/开头
    if (0 != topic.find("/lite/")) {
        return false;
    }

    // 判断是否以/info结尾
    std::string suffix = "/info";
    if (topic.rfind(suffix) != topic.length() - suffix.length()) {
        return false;
    }

    return true;
}

//例   /reply//lite/lite1/method/cmd
bool IsLiteMethodCMDTopic(const std::string& topic) {

    // 判断是否以/reply//lite/或/request//lite/开头
    if (0 != topic.find("/reply//lite/") && 0 != topic.find("/request//lite/")) {
        return false;
    }

    // 判断是否以/method/cmd结尾
    std::string suffix = "/method/cmd";
    if (topic.rfind(suffix) != topic.length() - suffix.length()) {
        return false;
    }

    return true;
}

//例   /lite/lite1/event/workresult1
bool IsAdfTopic(const std::string& topic) {
    // 判断/字符数是否为4
    if (4 != std::count(topic.begin(), topic.end(), '/')) {
        return false;
    }

    // 判断是否以/lite/开头
    if (0 != topic.find("/lite/")) {
        return false;
    }

    // 判断第三个//是否是/event/
    std::string part_three;
    int32_t res = hozon::netaos::data_tool_common::GetPartFromCmTopic(topic, 2, part_three);
    if (res == -1) {
        COMMON_LOG_WARN << "can't get third name from topic: " << topic;
        return false;
    }
    if ("event" != part_three) {
        return false;
    }

    return true;
}

//例   /lite/lite1/event/workresult2
int32_t GetPartFromCmTopic(const std::string& topic, const int32_t index, std::string& value) {
    value = "";
    if (index > 3) {
        return -1;
    }
    if (topic.substr(0, 6) != "/lite/")
        return -1;
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
        }  //判断第3个字段，是否是event
        // else if (i == 2 && topic.substr(start, end - start) != "event") {
        //     return -1;
        // }
        else if (i == 3 && index == 3) {
            value = topic.substr(start);
        } else {
            value = topic.substr(start, end - start);
        }
        if (i == index)
            break;
        start = end + 1;
    }
    return 0;
}

void RequestCommand(const std::map<std::string, std::vector<std::string>>& topics_map, const std::string& cmd, const bool status) {
    hozon::netaos::cm::ProtoMethodClient<hozon::adf::lite::dbg::EchoRequest, hozon::adf::lite::dbg::GeneralResponse> _proto_client;
    for (auto iter = topics_map.begin(); iter != topics_map.end(); ++iter) {
        int32_t ret = _proto_client.Init(0, "/lite/" + iter->first + "/method/cmd");
        std::shared_ptr<hozon::adf::lite::dbg::EchoRequest> req(new hozon::adf::lite::dbg::EchoRequest);
        req->set_cmd(cmd);
        req->set_status(status);
        for (auto& topic : iter->second) {
            hozon::adf::lite::dbg::EchoRequest::Element* ele = req->add_elements();
            ele->set_topic(topic);
        }

        std::shared_ptr<hozon::adf::lite::dbg::GeneralResponse> resp(new hozon::adf::lite::dbg::GeneralResponse);
        _proto_client.WaitServiceOnline(500);
        ret = _proto_client.Request(req, resp, 1000);
        _proto_client.DeInit();
        if (ret < 0) {
            COMMON_LOG_ERROR << "Cmd: " << req->cmd() << "; Fail to request, ret " << ret;
        } else {
            COMMON_LOG_DEBUG << "Cmd: " << req->cmd() << " Resp= " << resp->res();
        }
    }
}

}  // namespace data_tool_common
}  //namespace netaos
}  //namespace hozon