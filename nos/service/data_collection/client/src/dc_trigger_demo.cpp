//
// Created by cheng on 23-6-20.
//

#include <unistd.h>
#include <vector>

#include "cm/include/method.h"
#include "idl/generated/data_collection_info.h"
#include "idl/generated/data_collection_infoPubSubTypes.h"
#include "log/include/default_logger.h"
#include "basic/trans_struct.h"
#include "utils/include/dc_logger.hpp"

//#include "proto/soc/for_test.pb.h"
//void cb(const std::shared_ptr<adf::lite::dbg::WorkflowResult>& msg) {
//    DF_LOG_INFO << "msg valL: " << msg->val1();
//}
using namespace hozon::netaos::cm;
using namespace std;
enum DcResultCode {
    DC_OK = 0,
    DC_TIMEOUT,
    DC_INNER_ERROR,
    DC_SERVICE_NO_READY,
    DC_INVALID_PARAM,
    DC_NO_PARAM,
    DC_PATH_NOT_FOUND,
    DC_UNSUPPORT
};
enum UploadDataType {
    file,
    folder,
    fileAndFolder,
    memory,
    null,
};
using namespace hozon::netaos::dc;
class DcClient{
   public:
    DcClient(): client_(std::make_shared<triggerInfoPubSubType>(),std::make_shared<triggerResultPubSubType>()){

    }
    DcResultCode Init(const std::string clientName, const uint32_t maxWaitMillis = 1000) {
        return DC_OK;
    }
    DcResultCode DeInit() {
        return DC_OK;
    }
    DcResultCode CollectTrigger(uint32_t trigger_id){
        return DC_OK;
    };

    DcResultCode Upload(UploadDataType dataType, std::vector<std::string> pathList, std::string fileType, std::string fileName, uint16_t cacheFileNum) {
        return DC_OK;
    }
   private:
    Client<triggerInfo, triggerResult> client_;
};

struct strCmp{
    bool operator ()(std::string &a, std::string &b) {
        return a.compare(b)<0;
    }
};

int main(int argc, char* argv[]) {
    std::vector<std::string> allFiles,tempFiles;
    tempFiles.emplace_back("ds");
    tempFiles.emplace_back("ds3");
    tempFiles.emplace_back("ds1");
    tempFiles.emplace_back("ds2");
    std::sort(tempFiles.begin(),tempFiles.end(), strCmp());
    allFiles.insert(allFiles.end(),tempFiles.begin()+1,tempFiles.begin()+tempFiles.size()-1);


    Client<triggerInfo, triggerResult> client(std::make_shared<triggerInfoPubSubType>(),std::make_shared<triggerResultPubSubType>());
    client.Init(0, "serviceDataCollection");
    int online;
    int maxRetryTimes =5;
    for (int i=1;i<=maxRetryTimes;i++) {
        online = client.WaitServiceOnline(50+500*i);  //用户需要去调等待服务
        if (online ==0 ) {
            cout<<"retry times:"<<i<<endl;
            break;
        }
        if (i==maxRetryTimes) {
            cout<<"service is not online after "<<maxRetryTimes<<" times try;"<<endl;
            return -1;
        }
    }
    cout<<"Service online result:"<<online<<endl;
    std::shared_ptr<triggerInfo> req_data = std::make_shared<triggerInfo>();
    req_data->clientName("dc_cm_test_client");
    req_data->type("trigger");
    req_data->value("emergencyBraking");
    std::shared_ptr<triggerResult> res_data= std::make_shared<triggerResult>();
    res_data->retCode(1);
    res_data->msg("Service not online");
    auto res = client.Request(req_data, res_data, 5000*5);
    cout<<"result:"<<res_data->msg()<<endl;
    client.Deinit();
    return 0;
}