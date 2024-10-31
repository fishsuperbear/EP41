/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
*
* @File: dc_client_trigger.cpp
* @Date: 2023/10/27
* @Author: cheng
* @Desc: --
*/

#include "client/include/dc_client.h"
#include "common/include/argvparser.h"

struct UploadData {
    std::vector<std::string> uploadFilePathVec;
    std::string uploadDataVec;
    std::string fileType;
    std::string fileName;
    int cacheNum;
};

struct triggerDesc {
    int triggerId{-1};
    uint64_t time{0};
    std::string desc{""};
};

int main(int argc, char** argv)  {
    using namespace argvparser;
    bool helpFlag = false;
    int triggerId = -1;
    UploadData uploadData;
    triggerDesc td;
    auto triggerArgv = (option("-h", "--help").set(helpFlag, true) % "print the help infomation",
                        (option("-t", "--trigger id") &
                            value("trigger-id", triggerId) % "[101-9999] the trigger id of specific scene"),
                        (option("-uf", "--upload files") &
                            values("upload-files", uploadData.uploadFilePathVec) % "upload files list"),
                        (option("-ud", "--upload datas") &
                            value("upload-datas", uploadData.uploadDataVec) % "upload datas"),
                        (option("-type", "--file-type") &
                            value("file-type", uploadData.fileType) % "upload file type"),
                        (option("-f", "--file-name") &
                            value("file-name", uploadData.fileName) % "upload file name"),
                        (option("-c", "--cache-number") &
                            value("cache-number", uploadData.cacheNum) % "the max cache number of files when network is slow"),
                        (option("-td", "--trigger-description") &
                         (value("triggerId", td.triggerId) % "trigger id(100-10000)")  &
                         (value("time", td.triggerId) % "trigger time(eg:1695736685)") &
                        (value("desc", td.desc) % "description"))
                       );
    if (parse(argc, argv, triggerArgv) && !helpFlag) {
        hozon::netaos::dc::DcClient client;
        client.Init("testTrigger", 3000);
        if ((triggerId >= 100)) {
            client.CollectTrigger(triggerId);
        }
        if (!uploadData.uploadFilePathVec.empty()) {
            client.Upload(uploadData.uploadFilePathVec, uploadData.fileType, uploadData.fileName, uploadData.cacheNum);
        }
        if (!uploadData.uploadDataVec.empty()) {
            std::vector<char> vec(uploadData.uploadDataVec.begin(), uploadData.uploadDataVec.end());
            client.Upload(vec, uploadData.fileType, uploadData.fileName, uploadData.cacheNum);
        }
        if (td.triggerId != -1) {
            if (!td.desc.empty()) {
                client.CollectTriggerDesc(td.triggerId,td.time,td.desc);
            } else {
                client.CollectTriggerDesc(td.triggerId,td.time);
            }
        }
        client.DeInit();
        return 0;
    } else {
        std::cout << argvparser::make_man_page(triggerArgv, argv[0]) << '\n';
    }
}