#include "wrapper.h"
#include "pandarSwiftSDK.h"
// #include <pcl/io/pcd_io.h>
// #include <pcl/point_types.h>
#include <iostream>
#ifdef __cplusplus
extern "C" {
#endif

#define PRINT_FLAG
int printFlag = 1; // 1:print, other number: don't print
int pcdFileWriteFlag = 0; // 1:write pcd, other number: don't write pcd
bool running = true;
int frameItem = 0;
int saveFrameIndex = 10;
boost::shared_ptr<PandarSwiftSDK> spPandarSwiftSDK;

void gpsCallback(double timestamp) {
    if(printFlag == 1)     
        printf("gps: %lf\n", timestamp);   
}

void lidarCallback(boost::shared_ptr<PPointCloud> cld, double timestamp) {
    if(printFlag == 1)       
        printf("timestamp: %lf,point_size: %ld\n", timestamp, cld->size());
    if(pcdFileWriteFlag == 1) {
        frameItem++;
        if(saveFrameIndex == frameItem) {
            int Num = cld->size();
            std::ofstream zos("./cloudpoints.csv");
            for (int i = 0; i < Num; i++)                             
            {
                zos <<  cld->at(i).x << "," << cld->at(i).y << "," << cld->at(i).z << "," << cld->at(i).intensity << "," << cld->at(i).timestamp << "," << cld->at(i).ring << std::endl;
            }
        }
    }         
}



void rawcallback(PandarPacketsArray *array) {
    // printf("array size: %d\n", array->size());
}

void faultmessagecallback(AT128FaultMessageInfo &faultMessage) {
}


void RunPandarSwiftSDK(char* deviceipaddr, int lidarport, int gpsport, char* correctionfile, char* firtimeflie, char* pcapfile, int viewMode,
						char* certFile, char* privateKeyFile, char* caFile, int runTime) {
    spPandarSwiftSDK.reset(new PandarSwiftSDK(deviceipaddr, "", lidarport, gpsport, std::string("Pandar128"), \
                                correctionfile, \
                                firtimeflie, \
                                pcapfile, lidarCallback, rawcallback, gpsCallback, faultmessagecallback,\
                                certFile, \
                                privateKeyFile, \
                                caFile, \
                                0, 0, viewMode, std::string("both_point_raw"), ""));  
  
    sleep(runTime);
    spPandarSwiftSDK->stop();
    return;
}

void SetPcdFileWriteFlag(int flag, int frameNum){
    saveFrameIndex = frameNum;
    pcdFileWriteFlag = flag;
    return;
}

void SetPrintFlag(int flag){
    printFlag = flag;
    return;
}
#ifdef __cplusplus
};
#endif
