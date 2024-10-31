#include <cstdint>
#include <iostream>
#include <memory>
#include <unistd.h>
#include <csignal>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "log/include/default_logger.h"
#include "camera_method_client.h"

#pragma pack(push, 1)
struct camera_interal_data_x3f_x8b {
    uint8_t time_year;
    uint8_t time_month;
    uint8_t time_day;
    uint8_t version;
    uint8_t model;
    double fx;
    double fy;
    double cx;
    double cy;
    double k1;
    double k2;
    double k3;
    double k4;
    double k5;
    double k6;
    double p1;
    double p2;
    double averang;
    double maximum;
};

struct camera_interal_data_x031 {
    float fx;
    float fy;
    float cx;
    float cy;
    float k5;
    float k6;
    float xx;
    float k1;
    float k2;
    float p1;
    float p2;
    float k3;
    float k4;
};
#pragma pack(pop)


int main(int argc, char** argv)
{
    DefaultLogger::GetInstance().InitLogger();
    DF_LOG_INFO << "Start..";

    CameraMethodClient camera_client;
    camera_client.WaitServiceOnline(2000);

    for (uint8_t i = 0; i < 12; i++) {
        if (i == 3) {
            // no sensor id 3
            continue;
        }

        CameraInternalData camera_data;
        int ret = camera_client.GetCameraInternalData(i, camera_data);
        if (ret < 0) {
            DF_LOG_ERROR << "Request error " << i;
        }

        //------------test printf------------//
        if (camera_data.module_name == SENSOR_0X8B40) {
            camera_interal_data_x3f_x8b *data = 
                    reinterpret_cast<camera_interal_data_x3f_x8b*>(camera_data.data.data());
            DF_LOG_INFO << FIXED << SET_PRECISION(6) << "Camera " <<  camera_data.sensor_id 
                << " fx " << data->fx << " fy " << data->fy << " cx " << data->cx << " cy : " << data->cy
                << " k1 " << data->k1 << " k2 " << data->k2 << " k3 " << data->k3 << " k4 : " << data->k4
                << " k5 " << data->k5 << " k6 " << data->k6 << " p1 " << data->p1 << " p2 : " << data->p2;
        } else if (camera_data.module_name == SENSOR_ISX021) {
            // test print
            camera_interal_data_x031 *data = 
                    reinterpret_cast<camera_interal_data_x031*>(camera_data.data.data());
            DF_LOG_INFO << FIXED << SET_PRECISION(6) << "Camera " <<  camera_data.sensor_id 
                << " fx " << data->fx << " fy " << data->fy << " cx " << data->cx << " cy : " << data->cy
                << " k1 " << data->k1 << " k2 " << data->k2 << " k3 " << data->k3 << " k4 : " << data->k4
                << " k5 " << data->k5 << " k6 " << data->k6 << " p1 " << data->p1 << " p2 : " << data->p2;
        } else if (camera_data.module_name == SENSOR_ISX031) {
            // test print
            camera_interal_data_x031 *data = 
                    reinterpret_cast<camera_interal_data_x031*>(camera_data.data.data());
            DF_LOG_INFO << FIXED << SET_PRECISION(6) << "Camera " <<  camera_data.sensor_id 
                << " fx " << data->fx << " fy " << data->fy << " cx " << data->cx << " cy : " << data->cy
                << " k1 " << data->k1 << " k2 " << data->k2 << " k3 " << data->k3 << " k4 : " << data->k4
                << " k5 " << data->k5 << " k6 " << data->k6 << " p1 " << data->p1 << " p2 : " << data->p2;
        }
    }

    camera_client.DeInit();

    return 0;
}
