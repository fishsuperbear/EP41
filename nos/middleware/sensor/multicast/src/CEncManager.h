#ifndef CENCMANAGER_H
#define CENCMANAGER_H

#include "include/multicast.h"

class CEncManager
{
public:
    static CEncManager& Instance();
    static void Destroy();

    void NotifyEncodedImage(int32_t sensor_id, hozon::netaos::multicast::Multicast_EncodedImage& encoded_image);
    void SetSensorImageCbMap(std::map<int, hozon::netaos::multicast::IEPConsumerCbs> sensor_enc_cbs_map);
private:
    CEncManager();
    ~CEncManager();

    std::map<int, hozon::netaos::multicast::IEPConsumerCbs> sensor_enc_cbs_map_;
};

#endif
