#ifndef OTA_API_SAMPLE
#define OTA_API_SAMPLE

#include <string>

#include "ota_comp_api/include/ota_api.h"


class OTAApiSample {
public:
    OTAApiSample();
    virtual ~OTAApiSample();

    void Init();
    void DeInit();
    int Run(std::string upgrade_file_path);

private:
    OTAApiSample(const OTAApiSample &);
    OTAApiSample & operator = (const OTAApiSample &);

private:
    uint8_t stop_flag_;
};

#endif  // OTA_API_SAMPLE
