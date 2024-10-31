#pragma once
// Standard header files
#include <memory>
#include <unordered_map>

// SIPL header files
#include "NvSIPLCommon.hpp"
#include "NvSIPLCamera.hpp"

// Sample application header files
#include "cam_utils.hpp"

// Other NVIDIA header files
#include "nvscibuf.h"
#include "nvmedia_6x/nvmedia_core.h"

using namespace nvsipl;

namespace hozon {
namespace netaos {
namespace camera {

#define MAX_NUM_IMAGE_OUTPUTS (4U)

class CImageManager
{
public:
    //! Initialize: allocates image groups and images and registers them with SIPL.
    SIPLStatus Init(INvSIPLCamera *siplCamera,
                    const std::unordered_map<uint32_t, NvSIPLPipelineConfiguration> &PieplineInfo);

    //! Deinitialize: deallocates image groups and images.
    void Deinit();

    //! Destructor: calls Deinit.
    ~CImageManager();

    SIPLStatus Allocate(uint32_t sensorId);
    SIPLStatus Register(uint32_t sensorId);
    SIPLStatus GetBuffers(uint32_t uSensorId, INvSIPLClient::ConsumerDesc::OutputType outputType, std::vector<NvSciBufObj> &buffers);
    SIPLStatus SetNvSciSyncCPUWaiter(uint32_t pip, bool isp0Enabled, bool isp1Enabled, bool isp2Enabled);

private:
    typedef struct {
        bool enable;
        size_t size;
        INvSIPLClient::ConsumerDesc::OutputType outputType;
        std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> attrList;
        std::vector<NvSciBufObj> sciBufObjs;
    } ImagePool;

    //! Allocates buffers to be used for either capture or processing.
    SIPLStatus AllocateBuffers(ImagePool &imagePool);

    void PrintISPOutputFormat(uint32_t pip,
                            INvSIPLClient::ConsumerDesc::OutputType outputType,
                            NvSciBufAttrList attrlist);

    INvSIPLCamera *m_siplCamera = nullptr;
    NvSIPLPipelineConfiguration m_pipelineCfg;
    NvSciBufModule m_sciBufModule = nullptr;
    NvSciSyncModule m_sciSyncModule = nullptr;
    ImagePool m_imagePools[MAX_NUM_SENSORS][MAX_NUM_IMAGE_OUTPUTS];
};

}
}
}
