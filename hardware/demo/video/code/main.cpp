#include "hardware.h"
#include <iostream>
#include <cstring>
#include <iostream>
#include <csignal>
#include <thread>
#include <chrono>
#include <ctime>
#include <atomic>
#include <cmath>
#include <fstream>
#include <vector>
#include <memory>
using namespace std;

#define TEST_VIDEO_MODULE_NAME      "libhw_nvmedia_groupa_vs.so"

#define SECONDS_PER_ITERATION			1

static void thread_outputpipeline_handleframe(struct hw_video_outputpipeline_ops_t* i_poutputpipeline_ops)
{
    s32 ret;
    hw_video_handleframe handleframe;
    HW_VIDEO_FRAMERETSTATUS frameretstatus;

    uint64_t uFrameCountDelta = 0u;
    uint64_t uTimeElapsedSum = 0u;

    u64 prevframecount = 0;
    u64 currframecount;
    u64 timesumcurr = 0;

    auto oStartTime = chrono::steady_clock::now();
    auto timecurr = chrono::steady_clock::now();

    while (1)
    {
        ret = i_poutputpipeline_ops->gethandleframe(i_poutputpipeline_ops, &handleframe, HW_TIMEOUT_US_DEFAULT, &frameretstatus);
        if (ret == 0)
        {
            // receive notification, handle it
            if (frameretstatus == HW_VIDEO_FRAMERETSTATUS_QUIT)
            {
                printf("Frame thread quit by restatus[quit]...\r\n");
                break;
            }
            else if (frameretstatus == HW_VIDEO_FRAMERETSTATUS_TIMEOUT)
            {
                printf("Frame receive timeout\r\n");
            }
            else if (frameretstatus == HW_VIDEO_FRAMERETSTATUS_GET)
            {
                ret = i_poutputpipeline_ops->handle(i_poutputpipeline_ops, handleframe);
                if (ret != 0)
                {
                    printf("Frame handle fail!\r\n");
                    break;
                }
            }
            else
            {
                printf("Sensor Unexpected notifretstatus value[%u]\r\n", frameretstatus);
            }
        }
        else
        {
            printf("Sensor Unexpected ret value[0x%x]\r\n", ret);
        }

        /*
        * Output the frame count every 2 second.
        */
        // Wait for SECONDS_PER_ITERATION
        timecurr = chrono::steady_clock::now();
        auto uTimeElapsedMs = chrono::duration<double, std::milli>(timecurr - oStartTime).count();
        oStartTime = timecurr;
        uTimeElapsedSum += (u64)uTimeElapsedMs;
        timesumcurr += (u64)uTimeElapsedMs;

        if (timesumcurr > SECONDS_PER_ITERATION * 1000) {
            if (i_poutputpipeline_ops->frame_ops.getframecount(i_poutputpipeline_ops, &currframecount) != 0) {
                printf("getframecount fail!\r\n");
                break;
            }
            uFrameCountDelta = currframecount - prevframecount;
            prevframecount = currframecount;
            auto fps = (double)uFrameCountDelta / ((double)timesumcurr / 1000.0);
            string profName = "Sensor block[" + to_string(i_poutputpipeline_ops->blockindex) +
                "]sensor[" + to_string(i_poutputpipeline_ops->sensorindex) + "]_Out"
                + to_string(i_poutputpipeline_ops->outputtype) + "\t";
            cout << profName << "Frame rate (fps):\t\t" << fps << ", frame:" << uFrameCountDelta << ", time:" << (double)((double)timesumcurr / 1000.0) << "s" << endl;

            timesumcurr = 0;
        }
    }
}

static void thread_sensorpipeline_notification(struct hw_video_sensorpipeline_ops_t* i_psensorpipeline_ops)
{
    s32 ret;
    struct hw_video_notification_t blocknotif;
    HW_VIDEO_NOTIFRETSTATUS notifretstatus;
    while (1)
    {
        ret = i_psensorpipeline_ops->getnotification(i_psensorpipeline_ops, &blocknotif, 1, HW_TIMEOUT_US_DEFAULT, &notifretstatus);
        if (ret == 0)
        {
            // receive notification, handle it
            if (notifretstatus == HW_VIDEO_NOTIFRETSTATUS_QUIT)
            {
                printf("Sensor notif thread quit by restatus[quit]...\r\n");
                break;
            }
            else if (notifretstatus == HW_VIDEO_NOTIFRETSTATUS_TIMEOUT)
            {
                printf("Sensor notif receive timeout\r\n");
            }
            else if (notifretstatus == HW_VIDEO_NOTIFRETSTATUS_GET)
            {
                printf("Sensor notif: %u\r\n", blocknotif.notiftype);
            }
            else
            {
                printf("Sensor Unexpected notifretstatus value[%u]\r\n", notifretstatus);
            }
        }
        else
        {
            printf("Sensor Unexpected ret value[0x%x]\r\n", ret);

        }
    }
}

static void thread_blockpipeline_notification(struct hw_video_blockpipeline_ops_t* i_pblockpipeline_ops)
{
    s32 ret;
    struct hw_video_notification_t blocknotif;
    HW_VIDEO_NOTIFRETSTATUS notifretstatus;
    while (1)
    {
        ret = i_pblockpipeline_ops->getnotification(i_pblockpipeline_ops, &blocknotif, 1, HW_TIMEOUT_US_DEFAULT, &notifretstatus);
        if (ret == 0)
        {
            // receive notification, handle it
            if (notifretstatus == HW_VIDEO_NOTIFRETSTATUS_QUIT)
            {
                printf("Block notif thread quit by restatus[quit]...\r\n");
                break;
            }
            else if (notifretstatus == HW_VIDEO_NOTIFRETSTATUS_TIMEOUT)
            {
                printf("Block notif receive timeout\r\n");
            }
            else if (notifretstatus == HW_VIDEO_NOTIFRETSTATUS_GET)
            {
                printf("Block notif: %u\r\n", blocknotif.notiftype);
            }
            else
            {
                printf("Block Unexpected notifretstatus value[%u]\r\n", notifretstatus);
            }
        }
        else
        {
            printf("Block Unexpected ret value[0x%x]\r\n", ret);

        }
    }
}

static u32 _testframecount_yuv420 = 0;

static u32 _binitfile_yuv420 = 0;
static FILE* _pfiletest_yuv420;

static void handle_yuv420_buffer(struct hw_video_bufferinfo_t* i_pbufferinfo)
{
    if (HW_UNLIKELY(_binitfile_yuv420 == 0))
    {
        //string filename = "testfile_common" + std::to_string(_blockindex) + "_" + std::to_string(_sensorindex) + fileext;
        string fileext;
        switch (i_pbufferinfo->format_maintype)
        {
        case HW_VIDEO_BUFFERFORMAT_MAINTYPE_YUV420:
            switch (i_pbufferinfo->format_subtype)
            {
            case HW_VIDEO_BUFFERFORMAT_SUBTYPE_YUV420_PL:
                fileext = ".yuv420pl";
                break;
            case HW_VIDEO_BUFFERFORMAT_SUBTYPE_YUV420_BL:
                fileext = ".yuv420bl";
                break;
            default:
                printf("Unexpected sub type!\r\n");
                return;
            }
            break;
        default:
            printf("Unexpected main type!\r\n");
            return;
        }
        string filename = "testfile" + fileext;
        remove(filename.c_str());
        _pfiletest_yuv420 = fopen(filename.c_str(), "wb");
        if (!_pfiletest_yuv420) {
            printf("Failed to create output file\r\n");
            return;
        }
        _binitfile_yuv420 = 1;
    }
    /*
    * Assume that the data is yuv420 pl(the inner pipeline already change nvidia bl format to common pl format)
    */
    if (_testframecount_yuv420 < 3)
    {
        fwrite(i_pbufferinfo->pbuff, i_pbufferinfo->size, 1U, _pfiletest_yuv420);
        _testframecount_yuv420++;
    }
}


static u32 _testframecount_rgb = 0;

static u32 _binitfile_rgb = 0;
static FILE* _pfiletest_rgb;

static void handle_raw12_buffer(struct hw_video_bufferinfo_t* i_pbufferinfo)
{
    if (HW_UNLIKELY(_binitfile_rgb == 0))
    {
        //string filename = "testfile_common" + std::to_string(_blockindex) + "_" + std::to_string(_sensorindex) + fileext;
        string fileext;
        switch (i_pbufferinfo->format_maintype)
        {
        case HW_VIDEO_BUFFERFORMAT_SUBTYPE_RAW12:
            fileext = ".raw12";
            break;
        default:
            printf("Unexpected main type!\r\n");
            return;
        }
        string filename = "testfile" + fileext;
        remove(filename.c_str());
        _pfiletest_rgb = fopen(filename.c_str(), "wb");
        if (!_pfiletest_rgb) {
            printf("Failed to create output file\r\n");
            return;
        }
        _binitfile_rgb = 1;
    }
    /*
    * Assume that the data is rgb pl(the inner pipeline already change nvidia bl format to common pl format)
    */
    if (_testframecount_rgb < 3)
    {
        fwrite(i_pbufferinfo->pbuff, i_pbufferinfo->size, 1U, _pfiletest_rgb);
        _testframecount_rgb++;
    }
}

int main(int argc, char* argv[])
{
    s32 ret;
    struct hw_module_t* pmodule;
    struct hw_video_t* pvideo;

    /*
    * Register default sig handler.
    */
    hw_plat_regsighandler_default();

#if 0
    void* handle = NULL;
    struct hw_module_t* hmi = NULL;
    const char* sym = HAL_MODULE_INFO_SYM_AS_STR;
    int i;
    for (i = 0; i < 2; i++)
    {
        handle = dlopen(TEST_VIDEO_MODULE_NAME, RTLD_NOW);
        hmi = (struct hw_module_t*)dlsym(handle, sym);
        void* pvoid;
        hmi->privapi.init(&pvoid);
        dlclose(handle);
    }
    return 1;
#else

    ret = hw_module_get(TEST_VIDEO_MODULE_NAME, &pmodule);
    if (ret < 0) {
        printf("hw_module_get fail!\n");
        return ret;
        //goto label_fail_ret;
    }
    ret = hw_module_device_get(pmodule, NULL, (hw_device_t**)&pvideo);
    if (ret < 0) {
        printf("hw_module_device_get fail!\n");
        return ret;
        //goto label_fail_hw_module_put;
    }

    u32 counti;
    struct hw_video_info_t videoinfo;
    struct hw_video_blockspipelineconfig_t pipelineconfig =
    {
        .parrayblock =
        {
            [0] =
            {
                .bused = 1,
                .blockindex = 0,
                .parraysensor =
                {
                    [0] =
                    {
                        .bused = 1,
                        .blockindex = 0,
                        .sensorindex = 0,
                        .bcaptureoutputrequested = 0,
                        .bisp0outputrequested = 1,
                        .bisp1outputrequested = 0,
                        .bisp2outputrequested = 0,
                        .datacbsconfig =
                        {
                            .arraynumdatacbs = 2,
                            .parraydatacbs =
                            {
                                [0] =
                                {
                                    .bused = 1,
                                    .type = HW_VIDEO_REGDATACB_TYPE_YUV420,
                                    .cb = handle_yuv420_buffer,
                                    .bsynccb = 1,
                                    // set it when you need
                                    .pcustom = nullptr,
                                },
                                [1] =
                                {
                                    .bused = 0,
                                    .type = HW_VIDEO_REGDATACB_TYPE_RAW12,
                                    .cb = handle_raw12_buffer,
                                    .bsynccb = 1,
                                    // set it when you need
                                    .pcustom = nullptr,
                                },
                            },
                        },
                    },
                },
            },
        },
    };
    struct hw_video_blockspipeline_ops_t* pblockspipeline_ops;
    struct hw_video_blockpipeline_ops_t* pblockpipeline_ops;
    struct hw_video_sensorpipeline_ops_t* psensorpipeline_ops;
    struct hw_video_outputpipeline_ops_t* poutputpipeline_ops;
    hw_video_handlepipeline handlepipeline;
    u32 blocki, numblocks, sensori, numsensors, outputi;
    std::vector<std::unique_ptr<std::thread>> vthreadpipelinenotif, vthreadblocknotif, vthreadoutput;

#if 1

    for (counti = 0; counti < 3; counti++)
    {
        ret = pvideo->ops.device_open(pvideo, &videoinfo);
        if (ret < 0) {
            printf("device_open fail!\r\n");
            return ret;
        }
        printf("device_open success!\r\n");
        numblocks = videoinfo.numblocks;
        ret = pvideo->ops.pipeline_open(pvideo, &handlepipeline, &pipelineconfig, &pblockspipeline_ops);
        if (ret < 0) {
            printf("pipeline_open fail!\r\n");
            return ret;
        }
        printf("pipeline_open success!\r\n");
#if 1
        for (blocki = 0; blocki < numblocks; blocki++)
        {
            pblockpipeline_ops = &pblockspipeline_ops->parrayblock[blocki];
            if (pblockpipeline_ops->bused)
            {
                vthreadblocknotif.push_back(std::make_unique<std::thread>(thread_blockpipeline_notification,
                    pblockpipeline_ops));
                numsensors = pblockpipeline_ops->numsensors;
                for (sensori = 0; sensori < numsensors; sensori++)
                {
                    psensorpipeline_ops = &pblockpipeline_ops->parraysensor[sensori];
                    if (psensorpipeline_ops->bused)
                    {
                        vthreadpipelinenotif.push_back(std::make_unique<std::thread>(thread_sensorpipeline_notification,
                            psensorpipeline_ops));
                        for (outputi = 0; outputi <= HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_MAX; outputi++)
                        {
                            poutputpipeline_ops = &psensorpipeline_ops->parrayoutput[outputi];
                            if (poutputpipeline_ops->bused)
                            {
                                vthreadoutput.push_back(std::make_unique<std::thread>(thread_outputpipeline_handleframe,
                                    poutputpipeline_ops));
                            }
                        }
                    }
                }
            }
        }
        ret = pvideo->ops.pipeline_start(pvideo, handlepipeline);
        if (ret < 0) {
            printf("pipeline_start fail!\r\n");
            return ret;
        }
        printf("sleep 3 seconds.\r\n");
        sleep(3);
        ret = pvideo->ops.pipeline_stop(pvideo, handlepipeline);
        if (ret < 0) {
            printf("pipeline_stop fail!\r\n");
            return ret;
        }
        printf("pipeline_stop success!\r\n");
        for (auto& upthread : vthreadpipelinenotif) {
            if (upthread != nullptr) {
                upthread->join();
                upthread.reset();
            }
        }
        for (auto& upthread : vthreadblocknotif) {
            if (upthread != nullptr) {
                upthread->join();
                upthread.reset();
            }
        }
        for (auto& upthread : vthreadoutput) {
            if (upthread != nullptr) {
                upthread->join();
                upthread.reset();
            }
        }
#endif
        ret = pvideo->ops.pipeline_close(pvideo, handlepipeline);
        if (ret < 0) {
            printf("pipeline_close fail!\r\n");
            return ret;
        }
        printf("pipeline_close success!\r\n");
        ret = pvideo->ops.device_close(pvideo);
        if (ret < 0) {
            printf("device_close fail!\r\n");
            return ret;
        }
        printf("device_close success!\r\n");
}
    ret = pmodule->privapi.device_put(pmodule, (struct hw_device_t*)pvideo);
    if (ret < 0) {
        printf("device_put fail!\r\n");
        return ret;
    }
    printf("device_put success!\r\n");
    return 0;

#else
    ret = pvideo->ops.device_open(pvideo, NULL);
    if (ret < 0) {
        printf("device_open fail!\n");
        goto label_fail_hw_module_device_put;
    }



    pvideo->ops.device_close(pvideo);
    hw_module_device_put(pmodule, (hw_device_t*)pvideo);
    hw_module_put(pmodule);
    return 0;

label_fail_hw_module_device_put:
    hw_module_device_put(pmodule, (hw_device_t*)pvideo);
label_fail_hw_module_put:
    hw_module_put(pmodule);
label_fail_ret:
    return -1;

#endif

#endif




#if 0
    struct hw_module_t* pmod_nvmedia;
    struct hw_video_t* pvideo;
    hw_module_get(TEST_VIDEO_MODULE_NAME, &pmod_nvmedia);
#endif

#if 0
    void* h;
    h = dlopen("libhw_nvmedia_imx728_vs.so", RTLD_LAZY);
    if (!h)
    {
        printf("not found!\n");
        return 0;
    }
    test_func tf;
    tf = (test_func)dlsym(h, "test");
    char* error;
    if ((error = dlerror()) != NULL) {
        fprintf(stderr, "%s\n", error);
        return 0;
    }
    tf();
    dlclose(h);
    return 0;
#else

#if 1
#if 0
    test();

    TestClass::test();
    ITest* ptest = TestClass::getitest();
    ptest->output();

    if (HW_LIKELY(ptest != NULL))
    {
        printf("run end\n");
    }

    printf("run end\n");
    return 0;
#endif
#else
    printf("hello\n");

    pthread_setname_np(pthread_self(), "Main");

    bQuit = false;


    printf("Checking SIPL version\n");
    NvSIPLVersion oVer;
    NvSIPLGetVersion(oVer);

    printf("NvSIPL library version: %u.%u.%u\n", oVer.uMajor, oVer.uMinor, oVer.uPatch);
    printf("NVSIPL header version: %u %u %u\n", NVSIPL_MAJOR_VER, NVSIPL_MINOR_VER, NVSIPL_PATCH_VER);
    if (oVer.uMajor != NVSIPL_MAJOR_VER || oVer.uMinor != NVSIPL_MINOR_VER || oVer.uPatch != NVSIPL_PATCH_VER) {
        printf("NvSIPL library and header version mismatch\n");
    }
    
    // INvSIPLQuery
    auto pQuery = INvSIPLQuery::GetInstance();
    CHK_PTR_AND_RETURN(pQuery, "INvSIPLQuery::GetInstance");

    SIPLStatus status = pQuery->ParseDatabase();
    CHK_STATUS_AND_RETURN(status, "INvSIPLQuery::ParseDatabase");

    return 0;
#endif


#endif
}
