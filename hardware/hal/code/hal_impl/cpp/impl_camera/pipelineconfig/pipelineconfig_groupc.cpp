#include "hal_camera_impl.h"

#if 0
static u32 _testframecount_yuv422 = 0;

static u32 _binitfile_yuv422 = 0;
static FILE* _pfiletest_yuv422;
#endif

#if 0
static void handle_yuv422_buffer_groupc(struct hw_video_bufferinfo_t* i_pbufferinfo)
{
#if 0
    if (HW_UNLIKELY(_binitfile_yuv422 == 0))
    {
        //string filename = "testfile_common" + std::to_string(_blockindex) + "_" + std::to_string(_sensorindex) + fileext;
        string fileext;
        switch (i_pbufferinfo->format_maintype)
        {
        case HW_VIDEO_BUFFERFORMAT_MAINTYPE_YUV422:
            switch (i_pbufferinfo->format_subtype)
            {
            case HW_VIDEO_BUFFERFORMAT_SUBTYPE_YUV422:
                fileext = ".yuv422";
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
        _pfiletest_yuv422 = fopen(filename.c_str(), "wb");
        if (!_pfiletest_yuv422) {
            printf("Failed to create output file\r\n");
            return;
        }
        _binitfile_yuv422 = 1;
    }
    /*
    * Assume that the data is yuv422 pl(the inner pipeline already change nvidia bl format to common pl format)
    */
    if (_testframecount_yuv422 < 3)
    {
        fwrite(i_pbufferinfo->pbuff, i_pbufferinfo->size, 1U, _pfiletest_yuv422);
        _testframecount_yuv422++;
    }
#endif
}
#endif

static struct hw_video_blockspipelineconfig_t _pipelineconfig_groupc =
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
                    .bcaptureoutputrequested = 1,
                    .bisp0outputrequested = 0,
                    .bisp1outputrequested = 0,
                    .bisp2outputrequested = 0,
                    .datacbsconfig =
                    {
                        .arraynumdatacbs = 0,
#if 0
                        /*
                        * The following is a sample.
                        */
                        .parraydatacbs =
                        {
                            [0] =
                            {
                                .bused = 1,
                                .type = HW_VIDEO_REGDATACB_TYPE_YUV422,
                                .cb = handle_yuv422_buffer_groupc,
                                .bsynccb = 1,
                                // set it when you need
                                .pcustom = nullptr,
                            },
                        },
#endif
                    },
                },
                [1] =
                {
                    .bused = 1,
                    .blockindex = 0,
                    .sensorindex = 1,
                    .bcaptureoutputrequested = 1,
                    .bisp0outputrequested = 0,
                    .bisp1outputrequested = 0,
                    .bisp2outputrequested = 0,
                    .datacbsconfig =
                    {
                        .arraynumdatacbs = 0,
#if 0
                        /*
                        * The following is a sample.
                        */
                        .parraydatacbs =
                        {
                            [0] =
                            {
                                .bused = 1,
                                .type = HW_VIDEO_REGDATACB_TYPE_YUV422,
                                .cb = handle_yuv422_buffer_groupc,
                                .bsynccb = 1,
                                // set it when you need
                                .pcustom = nullptr,
                            },
                        },
#endif
                    },
                },
                [2] =
                {
                    .bused = 1,
                    .blockindex = 0,
                    .sensorindex = 2,
                    .bcaptureoutputrequested = 1,
                    .bisp0outputrequested = 0,
                    .bisp1outputrequested = 0,
                    .bisp2outputrequested = 0,
                    .datacbsconfig =
                    {
                        .arraynumdatacbs = 0,
#if 0
                        /*
                        * The following is a sample.
                        */
                        .parraydatacbs =
                        {
                            [0] =
                            {
                                .bused = 1,
                                .type = HW_VIDEO_REGDATACB_TYPE_YUV422,
                                .cb = handle_yuv422_buffer_groupc,
                                .bsynccb = 1,
                                // set it when you need
                                .pcustom = nullptr,
                            },
                        },
#endif
                    },
                },
                [3] =
                {
                    .bused = 1,
                    .blockindex = 0,
                    .sensorindex = 3,
                    .bcaptureoutputrequested = 1,
                    .bisp0outputrequested = 0,
                    .bisp1outputrequested = 0,
                    .bisp2outputrequested = 0,
                    .datacbsconfig =
                    {
                        .arraynumdatacbs = 0,
#if 0
                        /*
                        * The following is a sample.
                        */
                        .parraydatacbs =
                        {
                            [0] =
                            {
                                .bused = 1,
                                .type = HW_VIDEO_REGDATACB_TYPE_YUV422,
                                .cb = handle_yuv422_buffer_groupc,
                                .bsynccb = 1,
                                // set it when you need
                                .pcustom = nullptr,
                            },
                        },
#endif
                    },
                },
            },
        },
    },
};

void Internal_SetPipelineConfig_ToDefault_Groupc(struct hw_video_blockspipelineconfig_t* i_ppipelineconfig)
{
    *i_ppipelineconfig = _pipelineconfig_groupc;
}
