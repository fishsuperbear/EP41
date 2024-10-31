#include "hal_camera_impl.h"

/*
 * We set the skip gap of time tsc.
 * It seams to be us unit.
 * One second correspondent to value 100*1000000.
 */
#define SKIPGAP_TSC_DEFAULT     (60000000ULL)
#define CAMERA_DEVICE_OUTPUTPIPELINE_SECONDS_PER_ITERATION          1
static u32 _hastscfirstset[4][4] = {0};
static u64 _tscfirst[4][4];
static u32 _donotneedtoskip[4][4] = {0};


void Internal_ThreadRoutine_OutputPipeline_Default(struct hw_video_outputpipeline_ops_t* i_poutputpipeline_ops, void* i_pcontext,
    u32 i_outputtype)
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

    struct hw_video_buffertimeinfo_t timeinfo;
    u64 gaptsc;

    u32 blockindex = i_poutputpipeline_ops->blockindex;
    u32 sensorindex = i_poutputpipeline_ops->sensorindex;


    while (1)
    {
        ret = i_poutputpipeline_ops->gethandleframe(i_poutputpipeline_ops, &handleframe, HW_TIMEOUT_US_DEFAULT, &frameretstatus);
        if (ret == 0)
        {
            // receive notification, handle it
            if (frameretstatus == HW_VIDEO_FRAMERETSTATUS_QUIT)
            {
                HAL_CAMERA_LOG_UNMASK("Frame thread quit by restatus[quit]...\r\n");
                break;
            }
            else if (frameretstatus == HW_VIDEO_FRAMERETSTATUS_TIMEOUT)
            {
                HAL_CAMERA_LOG_DEBUG("Frame receive timeout\r\n");
            }
            else if (frameretstatus == HW_VIDEO_FRAMERETSTATUS_GET)
            {
                if (_donotneedtoskip[blockindex][sensorindex] != 1) {
                    i_poutputpipeline_ops->getframetimeinfo(i_poutputpipeline_ops, handleframe, &timeinfo);
                    if (_hastscfirstset[blockindex][sensorindex] == 0) {
                        _tscfirst[blockindex][sensorindex] = timeinfo.framecapturetsc;
                        _hastscfirstset[blockindex][sensorindex] = 1;
                    }
                    gaptsc = timeinfo.framecapturetsc - _tscfirst[blockindex][sensorindex];
                    if (gaptsc < SKIPGAP_TSC_DEFAULT) {
                        // skip the frame
                        i_poutputpipeline_ops->skiphandle(i_poutputpipeline_ops, handleframe);
                        //printf("skip continue\r\n");
                        continue;
                    } else {
                        _donotneedtoskip[blockindex][sensorindex] = 1;
                    }
                }
                ret = i_poutputpipeline_ops->handle(i_poutputpipeline_ops, handleframe);
                if (ret != 0)
                {
                    HAL_CAMERA_LOG_ERR("Frame handle fail!\r\n");
                    break;
                }
            }
            else
            {
                HAL_CAMERA_LOG_ERR("Sensor Unexpected notifretstatus value[%u]\r\n", frameretstatus);
            }
        }
        else
        {
            HAL_CAMERA_LOG_ERR("Sensor Unexpected ret value[0x%x]\r\n", ret);
        }

        /*
        * Output the frame count every 2 second.
        */
        // Wait for CAMERA_DEVICE_OUTPUTPIPELINE_SECONDS_PER_ITERATION
        timecurr = chrono::steady_clock::now();
        auto uTimeElapsedMs = chrono::duration<double, std::milli>(timecurr - oStartTime).count();
        oStartTime = timecurr;
        uTimeElapsedSum += (u64)uTimeElapsedMs;
        timesumcurr += (u64)uTimeElapsedMs;

        if (timesumcurr > CAMERA_DEVICE_OUTPUTPIPELINE_SECONDS_PER_ITERATION * 1000) {
            if (i_poutputpipeline_ops->frame_ops.getframecount(i_poutputpipeline_ops, &currframecount) != 0) {
                HAL_CAMERA_LOG_ERR("getframecount fail!\r\n");
                break;
            }
            uFrameCountDelta = currframecount - prevframecount;
            prevframecount = currframecount;
            double fps = (double)uFrameCountDelta / ((double)timesumcurr / 1000.0);
            string profName = "Sensor block[" + to_string(i_poutputpipeline_ops->blockindex) +
                "]sensor[" + to_string(i_poutputpipeline_ops->sensorindex) + "]_Out"
                + to_string(i_poutputpipeline_ops->outputtype) + "\t";

            HAL_CAMERA_LOG_INFO("%sFrame rate (fps):\t\t%f, frame:%llx, time:%fs\r\n", profName.c_str(), fps, uFrameCountDelta, (double)((double)timesumcurr / 1000.0));
            cout << profName << "Frame rate (fps):\t\t" << fps << ", frame:" << uFrameCountDelta << ", time:" << (double)((double)timesumcurr / 1000.0) << "s" << endl;

            timesumcurr = 0;
        }
    }
}
