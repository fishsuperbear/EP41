#pragma once

#include <iostream>
#include <thread>

#include "NvSIPLCommon.hpp"
#include "NvSIPLCamera.hpp"
#include "NvSIPLClient.hpp"
#include "NvSIPLPipelineMgr.hpp"

#include "cam_logger.hpp"

using namespace nvsipl;

namespace hozon {
namespace netaos {
namespace camera {

/**
 * @brief A utility class for reading items from a queue and invoking a callback.
 *
 * QUEUE_TYPE is the type of the queue being read.
 * PAYLOAD_TYPE is the type of items within that queue.
 */
template <typename QUEUE_TYPE, typename PAYLOAD_TYPE>
class QueueHandler final
{
public:

    /**
     * The interface used for callbacks to the client.
     * Each call to process() returns a single item read from the queue.
     */
    class ICallback
    {
    public:
        virtual void process(const PAYLOAD_TYPE& data) = 0;
    protected:
        ICallback() = default;
        virtual ~ICallback() = default;
    };

    QueueHandler() = default;
    ~QueueHandler() = default;

    /**
     * Begin reading from the queue.
     *
     * @param[in] callback The callback function to call every time an item is read from the queue.
     * @param[in] timeoutInUsec The timeout (in microseconds) for reading from the queue.
     *
     * @retval NVSIPL_STATUS_INVALID_STATE if this handler has ever been started.
     * @retval NVSIPL_STATUS_BAD_ARGUMENT if the callback or queue is null.
     * @retval NVSIPL_STATUS_OK if the object has started successfully reading from the queue.
     */
    SIPLStatus Start(QUEUE_TYPE* queue, ICallback* callback, size_t timeoutInUsec)
    {
        if (m_started) {
            CAM_LOG_ERROR << "Handler is already started\n";
            return NVSIPL_STATUS_INVALID_STATE;
        }
        if (m_stopThread) {
            CAM_LOG_ERROR << "Handler has already been stopped; restart is not supported\n";
            return NVSIPL_STATUS_INVALID_STATE;
        }
        if (callback == nullptr || queue == nullptr) {
            CAM_LOG_DEBUG << "Null callback or queue provided\n";
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }

        m_queue = queue;
        m_callback = callback;
        m_timeoutInUsec = timeoutInUsec;
        m_thread = std::thread(StaticThreadFunc, this);

        m_started = true;
        return NVSIPL_STATUS_OK;
    }

    /**
     * Stop reading from the queue.
     * After this call, this object cannot be used again.
     *
     * @retval NVSIPL_STATUS_INVALID_STATE if this handler has not been started.
     * @retval NVSIPL_STATUS_OK if the handler was successfully stopped.
     */
    SIPLStatus Stop()
    {
        if (!m_started) {
            CAM_LOG_ERROR << "Handler is not started\n";
            return NVSIPL_STATUS_INVALID_STATE;
        }

        m_stopThread = true;
        m_thread.join();

        m_started = false;
        return NVSIPL_STATUS_OK;
    }

    /**
     * Is the handler currently active?
     */
    bool IsRunning() const
    {
        return m_started;
    }

    /**
     * Execute the loop for reading items from the queue and making callbacks.
     */
    void RunThread()
    {
        SIPLStatus status = NVSIPL_STATUS_OK;
        PAYLOAD_TYPE payload;

        while (!m_stopThread) {
            status = m_queue->Get(payload, m_timeoutInUsec);
            if (status == NVSIPL_STATUS_OK) {
                m_callback->process(payload);
            } else if (status == NVSIPL_STATUS_TIMED_OUT) {
                CAM_LOG_INFO << "Queue timeout";
            } else if (status == NVSIPL_STATUS_EOF) {
                m_stopThread = true;
            } else {
                CAM_LOG_ERROR << "Unexpected queue return status\n";
                m_stopThread = true;
            }
        }

        CAM_LOG_INFO << "Queue handler thread completion.";
    }

    /**
     * Wrap thread execution so that it can be specified as a method on this object
     * (RunThread) instead of putting all that logic into a static method.
     */
    static void StaticThreadFunc(QueueHandler<QUEUE_TYPE, PAYLOAD_TYPE>* pThis)
    {
        pThis->RunThread();
    }

private:

    QUEUE_TYPE* m_queue {nullptr};      ///< The queue to read from.
    ICallback* m_callback {nullptr};    ///< The callback to invoke for every item read.
    size_t m_timeoutInUsec {0ULL};      ///< The queue reading timeout.
    std::thread m_thread;               ///< The thread that reads from the queue.
    bool m_started {false};             ///< Has this handler been started?
    bool m_stopThread {false};          ///< If true, requests the thread to stop executing.
};


using NotificationQueueHandler =
        QueueHandler<INvSIPLNotificationQueue, NvSIPLPipelineNotifier::NotificationData>;

using FrameCompleteQueueHandler =
        QueueHandler<INvSIPLFrameCompletionQueue, INvSIPLClient::INvSIPLBuffer*>;

}
}
}
