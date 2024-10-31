#include "adf-lite/include/executor_mgr.h"
#include "adf-lite/include/adf_lite_internal_logger.h"
#include "adf/include/time_monitor.h"
#include "adf-lite/include/dbg_info.h"
#include "adf-lite/include/topic_manager.h"

using namespace hozon::netaos::adf;

namespace hozon {
namespace netaos {
namespace adf_lite {

ExecutorMgr::ExecutorMgr() {

}

ExecutorMgr::~ExecutorMgr() {

}

int32_t ExecutorMgr::InitConfig(const std::string& config_file) {
    DO_OR_ERROR(_config.Parse(config_file), "Fail to parse executor config file " << config_file);
    _config_file_path = config_file;

    // DO(InitLogger());
    DO(LoadExecutor());
    DO(InitScheduler());
    DO(InitThreadPool());
    DO(InitCommunication());
    return 0;
}

int32_t ExecutorMgr::Init() {
    int32_t ret = -1;
    DO_OR_ERROR(ret = _executor->AlgInit(), "Fail to init algorithm, ret " << ret);
    _need_stop = false;
    for (std::size_t i = 0; i < _config.triggers.size(); ++i) {
        _trigger_control_map[_config.triggers[i].name].pause_enable = false;
    }

    ADF_EXEC_LOG_INFO << "Succ to init executor " << _config.executor_name;

    return 0;
}

void ExecutorMgr::StartProcess() {
    for (auto& trigger : _config.triggers) {
        if (trigger.type == ExecutorConfig::Trigger::Type::FREE) {
            ADF_EXEC_LOG_INFO << "Bypass FREE trigger " << trigger.name;
            continue;
        }
        else if (!_executor->GetProcessFunc(trigger.name)) {
            ADF_EXEC_LOG_WARN << "Missing process function of " << trigger.name;
            if (!_executor->GetProcessWithProfilerFunc(trigger.name)) {
                ADF_EXEC_LOG_WARN << "Missing GetProcessWithProfilerFunc function of " << trigger.name;
                continue;
            }
        }
        for (auto& main_source : trigger.main_sources) {
            //将ExecutorMgr和trigger名称，加到map的main_source.name中
            TopicManager::GetInstance().AddTrigger(main_source.name, _config.executor_name, trigger.name);
        }

        _trigger_control_map[trigger.name].process_thread = 
                std::make_shared<std::thread>(std::bind(&ExecutorMgr::Routine, this, trigger));
        ADF_EXEC_LOG_INFO << "Created process thread of trigger " << trigger.name;
    }
    // TopicManager::GetInstance().PrintTopic();
}

void ExecutorMgr::PreStop() {
    ADF_EXEC_LOG_INFO << "Executor " << _config.executor_name << " going to prestop.";

    // join trigger routine threads
    _need_stop = true;
    // MonitorReader::GetInstance().Stop();
    // for (auto& proxy : _recv_instances_map) {
    //     proxy.second._recv_base->Stop();
    // }

    for (auto& ele : _trigger_control_map) {
        ADF_EXEC_LOG_INFO << "Executor " << _config.executor_name << " trigger: " << ele.first << " start going to stop.";
        ele.second.resume_cv.notify_all();
        ele.second.pause_ready_cv.notify_all();
        ele.second.align_data_arrived_cv.notify_all();
    }

    _executor->AlgPreRelease();
}

void ExecutorMgr::Stop() {
    ADF_EXEC_LOG_INFO << "Executor " << _config.executor_name << " going to stop.";

    for (auto& ele : _trigger_control_map) {
        ADF_EXEC_LOG_INFO << "Executor " << _config.executor_name << " trigger: " << ele.first << " going to stop.";
        for (auto& th : ele.second.align_recv_threads) {
            th->join();
        }

        ADF_EXEC_LOG_INFO << "Executor " << _config.executor_name << " trigger: " << ele.first << " process_thread going to stop.";
        if (ele.second.process_thread) {
            ele.second.process_thread->join();
            ele.second.process_thread = nullptr;
        }
        ADF_EXEC_LOG_INFO << "Executor " << _config.executor_name << " trigger: " << ele.first << " succ to stop.";
    }


    ADF_EXEC_LOG_INFO << "Executor " << _config.executor_name << " thread pool going to stop.";
    if (_data_recv_thread_pool != nullptr) {
        _data_recv_thread_pool->Stop();
    }

    _executor->AlgRelease();
    ADF_EXEC_LOG_INFO << "Executor " << _config.executor_name << " succ to stop.";
    // _executor.reset();
    // _executor_loader.Unload();
    ADF_EXEC_LOG_INFO << "Executor library: " << _config.library << " Unload";
}

int32_t ExecutorMgr::LoadExecutor() {
    char* curr_ld_library_path = getenv("LD_LIBRARY_PATH");
    std::string new_ld_library_path;
    if (curr_ld_library_path) {
        new_ld_library_path = std::string(curr_ld_library_path);
    }

    /*for (auto& dep_lib_path : _config.dep_lib_path) {
        new_ld_library_path = new_ld_library_path +  ":" + dep_lib_path;
    }

    ADF_EXEC_LOG_DEBUG << "Set LD_LIBRARY_PATH to " << new_ld_library_path;
    if (setenv("LD_LIBRARY_PATH", new_ld_library_path.c_str(), 1) < 0) {
        ADF_EXEC_LOG_ERROR << "Fail to set LD_LIBRARY_PATH";
        return -1;
    }*/

    DO_OR_ERROR(_executor_loader.Load(_config.library), "Fail to load library " << _config.library);
    _executor = g_class_loader.Create(_config.executor_name);
    if (_executor == nullptr) {
        ADF_EXEC_LOG_ERROR << "Fail to create executor " << _config.executor_name << " in " << _config.library;
        return -1;
    }

    ADF_EXEC_LOG_DEBUG << "Succ to create executor in " << _config_file_path << " of name " << _config.executor_name;
    _executor->SetConfig(&_config);
    _executor->SetConfigFilePath(_config_file_path);
    _executor->RegistPauseTriggerCb(std::bind(&ExecutorMgr::PauseTrigger, this, std::placeholders::_1, std::placeholders::_2));
    
    return 0;
}

int32_t ExecutorMgr::InitScheduler() {
    sched_param param;
    if (!_config.schedule.HasValue()) {
        return 0;
    }

    if (_config.schedule.Value().policy == SCHED_OTHER) {
        param.sched_priority = 0;
    }
    else if ((_config.schedule.Value().policy == SCHED_FIFO)
        || (_config.schedule.Value().policy == SCHED_RR)) {
        param.sched_priority = _config.schedule.Value().priority;
    }
    else {
        ADF_EXEC_LOG_ERROR << "Unsupported scheduler.";
        return -1;
    }

    DO_OR_ERROR(sched_setscheduler(0, _config.schedule.Value().policy, &param), "Fail to set scheduler.");
    DO_OR_ERROR(sched_setaffinity(0, sizeof(_config.schedule.Value().affinity), &(_config.schedule.Value().affinity)), "Fail to set cpu affinity.");

    return 0;
}

int32_t ExecutorMgr::InitThreadPool() {
    uint32_t data_recv_thread_num = 0;
    for (auto& trigger : _config.triggers) {
        for (auto& main_source : trigger.main_sources) {
            if (main_source.timeout_ms > 0) {
                data_recv_thread_num++;
            }
        }
    }

    if (data_recv_thread_num != 0) {
        _data_recv_thread_pool = std::make_shared<ThreadPool>(data_recv_thread_num);
        ADF_EXEC_LOG_INFO << "Create internal data recv thread pool with " << data_recv_thread_num << " threads.";
    }

    return 0;
}

int32_t ExecutorMgr::InitCommunication() {
    for (auto& input_config : _config.inputs) {
        std::shared_ptr<Reader> reader(new Reader);

        reader->Init(input_config.topic, input_config.capacity);
        _readers[input_config.topic] = reader;
    }

    return 0;
}

/*暂停/恢复接收trigger，如果trigger的source没有其它trigger在接收，会暂停trigger的source接收，
如果source是来源于CmTopic，则也会停止CmTopic的接收。
如果有其它trigger也在接收该source，则只设置状态。当接收该source的所有trigger都不接收时，
则停止CmTopic的接收。
*/
int32_t ExecutorMgr::PauseTrigger(const std::string& trigger, const bool pause) {
    if (trigger == ALLTRIGGER_NAME) {
        for (int i = 0; i < static_cast<int>(_config.triggers.size()); i++) {
            PauseTriggerPtr(_config.triggers[i].name, &(_config.triggers[i]), pause);
        }
        return 0;
    }
    for (int i = 0; i < static_cast<int>(_config.triggers.size()); i++) {
        if (_config.triggers[i].name == trigger) {
            PauseTriggerPtr(trigger, &(_config.triggers[i]), pause);
            break;
        }
    }
    return 0;
}

int32_t ExecutorMgr::PauseTriggerPtr(const std::string& trigger, hozon::netaos::adf_lite::ExecutorConfig::Trigger* trigger_ptr, const bool pause) {
    if (trigger_ptr == nullptr) {
        ADF_EXEC_LOG_INFO << "trigger_ptr is nullptr";
        return -1;
    }
    ADF_EXEC_LOG_INFO << "trigger name: " << trigger_ptr->name << " set pause status:" << pause;
    if (pause) {
        std::unique_lock<std::mutex> pause_lk(_trigger_control_map[trigger].pause_mutex);
        _trigger_control_map[trigger_ptr->name].pause_enable = true;
    }
    for (auto main_source : trigger_ptr->main_sources) {
        TopicManager::GetInstance().ModifyTriggerStatus(main_source.name, _config.executor_name, trigger, !pause);
    }

    for (auto aux_source : trigger_ptr->aux_sources) {
        TopicManager::GetInstance().ModifyTriggerStatus(aux_source.name, _config.executor_name, trigger, !pause);
    }
    if (!pause) {
        std::unique_lock<std::mutex> pause_lk(_trigger_control_map[trigger].pause_mutex);
        _trigger_control_map[trigger_ptr->name].pause_enable = false;
        _trigger_control_map[trigger].resume_cv.notify_all();
    }
    // TopicManager::GetInstance().PrintTopic();
    return 0;
}

void ExecutorMgr::CheckForPause(const std::string& trigger) {
    std::unique_lock<std::mutex> pause_lk(_trigger_control_map[trigger].pause_mutex);
    if (_trigger_control_map[trigger].pause_enable == true) {
        _trigger_control_map[trigger].pause_ready = true;
        _trigger_control_map[trigger].pause_ready_cv.notify_all();

        while ((_trigger_control_map[trigger].pause_enable == true) && !_need_stop) {
            ADF_EXEC_LOG_DEBUG << "Trigger: " << trigger << " going pause.";
            _trigger_control_map[trigger].resume_cv.wait(pause_lk);
        }

        ADF_EXEC_LOG_DEBUG << "Trigger: " << trigger << " resumed.";
    }
}

void ExecutorMgr::Routine(ExecutorConfig::Trigger& trigger) {
    pthread_setname_np(pthread_self(), (std::string("trg_") + trigger.name).c_str());
                

    std::chrono::steady_clock::time_point period_wakeup_timepoint = std::chrono::steady_clock::now();
    CheckpointProfiler cpt_profiler(_config.profiler.name + "." + trigger.name);
    LatencyProfiler lat_profiler(_config.profiler.name + "." + trigger.name);
    std::vector<NodeConfig::LatencyShow> latency_shows;

    InitLatencyProfiler(trigger, latency_shows, lat_profiler);

    if (trigger.type == ExecutorConfig::Trigger::Type::TS_ALIGN) {
        for (auto& main_source : trigger.main_sources) {
            _trigger_control_map[trigger.name].source_latest_timestamp[main_source.name] = 0;
        }

        for (auto& main_source : trigger.main_sources) {
            _trigger_control_map[trigger.name].align_recv_threads.emplace_back(
                std::make_shared<std::thread>(std::bind(
                    &ExecutorMgr::RecvAlignMainSourceRoutine,
                    this,
                    trigger,
                    main_source)));

            ADF_EXEC_LOG_INFO << "Create recv thread of " << main_source.name;
        }
    }

    hozon::netaos::adf::TimeMonitor exec_time_monitor;
    if (trigger.exp_exec_time_ms.HasValue()) {
        std::string trigger_name = trigger.name;
        uint64_t exp_exec_time = trigger.exp_exec_time_ms.Value();
        ADF_EXEC_LOG_INFO << "Init execution time monitor, expected time " << exp_exec_time << "(ms).";
        exec_time_monitor.Init(exp_exec_time, [trigger_name, exp_exec_time, this](uint64_t duration){
            ADF_EXEC_LOG_ERROR << "Trigger " << trigger_name << " process timeout, exp: " << exp_exec_time << "(ms), curr: " << duration << "(ms).";
        });
    }

    while (!_need_stop) {
        Bundle input;
        ADF_EXEC_LOG_VERBOSE << "Trigger " << trigger.name << " loop enter.";

        CheckForPause(trigger.name);

        std::chrono::steady_clock::time_point data_recv_begin_time = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point data_recv_end_time = std::chrono::steady_clock::now();
        
        if (trigger.type == ExecutorConfig::Trigger::Type::EVENT) {
            data_recv_begin_time = std::chrono::steady_clock::now();
            // main source
            if (RecvMainSources(input, trigger) < 0) {
                continue;
            }
        
            // aux source
            if (RecvAuxSources(input, trigger) < 0) {
                continue;
            }
            data_recv_end_time = std::chrono::steady_clock::now();
        }
        else if (trigger.type == ExecutorConfig::Trigger::Type::PERIOD) {
            ADF_EXEC_LOG_DEBUG << "PERIOD Trigger " << trigger.name << " going to sleep.";
            std::this_thread::sleep_until(period_wakeup_timepoint + std::chrono::milliseconds(trigger.period_ms));
            period_wakeup_timepoint = std::chrono::steady_clock::now();
            ADF_EXEC_LOG_DEBUG << "PERIOD Trigger " << trigger.name << " wakeup.";

            data_recv_begin_time = std::chrono::steady_clock::now();
            // aux source
            if (RecvAuxSources(input, trigger) < 0) {
                continue;
            }
            data_recv_end_time = std::chrono::steady_clock::now();
        }
        else if (trigger.type == ExecutorConfig::Trigger::Type::TS_ALIGN) {
            data_recv_begin_time = std::chrono::steady_clock::now();
            // if (RecvAlignedMainSources(input, trigger) < 0) {
            //     continue;
            // }
            if (GetAlignedMainSources(input, trigger) < 0) {
                continue;
            }

            if (RecvAuxSources(input, trigger) < 0) {
                continue;
            }
            data_recv_end_time = std::chrono::steady_clock::now();
        }
        else {
            ADF_EXEC_LOG_ERROR << "Unsupported trigger type " 
                << static_cast<std::underlying_type<ExecutorConfig::Trigger::Type>::type>(trigger.type);
            return;
        }
        double data_recv_time_cost = std::chrono::duration<double, std::milli>(data_recv_end_time - data_recv_begin_time).count();
        ADF_EXEC_LOG_DEBUG << "Trigger " << trigger.name << " recv sources end, time cost " << data_recv_time_cost << "(ms).";

        CptProfilerBegin(cpt_profiler);
        CalcLatencyProfiler(trigger, latency_shows, lat_profiler, input, false);

        // alg process 
        int32_t ret = -1;
        std::chrono::steady_clock::time_point process_begin_time = std::chrono::steady_clock::now();
        exec_time_monitor.FeedBegin();
        ADF_EXEC_LOG_DEBUG << "Trigger " << trigger.name << " process begin.";
        auto alg_process_func = _executor->GetProcessFunc(trigger.name);
        if (alg_process_func) {
            ret = alg_process_func(&input);
            if (ret < 0) {
                ADF_EXEC_LOG_WARN << "Trigger " << trigger.name << " process failed, ret " << ret;
            }
        }
        else {
            auto alg_process_with_profiler_func = _executor->GetProcessWithProfilerFunc(trigger.name);
            if (alg_process_with_profiler_func) {
                ProfileToken token;
                GenProfileToken(input, token);
                ret = alg_process_with_profiler_func(&input, token);
                if (ret < 0) {
                    ADF_EXEC_LOG_WARN << "Trigger " << trigger.name << " process with profiler failed, ret " << ret;
                }
            }
            else {
                ADF_EXEC_LOG_ERROR << "Missing process function of " << trigger.name;
            }
        }
        CalcLatencyProfiler(trigger, latency_shows, lat_profiler, input, true);
        exec_time_monitor.FeedEnd();
        std::chrono::steady_clock::time_point process_end_time = std::chrono::steady_clock::now();
        double process_time_cost = std::chrono::duration<double, std::milli>(process_end_time - process_begin_time).count();
        ADF_EXEC_LOG_DEBUG << "Trigger " << trigger.name << " process end, process time cost " << process_time_cost << "(ms).";
        if ((trigger.period_ms != 0) && (process_time_cost > trigger.period_ms)) {
            ADF_EXEC_LOG_DEBUG << "Trigger " << trigger.name << " takes " << process_time_cost 
                    << "(ms) to process, longer than period " << trigger.period_ms << "(ms).";
        }
        CptProfilerEnd(cpt_profiler);
    }

    if (trigger.exp_exec_time_ms.HasValue()) {
        exec_time_monitor.DeInit();
    }
}

int32_t ExecutorMgr::RecvMainSources(Bundle& input, ExecutorConfig::Trigger& trigger) {
    std::unordered_map<std::string, std::future<BaseDataTypePtr>> future_results;

    bool comm_fault_occur = false;
    for (auto& main_source_config : trigger.main_sources) {
        if (_data_recv_thread_pool == nullptr) {
            ADF_EXEC_LOG_FATAL << "Missing data recv thread pool.";
            return -1;
        }

        future_results[main_source_config.name] = _data_recv_thread_pool->Commit(
                std::bind(&hozon::netaos::adf_lite::Reader::GetLatestOneBlocking, _readers[main_source_config.name], std::placeholders::_1, std::placeholders::_2),
                main_source_config.timeout_ms,
                false);
    }

    for (auto& ele : future_results) {
        BaseDataTypePtr main_source_ptr = future_results[ele.first].get();
        if (main_source_ptr == nullptr) {
            ADF_EXEC_LOG_WARN << "Fail to get main source " << ele.first;
            comm_fault_occur = true;
            continue;
            // return -1;
        }

        input.Add(ele.first, main_source_ptr);
    }

    (void)comm_fault_occur;

    return 0;
}

int32_t ExecutorMgr::RecvAuxSources(Bundle& input, ExecutorConfig::Trigger& trigger) {
    for (auto aux_source_config : trigger.aux_sources) {
        std::vector<BaseDataTypePtr> aux_source_vec;
        if (aux_source_config.multi_frame > 0) {
            aux_source_vec = _readers[aux_source_config.name]->GetLatestNdata(aux_source_config.multi_frame, aux_source_config.read_clear);
            ADF_EXEC_LOG_VERBOSE << "PopNdata " << aux_source_vec.size() << ", multi " << aux_source_config.multi_frame;
        }
        else {
            BaseDataTypePtr aux_source_ptr = _readers[aux_source_config.name]->GetLatestOne(aux_source_config.read_clear);
            aux_source_vec.emplace_back(aux_source_ptr);
        }
        
        input.Set(aux_source_config.name, aux_source_vec);
    }
    
    return 0;
}

void ExecutorMgr::RecvAlignMainSourceRoutine(ExecutorConfig::Trigger& trigger, ExecutorConfig::Trigger::MainSource& main_source) {
    pthread_setname_np(pthread_self(), (std::string("al_") + main_source.name).c_str());

    while (!_need_stop) {
        auto& trigger_control_handle = _trigger_control_map[trigger.name];

        {
            std::unique_lock<std::mutex> pause_lk(trigger_control_handle.pause_mutex);
            if (trigger_control_handle.pause_enable == true) {
                while ((trigger_control_handle.pause_enable == true) && !_need_stop) {
                    ADF_EXEC_LOG_DEBUG << "AlignMainSourceRoutine: " << main_source.name << " going to pause.";
                    std::unique_lock<std::mutex> ali_lk(trigger_control_handle.align_mutex);
                    trigger_control_handle.align_list.clear();
                    trigger_control_handle.source_latest_timestamp[trigger.name] = 0;
                    ali_lk.unlock();
                    trigger_control_handle.resume_cv.wait(pause_lk);
                }

                ADF_EXEC_LOG_DEBUG << "AlignMainSourceRoutine: " << main_source.name << " resumed.";
            }
        }

        std::pair<std::string, BaseDataTypePtr> data_pair;
        data_pair.first = main_source.name;
        data_pair.second = _readers[main_source.name]->GetLatestOneBlocking(main_source.timeout_ms, true);
        if (!data_pair.second) {
            ADF_EXEC_LOG_WARN << "Fail to get main source " << main_source.name;
            continue;
        }

        std::unique_lock<std::mutex> align_lk(trigger_control_handle.align_mutex);
        if (data_pair.second->__header.timestamp_real_us > trigger_control_handle.source_latest_timestamp[main_source.name]) {
            trigger_control_handle.source_latest_timestamp[main_source.name] = data_pair.second->__header.timestamp_real_us;
        }

        auto lower_it = std::lower_bound(
            trigger_control_handle.align_list.begin(), 
            trigger_control_handle.align_list.end(), 
            data_pair,
            [](std::pair<std::string, BaseDataTypePtr> A, std::pair<std::string, BaseDataTypePtr> B) {
                return A.second->__header.timestamp_real_us < B.second->__header.timestamp_real_us;
            });
        trigger_control_handle.align_list.insert(lower_it, data_pair);
        ADF_EXEC_LOG_VERBOSE << "Insert align source: " << main_source.name << ", at " << data_pair.second->__header.timestamp_real_us / 1000
                << ", size: " << trigger_control_handle.align_list.size();

        for (auto it = trigger_control_handle.align_list.begin(); it != trigger_control_handle.align_list.end();) {
            if (it->first != main_source.name) {
                ++it;
                continue;
            }
            if (it->second->__header.timestamp_real_us < (trigger_control_handle.source_latest_timestamp[main_source.name] - trigger.align_validity_ms * 1000)) {
                ADF_EXEC_LOG_VERBOSE << "Erase out-of-date data: " << main_source.name << ", at " << it->second->__header.timestamp_real_us / 1000;
                it = trigger_control_handle.align_list.erase(it);
            }
            else {
                ADF_EXEC_LOG_VERBOSE << "No need to erase data: " << main_source.name 
                        << ", validity: " << trigger.align_validity_ms 
                        << ", latest: " << trigger_control_handle.source_latest_timestamp[main_source.name] / 1000
                        << ", curr: " << it->second->__header.timestamp_real_us / 1000
                        << ", diff: " << (trigger_control_handle.source_latest_timestamp[main_source.name] - it->second->__header.timestamp_real_us) / 1000;
                ++it;
            }
        }
        trigger_control_handle.align_data_arrived_cv.notify_all();
    }
}

int32_t ExecutorMgr::CheckDataAligned(ExecutorConfig::Trigger& trigger, std::unordered_map<std::string, BaseDataTypePtr>& data_map) {
    uint64_t max_us = 0;
    uint64_t min_us = UINT64_MAX;

    for (auto& main_source : trigger.main_sources) {
        if (data_map.find(main_source.name) == data_map.end()) {
            ADF_EXEC_LOG_VERBOSE << "Aligning timestamp, missing data " << main_source.name;
            return -1;
        }
    }

    for (auto& main_source : trigger.main_sources) {
        max_us = std::max(data_map[main_source.name]->__header.timestamp_real_us, max_us);
        min_us = std::min(data_map[main_source.name]->__header.timestamp_real_us, min_us);
        ADF_EXEC_LOG_VERBOSE << "Aligning timestamp of " << main_source.name << 
                        ", curr: " << data_map[main_source.name]->__header.timestamp_real_us / 1000
                        << " min: " << min_us / 1000 << " max: " << max_us / 1000;
    }

    if ((max_us - min_us) > (trigger.time_window_ms * 1000)) {
        ADF_EXEC_LOG_WARN << "Aligning timestamp failed, time diff " << (max_us - min_us) / 1000;
        return -1;
    }

    ADF_EXEC_LOG_VERBOSE << "Aligning timestamp succ, time diff " << (max_us - min_us) / 1000;

    return 0;
}

int32_t ExecutorMgr::AlignSources(Bundle& input, ExecutorConfig::Trigger& trigger) {
    auto& trigger_control_handle = _trigger_control_map[trigger.name];
    std::unordered_map<std::string, BaseDataTypePtr> data_map;

    for (auto rit = trigger_control_handle.align_list.rbegin(); rit != trigger_control_handle.align_list.rend(); ++rit) {
        data_map[rit->first] = rit->second;
        int ret = CheckDataAligned(trigger, data_map);
        if (ret == 0) {
            for (auto& ele : data_map) {
                input.Add(ele.first, ele.second);

                // TODO: use map::find
                for (auto it = trigger_control_handle.align_list.begin(); it != trigger_control_handle.align_list.end();) {
                    if (ele.second == it->second) {
                        it = trigger_control_handle.align_list.erase(it);
                        break;
                    }
                    else {
                        ++it;
                    }
                }
            }
            return 0;
        } 
    }

    return -1;
}

int32_t ExecutorMgr::GetAlignedMainSources(Bundle& input, ExecutorConfig::Trigger& trigger) {
    auto& trigger_control_handle = _trigger_control_map[trigger.name];
    std::unique_lock<std::mutex> align_lk(trigger_control_handle.align_mutex);

    std::chrono::steady_clock::time_point timeout_point;
    if (trigger.align_timeout_ms.HasValue()) {
        timeout_point = std::chrono::steady_clock::now() + std::chrono::milliseconds(trigger.align_timeout_ms.Value());
    }
    
    ADF_EXEC_LOG_VERBOSE << "Try to get aligned sources of " << trigger.name;
    int32_t ret = AlignSources(input, trigger);
    if (ret == 0) {
        ADF_EXEC_LOG_VERBOSE << "Get aligned sources from existing data " << trigger.name;
        return 0;
    }

    while (!_need_stop) {
        if (trigger.align_timeout_ms.HasValue()) {
            ADF_EXEC_LOG_VERBOSE << "Wait for incoming data " << trigger.name << " with timeout.";
            std::cv_status wait_ret = trigger_control_handle.align_data_arrived_cv.wait_until(align_lk, timeout_point);
            if (wait_ret == std::cv_status::timeout) {
                ADF_EXEC_LOG_WARN << "Fail to align trigger " << trigger.name;
                return -1;
            }
        }
        else {
            ADF_EXEC_LOG_VERBOSE << "Wait for incoming data " << trigger.name;
            trigger_control_handle.align_data_arrived_cv.wait(align_lk);
            ADF_EXEC_LOG_VERBOSE << "Wait for incoming data " << trigger.name << " successfully.";
        }

        ret = AlignSources(input, trigger);
        if (ret == 0) {
            return 0;
        }
        else {
            // continue to wait
        }
    }

    return -1;
}

void ExecutorMgr::GenProfileToken(Bundle& input, ProfileToken& token) {
    if (_config.profiler.enable && _config.profiler.latency.enable) {
        for (auto link : _config.profiler.latency.links) {
            auto recv_msgs = input.GetAll(link.recv_msg);
            if (recv_msgs.empty()) {
                ADF_EXEC_LOG_WARN << "Empty recv msg [" << link.recv_msg << "], set timestamp to 0.";
                token.latency_info.data[link.name] = {0, 0};
            } else {
                // use latest msg
                auto recv_msg = recv_msgs.back();
                ADF_EXEC_LOG_DEBUG << "GenProfileToken [" << link.recv_msg
                                   << "] recv_msg->__header.latency_info.data.size() : "
                                   << recv_msg->__header.latency_info.data.size();

                // empty latency info in recv msgs, copy timestamp to latency info
                if (recv_msg->__header.latency_info.data.find(link.name) ==
                    recv_msg->__header.latency_info.data.end()) {
                    struct AlgTime time;
                    time.sec = recv_msg->__header.timestamp_real_us / 1000 / 1000;
                    time.nsec = (recv_msg->__header.timestamp_real_us - time.sec * 1000 * 1000) * 1000;

                    token.latency_info.data[link.name] = time;
                    struct timespec time_now;
                    clock_gettime(CLOCK_REALTIME, &time_now);

                    ADF_EXEC_LOG_WARN << "link.name=" << link.name << " sec: " << time.sec << " nsec: " << time.nsec
                                      << " data.size(): " << token.latency_info.data.size()
                                      << " time_now tv_sec: " << time_now.tv_sec << " tv_nsec: " << time_now.tv_nsec;
                    recv_msg->__header.latency_info.data[link.name] = time;
                }
                // if latency info already exists in recv msg, just pass through it
                else {
                    token.latency_info.data[link.name] = recv_msg->__header.latency_info.data[link.name];
                    ADF_EXEC_LOG_DEBUG << "time_now tv_sec: " << token.latency_info.data[link.name].sec
                                       << " nsec: " << token.latency_info.data[link.name].nsec;
                }
            }
        }
    }
}

void ExecutorMgr::InitLatencyProfiler(ExecutorConfig::Trigger& trigger,
                                      std::vector<NodeConfig::LatencyShow>& latency_shows,
                                      LatencyProfiler& lat_profiler) {
    std::vector<std::string> lat_profiler_links;
    for (auto show : _config.profiler.latency.shows) {
        if (show.trigger_name == trigger.name) {
            lat_profiler_links.emplace_back(show.link_name + "(ms)");
        }
        latency_shows.emplace_back(show);
    }
    lat_profiler.Init(lat_profiler_links);
}

void ExecutorMgr::CalcLatencyProfiler(ExecutorConfig::Trigger& trigger,
                                      std::vector<NodeConfig::LatencyShow>& latency_shows,
                                      LatencyProfiler& lat_profiler, Bundle& input, bool after_process) {
    if (_config.profiler.enable && _config.profiler.latency.enable && !latency_shows.empty()) {
        struct timespec time_now;
        clock_gettime(CLOCK_REALTIME, &time_now);

        std::vector<double> lantencies_us;
        for (auto show : latency_shows) {
            if (show.trigger_name != trigger.name) {
                continue;
            }

            auto input_data = input.GetOne(show.from_msg);
            ADF_EXEC_LOG_DEBUG << "CalcLatencyProfiler " << show.from_msg;
            if (input_data != nullptr) {
                if (input_data->__header.latency_info.data.find(show.link_name) !=
                    input_data->__header.latency_info.data.end()) {
                    int sec_diff = time_now.tv_sec - input_data->__header.latency_info.data[show.link_name].sec;
                    int nsec_diff = time_now.tv_nsec - input_data->__header.latency_info.data[show.link_name].nsec;
                    double diff_us = sec_diff * 1000 * 1000 + nsec_diff / 1000;
                    lantencies_us.emplace_back(diff_us / 1000);
                    ADF_EXEC_LOG_DEBUG << "time_now tv_sec: " << time_now.tv_sec << " tv_nsec: " << time_now.tv_nsec;
                    ADF_EXEC_LOG_DEBUG << "data latency " << show.link_name << ": sec "
                                       << input_data->__header.latency_info.data[show.link_name].sec << ", nsec "
                                       << input_data->__header.latency_info.data[show.link_name].nsec << ", diff "
                                       << diff_us / 1000 << "(ms)";
                } else {
                    double diff_ms = time_now.tv_sec * 1000 + time_now.tv_nsec / 1000 / 1000 -
                                     input_data->__header.timestamp_real_us / 1000;
                    lantencies_us.emplace_back(diff_ms);
                    ADF_EXEC_LOG_DEBUG << "first latency " << show.link_name << ", diff " << diff_ms << "(ms)";
                }
            } else {
                ADF_EXEC_LOG_DEBUG << "CalcLatencyProfiler " << show.from_msg << " hasnot msg";
            }
        }

        if (lantencies_us.size() != 0) {
            ADF_EXEC_LOG_DEBUG << "Show latency.";
            lat_profiler.Show(lantencies_us, after_process);
        }
    }
}

void ExecutorMgr::CptProfilerBegin(CheckpointProfiler& cpt_profiler) {
    if (_config.profiler.enable && _config.profiler.checkpoint.enable) {
        cpt_profiler.Begin();
    }
}

void ExecutorMgr::CptProfilerEnd(CheckpointProfiler& cpt_profiler) {
    if (_config.profiler.enable && _config.profiler.checkpoint.enable) {
        cpt_profiler.End();
    }
}
}
}
}