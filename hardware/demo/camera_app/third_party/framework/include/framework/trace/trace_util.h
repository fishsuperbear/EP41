#pragma once
#if defined(ENABLE_TRACE)

#include <string>
#include <sstream>
#include <sys/time.h>
#include <unistd.h>
#include <sys/types.h>
#include <ifaddrs.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <array>
#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>
#include <cstring>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <typeinfo>
#include <cxxabi.h>
#include <fstream>
#include <functional>

#include "framework/common/log.h"

namespace netaos {
namespace framework {

class FrameworkTraceModule {
  public:
    constexpr static auto CREATEREADER = "CRRE";
    constexpr static auto CREATEWRITER = "CRWR";
    constexpr static auto CREATECLIENT = "CRCL";
    constexpr static auto CREATESERVICE = "CRSV";
    constexpr static auto READERCALLBACK = "RECB";
    constexpr static auto WRITERWRITE = "WRWT";
    constexpr static auto CLIENTSENDREQUEST = "CLRQ";
    constexpr static auto SERVICECALLBACK = "SVCB";
};

inline bool check_trace_status(const std::string &module) {
    return true; // 检测module是否开启埋点收集
}

class TraceRecordStream {
  public:
    explicit TraceRecordStream() {}
    bool init(const std::string &file_path) {
        if ((m_fd = open(file_path.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR)) < 0) {
            AERROR << "TraceRecordStream open " << file_path << " error";
            return false;
        }
        if (!get_stream(SEGMENT_SIZE)) {
            return false;
        }
        m_ok = true;
        return m_ok;
    }
    ~TraceRecordStream() {
        if (m_fd > 0) {
            close(m_fd);
        }
        if (m_stream_buffer != MAP_FAILED) {
            msync(m_stream_buffer, m_stream_size, MS_SYNC);
            munmap(m_stream_buffer, m_stream_size);
        }
        m_ok = false;
    }
    operator bool() {
        return m_ok;
    }
    TraceRecordStream &operator << (const std::string &content) {
        std::unique_lock<std::mutex> lock(m_mutex);
        int inc_segment_count = (m_write_pos % SEGMENT_SIZE + content.size()) / SEGMENT_SIZE - 1;
        if (inc_segment_count > 0) {
            if (!get_stream(m_stream_size + SEGMENT_SIZE * inc_segment_count)) {
                return *this;
            }
        }
        memcpy((uint8_t *)m_stream_buffer + m_write_pos, content.c_str(), content.size());
        m_write_pos += content.size();
        // printf("size: %lu m_write_pos: %lu\n", content.size(), m_write_pos);
        return *this;
    }
  private:
    const uint32_t SEGMENT_SIZE {1 * 1024 * 1024};
    const uint32_t MAX_BUFFER_SIZE {50 * 1024 * 1024};

    bool m_ok {false};
    int m_fd {-1};
    uint32_t m_stream_size {0};
    uint32_t m_write_pos {0};
    void *m_stream_buffer {MAP_FAILED};
    std::mutex m_mutex;

    bool get_stream(uint32_t size) {
        if (m_stream_size >= MAX_BUFFER_SIZE) {
            AERROR << "TraceRecordStream trace file too large";
            return false;
        }
        if (ftruncate(m_fd, size) != 0) {
            AERROR << "TraceRecordStream ftruncate error";
            return false;
        }
        if (m_stream_buffer != MAP_FAILED) {
            munmap(m_stream_buffer, m_stream_size);
            m_stream_buffer = MAP_FAILED;
        }
        m_stream_buffer = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, m_fd, 0);
        if (m_stream_buffer == MAP_FAILED) {
            AERROR << "TraceRecordStream mmap error";
            return false;
        }
        m_stream_size = size;
        return true;
    }
};

extern TraceRecordStream g_trace_record_stream;

class TraceContext;

class SourceInfo {
  public:
    std::string m_file {""};
    int m_line {0};
    std::string m_function {""};
    static SourceInfo get_current_source_info(const std::string file = __builtin_FILE(),
            int line = __builtin_LINE(),
            const std::string function = __builtin_FUNCTION()) {
        SourceInfo cbinfo;
        cbinfo.m_file = file;
        cbinfo.m_line = line;
        cbinfo.m_function = function;
        return cbinfo;
    }
  private:
    SourceInfo() {}
    friend TraceContext;
};

class TraceContext {
  public:
    enum Role {
        Unknown = -1,
        Reader = 10,
        Writer,
        Client,
        Service,
    };
    explicit TraceContext() {}
    void set_node_name(const std::string &node_name) {
        m_node_name = node_name;
    }
    const std::string &get_node_name() const { // 获取Node名
        return m_node_name;
    }
    void set_role(Role role) {
        m_role = role;
    }
    Role get_role() const { // -1 Unknown, 1 Reader, 2 Writer, 3 Client, 4 Service
        return m_role;
    }
    void set_attribute_info(const std::string &attribute_info) {
        m_attribute_info = attribute_info;
    }
    const std::string &get_attribute_info() const { // 获取属性信息
        return m_attribute_info;
    }
    void set_source_info(const SourceInfo &source_info) {
        m_source_info = source_info;
    }
    const SourceInfo &get_source_info() const { // 获取源文件信息
        return m_source_info;
    }
  private:
    std::string m_node_name {""};
    Role m_role {Role::Unknown};
    std::string m_attribute_info {""};
    SourceInfo m_source_info;
};

class TraceEvent {
  public:
    enum Code {
        STR = 200,
        MODULE,
        FILE,
        LINE,
        FUNCTION,
        TIME,
        HOST,
        IP,
        PID,
        TID,
        NODENAME = 400,
        ROLE, // Reader, Writer, Client, Service
        ATTRIBUTE, // Channle, Client Name, Service Name
        SENDMSG,
        RECEIVEMSG,
        CALLBACK,
        PROCESSCPU = 600,
        THREADCPU,
        MEM,
        STARTL = 9901,
        ENDL = 9999,
    };
    virtual Code code() const = 0; // 埋点事件的类型，不能重复
    virtual uint32_t size() const = 0; // 埋点事件记录的数据大小：字节数
    virtual const void *data() const = 0;  // 埋点事件记录的数据
};

class TraceStr : public TraceEvent { // 记录字符串信息
  public:
    virtual TraceEvent::Code code() const override {
        return Code::STR;
    }
    virtual uint32_t size() const override {
        return m_str.length();
    }
    virtual const void *data() const override {
        return m_str.c_str();
    }
  protected:
    TraceStr(const std::string &str) : m_str(str) {}
    std::string m_str {""};
};

class TraceModule : public TraceStr { // 记录Module信息
  public:
    TraceModule(const std::string &module) : TraceStr(module) {}
    virtual TraceEvent::Code code() const override {
        return Code::MODULE;
    }
};

class TraceFile : public TraceEvent { // 记录源文件文件名
  public:
    TraceFile(const std::string &str) : m_str(str) {}
    virtual TraceEvent::Code code() const override {
        return Code::FILE;
    }
    virtual uint32_t size() const override {
        return m_str.length();
    }
    virtual const void *data() const override {
        return m_str.c_str();
    }
  private:
    std::string m_str {""};
};

class TraceLine : public TraceEvent { // 记录源文件行号
  public:
    TraceLine(const std::string &str) : m_str(str) {}
    virtual TraceEvent::Code code() const override {
        return Code::LINE;
    }
    virtual uint32_t size() const override {
        return m_str.length();
    }
    virtual const void *data() const override {
        return m_str.c_str();
    }
  private:
    std::string m_str {""};
};

class TraceFunction : public TraceEvent { // 记录源文件函数名
  public:
    TraceFunction(const std::string &str) : m_str(str) {}
    virtual TraceEvent::Code code() const override {
        return Code::FUNCTION;
    }
    virtual uint32_t size() const override {
        return m_str.length();
    }
    virtual const void *data() const override {
        return m_str.c_str();
    }
  private:
    std::string m_str {""};
};

class TraceTime : public TraceEvent { // 记录当前时间
  public:
    TraceTime() {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        m_time[0] = (uint64_t)(ts.tv_sec);
        m_time[1] = (uint64_t)(ts.tv_nsec);
    }
    virtual TraceEvent::Code code() const override {
        return Code::TIME;
    }
    virtual uint32_t size() const override {
        return m_time.size() * sizeof(m_time[0]);
    }
    virtual const void *data() const override {
        return &m_time[0];
    }
  private:
    std::array<uint64_t, 2> m_time {0, 0};
};

class TraceHost : public TraceEvent { // 记录当前host
  public:
    TraceHost() {
        char host[1024];
        if (gethostname(host, sizeof(host)) == 0) {
            m_host = host;
        }
    }
    virtual TraceEvent::Code code() const override {
        return Code::HOST;
    }
    virtual uint32_t size() const override {
        return m_host.length();
    }
    virtual const void *data() const override {
        return m_host.c_str();
    }
  private:
    std::string m_host {""};
};

class TraceIP : public TraceEvent { // 记录当前IP
  public:
    TraceIP(bool support_ipv6 = false) {
        struct ifaddrs *ifAddrStruct = nullptr;
        void *tmpAddrPtr = nullptr;
        if (getifaddrs(&ifAddrStruct) == 0) {
            while (ifAddrStruct != nullptr) {
                if (ifAddrStruct->ifa_addr->sa_family == AF_INET) { // IPV4
                    tmpAddrPtr = &((struct sockaddr_in *)ifAddrStruct->ifa_addr)->sin_addr;
                    char addressBuffer[INET_ADDRSTRLEN] {0};
                    inet_ntop(AF_INET, tmpAddrPtr, addressBuffer, INET_ADDRSTRLEN);
                    m_ip += addressBuffer;
                    m_ip += " ";
                } else if (ifAddrStruct->ifa_addr->sa_family == AF_INET6 && support_ipv6) { // IPV6
                    tmpAddrPtr = &((struct sockaddr_in *)ifAddrStruct->ifa_addr)->sin_addr;
                    char addressBuffer[INET6_ADDRSTRLEN] {0};
                    inet_ntop(AF_INET6, tmpAddrPtr, addressBuffer, INET6_ADDRSTRLEN);
                    m_ip += addressBuffer;
                    m_ip += " ";
                }
                ifAddrStruct = ifAddrStruct->ifa_next;
            }
        }
    }
    virtual TraceEvent::Code code() const override {
        return Code::IP;
    }
    virtual uint32_t size() const override {
        return m_ip.length() > 0 ? m_ip.length() - 1 : 0; // 去掉结尾空格
    }
    virtual const void *data() const override {
        return m_ip.c_str();
    }
  private:
    std::string m_ip {""};
};

class TracePID : public TraceEvent { // 记录当前进程的ID
  public:
    TracePID() {
        m_pid = getpid();
    }
    virtual TraceEvent::Code code() const override {
        return Code::PID;
    }
    virtual uint32_t size() const override {
        return sizeof(m_pid);
    }
    virtual const void *data() const override {
        return &m_pid;
    }
  private:
    uint64_t m_pid {0};
};

class TraceTID : public TraceEvent { // 记录当前线程的ID
  public:
    TraceTID() {
        m_tid = pthread_self();
    }
    virtual TraceEvent::Code code() const override {
        return Code::TID;
    }
    virtual uint32_t size() const override {
        return sizeof(m_tid);
    }
    virtual const void *data() const override {
        return &m_tid;
    }
  private:
    uint64_t m_tid {0};
};

class TraceProcessCPU : public TraceEvent { // 记录当前进程CPU使用数据
  public:
    TraceProcessCPU() {
        try {
            std::string line;
            std::vector<std::string> token_vec;
            std::string stat_file = get_proc_stat_file();
            std::ifstream stream(stat_file);
            if (stream) {
                std::getline(stream, line);
                // printf("%s\n", line.c_str());
                std::istringstream line_stream(line);
                std::string token;
                while (line_stream >> token) {
                    token_vec.emplace_back(token);
                }
            }
            if (token_vec.size() >= 16) {
                // https://linux.die.net/man/5/proc
                m_cpu[0] = std::stoll(token_vec[13]); // utime
                m_cpu[1] = std::stoll(token_vec[14]); // stime
                m_cpu[2] = std::stoll(token_vec[15]); // cutime
                m_cpu[3] = std::stoll(token_vec[16]); // cstime
                // printf("%lu %lu %lu %lu\n", m_cpu[0], m_cpu[1], m_cpu[2], m_cpu[3]);
            }
        } catch (...) {
        }
    }
    virtual TraceEvent::Code code() const override {
        return Code::PROCESSCPU;
    }
    virtual uint32_t size() const override {
        return m_cpu.size() * sizeof(m_cpu[0]);
    }
    virtual const void *data() const override {
        return &m_cpu[0];
    }
  protected:
    std::array<uint64_t, 4> m_cpu {0, 0, 0, 0};
    virtual std::string get_proc_stat_file() {
        return "/proc/" + std::to_string(getpid()) + "/stat";
    }
};

class TraceThreadCPU : public TraceProcessCPU { // 记录当前线程CPU使用数据
  public:
    TraceThreadCPU() : TraceProcessCPU() {}
    virtual TraceEvent::Code code() const override {
        return Code::THREADCPU;
    }
  protected:
    virtual std::string get_proc_stat_file() {
        // printf("%s\n", ("/proc/" + std::to_string(getpid()) + "/task/" + std::to_string(pthread_self()) + "/stat").c_str());
        return "/proc/" + std::to_string(getpid()) + "/task/" + std::to_string(pthread_self()) + "/stat";
    }

};

class TraceMem : public TraceEvent { // 记录当前内存使用数据
  public:
    TraceMem() {
        try {
            std::string line;
            std::vector<std::string> token_vec;
            std::string stat_file = "/proc/" + std::to_string(getpid()) + "/statm";
            std::ifstream stream(stat_file);
            if (stream) {
                std::getline(stream, line);
                std::istringstream line_stream(line);
                std::string token;
                while (line_stream >> token) {
                    token_vec.emplace_back(token);
                }
            }
            if (token_vec.size() >= 2) {
                // https://linux.die.net/man/5/proc
                m_mem[0] = std::stoll(token_vec[0]); // VmSize
                m_mem[1] = std::stoll(token_vec[1]); // VmRSS
            }
        } catch (...) {
        }
    }
    virtual TraceEvent::Code code() const override {
        return Code::MEM;
    }
    virtual uint32_t size() const override {
        return m_mem.size() * sizeof(m_mem[0]);
    }
    virtual const void *data() const override {
        return &m_mem[0];
    }
  private:
    std::array<uint64_t, 2> m_mem {0, 0};
};

class TraceNodeName : public TraceStr { // 记录节点名称信息
  public:
    TraceNodeName(const std::string &str) : TraceStr(str) {}
    virtual TraceEvent::Code code() const override {
        return Code::NODENAME;
    }
};

class TraceRole : public TraceEvent { // 记录类型信息
  public:
    TraceRole(TraceContext::Role role) {
        switch (role) {
        case TraceContext::Role::Client: {
            m_str = "CLIENT";
            break;
        }
        case TraceContext::Role::Service: {
            m_str = "SERVICE";
            break;
        }
        case TraceContext::Role::Reader: {
            m_str = "READER";
            break;
        }
        case TraceContext::Role::Writer: {
            m_str = "WRITER";
            break;
        }
        default: {
            m_str = "UNKNOWN";
            break;
        }
        }
    }
    virtual TraceEvent::Code code() const override {
        return Code::ROLE;
    }
    virtual uint32_t size() const override {
        return m_str.length();
    }
    virtual const void *data() const override {
        return m_str.c_str();
    }
  private:
    std::string m_str {""};
};

class TraceAttribute : public TraceStr { // 记录属性信息
  public:
    TraceAttribute(const std::string &attribute) : TraceStr(attribute) {}
    virtual TraceEvent::Code code() const override {
        return Code::ATTRIBUTE;
    }
};

class TraceSendMsg : public TraceEvent { // 记录发送消息信息
  public:
    TraceSendMsg(const std::type_info &info, uint32_t msg_size) {
        m_info = abi::__cxa_demangle(info.name(), 0, 0, 0 );
        m_info += "|" + std::to_string(msg_size);
    }
    virtual TraceEvent::Code code() const override {
        return Code::SENDMSG;
    }
    virtual uint32_t size() const override {
        return m_info.length();
    }
    virtual const void *data() const override {
        return m_info.c_str();
    }
  private:
    std::string m_info {""};
};

class TraceReceiveMsg : public TraceSendMsg { // 记录接收消息信息
  public:
    TraceReceiveMsg(const std::type_info &info, uint32_t msg_size) : TraceSendMsg(info, msg_size) {}
    virtual TraceEvent::Code code() const override {
        return Code::RECEIVEMSG;
    }
};

class TraceCallBack : public TraceEvent { // 记录回调function信息
  public:
    TraceCallBack(const std::type_info &info) {
        m_info = abi::__cxa_demangle(info.name(), 0, 0, 0 );
    }
    virtual TraceEvent::Code code() const override {
        return Code::CALLBACK;
    }
    virtual uint32_t size() const override {
        return m_info.length();
    }
    virtual const void *data() const override {
        return m_info.c_str();
    }
  private:
    std::string m_info {""};
};

class TraceRecorder {
  public:
    explicit TraceRecorder(const std::string &module) : m_module(module) {
        m_ss << TraceEvent::Code::STARTL;
    }
    ~TraceRecorder() {
        *this << TraceTime(); // 记录离开作用域的时间
        m_ss << TraceEvent::Code::ENDL;
        g_trace_record_stream << m_ss.str();
    }
    TraceRecorder &operator << (const TraceEvent &event) { // 记录埋点事件，相当于record_event
        static auto init_res = [this]() -> bool {
            uint64_t pid = getpid();
            auto time_t_val = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            auto p_tm = std::localtime(&time_t_val);
            std::stringstream ss;
            ss << "tracerecord_" << pid << "_" << std::put_time(p_tm, "%F_%T");
            return g_trace_record_stream.init(ss.str());
        }();
        if (!init_res) {
            return *this;
        }
        if (!check_trace_status(m_module)) {
            return *this;
        }
        auto code = (int)event.code();
        switch (code) {
        case TraceEvent::STR:
        case TraceEvent::FILE:
        case TraceEvent::LINE:
        case TraceEvent::FUNCTION:
        case TraceEvent::HOST:
        case TraceEvent::IP:
        case TraceEvent::PROCESSCPU:
        case TraceEvent::THREADCPU:
        case TraceEvent::MEM:
        case TraceEvent::NODENAME:
        case TraceEvent::ROLE:
        case TraceEvent::ATTRIBUTE:
        case TraceEvent::SENDMSG:
        case TraceEvent::RECEIVEMSG:
        case TraceEvent::CALLBACK: {
            m_ss << " code[" << code << "]";
            auto size = event.size();
            m_ss << " size[" << size << "]";
            auto buffer = (const char *)event.data();
            std::string _tmp_str(&buffer[0], &buffer[size]);
            m_ss << " data[" << _tmp_str << "]";
            break;
        }
        case TraceEvent::TIME: {
            m_ss << " code[" << code << "]";
            auto size = event.size() / sizeof(uint64_t);
            m_ss << " size[" << size << "]";
            auto buffer = (const uint64_t *)event.data();
            std::string _tmp_str{""};
            for (auto i = 0; i < size; ++i) {
                _tmp_str += std::to_string(buffer[i]);
                _tmp_str += i < size - 1 ? ":" : "";

            }
            m_ss << " data[" << _tmp_str << "]";
            break;
        }
        case TraceEvent::PID:
        case TraceEvent::TID: {
            m_ss << " code[" << code << "]";
            auto size = event.size();
            m_ss << " size[" << size << "]";
            auto buffer = (const uint64_t *)event.data();
            std::string _tmp_str;
            _tmp_str = std::to_string(*buffer);
            m_ss << " data[" << _tmp_str << "]";
            break;
        }
        default: {
            AERROR << "TraceRecorder bad code " << code;
        }
        }
        return *this;
    }
    std::stringstream &buffer() {
        return m_ss;
    }
  private:
    std::string m_module {""};
    std::stringstream m_ss;
};

} // namespace framework
} // namespace netaos

#define TRACE_ENTRY(module, source_info) _TRACE_ENTRY_DETAIL_(module, source_info.m_file, source_info.m_line, source_info.m_function)
#define _TRACE_ENTRY_DETAIL_(module, file, line, function) \
        netaos::framework::TraceRecorder _trace_rec_(module); \
        _trace_rec_ << netaos::framework::TraceModule(module) << netaos::framework::TraceFile(file) << netaos::framework::TraceLine(std::to_string(line)) << netaos::framework::TraceFunction(function) \
                    << netaos::framework::TraceTime() << netaos::framework::TracePID() << netaos::framework::TraceTID()
#define TRACE_ENTRY_EXT(module, context) _TRACE_ENTRY_EXT_DETAIL_(module, context.get_source_info().m_file, context.get_source_info().m_line, context.get_source_info().m_function, context.get_node_name(), context.get_role(), context.get_attribute_info())
#define _TRACE_ENTRY_EXT_DETAIL_(module, file, line, function, node_name, role, attribute) \
        netaos::framework::TraceRecorder _trace_rec_(module); \
        _trace_rec_ << netaos::framework::TraceModule(module) << netaos::framework::TraceFile(file) << netaos::framework::TraceLine(std::to_string(line)) << netaos::framework::TraceFunction(function) \
                    << netaos::framework::TraceTime() << netaos::framework::TracePID() << netaos::framework::TraceTID() \
                    << netaos::framework::TraceNodeName(node_name) << netaos::framework::TraceRole(role) << netaos::framework::TraceAttribute(attribute)
#define TRACE_SYS_PERF() \
        _trace_rec_ << netaos::framework::TraceProcessCPU() << netaos::framework::TraceThreadCPU() << netaos::framework::TraceMem()
#define TRACE_SYS_INFO() \
        _trace_rec_ << netaos::framework::TraceHost() << netaos::framework::TraceIP()
#define TRACE_CALLBACK_INFO(callback) \
        _trace_rec_ << netaos::framework::TraceCallBack(callback.target_type())
#define TRACE_SEND_MSG(msg) \
        _trace_rec_ << netaos::framework::TraceSendMsg(typeid(msg), 0)
#define TRACE_RECEIVE_MSG(msg) \
        _trace_rec_ << netaos::framework::TraceReceiveMsg(typeid(msg), 0)

#endif
