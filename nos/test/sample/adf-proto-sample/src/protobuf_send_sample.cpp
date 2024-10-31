#include <memory>
#include <map>
#include <unistd.h>
#include "adf/include/log.h"
#include "proto/test/soc/for_test.pb.h"
#include "adf/include/base.h"
#include "adf/include/node_base.h"
#include "cm/include/proto_method.h"

using namespace hozon::netaos::log;
using namespace hozon::netaos::adf;

class ProtobufSendSample : public hozon::netaos::adf::NodeBase {
public:
    ProtobufSendSample() 
        : _server(std::bind(&ProtobufSendSample::SumProcess, this, std::placeholders::_1, std::placeholders::_2)) {

    }

    ~ProtobufSendSample() {}

    virtual int32_t AlgInit() {
        REGISTER_PROTO_MESSAGE_TYPE("workresult", adf::lite::dbg::WorkflowResult)
        RegistAlgProcessWithProfilerFunc("main", std::bind(&ProtobufSendSample::AlgProcess1, this, std::placeholders::_1, std::placeholders::_2));
        
        _server.Start(0, "/method_test");
        return 0;
    }

    virtual int32_t AlgProcess1(hozon::netaos::adf::NodeBundle* input,
                            const hozon::netaos::adf::ProfileToken& token) {
        BaseDataTypePtr base_ptr(new BaseData);

        std::shared_ptr<adf::lite::dbg::WorkflowResult> result = std::make_shared<adf::lite::dbg::WorkflowResult>();
        base_ptr->proto_msg = result;
        static int i = 0;
        ++i;

        result->set_val1(i);
        result->mutable_header()->set_publish_stamp(GetRealTimestamp());

        APP_OP_LOG_WARN << "Test for op log.";
        NODE_LOG_INFO << "Send result " << result->val1() 
            << ", time sec " << result->mutable_header()->publish_stamp();


        NodeBundle out_bundle;
        out_bundle.Add("workresult", base_ptr);
        SendOutput(&out_bundle, token);

        return 0;
    }

    virtual void AlgRelease() {
        _server.Stop();
    }

    int32_t SumProcess(const std::shared_ptr<adf::lite::dbg::WorkflowResult>& req, std::shared_ptr<adf::lite::dbg::WorkflowResult>& resp) {
        resp->set_val3(req->val1() + req->val2());

        return 0;
    }

private:
    hozon::netaos::cm::ProtoMethodServer<adf::lite::dbg::WorkflowResult, adf::lite::dbg::WorkflowResult> _server;
};

int main(int argc, char* argv[]) {
    ProtobufSendSample send_node;

    send_node.Start(std::string(argv[1]));
    send_node.NeedStopBlocking();
    send_node.Stop();

    return 0;
}

