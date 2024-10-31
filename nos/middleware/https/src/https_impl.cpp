#include "https_impl.h"

namespace hozon {
namespace netaos {
namespace https {

HttpsImpl::HttpsImpl() {
  
}

HttpsImpl::~HttpsImpl() {

}

void HttpsImpl::Init() {
  http_client_.Init();
  http_client_.Start();
}

int HttpsImpl::HttpRequest( RequestPtr req_ptr, ResponseHandler handler) {
  return http_client_.HttpRequest(req_ptr, handler);
}

bool HttpsImpl::CancelRequest(int id) {
  // return http_client_.CancelRequest(id);
}

}
}
}