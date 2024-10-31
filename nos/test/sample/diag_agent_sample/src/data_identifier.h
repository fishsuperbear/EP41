#ifndef DATA_IDENTIFIER_H
#define DATA_IDENTIFIER_H

#include <mutex>
#include <vector>
#include "diag/diag_agent/include/service/diag_agent_data_identifier.h"

using namespace hozon::netaos::diag::diag_agent;

class DataIdentifier: public DiagAgentDataIdentifier {
public:
    DataIdentifier();
    virtual ~DataIdentifier();

    virtual bool Read(const uint16_t did, std::vector<uint8_t>& resData);
    virtual bool Write(const uint16_t did, const std::vector<uint8_t>& reqData, std::vector<uint8_t>& resData);

private:
    DataIdentifier(const DataIdentifier &);
    DataIdentifier & operator = (const DataIdentifier &);

private:
    std::vector<uint8_t> test_did_data_a500_;
    std::vector<uint8_t> test_did_data_a501_;
};

#endif  // DATA_IDENTIFIER_H
