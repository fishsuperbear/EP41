#include <iostream>
#include "data_identifier.h"
#include "lidar_calibration.h"

DataIdentifier::DataIdentifier()
: DiagAgentDataIdentifier()
, test_did_data_a500_({0x31, 0x32, 0x33})
, test_did_data_a501_({0x31, 0x32, 0x33, 0x34})
{
}

DataIdentifier::~DataIdentifier()
{
}

bool
DataIdentifier::Read(const uint16_t did, std::vector<uint8_t>& resData)
{
    std::cout << "DataIdentifier::Read did: " << did << std::endl;
    bool bResult = false;
    switch (did)
    {
        case 0xA500:
            // get did data success
            resData.push_back(0xA5);
            resData.push_back(0x00);
            for (auto& item : test_did_data_a500_) {
                resData.push_back(static_cast<uint8_t>(item));
            }

            bResult = true;
            break;
        case 0xA501:
            // get did data success
            resData.push_back(0xA5);
            resData.push_back(0x01);
            for (auto& item : test_did_data_a501_) {
                resData.push_back(static_cast<uint8_t>(item));
            }

            bResult = true;
            break;
        case 0xFD14:
            {
                resData.push_back(0xFD);
                resData.push_back(0x14);
                std::vector<uint8_t> data;
                bResult = LidarCalibration::getInstance()->GetCalibrateResult(data);
                if (data.size() > 0) {
                    for (uint i = 0; i < data.size(); ++i) {
                        resData.emplace_back(data[i]);
                    }
                }
            }
            break;
        default:
            break;
    }

    return bResult;
}

bool
DataIdentifier::Write(const uint16_t did, const std::vector<uint8_t>& reqData, std::vector<uint8_t>& resData)
{
    std::cout << "DataIdentifier::Read did: " << did << std::endl;
    bool bResult = false;
    switch (did)
    {
        case 0xA501:
            // write did data success
            test_did_data_a501_.assign(reqData.begin(), reqData.end());
            resData.push_back(0xA5);
            resData.push_back(0x01);
            bResult = true;
            break;
        default:
            break;
    }

    return bResult;
}
