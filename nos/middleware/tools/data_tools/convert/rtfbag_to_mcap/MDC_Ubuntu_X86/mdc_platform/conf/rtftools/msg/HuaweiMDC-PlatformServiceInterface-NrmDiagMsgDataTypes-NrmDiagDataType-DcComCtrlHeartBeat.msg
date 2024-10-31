{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/NrmDiagMsgDataTypes/NrmDiagDataType/DcComCtrlHeartBeat",
  "Namespace": "::mdc::nrm",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "version",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint8_t"
    },
    {
      "Name": "seq",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint8_t",
      "Padding": "6"
    },
    {
      "Name": "tsSecs",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint64_t"
    },
    {
      "Name": "tsNsecs",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint64_t"
    }
  ],
  "IsPlane": "true"
}