{
  "ShortName": "/HuaweiMDC/FunctionalSoftware/TrajPredictMsgDataType/TrajPredictArray",
  "Namespace": "::adsfi",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "header",
      "Type": "/HuaweiMDC/CommonDataTypes/CommonHeader"
    },
    {
      "Name": "object",
      "Type": "/HuaweiMDC/FunctionalSoftware/TrajPredictMsgDataType/PredictObjectVector"
    },
    {
      "Name": "startTime",
      "Type": "/HuaweiMDC/CommonDataTypes/CommonTime"
    },
    {
      "Name": "endTime",
      "Type": "/HuaweiMDC/CommonDataTypes/CommonTime"
    },
    {
      "Name": "changeOriginFlag",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "selfIntent",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "scenario",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16"
    }
  ],
  "IsPlane": "false"
}