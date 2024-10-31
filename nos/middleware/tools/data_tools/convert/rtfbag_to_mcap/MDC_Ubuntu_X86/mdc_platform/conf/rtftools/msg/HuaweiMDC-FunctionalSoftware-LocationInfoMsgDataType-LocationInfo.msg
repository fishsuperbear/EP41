{
  "ShortName": "/HuaweiMDC/FunctionalSoftware/LocationInfoMsgDataType/LocationInfo",
  "Namespace": "::adsfi",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "header",
      "Type": "/HuaweiMDC/CommonDataTypes/CommonHeader"
    },
    {
      "Name": "locationState",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16"
    },
    {
      "Name": "pose",
      "Type": "/HuaweiMDC/ApplicationDataType/PoseWithCovariance"
    },
    {
      "Name": "velocity",
      "Type": "/HuaweiMDC/ApplicationDataType/TwistWithCovariance"
    },
    {
      "Name": "acceleration",
      "Type": "/HuaweiMDC/ApplicationDataType/AccelWithCovariance"
    }
  ],
  "IsPlane": "false"
}