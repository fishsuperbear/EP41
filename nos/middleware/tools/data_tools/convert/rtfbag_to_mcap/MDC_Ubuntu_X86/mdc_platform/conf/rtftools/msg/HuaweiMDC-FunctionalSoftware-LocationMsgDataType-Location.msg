{
  "ShortName": "/HuaweiMDC/FunctionalSoftware/LocationMsgDataType/Location",
  "Namespace": "::ara::location",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "header",
      "Type": "/HuaweiMDC/FunctionalSoftware/LocationMsgDataType/Header"
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
    },
    {
      "Name": "odomType",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint8_t"
    }
  ],
  "IsPlane": "false"
}