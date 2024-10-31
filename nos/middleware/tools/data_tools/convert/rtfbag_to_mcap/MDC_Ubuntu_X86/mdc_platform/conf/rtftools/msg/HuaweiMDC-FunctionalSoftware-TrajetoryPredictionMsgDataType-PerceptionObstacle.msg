{
  "ShortName": "/HuaweiMDC/FunctionalSoftware/TrajetoryPredictionMsgDataType/PerceptionObstacle",
  "Namespace": "::ara::adsfi",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "id",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint32_t"
    },
    {
      "Name": "position",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "theta",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "velocity",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "hasVelocity",
      "Type": "/HuaweiMDC/StdCppBasicDataType/bool"
    },
    {
      "Name": "size",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "polygonPoint",
      "Type": "/HuaweiMDC/ApplicationDataType/PointArray"
    },
    {
      "Name": "trackingTime",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "type",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint8_t"
    },
    {
      "Name": "confidence",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "gpsTime",
      "Type": "/HuaweiMDC/FunctionalSoftware/ObjectArrayMsgDataType/Time"
    },
    {
      "Name": "confidenceType",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint8_t"
    },
    {
      "Name": "drops",
      "Type": "/HuaweiMDC/ApplicationDataType/PointArray"
    },
    {
      "Name": "acceleration",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "anchorPoint",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "boundingBox",
      "Type": "/HuaweiMDC/ApplicationDataType/PointArray"
    },
    {
      "Name": "subType",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint8_t"
    },
    {
      "Name": "heightAboveGround",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "positionCovariance",
      "Type": "/HuaweiMDC/FunctionalSoftware/TrajetoryPredictionMsgDataType/DoubleArray9"
    },
    {
      "Name": "velocityCovariance",
      "Type": "/HuaweiMDC/FunctionalSoftware/TrajetoryPredictionMsgDataType/DoubleArray9"
    },
    {
      "Name": "accelerationCovariance",
      "Type": "/HuaweiMDC/FunctionalSoftware/TrajetoryPredictionMsgDataType/DoubleArray9"
    },
    {
      "Name": "lightStatus",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint8_t"
    }
  ],
  "IsPlane": "false"
}