export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/orin/szf/cyber/output/arm/lib

/home/orin/szf/code/netaos/thirdparty/for_arm/protobuf/bin/protoc \
--proto_path=/home/orin/szf/code/hardware/demo/camera_app/proto \
--cpp_out=/home/orin/szf/code/hardware/demo/camera_app/proto \
/home/orin/szf/code/hardware/demo/camera_app/proto/camera_config.proto