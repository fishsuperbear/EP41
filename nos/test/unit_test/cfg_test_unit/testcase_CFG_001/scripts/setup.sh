

su - root -c "busybox killall -15 config_server"
current_path=$(pwd)
echo "current_path: $current_path"
su - root -c "mv /app/runtime_service/config_server/conf/config_server.json /app/runtime_service/config_server/conf/config_server.jsonbak"
su - root -c "cp  $current_path/../conf/config_server.json  /app/runtime_service/config_server/conf/config_server.json"



