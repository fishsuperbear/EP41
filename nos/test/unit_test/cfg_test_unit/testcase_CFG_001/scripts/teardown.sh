su - root -c "busybox killall -15 config_server"
su - root -c "mv /app/runtime_service/config_server/conf/config_server.jsonbak /app/runtime_service/config_server/conf/config_server.json"
su - root -c "rm -r /cfg_bak/*"