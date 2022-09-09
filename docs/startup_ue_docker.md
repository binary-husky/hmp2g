# 1. Start up docker container | 启动docker虚拟容器
## 1. Check | 检查
```sh
# . 检查docker是否可用 （如果已经身处某个docker容器内，则以下命令会失败，请找到宿主系统，然后再运行以下命令）
sudo docker ps
```

## 2. Start | 启动
```sh
# 启动docker容器
sudo docker run -itd   --name  $USER-swarm \
--net host --memory 500G --gpus all --shm-size=32G fuqingxu/hmp:unreal-trim


# 修改docker容器的ssh的端口到 34567，自行选择合适的空闲端口
sudo docker exec -it  $USER-swarm sed -i 's/2266/34567/g' /etc/ssh/sshd_config
# 运行docker容器的ssh
sudo docker exec -it  $USER-swarm service ssh start
```

## 3. Connect | 连接

Now find a computer to ssh into it: ```ssh hmp@your_host_ip -p 34567```
``` sh
# 用任意ssh工具连接刚刚创建的虚拟容器
IP Addr: same with the host
SSH Port 34567
UserName: hmp
Password: hmp
```

# 2. clone code | 下载代码
``` sh
# 下载代码
git clone https://github.com/binary-husky/hmp2g.git -b multiteam
# 进入目录
cd hmp2g
```


# 3. Run unreal-based training | 启动训练

``` sh
# 按照example.jsonc中的实验配置，启动实验
python main.py -c example.jsonc
```
