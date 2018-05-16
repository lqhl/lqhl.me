---
title: Linux Network Tuning - 10 GE
date: 2018-09-14T20:26:21+08:00
tags: ['linux', 'networking']
---

最近在研究如何为公司搭建 40/100 GE，在购买新设备之前，决定先在现有的 10 GE 网络上做一些 benchmark。
做完才发现现有的网络利用率还有不少的提高空间。

## 测试方式

测试机器

* Server-A：用于运行 NFS 服务器（IP: `10.1.1.1`）
* Client-B: 用于运行 NFS 的客户端

### 用 iperf 测试网速

在 Server-A 上：

```bash
iperf3 -s -p 12000 -i1
```

在 Client-B 上：

```bash
iperf3 -c storage0002 -p 12000 -i1 -t 30
```

### 用 fio 测试 NFS 的性能

在 Client-B 上 mount NFS（假设 Server-A 在 `/srv/nfs` 上配置了一个 NFS）：

```bash
sudo mount 10.1.1.1:/srv/nfs /mnt/tmp -v
```

Fio 的测试文件 `read.fio`:

```ini
[global]
bs=2M
iodepth=1
direct=1
ioengine=libaio
randrepeat=0
group_reporting
time_based
runtime=60
filesize=2G
numjobs=4

[job]
rw=read
filename=/mnt/tmp/test
name=read
```

运行 fio 测试：

```bash
fio read.fio
```

### Ubuntu 默认配置下的测试结果

iperf 的速度：

```text
- - - - - - - - - - - - - - - - - - - - - - - - -
[ ID] Interval           Transfer     Bandwidth       Retr
[  4]   0.00-30.00  sec  32.9 GBytes  9.41 Gbits/sec  291             sender
[  4]   0.00-30.00  sec  32.9 GBytes  9.41 Gbits/sec                  receiver
```

可以看到用 iperf 测试时，基本还是把 10 GE 的带宽用满了。

用 fio 在 NFS 上的测试结果是: `7.5 Gbits/sec`！大大的低于我们的预期。

## 调整 Linux 的 TCP 协议栈

TCP Tuning 可以参考 ESnet 的经验：<https://fasterdata.es.net/host-tuning/linux/test-measurement-host-tuning/>

Step 1. 配置  Jumbo frames（参考 <https://askubuntu.com/a/122835>）：

```bash
ip link set <NIC> mtu 9000  # <NIC> 是对外的网卡名称（比如 eth0）
```

Step 2. 调整 kernel 中网络相关的参数，修改 `/etc/sysctl.conf`：

```ini
# increase TCP max buffer size setable using setsockopt()
# allow testing with 256MB buffers
net.core.rmem_max = 268435456 
net.core.wmem_max = 268435456 
# increase Linux autotuning TCP buffer limits 
# min, default, and max number of bytes to use
# allow auto-tuning up to 128MB buffers
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
# recommended to increase this for CentOS6 with 10G NICS or higher
net.core.netdev_max_backlog = 250000
# don't cache ssthresh from previous connection
net.ipv4.tcp_no_metrics_save = 1
# Explicitly set htcp as the congestion control: cubic buggy in older 2.6 kernels
net.ipv4.tcp_congestion_control = htcp
# If you are using Jumbo Frames, also set this
net.ipv4.tcp_mtu_probing = 1
# recommended for CentOS7/Debian8 hosts
net.core.default_qdisc = fq
```

使我们对 `/etc/sysctl.conf` 的改动生效：

```bash
sysctl -p
service nfs-server restart # 因为我们要用 fio 测试 NFS，所以重启一下 NFS 服务
```

Step 3. 增大 TCP 的 Transmit Queue Length (txqueuelen)

```bash
ifconfig <NIC> txqueuelen 10000  # <NIC> 是对外的网卡名称（比如 eth0）
```

修改 `/etc/rc.local`，添加以下这行：

```bash
/sbin/ifconfig <NIC> txqueuelen 10000  # <NIC> 是对外的网卡名称（比如 eth0）
```

### 优化后的测试结果

iperf 的测试结果:

```text
- - - - - - - - - - - - - - - - - - - - - - - - -
[ ID] Interval           Transfer     Bandwidth       Retr
[  4]   0.00-30.00  sec  34.6 GBytes  9.90 Gbits/sec  419             sender
[  4]   0.00-30.00  sec  34.6 GBytes  9.90 Gbits/sec                  receiver
```

iperf 的结果只是稍微高了一点（提升空间本来也不大了）。

用 fio 在 NFS 上的测试结果：`9.76 Gbits/sec`。NFS 的性能提升是巨大的，比较有用的一个改动应该是 jumbo frames。
中间还尝试了 Google 的 BBR 拥塞控制算法，用处并不大。BBR 增对的是广域网上少量丢包的情况，在数据中心的网络中反而可能导致 NFS 的速度变慢。
