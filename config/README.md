Configuration Files
===================

Configuration files are replacing environment variables for AvA's settings.
Guestlib reads `/etc/ava/guest.conf` for user-defined configurations.

Run `setup.sh` to install an example configuration.

## Guest configuration

| Name             | Example        | Default        | Explanation                             |
|------------------|----------------|----------------|-----------------------------------------|
| channel          | "TCP"          | "TCP"          | Transport channel (TCP\|SHM\|VSOCK)     |
| manager_address  | "0.0.0.0:3334" | "0.0.0.0:3334" | AvA manager's address                   |
| instance_type    | "ava.xlarge"   | Ignored        | Service instance type                   |
| gpu_count        | 2              | Ignored        | Number of requested GPU, currently represented by `gpu_memory.size()` |
| gpu_memory       | [1024,512]     | []             | Requested GPU memory sizes, in MB       |
