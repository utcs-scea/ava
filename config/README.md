Configuration Files
===================

Configuration files are replacing environment variables for AvA's settings.
The guestlib reads `/etc/ava/guest.conf` by default for user-defined configurations,
this can be changed by setting the environment variable `AVA_CONFIG_FILE_PATH`
to the absolute path to a configuration file.

Run `setup.sh` to install an example configuration.

## Guest configuration

| Name             | Example        | Default        | Explanation                             |
|------------------|----------------|----------------|-----------------------------------------|
| channel          | "TCP"          | "TCP"          | Transport channel (TCP\|SHM\|VSOCK)     |
| connect_timeout  | 5000L          | 5000L          | Timeout for API server connection, in milliseconds |
| manager_address  | "0.0.0.0:3334" | "0.0.0.0:3334" | AvA manager's address                   |
| instance_type    | "ava.xlarge"   | Ignored        | Service instance type                   |
| gpu_count        | 2              | Ignored        | Number of requested GPU, currently represented by `gpu_memory.size()` |
| gpu_memory       | [1024L,512LL]  | []             | Requested GPU memory sizes, in MB       |
| log_level        | "info"         | "info"         | Logger severity (verbose\|debug\|info\|warning\|error\|fatal\|none) |
