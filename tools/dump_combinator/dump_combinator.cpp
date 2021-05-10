#include <assert.h>
#include <cuda.h>
#include <driver_types.h>
#include <fatbinary.h>
#include <fcntl.h>
#include <glib.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <sstream>
#include <string>

#include "debug.h"
#include "logging.h"

struct fatbin_wrapper {
  uint32_t magic;
  uint32_t seq;
  uint64_t ptr;
  uint64_t data_ptr;
};

struct kernel_arg {
  char is_handle;
  uint32_t size;
};

#define MAX_KERNEL_ARG 25
#define MAX_KERNEL_NAME_LEN 1024

struct fatbin_function {
  int argc;
  struct kernel_arg args[MAX_KERNEL_ARG];

  CUfunction cufunc;
  void *hostfunc;
  CUmodule module;  // unneeded
};

int fatfunction_fd_load = 0;
int fatfunction_fd_dump = 0;

GPtrArray *global_fatbin_funcs = NULL;
int global_num_fatbins = 0;

ssize_t read_all(int fd, void *buf_in, size_t len) {
  char *buf = (char *)buf_in;
  ssize_t ret;
  while (len != 0 && (ret = read(fd, buf, len)) != 0) {
    if (ret == -1) {
      if (errno == EINTR) continue;
      break;
    }
    len -= ret;
    buf += ret;
  }
  return ret;
}

ssize_t write_all(int fd, char *buf, size_t len) {
  ssize_t ret;
  while (len != 0 && (ret = write(fd, buf, len)) != 0) {
    if (ret == -1) {
      if (errno == EINTR) continue;
      break;
    }
    len -= ret;
    buf += ret;
  }
  return ret;
}

/**
 * Loads the function argument information from dump.
 */
GHashTable *__helper_load_function_arg_info(char *load_dir, char *dump_dir, int fatbin_num) {
  GPtrArray *fatbin_funcs;
  if (global_fatbin_funcs == NULL) {
    global_fatbin_funcs = g_ptr_array_new_with_free_func(g_free);
    g_ptr_array_add(global_fatbin_funcs, (gpointer)NULL);  // func_id starts from 1
  }
  fatbin_funcs = global_fatbin_funcs;

  GHashTable *ht = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, NULL);

  int fd, fd_w, read_ret, write_ret;
  std::stringstream ss;
  ss << load_dir << "/function_arg-" << fatbin_num << ".ava";
  std::string filename = ss.str();
  fd = open(filename.c_str(), O_RDONLY, 0666);
  ss.str(std::string());
  ss.clear();

  ss << dump_dir << "/function_arg-" << global_num_fatbins << ".ava";
  filename = ss.str();
  fd_w = open(filename.c_str(), O_WRONLY | O_TRUNC | O_CREAT, 0666);
  AVA_DEBUG.printf("Dump function argument info to %s\n", filename.c_str());
  ss.str(std::string());
  ss.clear();

  struct fatbin_function *func;
  size_t name_size = 0;
  char func_name[MAX_KERNEL_NAME_LEN];

  while (1) {
    read_ret = read_all(fd, (void *)&name_size, sizeof(size_t));
    if (read_ret == 0) break;
    if (read_ret == -1) {
      fprintf(stderr, "read [errno=%d, errstr=%s] at %s:%d", read_ret, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
    assert(name_size < MAX_KERNEL_NAME_LEN && "name_size >= MAX_KERNEL_NAME_LEN");
    read_ret = read_all(fd, (void *)func_name, name_size);
    if (read_ret == -1) {
      fprintf(stderr, "read [errno=%d, errstr=%s] at %s:%d", read_ret, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }

    func = g_new(struct fatbin_function, 1);
    read_ret = read_all(fd, (void *)func, sizeof(struct fatbin_function));
    if (read_ret == -1) {
      fprintf(stderr, "read [errno=%d, errstr=%s] at %s:%d", read_ret, strerror(errno), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
    AVA_DEBUG.printf("function %d (%s) has argc = %d\n", fatbin_funcs->len - 1, func_name, func->argc);
    /* Insert into the function table */
    g_ptr_array_add(fatbin_funcs, (gpointer)func);

    /* Add name->index mapping */
    if (g_hash_table_lookup(ht, func_name) == NULL) {
      assert(fatbin_funcs->len > 1 && "fatbin_funcs->len <= 1");
      g_hash_table_insert(ht, g_strdup(func_name), (gpointer)((uintptr_t)fatbin_funcs->len - 1));
    }

    /* Dump the function argument sizes to file */
    write_ret = write(fd_w, (void *)&name_size, sizeof(size_t));
    if (write_ret == -1) {
      fprintf(stderr, "Unexpected error write [errno=%d, errstr=%s] at %s:%d", write_ret, strerror(errno), __FILE__,
              __LINE__);
      exit(EXIT_FAILURE);
    }
    write_ret = write(fd_w, (void *)func_name, name_size);
    if (write_ret == -1) {
      fprintf(stderr, "Unexpected error write [errno=%d, errstr=%s] at %s:%d", write_ret, strerror(errno), __FILE__,
              __LINE__);
      exit(EXIT_FAILURE);
    }
    write_ret = write(fd_w, (void *)func, sizeof(struct fatbin_function));
    if (write_ret == -1) {
      fprintf(stderr, "Unexpected error write [errno=%d, errstr=%s] at %s:%d", write_ret, strerror(errno), __FILE__,
              __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  close(fd);

  return ht;
}

void __helper_load_and_dump_fatbin(char *load_dir, char *dump_dir, int fatbin_num, void *fatCubin) {
  /* Read fatbin dump */
  int fd;
  int read_ret, write_ret;
  struct stat file_stat;

  struct fatbin_wrapper *wp = (struct fatbin_wrapper *)fatCubin;

  // load fatbin
  std::stringstream ss;
  ss << load_dir << "/fatbin-" << fatbin_num << ".ava";
  std::string filename = ss.str();
  AVA_DEBUG.printf("loading %s\n", filename.c_str());
  fd = open(filename.c_str(), O_RDONLY, 0666);
  ss.str(std::string());
  ss.clear();

  fstat(fd, &file_stat);
  size_t fatbin_size = (size_t)file_stat.st_size;
  void *fatbin = malloc(fatbin_size);
  read_ret = read_all(fd, fatbin, fatbin_size);
  if (read_ret == -1) {
    fprintf(stderr, "Unexpected error read [errno=%d, errstr=%s] at %s:%d", read_ret, strerror(errno), __FILE__,
            __LINE__);
    exit(EXIT_FAILURE);
  }
  close(fd);

  struct fatBinaryHeader *fbh = (struct fatBinaryHeader *)fatbin;
  AVA_DEBUG.printf("Read fatbin-%d.ava size = %lu, should be %llu\n", fatbin_num, fatbin_size,
                   fbh->headerSize + fbh->fatSize);
  assert(fatbin_size == fbh->headerSize + fbh->fatSize && "fatbin size is wrong");
  (void)fbh;

  wp->ptr = (uint64_t)fatbin;

  /* Dump fat binary to a file */
  ss << dump_dir << "/fatbin-" << global_num_fatbins << ".ava";
  std::string fatbin_filename = ss.str();
  fd = open(fatbin_filename.c_str(), O_WRONLY | O_TRUNC | O_CREAT, 0666);
  if (fd == -1) {
    fprintf(stderr, "Unexpected error open [errno=%d, errstr=%s] at %s:%d", fd, strerror(errno), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  AVA_DEBUG.printf("Dump fatbinary to %s\n", fatbin_filename.c_str());
  write_ret = write(fd, (const void *)wp->ptr, fbh->headerSize + fbh->fatSize);
  if (write_ret == -1) {
    fprintf(stderr, "Unexpected error write [errno=%d, errstr=%s] at %s:%d", write_ret, strerror(errno), __FILE__,
            __LINE__);
    exit(EXIT_FAILURE);
  }
  close(fd);
  ss.str(std::string());
  ss.clear();

  if (fatfunction_fd_dump == 0) {
    ss << dump_dir << "/fatfunction.ava";
    filename = ss.str();
    AVA_DEBUG.printf("fatfunction ava file to dump is %s\n", filename.c_str());
    fatfunction_fd_dump = open(filename.c_str(), O_WRONLY | O_TRUNC | O_CREAT, 0666);
    ss.str(std::string());
    ss.clear();
  }

  /* Load function argument information */
  GHashTable *ht = __helper_load_function_arg_info(load_dir, dump_dir, fatbin_num);

  /* Register CUDA functions */
  GPtrArray *fatbin_funcs = global_fatbin_funcs;
  struct fatbin_function *func;

  if (fatfunction_fd_load == 0) {
    ss << load_dir << "/fatfunction.ava";
    filename = ss.str();
    fprintf(stderr, "fatfunction ava file is %s\n", filename.c_str());
    fatfunction_fd_load = open(filename.c_str(), O_RDONLY, 0666);
    fprintf(stderr, "fatfunction_fd is %d\n", fatfunction_fd_load);
    ss.str(std::string());
    ss.clear();
  }
  fd = fatfunction_fd_load;

  void *func_id = NULL;
  size_t size = 0;
  int exists = 0;
  char *deviceFun = NULL;
  char *deviceName = NULL;
  int thread_limit;
  uint3 *tid = NULL;
  uint3 *bid = NULL;
  dim3 *bDim = NULL;
  dim3 *gDim = NULL;
  int *wSize = NULL;
  while (1) {
    // load size and deviceFun
    read_ret = read_all(fd, (void *)&size, sizeof(size_t));
    if (read_ret == 0) {  // EOF
      close(fd);
      break;
    }
    if (size == 0) {  // Meet separator
      AVA_DEBUG.printf("Finish reading functions for fatbin-%d.ava\n", fatbin_num);
      break;
    }
    if (read_ret == -1) {
      fprintf(stderr, "Unexpected error read [errno=%d, errstr=%s] at %s:%d", read_ret, strerror(errno), __FILE__,
              __LINE__);
      exit(EXIT_FAILURE);
    }
    deviceFun = (char *)malloc(size);
    read_ret = read_all(fd, (void *)deviceFun, size);
    if (read_ret == -1) {
      fprintf(stderr, "Unexpected error read [errno=%d, errstr=%s] at %s:%d", read_ret, strerror(errno), __FILE__,
              __LINE__);
      exit(EXIT_FAILURE);
    }

    // dump size and deviceFun
    write_ret = write(fatfunction_fd_dump, (const void *)&size, sizeof(size_t));
    if (write_ret == -1) {
      fprintf(stderr, "Unexpected error write [errno=%d, errstr=%s] at %s:%d", write_ret, strerror(errno), __FILE__,
              __LINE__);
      exit(EXIT_FAILURE);
    }
    write_ret = write(fatfunction_fd_dump, (const void *)deviceFun, size);
    if (write_ret == -1) {
      fprintf(stderr, "Unexpected error write [errno=%d, errstr=%s] at %s:%d", write_ret, strerror(errno), __FILE__,
              __LINE__);
      exit(EXIT_FAILURE);
    }

    // load size and deviceName
    read_ret = read_all(fd, (void *)&size, sizeof(size_t));
    if (read_ret == -1) {
      fprintf(stderr, "Unexpected error read [errno=%d, errstr=%s] at %s:%d", read_ret, strerror(errno), __FILE__,
              __LINE__);
      exit(EXIT_FAILURE);
    }
    deviceName = (char *)malloc(size);
    read_ret = read_all(fd, (void *)deviceName, size);
    if (read_ret == -1) {
      fprintf(stderr, "Unexpected error read [errno=%d, errstr=%s] at %s:%d", read_ret, strerror(errno), __FILE__,
              __LINE__);
      exit(EXIT_FAILURE);
    }

    // dump size and deviceName
    write_ret = write(fatfunction_fd_dump, (const void *)&size, sizeof(size_t));
    if (write_ret == -1) {
      fprintf(stderr, "Unexpected error write [errno=%d, errstr=%s] at %s:%d", write_ret, strerror(errno), __FILE__,
              __LINE__);
      exit(EXIT_FAILURE);
    }
    write_ret = write(fatfunction_fd_dump, (const void *)deviceName, size);
    if (write_ret == -1) {
      fprintf(stderr, "Unexpected error write [errno=%d, errstr=%s] at %s:%d", write_ret, strerror(errno), __FILE__,
              __LINE__);
      exit(EXIT_FAILURE);
    }

    read_ret = read_all(fd, (void *)&thread_limit, sizeof(int));
    if (read_ret == -1) {
      fprintf(stderr, "Unexpected error read [errno=%d, errstr=%s] at %s:%d", read_ret, strerror(errno), __FILE__,
              __LINE__);
      exit(EXIT_FAILURE);
    }
    write_ret = write(fatfunction_fd_dump, (const void *)&thread_limit, sizeof(int));
    if (write_ret == -1) {
      fprintf(stderr, "Unexpected error write [errno=%d, errstr=%s] at %s:%d", write_ret, strerror(errno), __FILE__,
              __LINE__);
      exit(EXIT_FAILURE);
    }

    read_ret = read_all(fd, (void *)&exists, sizeof(int));
    if (read_ret == -1) {
      fprintf(stderr, "Unexpected error read [errno=%d, errstr=%s] at %s:%d", read_ret, strerror(errno), __FILE__,
              __LINE__);
      exit(EXIT_FAILURE);
    }
    write_ret = write(fatfunction_fd_dump, (const void *)&exists, sizeof(int));
    if (write_ret == -1) {
      fprintf(stderr, "Unexpected error write [errno=%d, errstr=%s] at %s:%d", write_ret, strerror(errno), __FILE__,
              __LINE__);
      exit(EXIT_FAILURE);
    }

    if (exists) {
      tid = (uint3 *)malloc(sizeof(uint3));
      read_ret = read_all(fd, (void *)tid, sizeof(uint3));
      if (read_ret == -1) {
        fprintf(stderr, "Unexpected error read [errno=%d, errstr=%s] at %s:%d", read_ret, strerror(errno), __FILE__,
                __LINE__);
        exit(EXIT_FAILURE);
      }
      write_ret = write(fatfunction_fd_dump, (const void *)tid, sizeof(uint3));
      if (write_ret == -1) {
        fprintf(stderr, "Unexpected error write [errno=%d, errstr=%s] at %s:%d", write_ret, strerror(errno), __FILE__,
                __LINE__);
        exit(EXIT_FAILURE);
      }
    } else
      tid = NULL;

    read_ret = read_all(fd, (void *)&exists, sizeof(int));
    if (read_ret == -1) {
      fprintf(stderr, "Unexpected error read [errno=%d, errstr=%s] at %s:%d", read_ret, strerror(errno), __FILE__,
              __LINE__);
      exit(EXIT_FAILURE);
    }
    write_ret = write(fatfunction_fd_dump, (const void *)&exists, sizeof(int));
    if (write_ret == -1) {
      fprintf(stderr, "Unexpected error write [errno=%d, errstr=%s] at %s:%d", write_ret, strerror(errno), __FILE__,
              __LINE__);
      exit(EXIT_FAILURE);
    }

    if (exists) {
      bid = (uint3 *)malloc(sizeof(uint3));
      read_ret = read_all(fd, (void *)bid, sizeof(uint3));
      if (read_ret == -1) {
        fprintf(stderr, "Unexpected error read [errno=%d, errstr=%s] at %s:%d", read_ret, strerror(errno), __FILE__,
                __LINE__);
        exit(EXIT_FAILURE);
      }
      write_ret = write(fatfunction_fd_dump, (const void *)bid, sizeof(uint3));
      if (write_ret == -1) {
        fprintf(stderr, "Unexpected error write [errno=%d, errstr=%s] at %s:%d", write_ret, strerror(errno), __FILE__,
                __LINE__);
        exit(EXIT_FAILURE);
      }
    } else
      bid = NULL;

    read_ret = read_all(fd, (void *)&exists, sizeof(int));
    if (read_ret == -1) {
      fprintf(stderr, "Unexpected error read [errno=%d, errstr=%s] at %s:%d", read_ret, strerror(errno), __FILE__,
              __LINE__);
      exit(EXIT_FAILURE);
    }
    write_ret = write(fatfunction_fd_dump, (const void *)&exists, sizeof(int));
    if (write_ret == -1) {
      fprintf(stderr, "Unexpected error write [errno=%d, errstr=%s] at %s:%d", write_ret, strerror(errno), __FILE__,
              __LINE__);
      exit(EXIT_FAILURE);
    }

    if (exists) {
      bDim = (dim3 *)malloc(sizeof(dim3));
      read_ret = read_all(fd, (void *)bDim, sizeof(dim3));
      if (read_ret == -1) {
        fprintf(stderr, "Unexpected error read [errno=%d, errstr=%s] at %s:%d", read_ret, strerror(errno), __FILE__,
                __LINE__);
        exit(EXIT_FAILURE);
      }
      write_ret = write(fatfunction_fd_dump, (const void *)bDim, sizeof(dim3));
      if (write_ret == -1) {
        fprintf(stderr, "Unexpected error write [errno=%d, errstr=%s] at %s:%d", write_ret, strerror(errno), __FILE__,
                __LINE__);
        exit(EXIT_FAILURE);
      }
    } else
      bDim = NULL;

    read_ret = read_all(fd, (void *)&exists, sizeof(int));
    if (read_ret == -1) {
      fprintf(stderr, "Unexpected error read [errno=%d, errstr=%s] at %s:%d", read_ret, strerror(errno), __FILE__,
              __LINE__);
      exit(EXIT_FAILURE);
    }
    write_ret = write(fatfunction_fd_dump, (const void *)&exists, sizeof(int));
    if (write_ret == -1) {
      fprintf(stderr, "Unexpected error write [errno=%d, errstr=%s] at %s:%d", write_ret, strerror(errno), __FILE__,
              __LINE__);
      exit(EXIT_FAILURE);
    }
    if (exists) {
      gDim = (dim3 *)malloc(sizeof(dim3));
      read_ret = read_all(fd, (void *)gDim, sizeof(dim3));
      if (read_ret == -1) {
        fprintf(stderr, "Unexpected error read [errno=%d, errstr=%s] at %s:%d", read_ret, strerror(errno), __FILE__,
                __LINE__);
        exit(EXIT_FAILURE);
      }
      write_ret = write(fatfunction_fd_dump, (const void *)gDim, sizeof(dim3));
      if (write_ret == -1) {
        fprintf(stderr, "Unexpected error write [errno=%d, errstr=%s] at %s:%d", write_ret, strerror(errno), __FILE__,
                __LINE__);
        exit(EXIT_FAILURE);
      }
    } else
      gDim = NULL;

    read_ret = read_all(fd, (void *)&exists, sizeof(int));
    if (read_ret == -1) {
      fprintf(stderr, "Unexpected error read [errno=%d, errstr=%s] at %s:%d", read_ret, strerror(errno), __FILE__,
              __LINE__);
      exit(EXIT_FAILURE);
    }
    write_ret = write(fatfunction_fd_dump, (const void *)&exists, sizeof(int));
    if (write_ret == -1) {
      fprintf(stderr, "Unexpected error write [errno=%d, errstr=%s] at %s:%d", write_ret, strerror(errno), __FILE__,
              __LINE__);
      exit(EXIT_FAILURE);
    }
    if (exists) {
      wSize = (int *)malloc(sizeof(int));
      read_ret = read_all(fd, (void *)wSize, sizeof(int));
      if (read_ret == -1) {
        fprintf(stderr, "Unexpected error read [errno=%d, errstr=%s] at %s:%d", read_ret, strerror(errno), __FILE__,
                __LINE__);
        exit(EXIT_FAILURE);
      }
      write_ret = write(fatfunction_fd_dump, (const void *)wSize, sizeof(int));
      if (write_ret == -1) {
        fprintf(stderr, "Unexpected error write [errno=%d, errstr=%s] at %s:%d", write_ret, strerror(errno), __FILE__,
                __LINE__);
        exit(EXIT_FAILURE);
      }
    } else
      wSize = NULL;

    AVA_DEBUG.printf("Register function deviceName = %s\n", deviceName);
    func_id = (void *)g_hash_table_lookup(ht, deviceName);
    assert(func_id != NULL && "func_id should not be NULL");
    func = (fatbin_function *)g_ptr_array_index(fatbin_funcs, (intptr_t)func_id);
    // __helper_register_function(func, (const char *)func_id, mod, deviceName);

    free(deviceFun);
    free(deviceName);
    if (tid) free(tid);
    if (bid) free(bid);
    if (bDim) free(bDim);
    if (gDim) free(gDim);
    if (wSize) free(wSize);
  }

  size = 0;
  write_ret = write(fatfunction_fd_dump, (const void *)&size, sizeof(size_t));
  if (write_ret == -1) {
    fprintf(stderr, "Unexpected error write [errno=%d, errstr=%s] at %s:%d", write_ret, strerror(errno), __FILE__,
            __LINE__);
    exit(EXIT_FAILURE);
  }
  ++(global_num_fatbins);

  // dump size and wrapper to fatbin-info
  ss << dump_dir << "/fatbin-info.ava";
  filename = ss.str();
  fd = open(filename.c_str(), O_RDWR | O_CREAT, 0666);
  AVA_DEBUG.printf("Fatbinary counter = %d\n", global_num_fatbins);
  write_ret = write(fd, (const void *)&global_num_fatbins, sizeof(int));
  if (write_ret == -1) {
    fprintf(stderr, "Unexpected error write [errno=%d, errstr=%s] at %s:%d", write_ret, strerror(errno), __FILE__,
            __LINE__);
    exit(EXIT_FAILURE);
  }
  lseek(fd, 0, SEEK_END);
  write_ret = write(fd, (const void *)wp, sizeof(struct fatbin_wrapper));
  if (write_ret == -1) {
    fprintf(stderr, "Unexpected error write [errno=%d, errstr=%s] at %s:%d", write_ret, strerror(errno), __FILE__,
            __LINE__);
    exit(EXIT_FAILURE);
  }
  close(fd);

  g_hash_table_destroy(ht);
  // return fatbin_handle;
}

void load_dump_fatbin(char *load_dir, char *dump_dir) {
  /* Read cubin number */
  int fd = 0, ret = 0, fatbin_num = 0;
  std::stringstream ss;
  ss << load_dir << "/fatbin-info.ava";
  std::string filename = ss.str();
  fd = open(filename.c_str(), O_RDONLY, 0666);
  ret = read_all(fd, (void *)&fatbin_num, sizeof(int));
  if (ret == -1) {
    fprintf(stderr, "Unexpected error read [errno=%d, errstr=%s] at %s:%d", ret, strerror(errno), __FILE__, __LINE__);
  }
  AVA_DEBUG.printf("%s fatbin num is %d\n", load_dir, fatbin_num);

  int i;
  void *fatCubin;
  // void **fatbin_handle;
  for (i = 0; i < fatbin_num; i++) {
    fatCubin = malloc(sizeof(struct fatbin_wrapper));
    ret = read_all(fd, fatCubin, sizeof(struct fatbin_wrapper));
    if (ret == -1) {
      fprintf(stderr, "Unexpected error read [errno=%d, errstr=%s] at %s:%d", ret, strerror(errno), __FILE__, __LINE__);
    }
    __helper_load_and_dump_fatbin(load_dir, dump_dir, i, fatCubin);
  }
  close(fd);
  // reset metadata
  fatfunction_fd_load = 0;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "<usage>: %s <load_dir1> <load_dir2> <dump_dir>\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  char *load_dir1 = argv[1];
  char *load_dir2 = argv[2];
  char *dump_dir = argv[3];

  load_dump_fatbin(load_dir1, dump_dir);
  fprintf(stderr, "done loading to %s and dumping to %s\n", load_dir1, dump_dir);
  load_dump_fatbin(load_dir2, dump_dir);
  fprintf(stderr, "done loading to %s and dumping to %s\n", load_dir2, dump_dir);
}
