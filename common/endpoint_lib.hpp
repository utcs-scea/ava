#ifndef AVA_COMMON_ENDPOINT_LIB_HPP_
#define AVA_COMMON_ENDPOINT_LIB_HPP_

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef __cplusplus
#include <atomic>
using namespace std;
#else
#include <stdatomic.h>
#endif

#include <glib.h>
#include <glib/ghash.h>
#include <gmodule.h>
#include <string.h>
#include <sys/time.h>

#include "common/cmd_channel.hpp"
#include "common/cmd_handler.hpp"
#include "common/murmur3.h"
#include "common/shadow_thread_pool.hpp"

#ifdef __cplusplus
extern "C" {
#endif

//// Utilities

#define __ava_check_type(type, expr) \
  ({                                 \
    type __tmp = (expr);             \
    __tmp;                           \
  })

struct nw_handle_pool;

struct nw_handle_pool *nw_handle_pool_new();
void nw_handle_pool_free(struct nw_handle_pool *pool);
void *nw_handle_pool_insert(struct nw_handle_pool *pool, const void *handle);
void *nw_handle_pool_lookup_or_insert(struct nw_handle_pool *pool, const void *handle);
GPtrArray *nw_handle_pool_get_live_handles(struct nw_handle_pool *pool);
void *nw_handle_pool_deref(struct nw_handle_pool *pool, const void *id);
void *nw_handle_pool_deref_and_remove(struct nw_handle_pool *pool, const void *id);
void nw_handle_pool_assign_handle(struct nw_handle_pool *pool, const void *id, const void *handle);

gboolean nw_hash_table_remove_flipped(gconstpointer key, GHashTable *hash_table);

gpointer nw_hash_table_steal_value(GHashTable *hash_table, gconstpointer key);

#ifdef NDEBUG
#define abort_with_reason(reason) abort()
#else
#define abort_with_reason(reason) __assert_fail(reason, __FILE__, __LINE__, __FUNCTION__)
#endif
#define AVA_CHECK_RET(code) \
  if (G_UNLIKELY(!(code))) abort_with_reason("Function returned failure: " __STRING(code))

#ifndef AVA_RELEASE
#define AVA_DEBUG_ASSERT(code) assert(code)
#else
#define AVA_DEBUG_ASSERT(code)
#endif

#define __INTERNAL_PRAGMA(t) _Pragma(#t)
#define __PRAGMA(t) __INTERNAL_PRAGMA(t)

/// A combined compile time warning and runtime abort_with_reason.
#define ABORT_TODO(t) __PRAGMA(GCC warning __STRING(TODO : t)) abort_with_reason(__STRING(TODO : t))

/* Sentinel to tell worker there is a buffer to return data into. */
#define HAS_OUT_BUFFER_SENTINEL ((void *)1)

/// Extract the explicit state of the object `o` and return it as a malloc'd buffer.
/// The caller takes ownership of the buffer. The length of the buffer must be written to `*length`.
typedef void *(*ava_extract_function)(void *obj, size_t *length);

/// Replace (reconstruct) the explicit state of the object `o` from data (which has length `length`).
typedef void (*ava_replace_function)(void *obj, void *data, size_t length);

typedef void *(*ava_allocator)(size_t size);
typedef void (*ava_deallocator)(void *ptr);

//// Library functions/macros expected by metadata expressions

enum ava_sync_mode_t { NW_ASYNC = 0, NW_SYNC, NW_FLUSH };

enum ava_transfer_t { NW_NONE = 0, NW_HANDLE, NW_OPAQUE, NW_BUFFER, NW_CALLBACK, NW_CALLBACK_REGISTRATION, NW_FILE };

enum ava_lifetime_t {
  AVA_CALL = 0,
  AVA_COUPLED,
  AVA_STATIC,
  AVA_MANUAL,
};

struct ava_metadata_base {
  //! For handles and buffers
  GPtrArray * /* ava_shadow_record_t* */ coupled_shadow_buffers;
  struct ava_shadow_record_t *shadow;

  //! For handles (worker only)
  ava_extract_function extract;
  ava_replace_function replace;
  GPtrArray * /* ava_offset_pair_t* */ recorded_calls;
  GPtrArray * /* handle */ dependencies;

  //! For buffers
  ava_deallocator deallocator;
};

struct call_id_and_handle_t {
  int call_id;
  const void *handle;
};

__attribute__((const)) guint nw_hash_mix64variant13(gconstpointer ptr);

__attribute__((pure)) static inline guint nw_hash_struct(gconstpointer ptr, int size) {
  guint ret;
  MurmurHash3_x86_32(ptr, size, 0xfbcdabc7 + size, &ret);
  return ret;
}

__attribute__((pure)) guint nw_hash_call_id_and_handle(gconstpointer ptr);

__attribute__((pure)) gint nw_equal_call_id_and_handle(gconstpointer ptr1, gconstpointer ptr2);

__attribute__((const)) static inline guint nw_hash_pointer(gconstpointer ptr) { return nw_hash_mix64variant13(ptr); }

struct ava_handle_pair_t {
  void *a;
  void *b;
};

struct ava_offset_pair_t {
  size_t a;
  size_t b;
};

static inline struct ava_offset_pair_t *ava_new_offset_pair(size_t a, size_t b) {
  struct ava_offset_pair_t *ret = (struct ava_offset_pair_t *)malloc(sizeof(struct ava_offset_pair_t));
  ret->a = a;
  ret->b = b;
  return ret;
}

/// Create a new metadata map.
GHashTable *metadata_map_new();

extern struct nw_handle_pool *nw_global_handle_pool;
extern struct shadow_thread_pool_t *nw_shadow_thread_pool;
extern GHashTable *nw_global_metadata_map;
extern pthread_mutex_t nw_global_metadata_map_mutex;

struct ava_replay_command_t;

/**
 * Extract state from a set of objects and write it to output_chan.
 * @param output_chan The output channel which will receive the state.
 * @param log_chan The log channel which has been used by the system.
 * @param to_extract The set of root objects to extract.
 */
void ava_extract_objects(struct command_channel *output_chan, struct command_channel_log *log_chan,
                         GPtrArray *to_extract);

/**
 * Extract state from a set of objects, combine each pair into a large command and write it to output_chan.
 * @param output_chan The output channel which will receive the state.
 * @param log_chan The log channel which has been used by the system.
 * @param to_extract The set of root objects to extract.
 */
void ava_extract_objects_in_pair(struct command_channel *output_chan, struct command_channel_log *log_chan,
                                 GPtrArray *to_extract);

/**
 * Execute the replacement based on cmd.
 * @param chan
 * @param cmd
 */
void ava_handle_replace_explicit_state(struct command_channel *chan, struct nw_handle_pool *handle_pool,
                                       struct ava_replay_command_t *cmd);

//! Custom allocator/deallocator handling
struct ava_buffer_with_deallocator;

/** Create a container with the deallocator.
 *
 * The new object takes ownership of buffer.
 * @param deallocator The deallocator for buffer.
 * @param buffer A pointer of some kind.
 * @return
 */
struct ava_buffer_with_deallocator *ava_buffer_with_deallocator_new(void (*deallocator)(void *), void *buffer);

/** Use the stored deallocator to free the buffer.
 *
 * @param buffer
 */
void ava_buffer_with_deallocator_free(struct ava_buffer_with_deallocator *buffer);

//! Callback handling

/**
 * The representation of a callback closure. These are stored in endpoint where the callback will execute.
 */
struct ava_callback_user_data {
  void *userdata;
  void *function_pointer;
};

//! The endpoint representation itself

#define metadata_map nw_global_metadata_map
#define metadata_map_mutex nw_global_metadata_map_mutex

struct ava_shadow_buffer_pool {
  pthread_mutex_t mutex;
  GHashTable *buffers_by_id;
  // The mapping by local is in the metadata_map
};

/**
 * The representation of an endpoint used by many endpoint utility functions.
 *
 * Generated code does not access these fields directly. Instead it uses the functions below as "public methods".
 */
struct ava_endpoint {
  size_t metadata_size;
  GHashTable *managed_buffer_map;
  GHashTable *managed_by_coupled_map;
  pthread_mutex_t managed_buffer_map_mutex;
  // GHashTable* metadata_map;
  // pthread_mutex_t metadata_map_mutex = PTHREAD_MUTEX_INITIALIZER;
  GHashTable *call_map;
  pthread_mutex_t call_map_mutex;
  atomic_long call_counter;
  // struct ava_zcopy_region *zcopy_region;
  struct ava_shadow_buffer_pool shadow_buffers;
#ifdef AVA_BENCHMARKING_MIGRATE
  intptr_t migration_call_id;
#endif
};

extern struct ava_endpoint __ava_endpoint;

/**
 * Get the metadata for an object. This uses a lock to protect the hash_table.
 *
 * The `pure` annotation is a bit of a
 * stretch since we do insert elements into the hash table, but having the compiler perform CSE on this function is
 * pretty important and this function is idempotent.
 * @param endpoint
 * @param p The handle with which that returned metadata is associated.
 * @return
 */
__attribute__((pure)) struct ava_metadata_base *ava_internal_metadata(struct ava_endpoint *endpoint, const void *p);

/**
 * Get the next call ID.
 * @param endpoint The endpoint structure.
 * @return A new call ID.
 */
intptr_t ava_get_call_id(struct ava_endpoint *endpoint);

/**
 * Add a call record to the collection on in-flight calls.
 * @param endpoint
 * @param id The call ID.
 * @param ptr The call record itself.
 */
void ava_add_call(struct ava_endpoint *endpoint, intptr_t id, void *ptr);

/**
 * Find and remove a call record by its ID.
 * @param endpoint
 * @param id The call ID.
 * @return The call record.
 */
void *ava_remove_call(struct ava_endpoint *endpoint, intptr_t id);

/**
 * Allocate or get the buffer associated with `coupled`.
 * The buffer will be freed when `coupled` is deallocated.
 * @param endpoint
 * @param cmd_id The command ID to cache the buffer for.
 * @param coupled The opaque value to cache the buffer for (generally a handle).
 * @param size The size of the buffer.
 * @return The new or cached buffer.
 */
void *ava_cached_alloc(struct ava_endpoint *endpoint, int cmd_id, const void *coupled, size_t size);

/**
 * Allocate a new buffer associated with `coupled`.
 * The buffer will be freed when `coupled` is deallocated.
 * @param endpoint
 * @param coupled The opaque value to cache the buffer for (generally a handle).
 * @param size The size of the buffer.
 * @return The new buffer.
 */
void *ava_uncached_alloc(struct ava_endpoint *endpoint, const void *coupled, size_t size);

/**
 * Allocate or get the buffer associated with a specific command ID.
 * @param endpoint
 * @param cmd_id The command ID to cache the buffer for.
 * @param size The size of the buffer.
 * @return The new or cached buffer.
 */
void *ava_static_alloc(struct ava_endpoint *endpoint, int cmd_id, size_t size);

/**
 * Free all buffers coupled to `coupled`.
 * @param endpoint
 * @param coupled
 */
void ava_coupled_free(struct ava_endpoint *endpoint, const void *coupled);

// __attribute_alloc_size__((2)) __attribute_malloc__ __attribute_used__
//     static void *ava_endpoint_zerocopy_alloc(struct ava_endpoint *endpoint, size_t size) {
//   return ava_zcopy_region_alloc(endpoint->zcopy_region, size);
// }

// __attribute_used__ static void ava_endpoint_zerocopy_free(struct ava_endpoint *endpoint, void *ptr) {
//   ava_zcopy_region_free(endpoint->zcopy_region, ptr);
// }

// __attribute_used__ static uintptr_t ava_endpoint_zerocopy_get_physical_address(struct ava_endpoint *endpoint,
//                                                                                void *ptr) {
//   return ava_zcopy_region_get_physical_address(endpoint->zcopy_region, ptr);
// }

/**
 * Record a call with for the object `handle`.
 * @param endpoint
 * @param handle The object to record the call for.
 * @param pair The pair of offsets representing the call and ret commands.
 */
void ava_add_recorded_call(struct ava_endpoint *endpoint, void *handle, struct ava_offset_pair_t *pair);

/**
 * Mark all commands associated with `handle` as deleted in the log.
 * @param endpoint
 * @param log The log in which the commands should be marked.
 * @param handle The handle to expunge.
 */
void ava_expunge_recorded_calls(struct ava_endpoint *endpoint, struct command_channel_log *log, void *handle);

/**
 * Add a dependency of `a` on `b` for use during state extraction.
 * @param endpoint
 * @param a
 * @param b
 */
void ava_add_dependency(struct ava_endpoint *endpoint, void *a, void *b);

/**
 * Assign `extract` and `replace` function for `handle`.
 * @param endpoint
 * @param handle
 * @param extract
 * @param replace
 */
void ava_assign_record_replay_functions(struct ava_endpoint *endpoint, const void *handle, ava_extract_function extract,
                                        ava_replace_function replace);

/**
 * Initialize an ava_endpoint.
 * @param endpoint
 * @param metadata_size The size of the metadata structure used by this endpoint.
 * @param counter_tag The tag placed in the low 4 bits of counters to make sure they are different between different ID
 * generation scopes.
 */
void ava_endpoint_init(struct ava_endpoint *endpoint, size_t metadata_size, uint8_t counter_tag);

#define AVA_COUNTER_TAG_WORKER 1
#define AVA_COUNTER_TAG_GUEST 2

/**
 * Destroy all structures associated with `endpoint`.
 * @param endpoint
 */
void ava_endpoint_destroy(struct ava_endpoint *endpoint);

//! Shadow buffer handling

struct ava_buffer_header_t {
  /**
   * The buffer on the sender that provided this data. This is used as an ID
   * to identify the shadow buffer.
   */
  void *id;

  /**
   * True (non-zero) if this buffer has real data. If this is False then there
   * is no data attached to this buffer and the content can be undefined.
   */
  uint8_t has_data;

  /**
   * The size of the buffer. This is not strictly needed, but provides a very
   * useful check to make sure size computations are consistent. It can be
   * removed if the space this value takes up becomes an issue.
   */
  size_t size;
};

__attribute__((pure)) static inline size_t ava_shadow_buffer_size(struct ava_endpoint *endpoint,
                                                                  struct command_channel *chan, size_t size) {
  return command_channel_buffer_size(chan, sizeof(struct ava_buffer_header_t)) +
         command_channel_buffer_size(chan, size);
}

__attribute__((pure)) static inline size_t ava_shadow_buffer_size_without_data(struct ava_endpoint *endpoint,
                                                                               struct command_channel *chan,
                                                                               size_t size) {
  return command_channel_buffer_size(chan, sizeof(struct ava_buffer_header_t));
}

/**
 * Attach a buffer to a command with all information need to make or update a shadow buffer.
 * @param endpoint
 * @param chan
 * @param cmd
 * @param local Attach the buffer as if it where this one.
 * @param data_buffer Use the data from this buffer.
 * @param size
 * @param lifetime
 * @param alloc
 * @param dealloc
 * @param header
 * @return
 */
void *ava_shadow_buffer_attach_buffer(struct ava_endpoint *endpoint, struct command_channel *chan,
                                      struct command_base *cmd, const void *local, const void *data_buffer, size_t size,
                                      enum ava_lifetime_t lifetime, ava_allocator alloc, ava_deallocator dealloc,
                                      struct ava_buffer_header_t *header);

/**
 * Pseudo-attach a buffer to a command with all the information needed to create or identify a shadow buffer.
 * This effectively transports a buffer without updating the data in it.
 * @param endpoint
 * @param chan
 * @param cmd
 * @param local
 * @param size
 * @param lifetime
 * @param alloc
 * @param dealloc
 * @param header
 * @return
 */
void *ava_shadow_buffer_attach_buffer_without_data(struct ava_endpoint *endpoint, struct command_channel *chan,
                                                   struct command_base *cmd, const void *local, const void *data_buffer,
                                                   size_t size, enum ava_lifetime_t lifetime, ava_allocator alloc,
                                                   ava_deallocator dealloc, struct ava_buffer_header_t *header);

__attribute__((pure)) void *ava_shadow_buffer_get_buffer(struct ava_endpoint *endpoint, struct command_channel *chan,
                                                         struct command_base *cmd, void *offset,
                                                         enum ava_lifetime_t lifetime, void *lifetime_coupled,
                                                         size_t *size_out, ava_allocator alloc,
                                                         ava_deallocator dealloc);

void ava_shadow_buffer_free_coupled(struct ava_endpoint *endpoint, void *obj);

#ifdef __cplusplus
}
#endif

#endif  // AVA_COMMON_ENDPOINT_LIB_HPP_
