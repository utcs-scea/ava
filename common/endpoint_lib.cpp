#include "common/endpoint_lib.hpp"

#include <glib.h>
#include <plog/Log.h>
#include <pthread.h>
#include <stdlib.h>

#include <atomic>
#include <cstdint>

#include "common/shadow_thread_pool.hpp"

struct ava_endpoint __ava_endpoint;

struct nw_handle_pool {
  GHashTable *to_handle;
  GHashTable *to_id;
  pthread_mutex_t lock;
};

struct ava_replay_command_t {
  struct command_base base;
  void *id;
  void *data;
  size_t data_length;
};

#define counter_tag_mask 0xfL
#define counter_count_shift 4
#define counter_count_min 1024 / (1 << counter_count_shift)
#define counter_count_prefix (0xffffffUL << 40)

// TODO: These values should be inside the endpoint, but this makes the code for
// handle_pool confused.
static uintptr_t global_counter_tag = 0;
static std::atomic<std::uint64_t> global_counter(counter_count_min);

static inline void *next_id() {
  uintptr_t n;
  while ((n = global_counter.fetch_add(1)) < counter_count_min)
    ;
  // Shift the id and add the tag.
  uintptr_t tmp = (n << counter_count_shift);
  return (void *)(tmp | global_counter_tag | counter_count_prefix);
}

static inline void update_next_id(uintptr_t v) {
  // Remove the tag and add 1 to give a reasonable "next" based on v.
  v = (v >> counter_count_shift) + 1;
  // Remove the prefix to give the counter value
  v &= (~counter_count_prefix);
  std::uint64_t old;
  do {
    old = global_counter;
  } while (old < v && !global_counter.compare_exchange_strong(old, v));
}

// TODO: Perhaps expose this function to the spec as ava_is_handle, so
// that the spec would have an explicit conditional for handle and
// non-handle cases.
static int is_handle(const void *id) {
  uintptr_t prefix = (uintptr_t)id & counter_count_prefix;
  return !(prefix ^ counter_count_prefix);
}

struct nw_handle_pool *nw_handle_pool_new() {
  struct nw_handle_pool *ret = (struct nw_handle_pool *)malloc(sizeof(struct nw_handle_pool));
  ret->to_handle = metadata_map_new();
  ret->to_id = metadata_map_new();
  pthread_mutex_init(&ret->lock, NULL);
  return ret;
}

void nw_handle_pool_free(struct nw_handle_pool *pool) {
  pthread_mutex_lock(&pool->lock);
  g_hash_table_unref(pool->to_handle);
  g_hash_table_unref(pool->to_id);
  pthread_mutex_unlock(&pool->lock);
  free(pool);
}

static void *internal_handle_pool_insert(struct nw_handle_pool *pool, const void *handle) {
  void *id = next_id();
  gboolean b = g_hash_table_insert(pool->to_id, (gpointer)handle, id);
  assert(b && "handle already exists");
  b = g_hash_table_insert(pool->to_handle, id, (gpointer)handle);
  assert(b && "id already exists");
  (void)b;
  return id;
}

void *nw_handle_pool_insert(struct nw_handle_pool *pool, const void *handle) {
  if (handle == NULL || pool == NULL) return (void *)handle;
  pthread_mutex_lock(&pool->lock);
  void *id = internal_handle_pool_insert(pool, handle);
  pthread_mutex_unlock(&pool->lock);
  return id;
}

void *nw_handle_pool_lookup_or_insert(struct nw_handle_pool *pool, const void *handle) {
  if (handle == NULL || pool == NULL) return (void *)handle;
  pthread_mutex_lock(&pool->lock);
  void *id = g_hash_table_lookup(pool->to_id, handle);
  if (id == NULL) id = internal_handle_pool_insert(pool, handle);
  pthread_mutex_unlock(&pool->lock);
  return id;
}

GPtrArray *nw_handle_pool_get_live_handles(struct nw_handle_pool *pool) {
  pthread_mutex_lock(&pool->lock);
  GPtrArray *ret = g_ptr_array_sized_new(g_hash_table_size(pool->to_id));
  GHashTableIter iter;
  gpointer key;
  g_hash_table_iter_init(&iter, pool->to_id);
  while (g_hash_table_iter_next(&iter, &key, NULL)) {
    g_ptr_array_add(ret, key);
  }
  pthread_mutex_unlock(&pool->lock);
  return ret;
}

void *nw_handle_pool_deref(struct nw_handle_pool *pool, const void *id) {
  /* Currently ava_handle implicitly treats non-handles as ava_opaque
   * and this will probably change in the future. */
  if (id == NULL || pool == NULL || !is_handle(id)) return (void *)id;
  pthread_mutex_lock(&pool->lock);
  void *handle = g_hash_table_lookup(pool->to_handle, id);
  assert(handle != NULL);
  pthread_mutex_unlock(&pool->lock);
  return handle;
}

void *nw_handle_pool_deref_and_remove(struct nw_handle_pool *pool, const void *id) {
  if (id == NULL || pool == NULL) return (void *)id;
  pthread_mutex_lock(&pool->lock);
  void *handle = nw_hash_table_steal_value(pool->to_handle, id);
  assert(handle != NULL);
  AVA_CHECK_RET(g_hash_table_remove(pool->to_id, handle));
  pthread_mutex_unlock(&pool->lock);
  return handle;
}

void nw_handle_pool_assign_handle(struct nw_handle_pool *pool, const void *id, const void *handle) {
  assert((id == NULL) == (handle == NULL) && "Either id and handle must be NULL or neither");
  if (id == NULL || pool == NULL) return;

  pthread_mutex_lock(&pool->lock);
  update_next_id((uintptr_t)id);

  // Check that there is not already a handle with this ID
  void *old_handle = g_hash_table_lookup(pool->to_handle, id);
  assert((old_handle == NULL || old_handle == handle) && "Handle ID assigned to a different handle during replay");
  // Do not check if there is already an ID with this handle since that could
  // happen legitimately (e.g., if a handle is reused by the underlying library
  // in the replay, but not in the original execution).

  // Allow overwriting an old ID with this ID.
  g_hash_table_insert(pool->to_id, (gpointer)handle, (gpointer)id);

  // Insert the handle if it is not already present
  if (old_handle == NULL) g_hash_table_insert(pool->to_handle, (gpointer)id, (gpointer)handle);

  pthread_mutex_unlock(&pool->lock);
}

gboolean nw_hash_table_remove_flipped(gconstpointer key, GHashTable *hash_table) {
  return g_hash_table_remove(hash_table, key);
}

gpointer nw_hash_table_steal_value(GHashTable *hash_table, gconstpointer key) {
  gpointer value = g_hash_table_lookup(hash_table, key);
  gboolean b = g_hash_table_steal(hash_table, key);
  /* In GLIB 2.58 we could use the following:
     g_hash_table_steal_extended(hash_table, key, NULL, &value); */
  if (!b) {
    return NULL;
  } else {
    assert(value != NULL);
    return value;
  }
}

guint nw_hash_call_id_and_handle(gconstpointer ptr) { return nw_hash_struct(ptr, sizeof(struct call_id_and_handle_t)); }

gint nw_equal_call_id_and_handle(gconstpointer ptr1, gconstpointer ptr2) {
  return memcmp(ptr1, ptr2, sizeof(struct call_id_and_handle_t)) == 0;
}

/// 64-bit to 64-bit hash from http://dx.doi.org/10.1145/2714064.2660195.
/// Modified to produce a 32-bit result.
/// This hash was chosen based on the paper above and
/// https://nullprogram.com/blog/2018/07/31/.
guint nw_hash_mix64variant13(gconstpointer ptr) {
  uintptr_t x = (uintptr_t)ptr;
  x ^= x >> 30;
  x *= UINT64_C(0xbf58476d1ce4e5b9);
  x ^= x >> 27;
  x *= UINT64_C(0x94d049bb133111eb);
  x ^= x >> 31;
  return (guint)x;
}

GHashTable *metadata_map_new() { return g_hash_table_new(nw_hash_mix64variant13, g_direct_equal); }

struct ava_extraction_state_t {
  GHashTable *const dependencies;
  GPtrArray *const offset_pairs;
  struct nw_handle_pool *const pool;
  GHashTable *const metadata_map;
  struct command_channel *output_chan;
};

static void ava_ptr_array_add_flipped(gpointer data, GPtrArray *array) { g_ptr_array_add(array, data); }

static gint _ava_order_by_a(const struct ava_offset_pair_t **x, const struct ava_offset_pair_t **y) {
  if ((*x)->a < (*y)->a) return -1;
  if ((*x)->a > (*y)->a) return 1;
  return 0;
}

static void _ava_extract_traverse(gpointer root, struct ava_extraction_state_t *state) {
  // Add the dependency to the overall set
  gboolean added = g_hash_table_add(state->dependencies, root);
  if (added) {
    struct ava_metadata_base *metadata = (struct ava_metadata_base *)g_hash_table_lookup(state->metadata_map, root);
    LOG_DEBUG << "root addr=" << std::hex << (uintptr_t)root << ", id=" << std::hex
              << (uintptr_t)g_hash_table_lookup(state->pool->to_id, root);
    if (metadata == NULL) return;

    // Copy recorded calls into the array
    if (metadata->recorded_calls)
      g_ptr_array_foreach(metadata->recorded_calls, (GFunc)ava_ptr_array_add_flipped, state->offset_pairs);

    // Recurse on all dependencies
    if (metadata->dependencies) g_ptr_array_foreach(metadata->dependencies, (GFunc)_ava_extract_traverse, state);

    // Base cases are either encountering a node which is already in
    // state->dependencies or a node with no dependencies.
  }
}

static void _ava_transfer_command(struct command_channel *output_chan, struct command_channel_log *log_chan,
                                  size_t offset) {
  uint32_t flags;
  struct command_base *msg = command_channel_log_load_command(log_chan, offset, &flags);
  if (flags != 1) {
    command_channel_transfer_command(output_chan, (struct command_channel *)log_chan, msg);
  }
  command_channel_free_command((struct command_channel *)log_chan, msg);
}

static void _ava_extract_explicit(gpointer obj, gpointer value, struct ava_extraction_state_t *state) {
  struct ava_metadata_base *metadata = (struct ava_metadata_base *)g_hash_table_lookup(state->metadata_map, obj);
  if (metadata == NULL) return;
  if (!metadata->extract) return;

  void *id = nw_handle_pool_lookup_or_insert(state->pool, obj);
  size_t len = -1;
  void *data = metadata->extract(obj, &len);
  assert(len != -1);

  size_t chan_len = command_channel_buffer_size(state->output_chan, len);

  struct ava_replay_command_t *cmd = (struct ava_replay_command_t *)command_channel_new_command(
      state->output_chan, sizeof(struct ava_replay_command_t), chan_len);
  cmd->base.api_id = COMMAND_HANDLER_API;
  cmd->base.command_id = COMMAND_HANDLER_REPLACE_EXPLICIT_STATE;
  cmd->id = id;
  cmd->data_length = len;
  cmd->data = command_channel_attach_buffer(state->output_chan, (struct command_base *)cmd, data, len);
  command_channel_send_command(state->output_chan, (struct command_base *)cmd);
}

void ava_extract_objects(struct command_channel *output_chan, struct command_channel_log *log_chan,
                         GPtrArray *to_extract) {
  GHashTable *const dependencies = g_hash_table_new(nw_hash_mix64variant13, g_direct_equal);
  GPtrArray *const offset_pairs = g_ptr_array_new_full(to_extract->len, NULL);
  struct ava_extraction_state_t state = {dependencies, offset_pairs, nw_global_handle_pool, nw_global_metadata_map,
                                         output_chan};

  // Add all globally recorded commands. NULL is the sentinel value for global.
  _ava_extract_traverse(NULL, &state);
  // Collect all offset pairs into an array, and all dependencies into a set
  g_ptr_array_foreach(to_extract, (GFunc)_ava_extract_traverse, &state);

  // Sort the array by call offset
  g_ptr_array_sort(offset_pairs, (GCompareFunc)_ava_order_by_a);

  // Extract all unique commands from the array into destination
  size_t prev_call_offset = -1;
  for (size_t i = 0; i < offset_pairs->len; i++) {
    const struct ava_offset_pair_t *pair = (const struct ava_offset_pair_t *)g_ptr_array_index(offset_pairs, i);
    if (pair->a != prev_call_offset) {
      _ava_transfer_command(output_chan, log_chan, pair->a);
      _ava_transfer_command(output_chan, log_chan, pair->b);
    }
    prev_call_offset = pair->a;
  }

  // Extract explicit state from all objects into destination
  g_hash_table_foreach(dependencies, (GHFunc)_ava_extract_explicit, &state);
}

// TODO: replace _ava_transfer_command
//  AMP: See comment on ava_extract_objects_in_pair
static void _ava_transfer_command_in_pair(struct command_channel *output_chan, struct command_channel_log *log_chan,
                                          size_t offset_a, size_t offset_b) {
  uint32_t flags_a, flags_b;
  struct command_base *call_cmd = command_channel_log_load_command(log_chan, offset_a, &flags_a);
  struct command_base *ret_cmd = command_channel_log_load_command(log_chan, offset_b, &flags_b);

  if (flags_a != 1 && flags_b != 1) {
    size_t chan_len = command_channel_buffer_size(output_chan, call_cmd->command_size + call_cmd->region_size) +
                      command_channel_buffer_size(output_chan, ret_cmd->command_size + ret_cmd->region_size);
    struct ava_replay_command_pair_t *combine = (struct ava_replay_command_pair_t *)command_channel_new_command(
        output_chan, sizeof(struct ava_replay_command_pair_t), chan_len);

    combine->base.api_id = COMMAND_HANDLER_API;
    combine->base.command_id = COMMAND_HANDLER_RECORDED_PAIR;
    combine->call_cmd = command_channel_attach_buffer(output_chan, (struct command_base *)combine, call_cmd,
                                                      call_cmd->command_size + call_cmd->region_size);
    combine->ret_cmd = command_channel_attach_buffer(output_chan, (struct command_base *)combine, ret_cmd,
                                                     ret_cmd->command_size + ret_cmd->region_size);

    command_channel_send_command(output_chan, (struct command_base *)combine);
  }

  command_channel_free_command((struct command_channel *)log_chan, call_cmd);
  command_channel_free_command((struct command_channel *)log_chan, ret_cmd);
}

// TODO: merge it into ava_extract_objects
//  AMP: I think we should be able to replace the original function with this
//  one. The paired replay commands should work in all cases.
void ava_extract_objects_in_pair(struct command_channel *output_chan, struct command_channel_log *log_chan,
                                 GPtrArray *to_extract) {
  GHashTable *const dependencies = g_hash_table_new(nw_hash_mix64variant13, g_direct_equal);
  GPtrArray *const offset_pairs = g_ptr_array_new_full(to_extract->len, NULL);
  struct ava_extraction_state_t state = {dependencies, offset_pairs, nw_global_handle_pool, nw_global_metadata_map,
                                         output_chan};

  // Add all globally recorded commands. NULL is the sentinel value for global.
  _ava_extract_traverse(NULL, &state);
  // Collect all offset pairs into an array, and all dependencies into a set
  g_ptr_array_foreach(to_extract, (GFunc)_ava_extract_traverse, &state);

  // Sort the array by call offset
  g_ptr_array_sort(offset_pairs, (GCompareFunc)_ava_order_by_a);

  // Extract all unique commands from the array into destination
  size_t prev_call_offset = -1;
  for (size_t i = 0; i < offset_pairs->len; i++) {
    const struct ava_offset_pair_t *pair = (const struct ava_offset_pair_t *)g_ptr_array_index(offset_pairs, i);
    if (pair->a != prev_call_offset) {
      LOG_DEBUG << "transfer log pair, offset=" << std::hex << pair->a;
      _ava_transfer_command_in_pair(output_chan, log_chan, pair->a, pair->b);
    }
    prev_call_offset = pair->a;
  }

  // Extract explicit state from all objects into destination
  g_hash_table_foreach(dependencies, (GHFunc)_ava_extract_explicit, &state);
}

void ava_handle_replace_explicit_state(struct command_channel *chan, struct nw_handle_pool *handle_pool,
                                       struct ava_replay_command_t *cmd) {
  assert(cmd->base.command_id == COMMAND_HANDLER_REPLACE_EXPLICIT_STATE);
  void *obj = nw_handle_pool_deref(handle_pool, cmd->id);
  struct ava_metadata_base *metadata = (struct ava_metadata_base *)g_hash_table_lookup(nw_global_metadata_map, obj);
  void *data = command_channel_get_buffer(chan, (struct command_base *)cmd, cmd->data);
  assert(metadata->replace);
  metadata->replace(obj, data, cmd->data_length);
}

struct nw_handle_pool *nw_global_handle_pool;
struct shadow_thread_pool_t *nw_shadow_thread_pool;
GHashTable *nw_global_metadata_map;
pthread_mutex_t nw_global_metadata_map_mutex = PTHREAD_MUTEX_INITIALIZER;

void __attribute__((constructor(0))) init_endpoint_lib(void) {
  nw_global_handle_pool = nw_handle_pool_new();
  nw_global_metadata_map = metadata_map_new();
  nw_shadow_thread_pool = shadow_thread_pool_new();
}

struct ava_buffer_with_deallocator {
  void (*deallocator)(void *);
  void *buffer;
};

struct ava_buffer_with_deallocator *ava_buffer_with_deallocator_new(void (*deallocator)(void *), void *buffer) {
  struct ava_buffer_with_deallocator *ret =
      (struct ava_buffer_with_deallocator *)malloc(sizeof(struct ava_buffer_with_deallocator));
  ret->buffer = buffer;
  ret->deallocator = deallocator;
  return ret;
}

void ava_buffer_with_deallocator_free(struct ava_buffer_with_deallocator *buffer) {
  buffer->deallocator(buffer->buffer);
  free(buffer);
}

struct ava_coupled_record_t {
  GPtrArray * /* elements: struct call_id_and_handle_t* */ key_list;
  GPtrArray *buffer_list;
};

void ava_coupled_record_free(struct ava_endpoint *endpoint, struct ava_coupled_record_t *r) {
  // Dealloc all keys and buffers (by removing them from the buffer map and
  // triggering their destroy callbacks)
  g_ptr_array_foreach(r->key_list, (GFunc)nw_hash_table_remove_flipped, endpoint->managed_buffer_map);
  // Dealloc the key_list itself (it has no destroy callback)
  g_ptr_array_unref(r->key_list);
  // Dealloc all buffers (by destroy callback)
  g_ptr_array_unref(r->buffer_list);
  free(r);
}

struct ava_coupled_record_t *ava_coupled_record_new() {
  struct ava_coupled_record_t *ret = (struct ava_coupled_record_t *)malloc(sizeof(struct ava_coupled_record_t));
  ret->key_list = g_ptr_array_new_full(1, NULL);
  ret->buffer_list = g_ptr_array_new_full(1, (GDestroyNotify)g_array_unref);
  return ret;
}

static struct ava_metadata_base *ava_internal_metadata_unlocked(struct ava_endpoint *endpoint, const void *ptr) {
  void *metadata = g_hash_table_lookup(metadata_map, ptr);
  if (metadata == NULL) {
    metadata = calloc(1, endpoint->metadata_size);
    g_hash_table_insert(metadata_map, (void *)ptr, metadata);
  }
  return (struct ava_metadata_base *)metadata;
}

struct ava_metadata_base *ava_internal_metadata(struct ava_endpoint *endpoint, const void *ptr) {
  pthread_mutex_lock(&metadata_map_mutex);
  struct ava_metadata_base *ret = ava_internal_metadata_unlocked(endpoint, ptr);
  pthread_mutex_unlock(&metadata_map_mutex);
  return ret;
}

struct ava_metadata_base *ava_internal_metadata_no_create(struct ava_endpoint *endpoint, const void *ptr) {
  pthread_mutex_lock(&metadata_map_mutex);
  struct ava_metadata_base *ret = (struct ava_metadata_base *)g_hash_table_lookup(metadata_map, ptr);
  pthread_mutex_unlock(&metadata_map_mutex);
  return ret;
}

intptr_t ava_get_call_id(struct ava_endpoint *endpoint) { return endpoint->call_counter.fetch_add(1); }

void ava_add_call(struct ava_endpoint *endpoint, intptr_t id, void *ptr) {
  pthread_mutex_lock(&endpoint->call_map_mutex);
  gboolean b = g_hash_table_insert(endpoint->call_map, (void *)id, ptr);
  assert(b && "Adding a call ID which currently exists.");
  (void)b;
  pthread_mutex_unlock(&endpoint->call_map_mutex);
}

void *ava_remove_call(struct ava_endpoint *endpoint, intptr_t id) {
  pthread_mutex_lock(&endpoint->call_map_mutex);
  void *ptr = nw_hash_table_steal_value(endpoint->call_map, (void *)id);
  assert(ptr != NULL && "Removing a call ID which does not exist");
  pthread_mutex_unlock(&endpoint->call_map_mutex);
  return ptr;
}

static struct ava_coupled_record_t *ava_get_coupled_record_unlocked(struct ava_endpoint *endpoint,
                                                                    const void *coupled) {
  struct ava_coupled_record_t *rec =
      (struct ava_coupled_record_t *)g_hash_table_lookup(endpoint->managed_by_coupled_map, coupled);
  if (rec == NULL) {
    rec = ava_coupled_record_new();
    g_hash_table_insert(endpoint->managed_by_coupled_map, (void *)coupled, rec);
  }
  return rec;
}

void *ava_cached_alloc(struct ava_endpoint *endpoint, int call_id, const void *coupled, size_t size) {
  pthread_mutex_lock(&endpoint->managed_buffer_map_mutex);
  struct call_id_and_handle_t key = {call_id, coupled};
  GArray *buffer = (GArray *)g_hash_table_lookup(endpoint->managed_buffer_map, &key);
  if (buffer == NULL) {
    buffer = g_array_sized_new(FALSE, TRUE, 1, size);
    struct call_id_and_handle_t *pkey = (struct call_id_and_handle_t *)malloc(sizeof(struct call_id_and_handle_t));
    *pkey = key;
    g_hash_table_insert(endpoint->managed_buffer_map, pkey, buffer);
    struct ava_coupled_record_t *rec = ava_get_coupled_record_unlocked(endpoint, coupled);
    g_ptr_array_add(rec->key_list, pkey);
    g_ptr_array_add(rec->buffer_list, buffer);
  }
  // TODO: This will probably never shrink the buffer. We may need to implement
  // that for large changes.
  g_array_set_size(buffer, size);
  pthread_mutex_unlock(&endpoint->managed_buffer_map_mutex);
  return buffer->data;
}

void *ava_uncached_alloc(struct ava_endpoint *endpoint, const void *coupled, size_t size) {
  pthread_mutex_lock(&endpoint->managed_buffer_map_mutex);
  GArray *buffer = g_array_sized_new(FALSE, TRUE, 1, size);
  struct ava_coupled_record_t *rec = ava_get_coupled_record_unlocked(endpoint, coupled);
  g_ptr_array_add(rec->buffer_list, buffer);
  pthread_mutex_unlock(&endpoint->managed_buffer_map_mutex);
  return buffer->data;
}

void ava_coupled_free(struct ava_endpoint *endpoint, const void *coupled) {
  pthread_mutex_lock(&endpoint->managed_buffer_map_mutex);
  g_hash_table_remove(endpoint->managed_by_coupled_map, coupled);
  pthread_mutex_unlock(&endpoint->managed_buffer_map_mutex);
}

void *ava_static_alloc(struct ava_endpoint *endpoint, int call_id, size_t size) {
  return ava_cached_alloc(endpoint, call_id, NULL, size);
}

void ava_add_recorded_call(struct ava_endpoint *endpoint, void *handle, struct ava_offset_pair_t *pair) {
  pthread_mutex_lock(&metadata_map_mutex);
  struct ava_metadata_base *__internal_metadata = ava_internal_metadata_unlocked(endpoint, handle);
  if (__internal_metadata->recorded_calls == NULL) {
    __internal_metadata->recorded_calls = g_ptr_array_new_full(1, free);
  }
  g_ptr_array_add(__internal_metadata->recorded_calls, pair);
  pthread_mutex_unlock(&metadata_map_mutex);
}

void ava_expunge_recorded_calls(struct ava_endpoint *endpoint, struct command_channel_log *log, void *handle) {
  pthread_mutex_lock(&metadata_map_mutex);
  struct ava_metadata_base *__internal_metadata = ava_internal_metadata_unlocked(endpoint, handle);
  if (__internal_metadata->recorded_calls != NULL) {
    for (size_t i = 0; i < __internal_metadata->recorded_calls->len; i++) {
      struct ava_offset_pair_t *pair =
          (struct ava_offset_pair_t *)g_ptr_array_index(__internal_metadata->recorded_calls, i);
      command_channel_log_update_flags(log, pair->a, 1);
      command_channel_log_update_flags(log, pair->b, 1);
    }
  }
  pthread_mutex_unlock(&metadata_map_mutex);
}

void ava_add_dependency(struct ava_endpoint *endpoint, void *a, void *b) {
  pthread_mutex_lock(&metadata_map_mutex);
  struct ava_metadata_base *__internal_metadata = ava_internal_metadata_unlocked(endpoint, a);
  if (__internal_metadata->dependencies == NULL) {
    __internal_metadata->dependencies = g_ptr_array_new_full(1, NULL);
  }
  g_ptr_array_add(__internal_metadata->dependencies, b);
  pthread_mutex_unlock(&metadata_map_mutex);
}

void ava_endpoint_init(struct ava_endpoint *endpoint, size_t metadata_size, uint8_t counter_tag) {
  assert(counter_tag == (counter_tag & 0xf) && "Only the low 4 bits of the tag may be used.");
  global_counter_tag = counter_tag;

  endpoint->metadata_size = metadata_size;
  // TODO(yuhc): Add back zero-copy.
  // endpoint->zcopy_region = zcopy_region;

#ifdef AVA_BENCHMARKING_MIGRATE
  endpoint->migration_call_id = -1;
  const char *migration_call_id_str = getenv("AVA_MIGRATION_CALL_ID");
  if (migration_call_id_str != NULL) {
    if (*migration_call_id_str == 'r') {
      long int limit = atol(migration_call_id_str + 1);
      assert(RAND_MAX / limit > 100 &&
             "Limit is large enough that migration call select bias may be "
             "more than 1%. Make AMP fix it.");
      struct timeval tv;
      gettimeofday(&tv, NULL);
      srand(tv.tv_sec + tv.tv_usec);
      endpoint->migration_call_id = rand() % limit;
    } else {
      endpoint->migration_call_id = atol(migration_call_id_str);
    }
  }
#endif

  endpoint->managed_buffer_map = g_hash_table_new_full(nw_hash_call_id_and_handle, nw_equal_call_id_and_handle, free,
                                                       (GDestroyNotify)g_array_unref);
  endpoint->managed_by_coupled_map =
      g_hash_table_new_full(nw_hash_pointer, g_direct_equal, NULL, (GDestroyNotify)ava_coupled_record_free);
  // endpoint->metadata_map = metadata_map_new();
  endpoint->call_map = metadata_map_new();
  endpoint->call_counter.store(0);
  pthread_mutex_init(&endpoint->managed_buffer_map_mutex, NULL);
  pthread_mutex_init(&endpoint->call_map_mutex, NULL);

  // shadow_buffers
  pthread_mutex_init(&endpoint->shadow_buffers.mutex, NULL);
  endpoint->shadow_buffers.buffers_by_id = g_hash_table_new_full(nw_hash_pointer, g_direct_equal, NULL, free);
}

void ava_endpoint_destroy(struct ava_endpoint *endpoint) {
#ifdef AVA_BENCHMARKING_MIGRATE
  printf(
      "INFO: Final call count = %lld\n(Set AVA_MIGRATION_CALL_ID=r%lld to "
      "choose a random call from this count. "
      "Note the 'r' before the number. Omit the 'r' to always migrate at "
      "exactly the specified call id.)\n",
      (long long int)endpoint->call_counter, (long long int)endpoint->call_counter);
  if (endpoint->migration_call_id >= 0) {
    printf("WARNING: Expected to migrate at call id %lld but did not.\n", (long long int)endpoint->migration_call_id);
  }
  if (endpoint->migration_call_id == 0) {
    printf("INFO: Migrated at expected call id.\n");
  }
#endif

  // TODO(yuhc): Add back zero-copy.
  // if (endpoint->zcopy_region) ava_zcopy_region_free_region(endpoint->zcopy_region);

  g_hash_table_unref(endpoint->managed_buffer_map);
  g_hash_table_unref(endpoint->managed_by_coupled_map);
  // g_hash_table_unref(endpoint->metadata_map);
  g_hash_table_unref(endpoint->call_map);
}

void ava_assign_record_replay_functions(struct ava_endpoint *endpoint, const void *handle, ava_extract_function extract,
                                        ava_replace_function replace) {
  struct ava_metadata_base *__internal_metadata = ava_internal_metadata(endpoint, handle);
  __internal_metadata->extract = extract;
  __internal_metadata->replace = replace;
}

//! Shadow buffers

void ava_shadow_buffer_free(struct ava_endpoint *endpoint, void *local);

void *ava_shadow_buffer_get(struct ava_endpoint *endpoint, void *id, size_t size, enum ava_lifetime_t lifetime,
                            void *lifetime_coupled, ava_allocator alloc, ava_deallocator dealloc);

void *ava_shadow_buffer_new_shadow(struct ava_endpoint *endpoint, void *id, size_t size, enum ava_lifetime_t lifetime,
                                   void *lifetime_coupled, ava_allocator alloc, ava_deallocator dealloc);

void ava_shadow_buffer_new_solid(struct ava_endpoint *endpoint, void *local, size_t size, enum ava_lifetime_t lifetime,
                                 ava_allocator alloc, ava_deallocator dealloc);

void ava_shadow_buffer_free_unlocked(struct ava_endpoint *endpoint, void *local);

void *ava_shadow_buffer_get_unlocked(struct ava_endpoint *endpoint, void *id, size_t size, enum ava_lifetime_t lifetime,
                                     void *lifetime_coupled, ava_allocator alloc, ava_deallocator dealloc);

void ava_shadow_buffer_new_solid_unlocked(struct ava_endpoint *endpoint, void *local, size_t size,
                                          enum ava_lifetime_t lifetime, ava_allocator alloc, ava_deallocator dealloc);

void *ava_shadow_buffer_new_shadow_unlocked(struct ava_endpoint *endpoint, void *id, size_t size,
                                            enum ava_lifetime_t lifetime, void *lifetime_coupled, ava_allocator alloc,
                                            ava_deallocator dealloc);

// Coupled lifetimes will be handled by registering the returned buffer as
// coupled. Manual lifetimes will have an deallocate call. Static lifetimes will
// never end and will never be deallocated.

struct ava_shadow_record_t {
  size_t size;
  ava_deallocator deallocator;
  void *id;
  void *local;
};

void *ava_shadow_buffer_get_unlocked(struct ava_endpoint *endpoint, void *id, size_t size, enum ava_lifetime_t lifetime,
                                     void *lifetime_coupled, ava_allocator alloc, ava_deallocator dealloc) {
  assert(lifetime != AVA_CALL);
  assert(id != NULL);
  struct ava_shadow_record_t *record =
      (struct ava_shadow_record_t *)g_hash_table_lookup(endpoint->shadow_buffers.buffers_by_id, id);
  assert(record != NULL &&
         "Only call ava_shadow_buffer_get directly when the buffer is known to "
         "already exist.");
  assert(record->id == id);
  void *local = record->local;

  /* Only reallocate the larger shadow buffer. The solid buffer should
   * be managed by the user program, and the record does not need to be
   * changed. */
  if (size > record->size && record->deallocator != NULL) {
    // TODO: Could be optimized to avoid several hash_table operations.
    ava_shadow_buffer_free_unlocked(endpoint, record->local);
    local = ava_shadow_buffer_new_shadow_unlocked(endpoint, id, size, lifetime, lifetime_coupled, alloc, dealloc);
  }
  return local;
}

struct ava_shadow_record_t *ava_shadow_buffer_new_record_unlocked(struct ava_endpoint *endpoint, void *id, size_t size,
                                                                  void *local, ava_deallocator dealloc,
                                                                  struct ava_metadata_base *metadata) {
  struct ava_shadow_record_t *record = (struct ava_shadow_record_t *)malloc(sizeof(struct ava_shadow_record_t));
  record->deallocator = dealloc;
  record->id = id;
  record->local = local;
  record->size = size;
  AVA_CHECK_RET(g_hash_table_insert(endpoint->shadow_buffers.buffers_by_id, id, record));
  assert(metadata->shadow == NULL);
  metadata->shadow = record;
  return record;
}

void ava_shadow_buffer_free_record_unlocked(const struct ava_endpoint *endpoint,
                                            const struct ava_shadow_record_t *record) {
  LOG_DEBUG << "shadow buffer: Deallocating shadow buffer: id=%#" << std::hex << (long int)record->id
            << ", local=" << std::hex << (long int)record->local << ", size=" << std::hex << (long int)record->size;
  assert(record->deallocator != NULL && "solid buffer is not supposed to be deallocated by AvA");
  record->deallocator(record->local);
  g_hash_table_remove(endpoint->shadow_buffers.buffers_by_id, record->id);
}

void *ava_shadow_buffer_new_shadow_unlocked(struct ava_endpoint *endpoint, void *id, size_t size,
                                            enum ava_lifetime_t lifetime, void *lifetime_coupled, ava_allocator alloc,
                                            ava_deallocator dealloc) {
  assert(lifetime != AVA_CALL);
  assert(id != NULL);
  assert(alloc != NULL);
  assert(dealloc != NULL);

  {
    // Check if the buffer already exists.
    struct ava_shadow_record_t *record =
        (struct ava_shadow_record_t *)g_hash_table_lookup(endpoint->shadow_buffers.buffers_by_id, id);
    if (record != NULL) {
      // TODO: Avoid additional lookup in ava_shadow_buffer_get_unlocked
      LOG_DEBUG << "Create shadow buffer: Existing shadow buffer: id=" << std::hex << (long int)record->id
                << ", local=" << std::hex << (long int)record->local << ", size=" << std::hex << (long int)record->size
                << "; new size=" << std::hex << (long int)size;
      return ava_shadow_buffer_get_unlocked(endpoint, id, size, lifetime, lifetime_coupled, alloc, dealloc);
    }
  }

  void *local = alloc(size);
  bzero(local, size);
  struct ava_metadata_base *metadata = ava_internal_metadata(endpoint, local);
  struct ava_shadow_record_t *record =
      ava_shadow_buffer_new_record_unlocked(endpoint, id, size, local, dealloc, metadata);
  LOG_DEBUG << "shadow buffer: Creating shadow buffer: id=" << std::hex << (long int)record->id
            << ", local=" << std::hex << (long int)record->local << ", size=" << std::hex << (long int)record->size;
  if (lifetime == AVA_MANUAL) lifetime_coupled = record->local;
  if (lifetime_coupled != NULL) {
    struct ava_metadata_base *coupled_metadata = ava_internal_metadata(endpoint, lifetime_coupled);
    if (coupled_metadata->coupled_shadow_buffers == NULL)
      coupled_metadata->coupled_shadow_buffers = g_ptr_array_new_full(1, NULL);
    g_ptr_array_add(coupled_metadata->coupled_shadow_buffers, record);
  }
  return record->local;
}

void ava_shadow_buffer_new_solid_unlocked(struct ava_endpoint *endpoint, void *local, size_t size,
                                          enum ava_lifetime_t lifetime, ava_allocator alloc, ava_deallocator dealloc) {
  // TODO: remove alloc and dealloc parameters.
  assert(dealloc == NULL);

  assert(lifetime != AVA_CALL);
  assert(local != NULL);

  struct ava_metadata_base *metadata = ava_internal_metadata(endpoint, local);
  struct ava_shadow_record_t *record = metadata->shadow;
  if (record != NULL)
    // There is already a shadow record and ID for this local buffer.
    return;

  void *id = next_id();
  record = ava_shadow_buffer_new_record_unlocked(endpoint, id, size, local, NULL, metadata);
  LOG_DEBUG << "shadow buffer: Attaching solid buffer: id=" << std::hex << (long int)record->id
            << ", local=" << std::hex << (long int)record->local << ", size=" << std::hex << (long int)record->size;
}

void ava_shadow_buffer_free_unlocked(struct ava_endpoint *endpoint, void *local) {
  struct ava_metadata_base *metadata = ava_internal_metadata_no_create(endpoint, local);
  if (!metadata) return;
  struct ava_shadow_record_t *record = metadata->shadow;
  if (!record) return;
  ava_shadow_buffer_free_record_unlocked(endpoint, record);
  metadata->shadow = NULL;
}

void ava_shadow_buffer_free_coupled_unlocked(struct ava_endpoint *endpoint, void *obj) {
  struct ava_metadata_base *metadata = ava_internal_metadata_no_create(endpoint, obj);
  if (metadata != NULL && metadata->coupled_shadow_buffers != NULL) {
    for (size_t i = 0; i < metadata->coupled_shadow_buffers->len; i++) {
      ava_shadow_buffer_free_unlocked(
          endpoint, ((struct ava_shadow_record_t *)g_ptr_array_index(metadata->coupled_shadow_buffers, i))->local);
    }
    g_ptr_array_free(metadata->coupled_shadow_buffers, TRUE);
    metadata->coupled_shadow_buffers = NULL;
  }
}

void *ava_shadow_buffer_attach_buffer(struct ava_endpoint *endpoint, struct command_channel *chan,
                                      struct command_base *cmd, const void *local, const void *data_buffer, size_t size,
                                      enum ava_lifetime_t lifetime, ava_allocator alloc, ava_deallocator dealloc,
                                      struct ava_buffer_header_t *header) {
  ava_shadow_buffer_new_solid(endpoint, (void *)local, size, lifetime, alloc, dealloc);
  struct ava_metadata_base *metadata = ava_internal_metadata(endpoint, local);
  assert(metadata->shadow != NULL && "The shadow buffer should have already been created.");
  header->id = (void *)metadata->shadow->id;
  header->has_data = 1;
  header->size = size;
  // TODO: This relies on the fact that buffers are allocated contiguously in
  // some larger space which is available to the command receiver.
  void *header_offset = command_channel_attach_buffer(chan, cmd, header, sizeof(struct ava_buffer_header_t));
  void *buffer_offset = command_channel_attach_buffer(chan, cmd, data_buffer, size);
  assert((std::int64_t)buffer_offset - (std::int64_t)header_offset == sizeof(struct ava_buffer_header_t));
  (void)header_offset;
  return buffer_offset;
}

void *ava_shadow_buffer_attach_buffer_without_data(struct ava_endpoint *endpoint, struct command_channel *chan,
                                                   struct command_base *cmd, const void *local, const void *data_buffer,
                                                   size_t size, enum ava_lifetime_t lifetime, ava_allocator alloc,
                                                   ava_deallocator dealloc, struct ava_buffer_header_t *header) {
  ava_shadow_buffer_new_solid(endpoint, (void *)local, size, lifetime, alloc, dealloc);
  struct ava_metadata_base *metadata = ava_internal_metadata(endpoint, local);
  assert(metadata->shadow != NULL && "The shadow buffer should have already been created.");
  header->id = (void *)metadata->shadow->id;
  header->has_data = 0;
  header->size = size;
  // TODO: This relies on the fact that buffers are allocated contiguously in
  // some larger space which is available to the command receiver.
  void *header_offset = command_channel_attach_buffer(chan, cmd, header, sizeof(struct ava_buffer_header_t));
  return header_offset + sizeof(struct ava_buffer_header_t);
}

void *ava_shadow_buffer_get_buffer(struct ava_endpoint *endpoint, struct command_channel *chan,
                                   struct command_base *cmd, void *offset, enum ava_lifetime_t lifetime,
                                   void *lifetime_coupled, size_t *size_out, ava_allocator alloc,
                                   ava_deallocator dealloc) {
  assert(lifetime != AVA_CALL);
  struct ava_buffer_header_t *header =
      (struct ava_buffer_header_t *)command_channel_get_buffer(chan, cmd, offset - sizeof(struct ava_buffer_header_t));
  // void *data = ((void *) header) + sizeof(struct ava_buffer_header_t);
  if (size_out) *size_out = header->size;
  void *shadow =
      ava_shadow_buffer_new_shadow(endpoint, header->id, header->size, lifetime, lifetime_coupled, alloc, dealloc);
  // memcpy(shadow, data, header->size);
  return shadow;
}

void ava_shadow_buffer_free(struct ava_endpoint *endpoint, void *local) {
  pthread_mutex_lock(&endpoint->shadow_buffers.mutex);
  ava_shadow_buffer_free_unlocked(endpoint, local);
  pthread_mutex_unlock(&endpoint->shadow_buffers.mutex);
}

void *ava_shadow_buffer_get(struct ava_endpoint *endpoint, void *id, size_t size, enum ava_lifetime_t lifetime,
                            void *lifetime_coupled, ava_allocator alloc, ava_deallocator dealloc) {
  pthread_mutex_lock(&endpoint->shadow_buffers.mutex);
  void *ret = ava_shadow_buffer_get_unlocked(endpoint, id, size, lifetime, lifetime_coupled, alloc, dealloc);
  pthread_mutex_unlock(&endpoint->shadow_buffers.mutex);
  return ret;
}

void *ava_shadow_buffer_new_shadow(struct ava_endpoint *endpoint, void *id, size_t size, enum ava_lifetime_t lifetime,
                                   void *lifetime_coupled, ava_allocator alloc, ava_deallocator dealloc) {
  pthread_mutex_lock(&endpoint->shadow_buffers.mutex);
  void *ret = ava_shadow_buffer_new_shadow_unlocked(endpoint, id, size, lifetime, lifetime_coupled, alloc, dealloc);
  pthread_mutex_unlock(&endpoint->shadow_buffers.mutex);
  return ret;
}

void ava_shadow_buffer_new_solid(struct ava_endpoint *endpoint, void *local, size_t size, enum ava_lifetime_t lifetime,
                                 ava_allocator alloc, ava_deallocator dealloc) {
  pthread_mutex_lock(&endpoint->shadow_buffers.mutex);
  ava_shadow_buffer_new_solid_unlocked(endpoint, local, size, lifetime, NULL, NULL);
  pthread_mutex_unlock(&endpoint->shadow_buffers.mutex);
}

void ava_shadow_buffer_free_coupled(struct ava_endpoint *endpoint, void *obj) {
  pthread_mutex_lock(&endpoint->shadow_buffers.mutex);
  ava_shadow_buffer_free_coupled_unlocked(endpoint, obj);
  pthread_mutex_unlock(&endpoint->shadow_buffers.mutex);
}
