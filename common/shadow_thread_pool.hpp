//
// Created by amp on 4/2/19.
//

#ifndef AVA_SHADOWN_THREAD_POOL_HPP
#define AVA_SHADOWN_THREAD_POOL_HPP

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations of structs to avoid dependency cycles in the includes.
struct command_channel;
struct command_base;

/**
 * A shadow thread pool manages a set of threads based on incoming command's `thread_id`s.
 * The pool will also handle "solid" threads: threads where are not managed by the pool,
 * and have a remote shadow at the other end of the AvA transport. A thread become a solid
 * thread as soon as it calls `shadow_thread_id(pool)`.
 */
struct shadow_thread_pool_t;

/**
 * @return A newly-constructed empty shadow_thread_pool_t.
 */
struct shadow_thread_pool_t *shadow_thread_pool_new();

/**
 * Destroy a shadow_thread_pool_t signaling all threads to exit. Any solid threads will not exit but will
 * disassociate from this pool.
 *
 * @param pool The pool.
 */
void shadow_thread_pool_free(struct shadow_thread_pool_t *pool);

/**
 * Execute a single command that is destined for this thread.
 * @param pool The pool this thread should be executing in.
 * @return 1 if this thread has been asked to exit, 0 otherwise.
 */
int shadow_thread_handle_single_command(struct shadow_thread_pool_t *pool);

/**
 * Get the cross channel id of the current thread in this pool. If the current thread is a shadow this is the
 * ID of the remote solid thread.
 * @param pool The pool.
 * @return The ID of the current thread.
 */
intptr_t shadow_thread_id(struct shadow_thread_pool_t *pool);

/**
 * Dispatch a single command to a thread pool. This call in non-blocking.
 *
 * @param pool The shadow_thread_pool_t
 * @param chan The channel from which `cmd` came.
 * @param cmd The command to dispatch.
 */
void shadow_thread_pool_dispatch(struct shadow_thread_pool_t *pool, struct command_channel *chan,
                                 struct command_base *cmd);

/**
 * Block until a command for this thread is executed and the
 * predicate is true.
 */
#define shadow_thread_handle_command_until(pool, predicate)                    \
  while (!(predicate)) {                                                       \
    int r = shadow_thread_handle_single_command(pool);                         \
    assert(r == 0 && ("Thread exit requested while waiting for " #predicate)); \
    (void)r;                                                                   \
  }

#ifdef __cplusplus
}
#endif

#endif  // AVA_SHADOWN_THREAD_POOL_HPP
