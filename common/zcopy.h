//
// Created by amp on 8/7/19.
//

#ifndef AVA_ZCOPY_H
#define AVA_ZCOPY_H

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ava_zcopy_region;

/**
 * \section Zero copy
 *
 * This API provides a simple way to access the zero-copy region provided by AvA.
 * The API is thread-safe.
 */

/**
 * Create a new zero-copy region. This version works in the worker.
 */
struct ava_zcopy_region *ava_zcopy_region_new_worker();

/**
 * Create a new zero-copy region. This version works in the guest.
 */
struct ava_zcopy_region *ava_zcopy_region_new_guest();

/**
 * Free the entire zero-copy region. All pointers into this region are invalidated.
 * @param region The region to be freed.
 */
void ava_zcopy_region_free_region(struct ava_zcopy_region *region);

/**
 * Allocate memory in region.
 * @param region The region from which to allocate.
 * @param size The number of bytes to allocate.
 * @return A pointer to the allocated memory (in virtual memory) or NULL for failure (with errno set to ENOMEM).
 */
void *ava_zcopy_region_alloc(struct ava_zcopy_region *region, size_t size) __attribute_malloc__
    __attribute_alloc_size__((2));

/**
 * Free memory allocated in the region.
 * @param region The region from which the data was allocated.
 * @param ptr A pointer to the region to free.
 */
void ava_zcopy_region_free(struct ava_zcopy_region *region, void *ptr);

/**
 * Get the physical pointer to a pointer in the given zero-copy region.
 * @param region The containing region.
 * @param ptr The pointer to convert as a pointer returned from ava_zcopy_region_alloc.
 * @return The physical address of ptr, or 0 if ptr is invalid. Sets errno to EFAULT on bad ptr.
 */
uintptr_t ava_zcopy_region_get_physical_address(struct ava_zcopy_region *region, const void *ptr) __attribute_pure__;

/**
 * Return the pointer in a form that can be decoded currectly in any environment with a connected zero-copy region.
 * @param region The zero-copy region.
 * @param ptr The pointer to encode.
 * @return The position independent value identifying ptr w.r.t region.
 */
void *ava_zcopy_region_encode_position_independent(struct ava_zcopy_region *region, const void *ptr) __attribute_pure__;

/**
 * Return the real local pointer associated with the provided position independent pointer.
 * @param region The zero-copy region.
 * @param ptr The pointer to decode.
 * @return The actual local ptr.
 */
void *ava_zcopy_region_decode_position_independent(struct ava_zcopy_region *region, const void *ptr) __attribute_pure__;

#ifdef __cplusplus
}
#endif

#endif  // AVA_ZCOPY_H
