#ifndef AVA_MIGRATION_BARRIER_H
#define AVA_MIGRATION_BARRIER_H

#ifdef __cplusplus
extern "C" {
#endif

void migration_barrier_init(void);
void migration_barrier_destroy(void);
void migration_barrier_wait(long long int call_id);

#ifdef __cplusplus
}
#endif

#endif
