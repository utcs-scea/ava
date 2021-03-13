#ifndef AVA_GUESTLIB_MIGRATION_H_
#define AVA_GUESTLIB_MIGRATION_H_

#ifdef __cplusplus
extern "C" {
#endif


void start_migration(void);
void start_self_migration(void);
void start_live_migration(void);


#ifdef __cplusplus
}
#endif

#endif  // AVA_GUESTLIB_MIGRATION_H_
