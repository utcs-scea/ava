#ifndef __EXECUTOR_OBJECT_LIST_H__
#define __EXECUTOR_OBJECT_LIST_H__


#include "common/ctype_util.h"
#include "common/object.h"
#include <stdint.h>
#include <glib.h>

#ifdef __cplusplus
extern "C" {
#endif

GHashTable *
InitObjectTable();

VOID
InsertNewObject(GHashTable *Table, HANDLE Handle, size_t Size);

PDEVICE_OBJECT_LIST
GetObjectByOriginalHandle(GHashTable *Table, HANDLE OriginalHandle);

VOID
RemoveObject(GHashTable *Table, HANDLE Handle);

#ifdef __cplusplus
}
#endif

#endif

