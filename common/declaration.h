#pragma once

// Put this in the declarations for a class to be uncopyable.
#define DISALLOW_COPY(TypeName) TypeName(const TypeName &) = delete

// Put this in the declarations for a class to be unassignable.
#define DISALLOW_COPY_ASSIGN(TypeName) TypeName &operator=(const TypeName &) = delete

#define DISALLOW_MOVE(TypeName) TypeName(TypeName &&other) noexcept = delete

#define DISALLOW_MOVE_ASSIGN(TypeName) TypeName &operator=(TypeName &&other) noexcept = delete

// Put this in the declarations for a class to be uncopyable and unassignable.
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  DISALLOW_COPY(TypeName);                 \
  DISALLOW_COPY_ASSIGN(TypeName)

#define DISALLOW_MOVE_AND_ASSIGN(TypeName) \
  DISALLOW_MOVE(TypeName);                 \
  DISALLOW_MOVE_ASSIGN(TypeName)

// Disable copy constructor, copy assignment, move constructor and move assignment
#define DISALLOW_COPY_AND_MOVE(TypeName) \
  DISALLOW_COPY_AND_ASSIGN(TypeName);    \
  DISALLOW_MOVE_AND_ASSIGN(TypeName)

// A macro to disallow all the implicit constructors, namely the
// default constructor, copy constructor and operator= functions.
// This is especially useful for classes containing only static methods.
#define DISALLOW_IMPLICIT_CONSTRUCTORS(TypeName) \
  TypeName() = delete;                           \
  DISALLOW_COPY_AND_ASSIGN(TypeName)

#define DEFAULT_COPY_AND_ASSIGN(TypeName)             \
  TypeName(const TypeName &other) noexcept = default; \
  TypeName &operator=(const TypeName &other) noexcept = default

#define DEFAULT_MOVE_AND_ASSIGN(TypeName) \
  TypeName(TypeName &&other) = default;   \
  TypeName &operator=(TypeName &&other) = default

// A macro to create default copy constructor, copy assignment,
// move constructor and move assignment
#define DEFAULT_IMPL(TypeName)       \
  DEFAULT_COPY_AND_ASSIGN(TypeName); \
  DEFAULT_MOVE_AND_ASSIGN(TypeName)

// Disable copy constructor, copy assignment. Use default for move constructor and move assignment
#define DISALLOW_COPY_DEFAULT_MOVE(TypeName) \
  DISALLOW_COPY_AND_ASSIGN(TypeName);        \
  DEFAULT_MOVE_AND_ASSIGN(TypeName)

#define __AVA_PREDICT_FALSE(x) __builtin_expect(x, 0)
#define __AVA_PREDICT_TRUE(x) __builtin_expect(false || (x), true)

#define AVA_NORETURN __attribute__((noreturn))
#define AVA_PREFETCH(addr) __builtin_prefetch(addr)
#define AVA_MUST_USE_RESULT __attribute__((warn_unused_result))

#if defined(__clang__)
// Only clang supports warn_unused_result as a type annotation.
#define AVA_MUST_USE_TYPE AVA_MUST_USE_RESULT
#else
#define AVA_MUST_USE_TYPE
#endif

#ifdef __cplusplus
#define AVA_UNUSED(NAME) NAME [[maybe_unused]]
#else
#define AVA_UNUSED(NAME) NAME __attribute__((unused))
#endif

#ifndef __AVA_FILE_CREAT_MODE
#define __AVA_FILE_CREAT_MODE 0664
#endif

#ifndef __AVA_DIR_CREAT_MODE
#define __AVA_DIR_CREAT_MODE 0775
#endif
