// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from ad_msgs:msg/LanePointDataArray.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__LANE_POINT_DATA_ARRAY__FUNCTIONS_H_
#define AD_MSGS__MSG__DETAIL__LANE_POINT_DATA_ARRAY__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "ad_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "ad_msgs/msg/detail/lane_point_data_array__struct.h"

/// Initialize msg/LanePointDataArray message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * ad_msgs__msg__LanePointDataArray
 * )) before or use
 * ad_msgs__msg__LanePointDataArray__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_ad_msgs
bool
ad_msgs__msg__LanePointDataArray__init(ad_msgs__msg__LanePointDataArray * msg);

/// Finalize msg/LanePointDataArray message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_ad_msgs
void
ad_msgs__msg__LanePointDataArray__fini(ad_msgs__msg__LanePointDataArray * msg);

/// Create msg/LanePointDataArray message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * ad_msgs__msg__LanePointDataArray__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_ad_msgs
ad_msgs__msg__LanePointDataArray *
ad_msgs__msg__LanePointDataArray__create();

/// Destroy msg/LanePointDataArray message.
/**
 * It calls
 * ad_msgs__msg__LanePointDataArray__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_ad_msgs
void
ad_msgs__msg__LanePointDataArray__destroy(ad_msgs__msg__LanePointDataArray * msg);

/// Check for msg/LanePointDataArray message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_ad_msgs
bool
ad_msgs__msg__LanePointDataArray__are_equal(const ad_msgs__msg__LanePointDataArray * lhs, const ad_msgs__msg__LanePointDataArray * rhs);

/// Copy a msg/LanePointDataArray message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_ad_msgs
bool
ad_msgs__msg__LanePointDataArray__copy(
  const ad_msgs__msg__LanePointDataArray * input,
  ad_msgs__msg__LanePointDataArray * output);

/// Initialize array of msg/LanePointDataArray messages.
/**
 * It allocates the memory for the number of elements and calls
 * ad_msgs__msg__LanePointDataArray__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_ad_msgs
bool
ad_msgs__msg__LanePointDataArray__Sequence__init(ad_msgs__msg__LanePointDataArray__Sequence * array, size_t size);

/// Finalize array of msg/LanePointDataArray messages.
/**
 * It calls
 * ad_msgs__msg__LanePointDataArray__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_ad_msgs
void
ad_msgs__msg__LanePointDataArray__Sequence__fini(ad_msgs__msg__LanePointDataArray__Sequence * array);

/// Create array of msg/LanePointDataArray messages.
/**
 * It allocates the memory for the array and calls
 * ad_msgs__msg__LanePointDataArray__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_ad_msgs
ad_msgs__msg__LanePointDataArray__Sequence *
ad_msgs__msg__LanePointDataArray__Sequence__create(size_t size);

/// Destroy array of msg/LanePointDataArray messages.
/**
 * It calls
 * ad_msgs__msg__LanePointDataArray__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_ad_msgs
void
ad_msgs__msg__LanePointDataArray__Sequence__destroy(ad_msgs__msg__LanePointDataArray__Sequence * array);

/// Check for msg/LanePointDataArray message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_ad_msgs
bool
ad_msgs__msg__LanePointDataArray__Sequence__are_equal(const ad_msgs__msg__LanePointDataArray__Sequence * lhs, const ad_msgs__msg__LanePointDataArray__Sequence * rhs);

/// Copy an array of msg/LanePointDataArray messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_ad_msgs
bool
ad_msgs__msg__LanePointDataArray__Sequence__copy(
  const ad_msgs__msg__LanePointDataArray__Sequence * input,
  ad_msgs__msg__LanePointDataArray__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // AD_MSGS__MSG__DETAIL__LANE_POINT_DATA_ARRAY__FUNCTIONS_H_
