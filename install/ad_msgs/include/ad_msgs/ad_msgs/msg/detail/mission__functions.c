// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from ad_msgs:msg/Mission.idl
// generated code does not contain a copyright notice
#include "ad_msgs/msg/detail/mission__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `objects`
#include "ad_msgs/msg/detail/mission_object__functions.h"
// Member `road_condition`
// Member `road_slope`
#include "rosidl_runtime_c/string_functions.h"

bool
ad_msgs__msg__Mission__init(ad_msgs__msg__Mission * msg)
{
  if (!msg) {
    return false;
  }
  // objects
  if (!ad_msgs__msg__MissionObject__Sequence__init(&msg->objects, 0)) {
    ad_msgs__msg__Mission__fini(msg);
    return false;
  }
  // road_condition
  if (!rosidl_runtime_c__String__init(&msg->road_condition)) {
    ad_msgs__msg__Mission__fini(msg);
    return false;
  }
  // road_slope
  if (!rosidl_runtime_c__String__init(&msg->road_slope)) {
    ad_msgs__msg__Mission__fini(msg);
    return false;
  }
  // speed_limit
  // parking
  return true;
}

void
ad_msgs__msg__Mission__fini(ad_msgs__msg__Mission * msg)
{
  if (!msg) {
    return;
  }
  // objects
  ad_msgs__msg__MissionObject__Sequence__fini(&msg->objects);
  // road_condition
  rosidl_runtime_c__String__fini(&msg->road_condition);
  // road_slope
  rosidl_runtime_c__String__fini(&msg->road_slope);
  // speed_limit
  // parking
}

bool
ad_msgs__msg__Mission__are_equal(const ad_msgs__msg__Mission * lhs, const ad_msgs__msg__Mission * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // objects
  if (!ad_msgs__msg__MissionObject__Sequence__are_equal(
      &(lhs->objects), &(rhs->objects)))
  {
    return false;
  }
  // road_condition
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->road_condition), &(rhs->road_condition)))
  {
    return false;
  }
  // road_slope
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->road_slope), &(rhs->road_slope)))
  {
    return false;
  }
  // speed_limit
  if (lhs->speed_limit != rhs->speed_limit) {
    return false;
  }
  // parking
  if (lhs->parking != rhs->parking) {
    return false;
  }
  return true;
}

bool
ad_msgs__msg__Mission__copy(
  const ad_msgs__msg__Mission * input,
  ad_msgs__msg__Mission * output)
{
  if (!input || !output) {
    return false;
  }
  // objects
  if (!ad_msgs__msg__MissionObject__Sequence__copy(
      &(input->objects), &(output->objects)))
  {
    return false;
  }
  // road_condition
  if (!rosidl_runtime_c__String__copy(
      &(input->road_condition), &(output->road_condition)))
  {
    return false;
  }
  // road_slope
  if (!rosidl_runtime_c__String__copy(
      &(input->road_slope), &(output->road_slope)))
  {
    return false;
  }
  // speed_limit
  output->speed_limit = input->speed_limit;
  // parking
  output->parking = input->parking;
  return true;
}

ad_msgs__msg__Mission *
ad_msgs__msg__Mission__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ad_msgs__msg__Mission * msg = (ad_msgs__msg__Mission *)allocator.allocate(sizeof(ad_msgs__msg__Mission), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(ad_msgs__msg__Mission));
  bool success = ad_msgs__msg__Mission__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
ad_msgs__msg__Mission__destroy(ad_msgs__msg__Mission * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    ad_msgs__msg__Mission__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
ad_msgs__msg__Mission__Sequence__init(ad_msgs__msg__Mission__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ad_msgs__msg__Mission * data = NULL;

  if (size) {
    data = (ad_msgs__msg__Mission *)allocator.zero_allocate(size, sizeof(ad_msgs__msg__Mission), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = ad_msgs__msg__Mission__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        ad_msgs__msg__Mission__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
ad_msgs__msg__Mission__Sequence__fini(ad_msgs__msg__Mission__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      ad_msgs__msg__Mission__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

ad_msgs__msg__Mission__Sequence *
ad_msgs__msg__Mission__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ad_msgs__msg__Mission__Sequence * array = (ad_msgs__msg__Mission__Sequence *)allocator.allocate(sizeof(ad_msgs__msg__Mission__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = ad_msgs__msg__Mission__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
ad_msgs__msg__Mission__Sequence__destroy(ad_msgs__msg__Mission__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    ad_msgs__msg__Mission__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
ad_msgs__msg__Mission__Sequence__are_equal(const ad_msgs__msg__Mission__Sequence * lhs, const ad_msgs__msg__Mission__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!ad_msgs__msg__Mission__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
ad_msgs__msg__Mission__Sequence__copy(
  const ad_msgs__msg__Mission__Sequence * input,
  ad_msgs__msg__Mission__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(ad_msgs__msg__Mission);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    ad_msgs__msg__Mission * data =
      (ad_msgs__msg__Mission *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!ad_msgs__msg__Mission__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          ad_msgs__msg__Mission__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!ad_msgs__msg__Mission__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
