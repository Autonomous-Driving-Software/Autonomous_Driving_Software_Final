// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from ad_msgs:msg/MissionRegion.idl
// generated code does not contain a copyright notice
#include "ad_msgs/msg/detail/mission_region__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `mission`
// Member `sub_type`
#include "rosidl_runtime_c/string_functions.h"

bool
ad_msgs__msg__MissionRegion__init(ad_msgs__msg__MissionRegion * msg)
{
  if (!msg) {
    return false;
  }
  // x
  // y
  // radius
  // mission
  if (!rosidl_runtime_c__String__init(&msg->mission)) {
    ad_msgs__msg__MissionRegion__fini(msg);
    return false;
  }
  // sub_type
  if (!rosidl_runtime_c__String__init(&msg->sub_type)) {
    ad_msgs__msg__MissionRegion__fini(msg);
    return false;
  }
  return true;
}

void
ad_msgs__msg__MissionRegion__fini(ad_msgs__msg__MissionRegion * msg)
{
  if (!msg) {
    return;
  }
  // x
  // y
  // radius
  // mission
  rosidl_runtime_c__String__fini(&msg->mission);
  // sub_type
  rosidl_runtime_c__String__fini(&msg->sub_type);
}

bool
ad_msgs__msg__MissionRegion__are_equal(const ad_msgs__msg__MissionRegion * lhs, const ad_msgs__msg__MissionRegion * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // x
  if (lhs->x != rhs->x) {
    return false;
  }
  // y
  if (lhs->y != rhs->y) {
    return false;
  }
  // radius
  if (lhs->radius != rhs->radius) {
    return false;
  }
  // mission
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->mission), &(rhs->mission)))
  {
    return false;
  }
  // sub_type
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->sub_type), &(rhs->sub_type)))
  {
    return false;
  }
  return true;
}

bool
ad_msgs__msg__MissionRegion__copy(
  const ad_msgs__msg__MissionRegion * input,
  ad_msgs__msg__MissionRegion * output)
{
  if (!input || !output) {
    return false;
  }
  // x
  output->x = input->x;
  // y
  output->y = input->y;
  // radius
  output->radius = input->radius;
  // mission
  if (!rosidl_runtime_c__String__copy(
      &(input->mission), &(output->mission)))
  {
    return false;
  }
  // sub_type
  if (!rosidl_runtime_c__String__copy(
      &(input->sub_type), &(output->sub_type)))
  {
    return false;
  }
  return true;
}

ad_msgs__msg__MissionRegion *
ad_msgs__msg__MissionRegion__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ad_msgs__msg__MissionRegion * msg = (ad_msgs__msg__MissionRegion *)allocator.allocate(sizeof(ad_msgs__msg__MissionRegion), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(ad_msgs__msg__MissionRegion));
  bool success = ad_msgs__msg__MissionRegion__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
ad_msgs__msg__MissionRegion__destroy(ad_msgs__msg__MissionRegion * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    ad_msgs__msg__MissionRegion__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
ad_msgs__msg__MissionRegion__Sequence__init(ad_msgs__msg__MissionRegion__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ad_msgs__msg__MissionRegion * data = NULL;

  if (size) {
    data = (ad_msgs__msg__MissionRegion *)allocator.zero_allocate(size, sizeof(ad_msgs__msg__MissionRegion), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = ad_msgs__msg__MissionRegion__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        ad_msgs__msg__MissionRegion__fini(&data[i - 1]);
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
ad_msgs__msg__MissionRegion__Sequence__fini(ad_msgs__msg__MissionRegion__Sequence * array)
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
      ad_msgs__msg__MissionRegion__fini(&array->data[i]);
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

ad_msgs__msg__MissionRegion__Sequence *
ad_msgs__msg__MissionRegion__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ad_msgs__msg__MissionRegion__Sequence * array = (ad_msgs__msg__MissionRegion__Sequence *)allocator.allocate(sizeof(ad_msgs__msg__MissionRegion__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = ad_msgs__msg__MissionRegion__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
ad_msgs__msg__MissionRegion__Sequence__destroy(ad_msgs__msg__MissionRegion__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    ad_msgs__msg__MissionRegion__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
ad_msgs__msg__MissionRegion__Sequence__are_equal(const ad_msgs__msg__MissionRegion__Sequence * lhs, const ad_msgs__msg__MissionRegion__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!ad_msgs__msg__MissionRegion__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
ad_msgs__msg__MissionRegion__Sequence__copy(
  const ad_msgs__msg__MissionRegion__Sequence * input,
  ad_msgs__msg__MissionRegion__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(ad_msgs__msg__MissionRegion);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    ad_msgs__msg__MissionRegion * data =
      (ad_msgs__msg__MissionRegion *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!ad_msgs__msg__MissionRegion__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          ad_msgs__msg__MissionRegion__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!ad_msgs__msg__MissionRegion__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
