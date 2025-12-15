// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from ad_msgs:msg/MissionObject.idl
// generated code does not contain a copyright notice
#include "ad_msgs/msg/detail/mission_object__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `object_type`
#include "rosidl_runtime_c/string_functions.h"

bool
ad_msgs__msg__MissionObject__init(ad_msgs__msg__MissionObject * msg)
{
  if (!msg) {
    return false;
  }
  // object_type
  if (!rosidl_runtime_c__String__init(&msg->object_type)) {
    ad_msgs__msg__MissionObject__fini(msg);
    return false;
  }
  // is_reach_end
  // x
  // y
  // yaw
  // velocity
  return true;
}

void
ad_msgs__msg__MissionObject__fini(ad_msgs__msg__MissionObject * msg)
{
  if (!msg) {
    return;
  }
  // object_type
  rosidl_runtime_c__String__fini(&msg->object_type);
  // is_reach_end
  // x
  // y
  // yaw
  // velocity
}

bool
ad_msgs__msg__MissionObject__are_equal(const ad_msgs__msg__MissionObject * lhs, const ad_msgs__msg__MissionObject * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // object_type
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->object_type), &(rhs->object_type)))
  {
    return false;
  }
  // is_reach_end
  if (lhs->is_reach_end != rhs->is_reach_end) {
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
  // yaw
  if (lhs->yaw != rhs->yaw) {
    return false;
  }
  // velocity
  if (lhs->velocity != rhs->velocity) {
    return false;
  }
  return true;
}

bool
ad_msgs__msg__MissionObject__copy(
  const ad_msgs__msg__MissionObject * input,
  ad_msgs__msg__MissionObject * output)
{
  if (!input || !output) {
    return false;
  }
  // object_type
  if (!rosidl_runtime_c__String__copy(
      &(input->object_type), &(output->object_type)))
  {
    return false;
  }
  // is_reach_end
  output->is_reach_end = input->is_reach_end;
  // x
  output->x = input->x;
  // y
  output->y = input->y;
  // yaw
  output->yaw = input->yaw;
  // velocity
  output->velocity = input->velocity;
  return true;
}

ad_msgs__msg__MissionObject *
ad_msgs__msg__MissionObject__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ad_msgs__msg__MissionObject * msg = (ad_msgs__msg__MissionObject *)allocator.allocate(sizeof(ad_msgs__msg__MissionObject), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(ad_msgs__msg__MissionObject));
  bool success = ad_msgs__msg__MissionObject__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
ad_msgs__msg__MissionObject__destroy(ad_msgs__msg__MissionObject * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    ad_msgs__msg__MissionObject__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
ad_msgs__msg__MissionObject__Sequence__init(ad_msgs__msg__MissionObject__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ad_msgs__msg__MissionObject * data = NULL;

  if (size) {
    data = (ad_msgs__msg__MissionObject *)allocator.zero_allocate(size, sizeof(ad_msgs__msg__MissionObject), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = ad_msgs__msg__MissionObject__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        ad_msgs__msg__MissionObject__fini(&data[i - 1]);
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
ad_msgs__msg__MissionObject__Sequence__fini(ad_msgs__msg__MissionObject__Sequence * array)
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
      ad_msgs__msg__MissionObject__fini(&array->data[i]);
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

ad_msgs__msg__MissionObject__Sequence *
ad_msgs__msg__MissionObject__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ad_msgs__msg__MissionObject__Sequence * array = (ad_msgs__msg__MissionObject__Sequence *)allocator.allocate(sizeof(ad_msgs__msg__MissionObject__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = ad_msgs__msg__MissionObject__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
ad_msgs__msg__MissionObject__Sequence__destroy(ad_msgs__msg__MissionObject__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    ad_msgs__msg__MissionObject__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
ad_msgs__msg__MissionObject__Sequence__are_equal(const ad_msgs__msg__MissionObject__Sequence * lhs, const ad_msgs__msg__MissionObject__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!ad_msgs__msg__MissionObject__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
ad_msgs__msg__MissionObject__Sequence__copy(
  const ad_msgs__msg__MissionObject__Sequence * input,
  ad_msgs__msg__MissionObject__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(ad_msgs__msg__MissionObject);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    ad_msgs__msg__MissionObject * data =
      (ad_msgs__msg__MissionObject *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!ad_msgs__msg__MissionObject__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          ad_msgs__msg__MissionObject__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!ad_msgs__msg__MissionObject__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
