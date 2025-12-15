// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from ad_msgs:msg/MissionDisplay.idl
// generated code does not contain a copyright notice
#include "ad_msgs/msg/detail/mission_display__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `objects`
#include "ad_msgs/msg/detail/mission_object__functions.h"
// Member `regions`
#include "ad_msgs/msg/detail/mission_region__functions.h"

bool
ad_msgs__msg__MissionDisplay__init(ad_msgs__msg__MissionDisplay * msg)
{
  if (!msg) {
    return false;
  }
  // objects
  if (!ad_msgs__msg__MissionObject__Sequence__init(&msg->objects, 0)) {
    ad_msgs__msg__MissionDisplay__fini(msg);
    return false;
  }
  // regions
  if (!ad_msgs__msg__MissionRegion__Sequence__init(&msg->regions, 0)) {
    ad_msgs__msg__MissionDisplay__fini(msg);
    return false;
  }
  return true;
}

void
ad_msgs__msg__MissionDisplay__fini(ad_msgs__msg__MissionDisplay * msg)
{
  if (!msg) {
    return;
  }
  // objects
  ad_msgs__msg__MissionObject__Sequence__fini(&msg->objects);
  // regions
  ad_msgs__msg__MissionRegion__Sequence__fini(&msg->regions);
}

bool
ad_msgs__msg__MissionDisplay__are_equal(const ad_msgs__msg__MissionDisplay * lhs, const ad_msgs__msg__MissionDisplay * rhs)
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
  // regions
  if (!ad_msgs__msg__MissionRegion__Sequence__are_equal(
      &(lhs->regions), &(rhs->regions)))
  {
    return false;
  }
  return true;
}

bool
ad_msgs__msg__MissionDisplay__copy(
  const ad_msgs__msg__MissionDisplay * input,
  ad_msgs__msg__MissionDisplay * output)
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
  // regions
  if (!ad_msgs__msg__MissionRegion__Sequence__copy(
      &(input->regions), &(output->regions)))
  {
    return false;
  }
  return true;
}

ad_msgs__msg__MissionDisplay *
ad_msgs__msg__MissionDisplay__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ad_msgs__msg__MissionDisplay * msg = (ad_msgs__msg__MissionDisplay *)allocator.allocate(sizeof(ad_msgs__msg__MissionDisplay), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(ad_msgs__msg__MissionDisplay));
  bool success = ad_msgs__msg__MissionDisplay__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
ad_msgs__msg__MissionDisplay__destroy(ad_msgs__msg__MissionDisplay * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    ad_msgs__msg__MissionDisplay__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
ad_msgs__msg__MissionDisplay__Sequence__init(ad_msgs__msg__MissionDisplay__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ad_msgs__msg__MissionDisplay * data = NULL;

  if (size) {
    data = (ad_msgs__msg__MissionDisplay *)allocator.zero_allocate(size, sizeof(ad_msgs__msg__MissionDisplay), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = ad_msgs__msg__MissionDisplay__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        ad_msgs__msg__MissionDisplay__fini(&data[i - 1]);
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
ad_msgs__msg__MissionDisplay__Sequence__fini(ad_msgs__msg__MissionDisplay__Sequence * array)
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
      ad_msgs__msg__MissionDisplay__fini(&array->data[i]);
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

ad_msgs__msg__MissionDisplay__Sequence *
ad_msgs__msg__MissionDisplay__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ad_msgs__msg__MissionDisplay__Sequence * array = (ad_msgs__msg__MissionDisplay__Sequence *)allocator.allocate(sizeof(ad_msgs__msg__MissionDisplay__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = ad_msgs__msg__MissionDisplay__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
ad_msgs__msg__MissionDisplay__Sequence__destroy(ad_msgs__msg__MissionDisplay__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    ad_msgs__msg__MissionDisplay__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
ad_msgs__msg__MissionDisplay__Sequence__are_equal(const ad_msgs__msg__MissionDisplay__Sequence * lhs, const ad_msgs__msg__MissionDisplay__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!ad_msgs__msg__MissionDisplay__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
ad_msgs__msg__MissionDisplay__Sequence__copy(
  const ad_msgs__msg__MissionDisplay__Sequence * input,
  ad_msgs__msg__MissionDisplay__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(ad_msgs__msg__MissionDisplay);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    ad_msgs__msg__MissionDisplay * data =
      (ad_msgs__msg__MissionDisplay *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!ad_msgs__msg__MissionDisplay__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          ad_msgs__msg__MissionDisplay__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!ad_msgs__msg__MissionDisplay__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
