// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from ad_msgs:msg/PolyfitLaneData.idl
// generated code does not contain a copyright notice
#include "ad_msgs/msg/detail/polyfit_lane_data__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `frame_id`
// Member `id`
#include "rosidl_runtime_c/string_functions.h"

bool
ad_msgs__msg__PolyfitLaneData__init(ad_msgs__msg__PolyfitLaneData * msg)
{
  if (!msg) {
    return false;
  }
  // frame_id
  if (!rosidl_runtime_c__String__init(&msg->frame_id)) {
    ad_msgs__msg__PolyfitLaneData__fini(msg);
    return false;
  }
  // id
  if (!rosidl_runtime_c__String__init(&msg->id)) {
    ad_msgs__msg__PolyfitLaneData__fini(msg);
    return false;
  }
  // a0
  // a1
  // a2
  // a3
  return true;
}

void
ad_msgs__msg__PolyfitLaneData__fini(ad_msgs__msg__PolyfitLaneData * msg)
{
  if (!msg) {
    return;
  }
  // frame_id
  rosidl_runtime_c__String__fini(&msg->frame_id);
  // id
  rosidl_runtime_c__String__fini(&msg->id);
  // a0
  // a1
  // a2
  // a3
}

bool
ad_msgs__msg__PolyfitLaneData__are_equal(const ad_msgs__msg__PolyfitLaneData * lhs, const ad_msgs__msg__PolyfitLaneData * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // frame_id
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->frame_id), &(rhs->frame_id)))
  {
    return false;
  }
  // id
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->id), &(rhs->id)))
  {
    return false;
  }
  // a0
  if (lhs->a0 != rhs->a0) {
    return false;
  }
  // a1
  if (lhs->a1 != rhs->a1) {
    return false;
  }
  // a2
  if (lhs->a2 != rhs->a2) {
    return false;
  }
  // a3
  if (lhs->a3 != rhs->a3) {
    return false;
  }
  return true;
}

bool
ad_msgs__msg__PolyfitLaneData__copy(
  const ad_msgs__msg__PolyfitLaneData * input,
  ad_msgs__msg__PolyfitLaneData * output)
{
  if (!input || !output) {
    return false;
  }
  // frame_id
  if (!rosidl_runtime_c__String__copy(
      &(input->frame_id), &(output->frame_id)))
  {
    return false;
  }
  // id
  if (!rosidl_runtime_c__String__copy(
      &(input->id), &(output->id)))
  {
    return false;
  }
  // a0
  output->a0 = input->a0;
  // a1
  output->a1 = input->a1;
  // a2
  output->a2 = input->a2;
  // a3
  output->a3 = input->a3;
  return true;
}

ad_msgs__msg__PolyfitLaneData *
ad_msgs__msg__PolyfitLaneData__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ad_msgs__msg__PolyfitLaneData * msg = (ad_msgs__msg__PolyfitLaneData *)allocator.allocate(sizeof(ad_msgs__msg__PolyfitLaneData), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(ad_msgs__msg__PolyfitLaneData));
  bool success = ad_msgs__msg__PolyfitLaneData__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
ad_msgs__msg__PolyfitLaneData__destroy(ad_msgs__msg__PolyfitLaneData * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    ad_msgs__msg__PolyfitLaneData__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
ad_msgs__msg__PolyfitLaneData__Sequence__init(ad_msgs__msg__PolyfitLaneData__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ad_msgs__msg__PolyfitLaneData * data = NULL;

  if (size) {
    data = (ad_msgs__msg__PolyfitLaneData *)allocator.zero_allocate(size, sizeof(ad_msgs__msg__PolyfitLaneData), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = ad_msgs__msg__PolyfitLaneData__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        ad_msgs__msg__PolyfitLaneData__fini(&data[i - 1]);
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
ad_msgs__msg__PolyfitLaneData__Sequence__fini(ad_msgs__msg__PolyfitLaneData__Sequence * array)
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
      ad_msgs__msg__PolyfitLaneData__fini(&array->data[i]);
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

ad_msgs__msg__PolyfitLaneData__Sequence *
ad_msgs__msg__PolyfitLaneData__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ad_msgs__msg__PolyfitLaneData__Sequence * array = (ad_msgs__msg__PolyfitLaneData__Sequence *)allocator.allocate(sizeof(ad_msgs__msg__PolyfitLaneData__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = ad_msgs__msg__PolyfitLaneData__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
ad_msgs__msg__PolyfitLaneData__Sequence__destroy(ad_msgs__msg__PolyfitLaneData__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    ad_msgs__msg__PolyfitLaneData__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
ad_msgs__msg__PolyfitLaneData__Sequence__are_equal(const ad_msgs__msg__PolyfitLaneData__Sequence * lhs, const ad_msgs__msg__PolyfitLaneData__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!ad_msgs__msg__PolyfitLaneData__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
ad_msgs__msg__PolyfitLaneData__Sequence__copy(
  const ad_msgs__msg__PolyfitLaneData__Sequence * input,
  ad_msgs__msg__PolyfitLaneData__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(ad_msgs__msg__PolyfitLaneData);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    ad_msgs__msg__PolyfitLaneData * data =
      (ad_msgs__msg__PolyfitLaneData *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!ad_msgs__msg__PolyfitLaneData__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          ad_msgs__msg__PolyfitLaneData__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!ad_msgs__msg__PolyfitLaneData__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
