// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from ad_msgs:msg/VehicleCommand.idl
// generated code does not contain a copyright notice
#include "ad_msgs/msg/detail/vehicle_command__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
ad_msgs__msg__VehicleCommand__init(ad_msgs__msg__VehicleCommand * msg)
{
  if (!msg) {
    return false;
  }
  // steering
  // accel
  // brake
  return true;
}

void
ad_msgs__msg__VehicleCommand__fini(ad_msgs__msg__VehicleCommand * msg)
{
  if (!msg) {
    return;
  }
  // steering
  // accel
  // brake
}

bool
ad_msgs__msg__VehicleCommand__are_equal(const ad_msgs__msg__VehicleCommand * lhs, const ad_msgs__msg__VehicleCommand * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // steering
  if (lhs->steering != rhs->steering) {
    return false;
  }
  // accel
  if (lhs->accel != rhs->accel) {
    return false;
  }
  // brake
  if (lhs->brake != rhs->brake) {
    return false;
  }
  return true;
}

bool
ad_msgs__msg__VehicleCommand__copy(
  const ad_msgs__msg__VehicleCommand * input,
  ad_msgs__msg__VehicleCommand * output)
{
  if (!input || !output) {
    return false;
  }
  // steering
  output->steering = input->steering;
  // accel
  output->accel = input->accel;
  // brake
  output->brake = input->brake;
  return true;
}

ad_msgs__msg__VehicleCommand *
ad_msgs__msg__VehicleCommand__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ad_msgs__msg__VehicleCommand * msg = (ad_msgs__msg__VehicleCommand *)allocator.allocate(sizeof(ad_msgs__msg__VehicleCommand), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(ad_msgs__msg__VehicleCommand));
  bool success = ad_msgs__msg__VehicleCommand__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
ad_msgs__msg__VehicleCommand__destroy(ad_msgs__msg__VehicleCommand * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    ad_msgs__msg__VehicleCommand__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
ad_msgs__msg__VehicleCommand__Sequence__init(ad_msgs__msg__VehicleCommand__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ad_msgs__msg__VehicleCommand * data = NULL;

  if (size) {
    data = (ad_msgs__msg__VehicleCommand *)allocator.zero_allocate(size, sizeof(ad_msgs__msg__VehicleCommand), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = ad_msgs__msg__VehicleCommand__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        ad_msgs__msg__VehicleCommand__fini(&data[i - 1]);
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
ad_msgs__msg__VehicleCommand__Sequence__fini(ad_msgs__msg__VehicleCommand__Sequence * array)
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
      ad_msgs__msg__VehicleCommand__fini(&array->data[i]);
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

ad_msgs__msg__VehicleCommand__Sequence *
ad_msgs__msg__VehicleCommand__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ad_msgs__msg__VehicleCommand__Sequence * array = (ad_msgs__msg__VehicleCommand__Sequence *)allocator.allocate(sizeof(ad_msgs__msg__VehicleCommand__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = ad_msgs__msg__VehicleCommand__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
ad_msgs__msg__VehicleCommand__Sequence__destroy(ad_msgs__msg__VehicleCommand__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    ad_msgs__msg__VehicleCommand__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
ad_msgs__msg__VehicleCommand__Sequence__are_equal(const ad_msgs__msg__VehicleCommand__Sequence * lhs, const ad_msgs__msg__VehicleCommand__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!ad_msgs__msg__VehicleCommand__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
ad_msgs__msg__VehicleCommand__Sequence__copy(
  const ad_msgs__msg__VehicleCommand__Sequence * input,
  ad_msgs__msg__VehicleCommand__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(ad_msgs__msg__VehicleCommand);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    ad_msgs__msg__VehicleCommand * data =
      (ad_msgs__msg__VehicleCommand *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!ad_msgs__msg__VehicleCommand__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          ad_msgs__msg__VehicleCommand__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!ad_msgs__msg__VehicleCommand__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
