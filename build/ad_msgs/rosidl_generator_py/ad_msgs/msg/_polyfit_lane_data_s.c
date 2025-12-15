// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from ad_msgs:msg/PolyfitLaneData.idl
// generated code does not contain a copyright notice
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdbool.h>
#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "numpy/ndarrayobject.h"
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif
#include "rosidl_runtime_c/visibility_control.h"
#include "ad_msgs/msg/detail/polyfit_lane_data__struct.h"
#include "ad_msgs/msg/detail/polyfit_lane_data__functions.h"

#include "rosidl_runtime_c/string.h"
#include "rosidl_runtime_c/string_functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool ad_msgs__msg__polyfit_lane_data__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[47];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("ad_msgs.msg._polyfit_lane_data.PolyfitLaneData", full_classname_dest, 46) == 0);
  }
  ad_msgs__msg__PolyfitLaneData * ros_message = _ros_message;
  {  // frame_id
    PyObject * field = PyObject_GetAttrString(_pymsg, "frame_id");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->frame_id, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }
  {  // id
    PyObject * field = PyObject_GetAttrString(_pymsg, "id");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->id, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }
  {  // a0
    PyObject * field = PyObject_GetAttrString(_pymsg, "a0");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->a0 = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // a1
    PyObject * field = PyObject_GetAttrString(_pymsg, "a1");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->a1 = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // a2
    PyObject * field = PyObject_GetAttrString(_pymsg, "a2");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->a2 = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // a3
    PyObject * field = PyObject_GetAttrString(_pymsg, "a3");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->a3 = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * ad_msgs__msg__polyfit_lane_data__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of PolyfitLaneData */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("ad_msgs.msg._polyfit_lane_data");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "PolyfitLaneData");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  ad_msgs__msg__PolyfitLaneData * ros_message = (ad_msgs__msg__PolyfitLaneData *)raw_ros_message;
  {  // frame_id
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->frame_id.data,
      strlen(ros_message->frame_id.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "frame_id", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // id
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->id.data,
      strlen(ros_message->id.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "id", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // a0
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->a0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "a0", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // a1
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->a1);
    {
      int rc = PyObject_SetAttrString(_pymessage, "a1", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // a2
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->a2);
    {
      int rc = PyObject_SetAttrString(_pymessage, "a2", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // a3
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->a3);
    {
      int rc = PyObject_SetAttrString(_pymessage, "a3", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
