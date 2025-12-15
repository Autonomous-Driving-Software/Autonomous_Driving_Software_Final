// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from ad_msgs:msg/PolyfitLaneDataArray.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__POLYFIT_LANE_DATA_ARRAY__STRUCT_HPP_
#define AD_MSGS__MSG__DETAIL__POLYFIT_LANE_DATA_ARRAY__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'polyfitlanes'
#include "ad_msgs/msg/detail/polyfit_lane_data__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__ad_msgs__msg__PolyfitLaneDataArray __attribute__((deprecated))
#else
# define DEPRECATED__ad_msgs__msg__PolyfitLaneDataArray __declspec(deprecated)
#endif

namespace ad_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PolyfitLaneDataArray_
{
  using Type = PolyfitLaneDataArray_<ContainerAllocator>;

  explicit PolyfitLaneDataArray_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->frame_id = "";
    }
  }

  explicit PolyfitLaneDataArray_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : frame_id(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->frame_id = "";
    }
  }

  // field types and members
  using _frame_id_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _frame_id_type frame_id;
  using _polyfitlanes_type =
    std::vector<ad_msgs::msg::PolyfitLaneData_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<ad_msgs::msg::PolyfitLaneData_<ContainerAllocator>>>;
  _polyfitlanes_type polyfitlanes;

  // setters for named parameter idiom
  Type & set__frame_id(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->frame_id = _arg;
    return *this;
  }
  Type & set__polyfitlanes(
    const std::vector<ad_msgs::msg::PolyfitLaneData_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<ad_msgs::msg::PolyfitLaneData_<ContainerAllocator>>> & _arg)
  {
    this->polyfitlanes = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    ad_msgs::msg::PolyfitLaneDataArray_<ContainerAllocator> *;
  using ConstRawPtr =
    const ad_msgs::msg::PolyfitLaneDataArray_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<ad_msgs::msg::PolyfitLaneDataArray_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<ad_msgs::msg::PolyfitLaneDataArray_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      ad_msgs::msg::PolyfitLaneDataArray_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<ad_msgs::msg::PolyfitLaneDataArray_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      ad_msgs::msg::PolyfitLaneDataArray_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<ad_msgs::msg::PolyfitLaneDataArray_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<ad_msgs::msg::PolyfitLaneDataArray_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<ad_msgs::msg::PolyfitLaneDataArray_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__ad_msgs__msg__PolyfitLaneDataArray
    std::shared_ptr<ad_msgs::msg::PolyfitLaneDataArray_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__ad_msgs__msg__PolyfitLaneDataArray
    std::shared_ptr<ad_msgs::msg::PolyfitLaneDataArray_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PolyfitLaneDataArray_ & other) const
  {
    if (this->frame_id != other.frame_id) {
      return false;
    }
    if (this->polyfitlanes != other.polyfitlanes) {
      return false;
    }
    return true;
  }
  bool operator!=(const PolyfitLaneDataArray_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PolyfitLaneDataArray_

// alias to use template instance with default allocator
using PolyfitLaneDataArray =
  ad_msgs::msg::PolyfitLaneDataArray_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace ad_msgs

#endif  // AD_MSGS__MSG__DETAIL__POLYFIT_LANE_DATA_ARRAY__STRUCT_HPP_
