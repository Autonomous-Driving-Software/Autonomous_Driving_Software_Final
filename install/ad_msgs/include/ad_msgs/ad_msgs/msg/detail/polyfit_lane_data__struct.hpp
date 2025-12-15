// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from ad_msgs:msg/PolyfitLaneData.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__POLYFIT_LANE_DATA__STRUCT_HPP_
#define AD_MSGS__MSG__DETAIL__POLYFIT_LANE_DATA__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__ad_msgs__msg__PolyfitLaneData __attribute__((deprecated))
#else
# define DEPRECATED__ad_msgs__msg__PolyfitLaneData __declspec(deprecated)
#endif

namespace ad_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PolyfitLaneData_
{
  using Type = PolyfitLaneData_<ContainerAllocator>;

  explicit PolyfitLaneData_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->frame_id = "";
      this->id = "";
      this->a0 = 0.0;
      this->a1 = 0.0;
      this->a2 = 0.0;
      this->a3 = 0.0;
    }
  }

  explicit PolyfitLaneData_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : frame_id(_alloc),
    id(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->frame_id = "";
      this->id = "";
      this->a0 = 0.0;
      this->a1 = 0.0;
      this->a2 = 0.0;
      this->a3 = 0.0;
    }
  }

  // field types and members
  using _frame_id_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _frame_id_type frame_id;
  using _id_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _id_type id;
  using _a0_type =
    double;
  _a0_type a0;
  using _a1_type =
    double;
  _a1_type a1;
  using _a2_type =
    double;
  _a2_type a2;
  using _a3_type =
    double;
  _a3_type a3;

  // setters for named parameter idiom
  Type & set__frame_id(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->frame_id = _arg;
    return *this;
  }
  Type & set__id(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->id = _arg;
    return *this;
  }
  Type & set__a0(
    const double & _arg)
  {
    this->a0 = _arg;
    return *this;
  }
  Type & set__a1(
    const double & _arg)
  {
    this->a1 = _arg;
    return *this;
  }
  Type & set__a2(
    const double & _arg)
  {
    this->a2 = _arg;
    return *this;
  }
  Type & set__a3(
    const double & _arg)
  {
    this->a3 = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    ad_msgs::msg::PolyfitLaneData_<ContainerAllocator> *;
  using ConstRawPtr =
    const ad_msgs::msg::PolyfitLaneData_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<ad_msgs::msg::PolyfitLaneData_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<ad_msgs::msg::PolyfitLaneData_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      ad_msgs::msg::PolyfitLaneData_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<ad_msgs::msg::PolyfitLaneData_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      ad_msgs::msg::PolyfitLaneData_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<ad_msgs::msg::PolyfitLaneData_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<ad_msgs::msg::PolyfitLaneData_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<ad_msgs::msg::PolyfitLaneData_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__ad_msgs__msg__PolyfitLaneData
    std::shared_ptr<ad_msgs::msg::PolyfitLaneData_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__ad_msgs__msg__PolyfitLaneData
    std::shared_ptr<ad_msgs::msg::PolyfitLaneData_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PolyfitLaneData_ & other) const
  {
    if (this->frame_id != other.frame_id) {
      return false;
    }
    if (this->id != other.id) {
      return false;
    }
    if (this->a0 != other.a0) {
      return false;
    }
    if (this->a1 != other.a1) {
      return false;
    }
    if (this->a2 != other.a2) {
      return false;
    }
    if (this->a3 != other.a3) {
      return false;
    }
    return true;
  }
  bool operator!=(const PolyfitLaneData_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PolyfitLaneData_

// alias to use template instance with default allocator
using PolyfitLaneData =
  ad_msgs::msg::PolyfitLaneData_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace ad_msgs

#endif  // AD_MSGS__MSG__DETAIL__POLYFIT_LANE_DATA__STRUCT_HPP_
