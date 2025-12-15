// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from ad_msgs:msg/MissionRegion.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__MISSION_REGION__STRUCT_HPP_
#define AD_MSGS__MSG__DETAIL__MISSION_REGION__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__ad_msgs__msg__MissionRegion __attribute__((deprecated))
#else
# define DEPRECATED__ad_msgs__msg__MissionRegion __declspec(deprecated)
#endif

namespace ad_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct MissionRegion_
{
  using Type = MissionRegion_<ContainerAllocator>;

  explicit MissionRegion_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->x = 0.0;
      this->y = 0.0;
      this->radius = 0.0;
      this->mission = "";
      this->sub_type = "";
    }
  }

  explicit MissionRegion_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : mission(_alloc),
    sub_type(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->x = 0.0;
      this->y = 0.0;
      this->radius = 0.0;
      this->mission = "";
      this->sub_type = "";
    }
  }

  // field types and members
  using _x_type =
    double;
  _x_type x;
  using _y_type =
    double;
  _y_type y;
  using _radius_type =
    double;
  _radius_type radius;
  using _mission_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _mission_type mission;
  using _sub_type_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _sub_type_type sub_type;

  // setters for named parameter idiom
  Type & set__x(
    const double & _arg)
  {
    this->x = _arg;
    return *this;
  }
  Type & set__y(
    const double & _arg)
  {
    this->y = _arg;
    return *this;
  }
  Type & set__radius(
    const double & _arg)
  {
    this->radius = _arg;
    return *this;
  }
  Type & set__mission(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->mission = _arg;
    return *this;
  }
  Type & set__sub_type(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->sub_type = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    ad_msgs::msg::MissionRegion_<ContainerAllocator> *;
  using ConstRawPtr =
    const ad_msgs::msg::MissionRegion_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<ad_msgs::msg::MissionRegion_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<ad_msgs::msg::MissionRegion_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      ad_msgs::msg::MissionRegion_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<ad_msgs::msg::MissionRegion_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      ad_msgs::msg::MissionRegion_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<ad_msgs::msg::MissionRegion_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<ad_msgs::msg::MissionRegion_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<ad_msgs::msg::MissionRegion_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__ad_msgs__msg__MissionRegion
    std::shared_ptr<ad_msgs::msg::MissionRegion_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__ad_msgs__msg__MissionRegion
    std::shared_ptr<ad_msgs::msg::MissionRegion_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const MissionRegion_ & other) const
  {
    if (this->x != other.x) {
      return false;
    }
    if (this->y != other.y) {
      return false;
    }
    if (this->radius != other.radius) {
      return false;
    }
    if (this->mission != other.mission) {
      return false;
    }
    if (this->sub_type != other.sub_type) {
      return false;
    }
    return true;
  }
  bool operator!=(const MissionRegion_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct MissionRegion_

// alias to use template instance with default allocator
using MissionRegion =
  ad_msgs::msg::MissionRegion_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace ad_msgs

#endif  // AD_MSGS__MSG__DETAIL__MISSION_REGION__STRUCT_HPP_
