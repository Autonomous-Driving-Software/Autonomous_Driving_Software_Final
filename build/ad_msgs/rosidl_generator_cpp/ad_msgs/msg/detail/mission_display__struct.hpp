// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from ad_msgs:msg/MissionDisplay.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__MISSION_DISPLAY__STRUCT_HPP_
#define AD_MSGS__MSG__DETAIL__MISSION_DISPLAY__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'objects'
#include "ad_msgs/msg/detail/mission_object__struct.hpp"
// Member 'regions'
#include "ad_msgs/msg/detail/mission_region__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__ad_msgs__msg__MissionDisplay __attribute__((deprecated))
#else
# define DEPRECATED__ad_msgs__msg__MissionDisplay __declspec(deprecated)
#endif

namespace ad_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct MissionDisplay_
{
  using Type = MissionDisplay_<ContainerAllocator>;

  explicit MissionDisplay_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
  }

  explicit MissionDisplay_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
    (void)_alloc;
  }

  // field types and members
  using _objects_type =
    std::vector<ad_msgs::msg::MissionObject_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<ad_msgs::msg::MissionObject_<ContainerAllocator>>>;
  _objects_type objects;
  using _regions_type =
    std::vector<ad_msgs::msg::MissionRegion_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<ad_msgs::msg::MissionRegion_<ContainerAllocator>>>;
  _regions_type regions;

  // setters for named parameter idiom
  Type & set__objects(
    const std::vector<ad_msgs::msg::MissionObject_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<ad_msgs::msg::MissionObject_<ContainerAllocator>>> & _arg)
  {
    this->objects = _arg;
    return *this;
  }
  Type & set__regions(
    const std::vector<ad_msgs::msg::MissionRegion_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<ad_msgs::msg::MissionRegion_<ContainerAllocator>>> & _arg)
  {
    this->regions = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    ad_msgs::msg::MissionDisplay_<ContainerAllocator> *;
  using ConstRawPtr =
    const ad_msgs::msg::MissionDisplay_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<ad_msgs::msg::MissionDisplay_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<ad_msgs::msg::MissionDisplay_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      ad_msgs::msg::MissionDisplay_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<ad_msgs::msg::MissionDisplay_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      ad_msgs::msg::MissionDisplay_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<ad_msgs::msg::MissionDisplay_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<ad_msgs::msg::MissionDisplay_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<ad_msgs::msg::MissionDisplay_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__ad_msgs__msg__MissionDisplay
    std::shared_ptr<ad_msgs::msg::MissionDisplay_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__ad_msgs__msg__MissionDisplay
    std::shared_ptr<ad_msgs::msg::MissionDisplay_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const MissionDisplay_ & other) const
  {
    if (this->objects != other.objects) {
      return false;
    }
    if (this->regions != other.regions) {
      return false;
    }
    return true;
  }
  bool operator!=(const MissionDisplay_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct MissionDisplay_

// alias to use template instance with default allocator
using MissionDisplay =
  ad_msgs::msg::MissionDisplay_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace ad_msgs

#endif  // AD_MSGS__MSG__DETAIL__MISSION_DISPLAY__STRUCT_HPP_
