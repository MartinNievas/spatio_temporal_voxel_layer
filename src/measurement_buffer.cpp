/*********************************************************************
 *
 * Software License Agreement
 *
 *  Copyright (c) 2018, Simbe Robotics, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Simbe Robotics, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: Steve Macenski (steven.macenski@simberobotics.com)
 *********************************************************************/

#include <spatio_temporal_voxel_layer/measurement_buffer.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

int *testmain(int num, int threads);
float *filter(float*,float*,float*,size_t,size_t);

namespace buffer
{

/*****************************************************************************/
MeasurementBuffer::MeasurementBuffer(const std::string& topic_name,          \
                                     const double& observation_keep_time,    \
                                     const double& expected_update_rate,     \
                                     const double& min_obstacle_height,      \
                                     const double& max_obstacle_height,      \
                                     const double& obstacle_range,           \
                                     tf2_ros::Buffer& tf,                    \
                                     const std::string& global_frame,        \
                                     const std::string& sensor_frame,        \
                                     const double& tf_tolerance,             \
                                     const double& min_d,                    \
                                     const double& max_d,                    \
                                     const double& vFOV,                     \
                                     const double& vFOVPadding,              \
                                     const double& hFOV,                     \
                                     const double& decay_acceleration,       \
                                     const bool& marking,                    \
                                     const bool& clearing,                   \
                                     const double& voxel_size,               \
                                     const bool& voxel_filter,               \
                                     const int& voxel_min_points,            \
                                     const bool& enabled,                    \
                                     const bool& clear_buffer_after_reading, \
                                     const ModelType& model_type) :
/*****************************************************************************/
    _buffer(tf), _observation_keep_time(observation_keep_time),
    _expected_update_rate(expected_update_rate),_last_updated(ros::Time::now()),
    _global_frame(global_frame), _sensor_frame(sensor_frame),
    _topic_name(topic_name), _min_obstacle_height(min_obstacle_height),
    _max_obstacle_height(max_obstacle_height), _obstacle_range(obstacle_range),
    _tf_tolerance(tf_tolerance), _min_z(min_d), _max_z(max_d),
    _vertical_fov(vFOV), _vertical_fov_padding(vFOVPadding),
    _horizontal_fov(hFOV), _decay_acceleration(decay_acceleration),
    _marking(marking), _clearing(clearing), _voxel_size(voxel_size),
    _voxel_filter(voxel_filter), _voxel_min_points(voxel_min_points),
    _enabled(enabled), _clear_buffer_after_reading(clear_buffer_after_reading),
    _model_type(model_type)
{
}

/*****************************************************************************/
MeasurementBuffer::~MeasurementBuffer(void)
/*****************************************************************************/
{
}

/*****************************************************************************/
void MeasurementBuffer::BufferROSCloud(const sensor_msgs::PointCloud2& cloud)
/*****************************************************************************/
{

  float start = 0.0, end = 0.0, elapsed = 0.0;

  // add a new measurement to be populated
  _observation_list.push_front(observation::MeasurementReading());

  const std::string origin_frame = \
                  _sensor_frame == "" ? cloud.header.frame_id : _sensor_frame;

  try
  {
    // transform into global frame
    geometry_msgs::PoseStamped  local_pose, global_pose;
    local_pose.pose.position.x=0;
    local_pose.pose.position.y=0;
    local_pose.pose.position.z=0;
    local_pose.pose.orientation.x=0;
    local_pose.pose.orientation.y=0;
    local_pose.pose.orientation.z=0;
    local_pose.pose.orientation.w=1;
    local_pose.header.stamp = cloud.header.stamp;
    local_pose.header.frame_id = origin_frame;

    // Check for timeout constant
    // I'm really checking here?
    _buffer.canTransform(_global_frame, local_pose.header.frame_id , \
                         local_pose.header.stamp, ros::Duration(0.5));
    _buffer.transform(local_pose, global_pose, _global_frame);

    _observation_list.front()._origin.x = global_pose.pose.position.x;
    _observation_list.front()._origin.y = global_pose.pose.position.y;
    _observation_list.front()._origin.z = global_pose.pose.position.z;

    _observation_list.front()._orientation = global_pose.pose.orientation;
    _observation_list.front()._obstacle_range_in_m = _obstacle_range;
    _observation_list.front()._min_z_in_m = _min_z;
    _observation_list.front()._max_z_in_m = _max_z;
    _observation_list.front()._vertical_fov_in_rad = _vertical_fov;
    _observation_list.front()._vertical_fov_padding_in_m = _vertical_fov_padding;
    _observation_list.front()._horizontal_fov_in_rad = _horizontal_fov;
    _observation_list.front()._decay_acceleration = _decay_acceleration;
    _observation_list.front()._clearing = _clearing;
    _observation_list.front()._marking = _marking;
    _observation_list.front()._model_type = _model_type;

    if (_clearing && !_marking)
    {
      // no need to buffer points
      return;
    }

    // transform the cloud in the global frame
    point_cloud_ptr cld_global(new sensor_msgs::PointCloud2());
    geometry_msgs::TransformStamped tf_stamped = _buffer.lookupTransform( \
                 _global_frame, cloud.header.frame_id, cloud.header.stamp);
    tf2::doTransform (cloud, *cld_global, tf_stamped);

    point_cloud_ptr output(new sensor_msgs::PointCloud2());
    pcl::PointCloud<pcl::PointXYZ> cloud_pcl;

    // pcl::PCLPointCloud2::Ptr cloud_pcl (new pcl::PCLPointCloud2 ());
    // pcl::PCLPointCloud2::Ptr cloud_filtered (new pcl::PCLPointCloud2 ());
    // pcl_conversions::toPCL(*cld_global, *cloud_pcl);

    // remove points that are below or above our height restrictions, and
    // in the same time, remove NaNs and if user wants to use it, combine with a
    if ( _voxel_filter )
    {

      // ROS_INFO("%s(%d,%d)\n", "cloud_msg size: ", (*cld_global).width, (*cld_global).height);
      sensor_msgs::PointCloud2 const& cloud_msg = *cld_global;
      sensor_msgs::PointCloud2ConstIterator<float> iter_x(cloud_msg, "x");
      sensor_msgs::PointCloud2ConstIterator<float> iter_y(cloud_msg, "y");
      sensor_msgs::PointCloud2ConstIterator<float> iter_z(cloud_msg, "z");

      // Memory allocation issue
      size_t size = (*cld_global).width*(*cld_global).height;
      size_t memory_size = size * sizeof(float);

      checkCudaErrors(cudaMallocManaged((void **)&(_observation_list.front()._h_inx), memory_size));
      checkCudaErrors(cudaMallocManaged((void **)&(_observation_list.front()._h_iny), memory_size));
      checkCudaErrors(cudaMallocManaged((void **)&(_observation_list.front()._h_inz), memory_size));

      // ROS_INFO("%s%d%s\n", "Filtering:", size, " points");

      if (_initialized == 0){
        ask_for_memory(memory_size);
        ROS_INFO("%s\n", "CUDA memory reservation");
        _initialized = 1;
      }

      // Fill in the cloud data
      cloud_pcl.width  = cloud.width;
      cloud_pcl.height = cloud.height;
      cloud_pcl.points.resize(cloud.width * cloud.height);

      for(size_t i=0; i < size; ++i){
        _observation_list.front()._h_inx[i] = *(iter_x+i);
        _observation_list.front()._h_iny[i] = *(iter_y+i);
        _observation_list.front()._h_inz[i] = *(iter_z+i);
      }

      ROS_INFO("%s\n", "CUDA filter");
      p = filter(
          _observation_list.front()._h_inx,
          _observation_list.front()._h_iny,
          _observation_list.front()._h_inz,
          size, THREADS_PER_BLOCK);

      ROS_INFO("%s\n", "Fill pointcloud");
      // ROS_INFO("%s_%f,%f,%f\n", "Fill pointcloud", h_inx[0], h_iny[0], h_inz[0]);

      for (size_t i = 0; i < cloud_pcl.points.size(); ++i)
      {
        cloud_pcl.points[i].x = _observation_list.front()._h_inx[i];
        cloud_pcl.points[i].y = _observation_list.front()._h_iny[i];
        cloud_pcl.points[i].z = _observation_list.front()._h_inz[i];
      }

      // ROS_INFO("%s\n", "Convert pointcloud");
      pcl::toROSMsg(cloud_pcl, *output);

      // start = omp_get_wtime();
      // pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
      // sor.setInputCloud (cloud_pcl);
      // sor.setFilterFieldName("z");
      // sor.setFilterLimits(_min_obstacle_height,_max_obstacle_height);
      // sor.setDownsampleAllData(false);
      // sor.setLeafSize ((float)_voxel_size,
      //                  (float)_voxel_size,
      //                  (float)_voxel_size);
      // sor.setMinimumPointsNumberPerVoxel(static_cast<unsigned int>(_voxel_min_points));
      // sor.filter(*cloud_filtered);
      // end = omp_get_wtime();
      // elapsed = end-start;
      // ROS_INFO("%s%f\n", "Filtered time:", elapsed);
    }
    else
    {
      // start = omp_get_wtime();
      // pcl::PassThrough<pcl::PCLPointCloud2> pass_through_filter;
      // pass_through_filter.setInputCloud(cloud_pcl);
      // pass_through_filter.setKeepOrganized(false);
      // pass_through_filter.setFilterFieldName("z");
      // pass_through_filter.setFilterLimits( \
      //             _min_obstacle_height,_max_obstacle_height);
      // pass_through_filter.filter(*cloud_filtered);
      // end = omp_get_wtime();
      // elapsed = end-start;
      // ROS_INFO("%s%f\n", "Non filtered time:", elapsed);
    }

    // pcl_conversions::fromPCL(*cloud_filtered, *cld_global);
    _observation_list.front()._cloud = output;
  }
  catch (tf::TransformException& ex)
  {
    // if fails, remove the empty observation
    _observation_list.pop_front();
    ROS_ERROR( \
      "TF Exception for sensor frame: %s, cloud frame: %s, %s", \
      _sensor_frame.c_str(), cloud.header.frame_id.c_str(), ex.what());
    return;
  }

  _last_updated = ros::Time::now();
  RemoveStaleObservations();
}

/*****************************************************************************/
void MeasurementBuffer::GetReadings( \
                    std::vector<observation::MeasurementReading>& observations)
/*****************************************************************************/
{
  RemoveStaleObservations();

  for (readings_iter it = _observation_list.begin(); \
                                          it != _observation_list.end(); ++it)
  {
    observations.push_back(*it);
  }
}

/*****************************************************************************/
void MeasurementBuffer::RemoveStaleObservations(void)
/*****************************************************************************/
{
  if (_observation_list.empty())
  {
    return;
  }

  readings_iter it = _observation_list.begin();
  if (_observation_keep_time == ros::Duration(0.0))
  {

    readings_iter cuda_iter = _observation_list.begin();
    for (cuda_iter = _observation_list.begin(); cuda_iter != _observation_list.end(); ++cuda_iter)
    {
      cudaFree((*cuda_iter)._h_inx);
      cudaFree((*cuda_iter)._h_iny);
      cudaFree((*cuda_iter)._h_inz);
    }

    _observation_list.erase(++it, _observation_list.end());
    return;
  }

  for (it = _observation_list.begin(); it != _observation_list.end(); ++it)
  {
    observation::MeasurementReading& obs = *it;
    const ros::Duration time_diff = _last_updated - obs._cloud->header.stamp;

    if (time_diff > _observation_keep_time)
    {
      ROS_INFO("%s\n", "CUDA Memory free");

      _observation_list.erase(it, _observation_list.end());
      return;
    }
  }
}

/*****************************************************************************/
void MeasurementBuffer::ask_for_memory(size_t size)
/*****************************************************************************/
{
  checkCudaErrors(cudaMallocManaged((void **)&h_inx, size));
  checkCudaErrors(cudaMallocManaged((void **)&h_iny, size));
  checkCudaErrors(cudaMallocManaged((void **)&h_inz, size));
  ROS_INFO("%s%d\n", "add)observation list siz ", _observation_list.size());
  ROS_INFO("%s\n%p\n%p\n%p\n", "cuda ALLOCATE ointer: ",
      h_inx,
      h_iny,
      h_inz
      );
  // ROS_INFO("%s\n", "Memory reservation");
}

/*****************************************************************************/
void MeasurementBuffer::ResetAllMeasurements(void)
/*****************************************************************************/
{
  _observation_list.clear();
}

/*****************************************************************************/
bool MeasurementBuffer::ClearAfterReading(void)
/*****************************************************************************/
{
  return _clear_buffer_after_reading;
}

/*****************************************************************************/
bool MeasurementBuffer::UpdatedAtExpectedRate(void) const
/*****************************************************************************/
{
  if (_expected_update_rate == ros::Duration(0.0))
  {
    return true;
  }

  const ros::Duration update_time = ros::Time::now() - _last_updated;
  const bool current = update_time.toSec() <= _expected_update_rate.toSec();
  if (!current)
  {
    ROS_WARN_THROTTLE(10.,
      "%s buffer updated in %.2fs, it should be updated every %.2fs.",
      _topic_name.c_str(), update_time.toSec(), _expected_update_rate.toSec());
  }
  return current;
}

/*****************************************************************************/
bool MeasurementBuffer::IsEnabled(void) const
/*****************************************************************************/
{
  return _enabled;
}

/*****************************************************************************/
void MeasurementBuffer::SetEnabled(const bool& enabled)
/*****************************************************************************/
{
  _enabled = enabled;
}

/*****************************************************************************/
void MeasurementBuffer::ResetLastUpdatedTime(void)
/*****************************************************************************/
{
  _last_updated = ros::Time::now();
}

/*****************************************************************************/
void MeasurementBuffer::Lock(void)
/*****************************************************************************/
{
  _lock.lock();
}

/*****************************************************************************/
void MeasurementBuffer::Unlock(void)
/*****************************************************************************/
{
  _lock.unlock();
}

}  // namespace buffer
