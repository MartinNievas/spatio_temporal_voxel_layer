#include <GL/freeglut.h>

//PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>

#include "cudaWrapper.h"

const unsigned int window_width  = 512;
const unsigned int window_height = 512;
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -20.0;
float translate_x, translate_y = 0.0;
float resolution = 0.5f;

pcl::PointCloud<pcl::PointXYZ> point_cloud;
CCudaWrapper cudaWrapper;

int main(int argc, char **argv)
{
  std::cout << "Lesson 1 - downsampling" << std::endl;

  if(argc < 2)
  {
    std::cout << "Usage:\n";
    std::cout << argv[0] <<" point_cloud_file.pcd parameters\n";
    std::cout << "-res resolution: default " << resolution << std::endl;

    std::cout << "Default:  ../../data/scan_Velodyne_VLP16.pcd\n";

    if(pcl::io::loadPCDFile("../../data/scan_Velodyne_VLP16.pcd", point_cloud) == -1)
    {
      return -1;
    }
  } else
  {
    std::vector<int> ind_pcd;
    ind_pcd = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");

    if(ind_pcd.size()!=1)
    {
      std::cout << "did you forget pcd file location? return" << std::endl;
      return -1;
    }

    if(pcl::io::loadPCDFile(argv[1], point_cloud) == -1)
    {
      return -1;
    }

    pcl::console::parse_argument (argc, argv, "-res", resolution);
    std::cout << "resolution for downsampling: " << resolution << std::endl;
  }

  cudaWrapper.warmUpGPU();

  clock_t begin_time;
  double computation_time;
  begin_time = clock();

  if(!cudaWrapper.downsampling(point_cloud, resolution))
  {
    cudaDeviceReset();
    std::cout << "cudaWrapper.downsampling NOT SUCCESFULL" << std::endl;
  }

  computation_time=(double)( clock () - begin_time ) /  CLOCKS_PER_SEC;
  std::cout << "cudaWrapper.downsampling computation_time: " << computation_time << std::endl;


  return 0;
}
