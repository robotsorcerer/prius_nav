/*
* Author: Harley Wiltzer
* New York, NY
* July 2023
*/

#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <iostream>
#include <mutex>

#include <pcl/features/normal_3d.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

#define LASER_TOPIC "/prius/center_laser/scan"
#define BUF_SIZE 1000

class laser_clouds {
private:
    pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloud;

    bool updateCloud;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

public:
    laser_clouds();

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewerCreator();

    void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event,
                               void* viewer_void);

    void laserCallback(const sensor_msgs::PointCloud::ConstPtr& msg) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloud (new pcl::PointCloud<pcl::PointXYZ>);

        readCloud(*msg, pclCloud);

        if(ros::ok() && !viewer->wasStopped())
        {
            viewer->setSize(400, 400);
            viewer->addPointCloud<pcl::PointXYZ>(this->pclCloud, "laser_cloud");
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "laser_cloud");
            viewer->spinOnce(10);
            boost::this_thread::sleep(boost::posix_time::microseconds(100));

            if(updateCloud)
            {
                viewer->removePointCloud("laser_cloud");
                viewer->updatePointCloud(this->pclCloud, "laser_cloud");
            }
        }
        updateCloud = true;
    }

    void readCloud(const sensor_msgs::PointCloud sensorCloud, pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloud)
    {
        pcl::PointXYZ points;
        for(auto it=sensorCloud.points.begin(); it!=sensorCloud.points.end(); ++it)
        {
            points.x = (*it).x;
            points.y = (*it).y;
            points.z = (*it).z;
            pclCloud->points.push_back(points);
        }
        this->pclCloud = pclCloud;
    }
};

laser_clouds::laser_clouds()
    : updateCloud(false){
        viewer = laser_clouds::viewerCreator();
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> laser_clouds::viewerCreator()
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer("Laser Scans 2D"));
    viewer->setBackgroundColor(0.2, 0.3, 0.3);
    viewer->addCoordinateSystem(1.0);
    viewer->setSize(400, 400);
    viewer->initCameraParameters();
    viewer->registerKeyboardCallback(&laser_clouds::keyboardEventOccurred, *this);
    return viewer;
}

unsigned int text_id = 0;

void laser_clouds::keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void* viewer_void)
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
    if (event.getKeySym() == "r" && event.keyDown())
    {
        std::cout << "r was pressed => removing all text" << std::endl;

        char str[512];
        for (unsigned int i=0; i < text_id; ++i)
        {
            sprintf(str, "text#%03d", i);
            viewer->removeShape(str);
        }
        text_id = 0;
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laser_scans");
    ros::NodeHandle n_laser;
    laser_clouds ls;
    ros::Subscriber sub = n_laser.subscribe(LASER_TOPIC, BUF_SIZE, &laser_clouds::laserCallback, &ls);
    ros::spin();
    return 0;
}