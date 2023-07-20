#include <thread>
#include <mutex>
#include <memory>
#include <chrono>
#include <iostream>
#include <cmath>

#include <ros/ros.h>
#include <ros/spinner.h>
#include <cv_bridge/cv_bridge.h>

#include <pcl/conversions.h>
#include <pcl/point_types.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
// #include <pcl/visualization/pcl_visualizer.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/LaserScan.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <laser_geometry/laser_geometry.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

/*aliases*/
using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;
// using pcl_viz = pcl::visualization::PCLVisualizer;

class Receiver
{
private:
  /*aliases*/
  using imageMsgSub = message_filters::Subscriber<sensor_msgs::Image>;
  using cloudMsgSub = message_filters::Subscriber<sensor_msgs::PointCloud>;
  using laserMsgSub = message_filters::Subscriber<sensor_msgs::LaserScan>;
  using syncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, 
                                sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image, 
                                sensor_msgs::LaserScan, sensor_msgs::PointCloud, sensor_msgs::LaserScan>;
  bool running, updateCloud, updateImage, updateLaser, save;
  size_t counter;
  std::ostringstream oss;

  const std::string cloudName;
  pcl::PCDWriter writer;

  ros::NodeHandle nh;
  std::mutex mutex;
  cv::Mat frontImage, leftImage, rightImage, backImage;
  std::string windowName;
//   PointCloudT cloud;
  sensor_msgs::LaserScan fl_scan, fr_scan;

  unsigned long const hardware_threads;
  ros::AsyncSpinner spinner;
  std::string subNameImageFront, subNameImageLeft, subNameImageRight, subNameImageBack;
  std::string subNameLaserFrontLeft, subNameLaserFrontRight, subNameLaserCenter;

  imageMsgSub subImageFront, subImageBack, subImageLeft, subImageRight;
  cloudMsgSub subCloudCenter;
  laserMsgSub subCloudFrontLeft, subCloudFrontRight;

  std::vector<std::thread> threads;

//   boost::shared_ptr<pcl_viz> viewer;
  message_filters::Synchronizer<syncPolicy> sync;

  laser_geometry::LaserProjection laser_projector;

//   boost::shared_ptr<visualizer> viz;
//   bool showLeftImage, showRightImage;

public:
  //constructor
  Receiver()
  : updateCloud(false), updateImage(false), save(false), counter(0),
  cloudName("detector_cloud"), windowName("Laser and image data"), basetopic("/prius"),
  suffix_cam_topic("_camera/image_raw"), suffix_laser_topic("_laser/scan"), hardware_threads(std::thread::hardware_concurrency()),  
  subNameImageFront(basetopic + "/front" + suffix_cam_topic), subNameImageBack(basetopic + "/back" + suffix_cam_topic),
  subNameImageLeft(basetopic + "/left" + suffix_cam_topic), subNameImageRight(basetopic + "/right" + suffix_cam_topic),
  subNameLaserFrontLeft(basetopic + "/front_left" + suffix_laser_topic),
  subNameLaserCenter(basetopic + "/center" + suffix_laser_topic),
  subNameLaserFrontRight(basetopic + "/front_right" + suffix_laser_topic),
  spinner(hardware_threads/2),
  subImageFront(nh, subNameImageFront, 1), subImageBack(nh, subNameImageBack, 1),
  subImageLeft(nh, subNameImageLeft, 1), subImageRight(nh, subNameImageRight, 1),
  subCloudCenter(nh, subNameLaserCenter, 1),
  subCloudFrontLeft(nh, subNameLaserFrontLeft, 1),
  subCloudFrontRight(nh, subNameLaserFrontRight, 1),
  sync(syncPolicy(10), subImageFront, subImageBack, subImageLeft, subImageRight,
                       subCloudFrontLeft, subCloudCenter, subCloudFrontRight)
  {
    sync.registerCallback(boost::bind(&Receiver::callback, this, _1, _2, _3, _4, _5, _6, _7));
    ROS_INFO_STREAM("#Hardware Concurrency: " << hardware_threads <<
      "\t. Spinning with " << hardware_threads/4 << " threads");
    ROS_INFO("I am running!!!");
  }
  //destructor
  ~Receiver()
  {
    // viz.reset();
    // viewer.reset();
  }

  Receiver(Receiver const&) =delete;
  Receiver& operator=(Receiver const&) = delete;

private:
    const std::string basetopic; 
    const std::string suffix_cam_topic;
    const std::string  suffix_laser_topic; 
    // global class image and cloud frames
    cv::Mat frontIm, leftIm, rightIm, backIm;
    PointCloudT centerCloud, frontLeftCloud, frontRightCloud;
  
  void run()
  {
    begin();
    end();
  }
private:
  void begin()
  {
    if(spinner.canStart())
    {
      spinner.start();
    }
    running = true;
    while(!updateImage || !updateCloud || !updateLaser)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // //spawn the threads
    // threads.push_back(std::thread(&Receiver::cloudDisp, this));
    // threads.push_back(std::thread(&Receiver::imageDisp, this));
    // //call join on each thread in turn
    // std::for_each(threads.begin(), threads.end(), \
    //               std::mem_fn(&std::thread::join));
  }

  void end()
  {
    spinner.stop();
    running = false;
  }

  void callback(const sensor_msgs::ImageConstPtr& frontImage, const sensor_msgs::ImageConstPtr& backImage,
                const sensor_msgs::ImageConstPtr& leftImage, const sensor_msgs::ImageConstPtr& rightImage,
                const sensor_msgs::LaserScanPtr& frontLeftScan, const sensor_msgs::PointCloudPtr& centerScan,
                const sensor_msgs::LaserScanPtr& frontRightScan)
  {
    cv::Mat frontIm, leftIm, rightIm, backIm;
    PointCloudT centerCloud, frontLeftCloud, frontRightCloud;
    
    getImage(frontImage, frontIm);
    getImage(leftImage, leftIm);
    getImage(rightImage, rightIm);
    getImage(backImage, backIm);

    getCloud(centerScan, centerCloud);

    sensor_msgs::PointCloud _frontLeftCloud, _frontRightCloud;

    projectScan(frontLeftScan, _frontLeftCloud);
    projectScan(frontRightScan, _frontRightCloud);

    getCloud(_frontLeftCloud, frontLeftCloud);
    getCloud(_frontRightCloud, frontRightCloud);

    std::lock_guard<std::mutex> lock(mutex);
    this->frontIm = frontIm;
    this->backIm = backIm;
    this->leftIm = leftIm;
    this->rightIm = rightIm;
    
    this->centerCloud = centerCloud;
    this->frontLeftCloud = frontLeftCloud;
    this->frontRightCloud = frontRightCloud;
    // if(showLeftImage)
    //   this->leftIr = leftIr;
    // if(showRightImage)
    //   this->rightIr = rightIr;
    updateImage = true;
    updateCloud = true;
    updateLaser = true;
  }

  void getImage(const sensor_msgs::ImageConstPtr msgImage, cv::Mat &image) const
  {
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvShare(msgImage, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    cv_ptr->image.copyTo(image);
  }

  void getCloud(const sensor_msgs::PointCloudConstPtr cb_cloud, PointCloudT& pcl_cloud) const
  {
    pcl::PointXYZ points;
    for(auto it=cb_cloud.points.begin(); it!=cb_cloud.points.end(); ++it)
    {
        points.x = (*it).x;
        points.y = (*it).y;
        points.z = (*it).z;
        pcl_cloud->points.push_back(points);
    }
  }

  void projectScan(const sensor_msgs::LaserScanPtr scan_in, \
                   sensor_msgs::PointCloudPtr& cloud_out)
  { 
    laser_projector.projectLaser(scan_in, cloud_out);
  }

/*
  void imageDisp()
  {
    cv::Mat ir, leftIr, rightIr;
    PointCloudT cloud;
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, 640, 480) ;

    const std::string leftWindowName = "leftImage";
    const std::string rightWindowName = "rightImage";

    //left image window
    if(showLeftImage)
    {
      cv::namedWindow(leftWindowName, cv::WINDOW_NORMAL);
      cv::resizeWindow(leftWindowName, 640, 480);
    }

    //right image window
    if(showRightImage)
    {
      cv::namedWindow(rightWindowName, cv::WINDOW_NORMAL);
      cv::resizeWindow(rightWindowName, 640, 480);
    }

    for(; running && ros::ok();)
    {
      if(updateImage)
      {
        std::lock_guard<std::mutex> lock(mutex);
        ir = this->ir;
        //detect and display features
        detectAndDisplay(ir);

        if(showLeftImage)
          leftIr = this->leftIr;
        if(showRightImage)
          rightIr = this->rightIr;
        cloud = this->cloud;
        updateImage = false;

        cv::imshow(windowName, ir);
        if(showLeftImage)
          cv::imshow(leftWindowName, leftIr);
        if(showRightImage)
          cv::imshow(rightWindowName, rightIr);
      }

      int key = cv::waitKey(1);
      switch(key & 0xFF)
      {
        case 27:
        case 'q':
          running = false;
          break;
        case ' ':
        case 's':
          // saveCloudAndImage(cloud, ir);
          save = true;
          break;
      }
    }
    cv::destroyAllWindows();
    cv::waitKey(100);
  }
*/
/*
  void cloudDisp()
  {
    viz = boost::shared_ptr<visualizer> (new visualizer());
    viewer = boost::shared_ptr<pcl_viz> (new pcl_viz);
    viewer= viz->createViewer();

    // PointCloudT cloud  = this->cloud;
    PointCloudT::Ptr cloud_ptr (&this->cloud);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> color_handler (cloud_ptr, 255, 150, 155);
    cv::Mat ir = this->ir;
    viewer->addPointCloud(cloud_ptr, color_handler, cloudName);

    for(; running && ros::ok() ;)
    {
      if(updateCloud)
      {
        std::lock_guard<std::mutex> lock(mutex);
        updateCloud = false;
        viewer->updatePointCloud(cloud_ptr, color_handler, cloudName);
      }
      if(save)
      {
        save = false;
      }
      // ROS_INFO_STREAM("facFeature: " << cloud_ptr->points(std::get<0>(faceFeature), std::get<1>(faceFeature)));
      viewer->spinOnce(10);
    }
    viewer->close();
  }
  */
};


int main(int argc, char** argv)
{
  ros::init(argc, argv, "ensensor_face_detection_node", ros::init_options::AnonymousName);

  ROS_INFO_STREAM("Started node " << ros::this_node::getName().c_str());

  Receiver r;
  r.run();

  if(!ros::ok())
  {
    return 0;
  }
  ros::shutdown();
}

