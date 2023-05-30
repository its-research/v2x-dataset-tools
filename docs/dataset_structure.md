# Dataset Structure

```txt
single-infrastructure-side              # DAIR-V2X-I Dataset
    ├── image
        ├── {id}.jpg
    ├── velodyne
        ├── {id}.pcd
    ├── calib
        ├── camera_intrinsic
            ├── {id}.json
        ├── virtuallidar_to_camera
            ├── {id}.json
    ├── label
        ├── camera                      # Labeled data in Infrastructure Virtual LiDAR Coordinate System fitting objects in image based on image frame time
            ├── {id}.json
        ├── virtuallidar                # Labeled data in Infrastructure Virtual LiDAR Coordinate System fitting objects in point cloud based on point cloud frame time
            ├── {id}.json
    ├── data_info.json                  # Relevant index information of the Infrastructure data
    ├── trajectories
        ├── {scene_id}.csv
    ├── traffic-light
        ├── {scene_id}.csv
single-vehicle-side                     # DAIR-V2X-V
    ├── image
        ├── {id}.jpg
    ├── velodyne
        ├── {id}.pcd
    ├── calib
        ├── camera_intrinsic
            ├── {id}.json
        ├── lidar_to_camera
            ├── {id}.json
    ├── label
        ├── camera                      # Labeled data in Vehicle LiDAR Coordinate System fitting objects in image based on image frame time
            ├── {id}.json
        ├── lidar                       # Labeled data in Vehicle LiDAR Coordinate System fitting objects in point cloud based on point cloud frame time
            ├── {id}.json
    ├── data_info.json                  # Relevant index information of the Vehicle data
    ├── trajectories
        ├── {scene_id}.csv
cooperative-vehicle-infrastructure      # DAIR-V2X-C
    ├── infrastructure-trajectories     # Trajectory forecasting dataset with infrastructure view
        ├── {scene_id}.csv
    ├── vehicle-trajectories            # Trajectory forecasting dataset with vehicle view
        ├── {scene_id}.csv
    ├── cooperative-trajectories        # Trajectory forecasting dataset with vehicle-infrastructure cooperative view
        ├── {scene_id}.csv
    ├── traffic-light
        ├── {scene_id}.csv
    ├── infrastructure-side             # DAIR-V2X-C-I
        ├── image
            ├── {id}.jpg
        ├── velodyne
            ├── {id}.pcd
        ├── calib
            ├── camera
                ├── {id}.json
            ├── virtuallidar_to_world
                ├── {id}.json
            ├── virtuallidar_to_camera
                ├── {id}.json
        ├── label
            ├── camera                  # Labeled data in Infrastructure Virtual LiDAR Coordinate System fitting objects in image based on image frame time
                ├── {id}.json
            ├── virtuallidar            # Labeled data in Infrastructure Virtual LiDAR Coordinate System fitting objects in point cloud based on point cloud frame time
                ├── {id}.json
        ├── data_info.json              # Relevant index information of Infrastructure data
    ├── vehicle-side                    # DAIR-V2X-C-V
        ├── image
            ├── {id}.jpg
        ├── velodyne
            ├── {id}.pcd
        ├── calib
            ├── camera_intrinsic
                ├── {id}.json
            ├── lidar_to_camera
                ├── {id}.json
            ├── lidar_to_novatel
                ├── {id}.json
            ├── novatel_to_world
                ├── {id}.json
        ├── label
            ├── camera                  # Labeled data in Vehicle LiDAR Coordinate System fitting objects in image based on image frame time
                ├── {id}.json
            ├── lidar                   # Labeled data in Vehicle LiDAR Coordinate System fitting objects in point cloud based on point cloud frame time
                ├── {id}.json
        ├── data_info.json              # Relevant index information of the Vehicle data
    ├── cooperative                     # Coopetative Files
        ├── label_world                 # Vehicle-Infrastructure Cooperative (VIC) Annotation files
            ├── {id}.json
        ├── data_info.json              # Relevant index information combined the Infrastructure data and the Vehicle data
V2X-Seq-SPD                             # sequential percpetion dataset
└──  infrastructure-side            # Infrastructure-side data
        ├── image
            ├── {id}.jpg
        ├── velodyne
            ├── {id}.pcd
        ├── calib
            ├── camera_intrinsic        # Camera intrinsic parameter
                ├── {id}.json
            ├── virtuallidar_to_world   # Extrinsic parameter from virtual LiDAR coordinate system to world coordinate system
                ├── {id}.json
            ├── virtuallidar_to_camera  # Extrinsic parameter from virtual LiDAR coordinate system to camera coordinate system
                ├── {id}.json
        ├── label
            ├── camera                  # Labeles in infrastructure virtual LiDAR coordinate system fitting objects in image with image camptured timestamp
                ├── {id}.json
            ├── virtuallidar            # Labeles in infrastructure virtual LiDAR coordinate system fitting objects in point cloud with point cloud captured timestamp
                ├── {id}.json
        └── data_info.json              # More detailed information for each infrastructure-side frame
    └── vehicle-side                    # Vehicle-side data
        ├── image
            ├── {id}.jpg
        ├── velodyne
            ├── {id}.pcd
        ├── calib
            ├── camera_intrinsic        # Camera intrinsic parameter
                ├── {id}.json
            ├── lidar_to_camera         # extrinsic parameter from LiDAR coordinate system to camera coordinate system
                ├── {id}.json
            ├── lidar_to_novatel        # extrinsic parameter from LiDAR coordinate system to NovAtel coordinate system
                ├── {id}.json
            ├── novatel_to_world        # location in the world coordinate system
                ├── {id}.json
        ├── label
            ├── camera                  # Labeles in vehicle LiDAR coordinate system fitting objects in image with image camptured timestamp
                ├── {id}.json
            ├── lidar                   # Labeles in vehicle LiDAR coordinate system fitting objects in point cloud with point cloud captured timestamp
                ├── {id}.json
        └── data_info.json              # More detailed information for each vehicle-side frame
    └── cooperative                     # Coopetative-view files
        ├── label                       # Vehicle-infrastructure cooperative (VIC) annotation files. Labeles in vehicle LiDAR coordinate system with the vehicle point cloud timestamp
            ├── {id}.json
        └── data_info.json              # More detailed information for vehicle-infrastructure cooperative frame pair
    └── maps                            # HD Maps for each intersection
```

```txt
            virtuallidar2world
inf lidar ------------------------> world coordinate <===== label_world
              补充   relative error      ^
                                         |
                                         |
                                         |
                                         |
                                         | novatel2world
                                         |
                                         |
                                         |
                                         |
                 lidar2novatel           |
veh lidar --------------------------> novatel

   (x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)


                        front z
                             /
                            /
              (x0, y0, z1) + -----------  + (x1, y0, z1)
                          /|            / |
                         / |           /  |
           (x0, y0, z0) + ----------- +   + (x1, y1, z1)
                        |  /      .   |  /
                        | / oriign    | /
           (x0, y1, z0) + ----------- + -------> x right
                        |             (x1, y1, z0)
                        |
                        v
                   down y

```
