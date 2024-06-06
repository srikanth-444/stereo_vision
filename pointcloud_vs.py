from open3d import * 


while True:
    cloud = io.read_point_cloud("pointcloud.ply") # Read point cloud
    visualization.draw_geometries([cloud]) 