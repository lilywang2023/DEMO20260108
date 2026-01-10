import voxelmorph as vxm
print(vxm.__file__)  # 查看实际加载路径
print(dir(vxm))      # 看是否有 'networks'
print(hasattr(vxm, 'networks'))  # 应该是 True