from setuptools import find_packages, setup

package_name = 'mcap_bag_processor'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'mcap',
        'mcap-ros2-support',
        'numpy',
        'pyyaml',
    ],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='Offline MCAP bag processor for adding tf_static, camera_info, and filtering pointclouds',
    license='MIT',
    entry_points={
        'console_scripts': [
            'mcap_processor = mcap_bag_processor.main:main',
        ],
    },
)

