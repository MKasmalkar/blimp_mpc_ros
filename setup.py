from setuptools import find_packages, setup

package_name = 'blimp_mpc_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mihir',
    maintainer_email='mihir@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'origin_lqr = blimp_mpc_ros.origin_lqr:main',
            'blimp_test = blimp_mpc_ros.blimp_test:main',
            'mpc_helix = blimp_mpc_ros.mpc_helix:main',
            'fdbk_lin_helix = blimp_mpc_ros.fdbk_lin_helix:main',
            'blimp_sim = blimp_mpc_ros.blimp_sim:main',
            'origin_pid = blimp_mpc_ros.origin_pid:main',
            'motion_test = blimp_mpc_ros.motion_test:main',
            'cs_nlp_origin = blimp_mpc_ros.cs_nlp_origin:main',
            'mpc_origin = blimp_mpc_ros.mpc_origin:main'
        ],
    },
)
