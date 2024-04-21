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
            'run_cbf_line       = blimp_mpc_ros.run_ctrl_cbf_line:main',
            'run_cbf_triangle   = blimp_mpc_ros.run_ctrl_cbf_triangle:main',
            'run_cbf_helix      = blimp_mpc_ros.run_ctrl_cbf_helix:main',
            'run_fbl_line       = blimp_mpc_ros.run_ctrl_fbl_line:main',
            'run_fbl_triangle   = blimp_mpc_ros.run_ctrl_fbl_triangle:main',
            'run_fbl_helix      = blimp_mpc_ros.run_ctrl_fbl_helix:main',
            'run_pid_line       = blimp_mpc_ros.run_ctrl_pid_line:main',
            'run_pid_triangle   = blimp_mpc_ros.run_ctrl_pid_triangle:main',
            'run_pid_helix      = blimp_mpc_ros.run_ctrl_pid_helix:main',
            'run_blimp_sim = blimp_mpc_ros.run_blimp_sim:main',
            'run_blimp_data = blimp_mpc_ros.run_blimp_data:main'
        ],
    },
)
