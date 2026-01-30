from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'cobot2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/resource', glob('resource/*') + ['resource/.env']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='june',
    maintainer_email='mhn06005@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'auth_action = cobot2.auth_action_server:main',
        ],
    },
)
