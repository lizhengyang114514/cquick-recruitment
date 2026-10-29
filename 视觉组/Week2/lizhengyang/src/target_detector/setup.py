from glob import glob

from setuptools import setup

package_name = "target_detector"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/test_images", glob("test_images/*.jpg")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="lizhengyang",
    maintainer_email="lizhengyang@example.com",
    description="颜色目标检测并发布目标位置",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "publisher = target_detector.publisher:main",
            "subscriber = target_detector.subscriber:main",
        ],
    },
)
