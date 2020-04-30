import setuptools

setuptools.setup(
    name="traffic_lights",
    version="0.0.1",
    author="Aidan Dunlop",
    author_email="aidandunlop@gmail.com",
    description="Traffic Light recognition",
    long_description='Traffic Light recognition using the LISA dataset',
    url="https://github.com/aidandunlop/traffic_light_recognition",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)