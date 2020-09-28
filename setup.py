import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tiempo_deshima", 
    version="0.1.0",
    author="Esmee Huijten (edits: Stefanie Brackenhoff)",
    maintainer= "Stefanie Brackenhoff <s.a.brackenhoff@student.tudelft.nl>",
    description="time-dependent end-to-end model for post-process optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Stefanie-B/DESHIMAmodel",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    package_data={"tiempo": ['Data/splines_Tb_sky/*.npy', 'DESHIMA/desim/*', 'DESHIMA/desim/data/*', 'DESHIMA/desim/lines/*', 'DESHIMA/MKID/*']},
)