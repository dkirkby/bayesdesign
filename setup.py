import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bayesdesign",
    author="David Kirkby",
    author_email="dkirkby@uci.edu",
    description="Package for Bayesian optimal experimental design",
    keywords="bayes,optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dkirkby/bayesdesign",
    project_urls={
        "Documentation": "https://github.com/dkirkby/bayesdesign",
        "Bug Reports": "https://github.com/dkirkby/bayesdesign/issues",
        "Source Code": "https://github.com/dkirkby/bayesdesign",
        # 'Funding': '',
        # 'Say Thanks!': '',
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    classifiers=[
        # see https://pypi.org/classifiers/
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    # install_requires=['Pillow'],
    extras_require={
        "dev": ["check-manifest"],
        # 'test': ['coverage'],
    },
    # entry_points={
    #     'console_scripts': [  # This can provide executable scripts
    #         'run=bed:main',
    # You can execute `run` in bash to run `main()` in src/bed/__init__.py
    #     ],
    # },
)
