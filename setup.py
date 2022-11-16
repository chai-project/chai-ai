from setuptools import setup

setup(
    name="chai-ai",
    packages=["chai_ai"],
    version="0.0.1",
    description="AI instance for the CHAI backend.",
    author="Kevin McAreavey, Kim Bauters",
    author_email="kevin.mcareavey@bristol.ac.uk, kim.bauters@bristol.ac.uk",
    license="Protected",
    install_requires=["pendulum",  # handle datetime instances with ease
                      "click",  # easy decorator style command line interface
                      "pg8000",  # pure Python PostgreSQL database adapter
                      "sqlalchemy",  # ORM for database access
                      "tomli",  # TOML configuration file parser
                      "tomli-w",  # TOML configuration file writer
                      "numpy",  # matrix operations for AI algorithm
                      "scipy",  # predictive confidence interval
                      ],
    extras_require={
        "compat": ["pylint", "perflint"],  # additional packages for code and performance checking
    },
    classifiers=[],
    include_package_data=True,
    platforms="any",

)
