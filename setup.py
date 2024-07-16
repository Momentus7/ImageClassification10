from setuptools import setup, find_packages

setup(
    name='image_classification',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'scikit-learn',
        'opencv-python',
        'tensorflow',
        'keras',
        'flask',
        'flask-restful',
    ],
)
