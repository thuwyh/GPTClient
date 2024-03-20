from setuptools import setup, find_packages

setup(
    name='GPTClient',
    version='0.1.0',
    url='https://github.com/thuwyh/GPTClient',
    author='thuwyh',
    author_email='wuyhthu@gmail.com',
    description='An Efficient GPT API client with cache.',
    packages=find_packages(),    
    install_requires=[
        "openai",
        "pydantic",
        "diskcache",
        "coloredlogs",
        "platformdirs"
    ],
)
